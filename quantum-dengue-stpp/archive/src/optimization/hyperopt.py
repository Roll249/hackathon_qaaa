"""
Hyperparameter optimization using Optuna for CNN-LSTM v2.

Optimizes model architecture and training hyperparameters
to maximize forecasting performance.
"""
import optuna
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CNNLSTMObjective:
    """
    Optuna objective function for CNN-LSTM v2 hyperparameter optimization.

    Searches over:
    - Architecture: conv_channels, lstm_hidden, lstm_layers, dropout
    - Training: lr, batch_size, seq_len
    - Spatial: grid_size
    - Loss: mse, poisson, nb
    """

    def __init__(
        self,
        train_data: tuple,
        val_data: tuple,
        device: str = "cpu",
        metric: str = "R2",
        seed: int = 42,
        max_epochs: int = 50,
    ):
        """
        Args:
            train_data: (X_train, y_train) tuples
            val_data: (X_val, y_val) tuples
            device: 'cpu' or 'cuda'
            metric: optimization metric ('R2', 'RMSE', 'neg_MSE')
            seed: random seed
            max_epochs: max training epochs per trial
        """
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.device = device
        self.metric = metric
        self.seed = seed
        self.max_epochs = max_epochs

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial."""
        from ..models.cnn_lstm_v2 import SpatioTemporalCNNv2, train_cnn_lstm_v2

        config = {
            'conv_channels': trial.suggest_categorical(
                'conv_channels',
                [
                    [32, 64],
                    [32, 64, 128],
                    [64, 128],
                    [64, 128, 256],
                ]
            ),
            'lstm_hidden': trial.suggest_categorical(
                'lstm_hidden',
                [64, 128, 256]
            ),
            'lstm_layers': trial.suggest_int(
                'lstm_layers',
                1, 3
            ),
            'dropout': trial.suggest_float(
                'dropout',
                0.1, 0.5,
                step=0.1
            ),
            'grid_size': trial.suggest_categorical(
                'grid_size',
                [24, 32, 48]
            ),
            'bidirectional': trial.suggest_categorical(
                'bidirectional',
                [True, False]
            ),
            'use_attention': trial.suggest_categorical(
                'use_attention',
                [True, False]
            ),
        }

        lr = trial.suggest_float(
            'lr',
            1e-5, 1e-2,
            log=True
        )

        batch_size = trial.suggest_categorical(
            'batch_size',
            [16, 32, 64]
        )

        loss_name = trial.suggest_categorical(
            'loss_name',
            ['mse', 'poisson', 'nb']
        )

        patience = trial.suggest_int(
            'patience',
            5, 20
        )

        try:
            model = SpatioTemporalCNNv2(
                input_channels=1,
                conv_channels=config['conv_channels'],
                lstm_hidden=config['lstm_hidden'],
                lstm_layers=config['lstm_layers'],
                dropout=config['dropout'],
                grid_size=config['grid_size'],
                forecast_horizon=1,
                bidirectional=config['bidirectional'],
                use_attention=config['use_attention'],
                loss=loss_name,
            )

            train_ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.X_train),
                torch.FloatTensor(self.y_train)
            )
            val_ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.X_val),
                torch.FloatTensor(self.y_val)
            )
            train_ld = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True
            )
            val_ld = torch.utils.data.DataLoader(
                val_ds,
                batch_size=batch_size
            )

            model = train_cnn_lstm_v2(
                model=model,
                train_loader=train_ld,
                val_loader=val_ld,
                epochs=self.max_epochs,
                lr=lr,
                device=self.device,
                patience=patience,
                seed=self.seed,
                verbose=False,
                loss_name=loss_name,
            )

            model.eval()
            with torch.no_grad():
                X_v = torch.FloatTensor(self.X_val).to(self.device)
                y_pred = model(X_v).cpu().numpy().flatten()

            from ..evaluation.metrics import compute_forecasting_metrics
            metrics = compute_forecasting_metrics(
                self.y_val.flatten(),
                y_pred
            )

            if self.metric == "R2":
                return metrics.get("R2", -float("inf"))
            elif self.metric == "RMSE":
                return -metrics.get("RMSE", float("inf"))
            else:
                return metrics.get(self.metric, -float("inf"))

        except Exception as e:
            warnings.warn(f"Trial failed: {e}")
            return -float("inf")


def optimize_cnn_lstm(
    train_data: tuple,
    val_data: tuple,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    device: str = "cpu",
    metric: str = "R2",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    seed: int = 42,
    max_epochs: int = 50,
    load_if_exists: bool = True,
) -> optuna.Study:
    """
    Run hyperparameter optimization for CNN-LSTM v2.

    Args:
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        n_trials: number of Optuna trials
        timeout: timeout in seconds
        device: 'cpu' or 'cuda'
        metric: optimization metric
        study_name: Optuna study name
        storage: Optuna storage URL (e.g., sqlite:///study.db)
        seed: random seed
        max_epochs: max epochs per trial
        load_if_exists: load existing study if found

    Returns:
        Optuna study with results
    """
    sampler = optuna.samplers.TPESampler(seed=seed)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize" if metric == "R2" else "minimize",
        load_if_exists=load_if_exists,
    )

    objective = CNNLSTMObjective(
        train_data=train_data,
        val_data=val_data,
        device=device,
        metric=metric,
        seed=seed,
        max_epochs=max_epochs,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    return study


def get_best_config(study: optuna.Study) -> Dict[str, Any]:
    """Extract best configuration from Optuna study."""
    best = study.best_params

    config = {
        'model': {
            'conv_channels': best.get('conv_channels'),
            'lstm_hidden': best.get('lstm_hidden'),
            'lstm_layers': best.get('lstm_layers'),
            'dropout': best.get('dropout'),
            'grid_size': best.get('grid_size'),
            'bidirectional': best.get('bidirectional'),
            'use_attention': best.get('use_attention'),
        },
        'training': {
            'lr': best.get('lr'),
            'batch_size': best.get('batch_size'),
            'patience': best.get('patience'),
            'loss_name': best.get('loss_name'),
        },
        'optimization': {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number,
        }
    }

    return config


def save_study_results(
    study: optuna.Study,
    output_path: Path,
    include_trials: bool = True
) -> None:
    """Save Optuna study results to JSON."""
    results = {
        'study_name': study.study_name,
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'n_completed': sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
    }

    if include_trials:
        trials_data = []
        for trial in study.trials:
            if trial.value is not None:
                trials_data.append({
                    'number': trial.number,
                    'value': float(trial.value),
                    'params': trial.params,
                    'state': str(trial.state),
                })
        results['trials'] = trials_data

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


class MultiModelOptimizer:
    """
    Optimize multiple models and select the best.

    Tests:
    - CNN-LSTM v1 (baseline)
    - CNN-LSTM v2 (with attention)
    - NEST (with fixes)
    """

    def __init__(
        self,
        train_data: tuple,
        val_data: tuple,
        device: str = "cpu",
        metric: str = "R2",
        seed: int = 42,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.metric = metric
        self.seed = seed
        self.results = {}

    def optimize_all(self, n_trials: int = 50) -> Dict[str, Any]:
        """Run optimization for all model types."""

        print("Optimizing CNN-LSTM v1...")
        cnn_lstm_v1 = optimize_cnn_lstm(
            self.train_data,
            self.val_data,
            n_trials=n_trials,
            device=self.device,
            metric=self.metric,
            study_name="cnn_lstm_v1",
            load_if_exists=False,
        )
        self.results['cnn_lstm_v1'] = {
            'study': cnn_lstm_v1,
            'best_value': cnn_lstm_v1.best_value,
            'best_params': get_best_config(cnn_lstm_v1),
        }

        print("Optimizing CNN-LSTM v2...")
        cnn_lstm_v2 = optimize_cnn_lstm(
            self.train_data,
            self.val_data,
            n_trials=n_trials,
            device=self.device,
            metric=self.metric,
            study_name="cnn_lstm_v2",
            load_if_exists=False,
        )
        self.results['cnn_lstm_v2'] = {
            'study': cnn_lstm_v2,
            'best_value': cnn_lstm_v2.best_value,
            'best_params': get_best_config(cnn_lstm_v2),
        }

        best_model = max(
            self.results.keys(),
            key=lambda k: self.results[k]['best_value']
        )

        return {
            'best_model': best_model,
            'best_value': self.results[best_model]['best_value'],
            'all_results': self.results,
        }
