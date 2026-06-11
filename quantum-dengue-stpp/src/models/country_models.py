"""
Country-specific models for dengue forecasting.

Different countries have different dynamics:
- Vietnam: High zero-inflation (31%), clustered
- Indonesia: Archipelago, strong clustering
- Singapore: Spatially regular, time-series focus
- Thailand: Largest dataset (63%), generalizable
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class CountrySpecificConfig:
    """Configuration for country-specific model parameters."""

    CONFIGS = {
        'VIET NAM': {
            'name': 'Vietnam',
            'characteristics': {
                'zero_inflation': 0.31,
                'clustering': 'high',
                'spatial_regularity': 'low',
            },
            'model': {
                'conv_channels': [32, 64, 128],
                'lstm_hidden': 128,
                'lstm_layers': 3,
                'dropout': 0.4,
                'loss': 'nb',
                'grid_size': 32,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 7, 'augmentation_factor': 2.0},
                'quantum': {'n_qubits': 10, 'augmentation_ratio': 4},
            },
        },
        'THAILAND': {
            'name': 'Thailand',
            'characteristics': {
                'zero_inflation': 0.15,
                'clustering': 'medium',
                'spatial_regularity': 'high',
            },
            'model': {
                'conv_channels': [32, 64],
                'lstm_hidden': 64,
                'lstm_layers': 2,
                'dropout': 0.3,
                'loss': 'mse',
                'grid_size': 32,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 5, 'augmentation_factor': 1.5},
                'quantum': {'n_qubits': 8, 'augmentation_ratio': 3},
            },
        },
        'INDONESIA': {
            'name': 'Indonesia',
            'characteristics': {
                'zero_inflation': 0.25,
                'clustering': 'very_high',
                'spatial_regularity': 'low',
            },
            'model': {
                'conv_channels': [64, 128, 256],
                'lstm_hidden': 256,
                'lstm_layers': 3,
                'dropout': 0.35,
                'loss': 'nb',
                'grid_size': 48,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 10, 'augmentation_factor': 2.5},
                'quantum': {'n_qubits': 12, 'augmentation_ratio': 5},
            },
        },
        'MALAYSIA': {
            'name': 'Malaysia',
            'characteristics': {
                'zero_inflation': 0.12,
                'clustering': 'medium',
                'spatial_regularity': 'medium',
            },
            'model': {
                'conv_channels': [32, 64],
                'lstm_hidden': 64,
                'lstm_layers': 2,
                'dropout': 0.25,
                'loss': 'mse',
                'grid_size': 24,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 5, 'augmentation_factor': 1.5},
                'quantum': {'n_qubits': 8, 'augmentation_ratio': 2},
            },
        },
        'SINGAPORE': {
            'name': 'Singapore',
            'characteristics': {
                'zero_inflation': 0.08,
                'clustering': 'low',
                'spatial_regularity': 'very_high',
            },
            'model': {
                'conv_channels': [16, 32],
                'lstm_hidden': 32,
                'lstm_layers': 1,
                'dropout': 0.2,
                'loss': 'mse',
                'grid_size': 16,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 3, 'augmentation_factor': 1.2},
                'quantum': {'n_qubits': 6, 'augmentation_ratio': 2},
            },
        },
        'CAMBODIA': {
            'name': 'Cambodia',
            'characteristics': {
                'zero_inflation': 0.22,
                'clustering': 'high',
                'spatial_regularity': 'low',
            },
            'model': {
                'conv_channels': [32, 64, 128],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'dropout': 0.35,
                'loss': 'nb',
                'grid_size': 24,
            },
            'augmentation': {
                'sop_v2': {'k_neighbors': 6, 'augmentation_factor': 2.0},
                'quantum': {'n_qubits': 8, 'augmentation_ratio': 3},
            },
        },
    }

    DEFAULT_CONFIG = {
        'model': {
            'conv_channels': [32, 64],
            'lstm_hidden': 64,
            'lstm_layers': 2,
            'dropout': 0.3,
            'loss': 'mse',
            'grid_size': 32,
        },
        'augmentation': {
            'sop_v2': {'k_neighbors': 5, 'augmentation_factor': 1.5},
            'quantum': {'n_qubits': 8, 'augmentation_ratio': 3},
        },
    }

    @classmethod
    def get_config(cls, country: str) -> Dict:
        """Get configuration for a specific country."""
        normalized = country.upper()
        for key in cls.CONFIGS:
            if key.upper() in normalized or normalized in key.upper():
                return cls.CONFIGS[key]
        return cls.DEFAULT_CONFIG

    @classmethod
    def get_all_countries(cls) -> List[str]:
        """Get list of all configured countries."""
        return list(cls.CONFIGS.keys())


class CountrySpecificModels:
    """
    Train and manage country-specific models.

    Each country gets its own optimized model based on
    country-specific characteristics.
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: int = 42,
        model_class=None,
    ):
        """
        Args:
            device: 'cpu' or 'cuda'
            seed: random seed
            model_class: model class to use (default: SpatioTemporalCNNv2)
        """
        self.device = device
        self.seed = seed
        self.model_class = model_class
        self.models: Dict[str, nn.Module] = {}
        self.configs: Dict[str, Dict] = {}

    def train_country_model(
        self,
        country: str,
        train_events: pd.DataFrame,
        val_events: pd.DataFrame,
        grid_size: int = 32,
        seq_len: int = 12,
        **train_kwargs
    ) -> Tuple[nn.Module, Dict]:
        """
        Train a model for a specific country.

        Args:
            country: country name
            train_events: training events
            val_events: validation events
            grid_size: spatial grid size
            seq_len: sequence length
            **train_kwargs: additional training arguments

        Returns:
            (trained_model, metrics)
        """
        from ..models.cnn_lstm_v2 import (
            SpatioTemporalCNNv2,
            train_cnn_lstm_v2,
            create_sequences_v2,
        )
        from ..evaluation.metrics import compute_forecasting_metrics

        if self.model_class is None:
            model_class = SpatioTemporalCNNv2
        else:
            model_class = self.model_class

        config = CountrySpecificConfig.get_config(country)
        self.configs[country] = config

        country_train = train_events[train_events['country'] == country]
        country_val = val_events[val_events['country'] == country]

        if len(country_train) < 100:
            print(f"  Warning: {country} has insufficient data ({len(country_train)} events)")

        from ..data.loader import create_spatial_grid
        train_grid, _, _ = create_spatial_grid(country_train, grid_size=grid_size)
        val_grid, _, _ = create_spatial_grid(country_val, grid_size=grid_size)

        X_train, y_train = create_sequences_v2(train_grid, seq_len=seq_len)
        X_val, y_val = create_sequences_v2(val_grid, seq_len=seq_len)

        if len(X_train) < 50:
            print(f"  Warning: {country} has insufficient sequences")
            return None, {}

        model_config = config['model']
        model = model_class(
            input_channels=1,
            conv_channels=model_config['conv_channels'],
            lstm_hidden=model_config['lstm_hidden'],
            lstm_layers=model_config['lstm_layers'],
            dropout=model_config['dropout'],
            grid_size=grid_size,
            forecast_horizon=1,
            loss=model_config['loss'],
        )

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=32)

        model = train_cnn_lstm_v2(
            model=model,
            train_loader=train_ld,
            val_loader=val_ld,
            device=self.device,
            seed=self.seed,
            **train_kwargs
        )

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_val).to(self.device)).cpu().numpy().flatten()

        metrics = compute_forecasting_metrics(y_val.flatten(), y_pred)

        self.models[country] = model

        return model, metrics

    def train_all(
        self,
        train_events: pd.DataFrame,
        val_events: pd.DataFrame,
        countries: Optional[List[str]] = None,
        grid_size: int = 32,
        **train_kwargs
    ) -> Dict[str, Dict]:
        """
        Train models for all countries.

        Args:
            train_events: training events
            val_events: validation events
            countries: list of countries (default: all in data)
            grid_size: spatial grid size
            **train_kwargs: training arguments

        Returns:
            Dict of country -> (model, metrics)
        """
        if countries is None:
            countries = train_events['country'].unique()

        results = {}

        for country in countries:
            print(f"\nTraining model for {country}...")
            model, metrics = self.train_country_model(
                country=country,
                train_events=train_events,
                val_events=val_events,
                grid_size=grid_size,
                **train_kwargs
            )
            results[country] = {
                'model': model,
                'metrics': metrics,
                'config': self.configs.get(country),
            }

            if metrics:
                print(f"  {country}: R2={metrics.get('R2', 'N/A'):.3f}, RMSE={metrics.get('RMSE', 'N/A'):.2f}")

        return results

    def predict(
        self,
        country: str,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict using country-specific model."""
        if country not in self.models:
            raise ValueError(f"No model for country: {country}")

        model = self.models[country]
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X).to(self.device)).cpu().numpy()

    def save_models(self, path: Path) -> None:
        """Save all models to disk."""
        path.mkdir(parents=True, exist_ok=True)

        for country, model in self.models.items():
            torch.save(model.state_dict(), path / f"model_{country.replace(' ', '_')}.pt")

        with open(path / "configs.json", 'w') as f:
            json.dump(self.configs, f, indent=2, default=str)

    def load_models(self, path: Path) -> None:
        """Load models from disk."""
        for config_file in path.glob("*.pt"):
            country = config_file.stem.replace("model_", "").replace("_", " ")
            model = torch.load(config_file, map_location=self.device)
            self.models[country] = model

        configs_path = path / "configs.json"
        if configs_path.exists():
            with open(configs_path) as f:
                self.configs = json.load(f)


class EnsembleForecaster:
    """
    Ensemble of country-specific and global models.

    Combines predictions from:
    1. Country-specific models (when available)
    2. Global model (fallback for unknown countries)
    """

    def __init__(
        self,
        global_model: nn.Module,
        country_models: Optional[CountrySpecificModels] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            global_model: fallback model for unknown countries
            country_models: country-specific models
            weights: ensemble weights {country: weight}
        """
        self.global_model = global_model
        self.country_models = country_models
        self.weights = weights or {}

    def predict(
        self,
        X: np.ndarray,
        country: Optional[str] = None,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Predict using ensemble.

        Args:
            X: input sequences
            country: country name (if known)
            device: device for prediction

        Returns:
            predictions
        """
        self.global_model.eval()

        with torch.no_grad():
            global_pred = self.global_model(torch.FloatTensor(X).to(device)).cpu().numpy()

        if country and self.country_models and country in self.country_models.models:
            country_pred = self.country_models.predict(country, X)
            weight = self.weights.get(country, 0.5)
            return weight * country_pred + (1 - weight) * global_pred

        return global_pred
