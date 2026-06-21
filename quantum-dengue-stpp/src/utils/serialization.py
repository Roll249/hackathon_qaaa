"""
Model serialization utilities for saving and loading trained models.
"""
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class ModelSerializer:
    """
    Handles saving and loading of trained models with metadata.
    """
    
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: Dict[str, Any],
        metrics: Dict[str, float],
        path: Union[str, Path],
        save_optimizer: bool = True
    ) -> Path:
        """
        Save a trained model with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer state
            config: Model configuration dict
            metrics: Training/validation metrics
            path: Save path (without extension)
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Path to saved checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "config": config,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
            "torch_version": torch.__version__,
        }
        
        if optimizer is not None and save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint_path = path.with_suffix(".pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata as JSON
        metadata = {
            "model_class": checkpoint["model_class"],
            "config": config,
            "metrics": metrics,
            "saved_at": checkpoint["saved_at"],
            "checkpoint_path": str(checkpoint_path),
        }
        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return checkpoint_path
    
    @staticmethod
    def load_model(
        path: Union[str, Path],
        model_class: Optional[type] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load a saved model checkpoint.
        
        Args:
            path: Path to checkpoint (.pt file)
            model_class: Optional model class to instantiate
            device: Device to load model on
            
        Returns:
            Dict with model, config, metrics, optimizer
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        result = {
            "model_state_dict": checkpoint["model_state_dict"],
            "config": checkpoint.get("config", {}),
            "metrics": checkpoint.get("metrics", {}),
            "saved_at": checkpoint.get("saved_at"),
        }
        
        if "optimizer_state_dict" in checkpoint:
            result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        
        return result
    
    @staticmethod
    def save_predictions(
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
        metadata: Dict[str, Any],
        path: Union[str, Path]
    ):
        """
        Save predictions with metadata for later analysis.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "predictions": predictions.cpu().numpy(),
            "targets": targets.cpu().numpy() if targets is not None else None,
            "metadata": metadata,
            "saved_at": datetime.now().isoformat(),
        }
        
        torch.save(data, path)
        
        # Also save as numpy for easier analysis
        np_path = path.with_suffix(".npz")
        np.savez(
            np_path,
            predictions=predictions.cpu().numpy(),
            targets=targets.cpu().numpy() if targets is not None else None,
            **metadata
        )


class ConfigManager:
    """
    Manages configuration files with validation.
    """
    
    DEFAULT_CONFIG = {
        "model": {
            "type": "cnn_lstm",
            "grid_size": 20,
            "seq_len": 12,
            "forecast_horizon": 1,
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "lr": 1e-3,
            "patience": 10,
        },
        "quantum": {
            "n_qubits": 4,
            "n_layers": 3,
            "use_local_pqc": True,
        },
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "zero_threshold": 0.9,
        },
    }
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls.DEFAULT_CONFIG.copy()
        
        with open(path) as f:
            config = json.load(f)
        
        return cls._merge_with_defaults(config)
    
    @classmethod
    def save(cls, config: Dict[str, Any], path: Union[str, Path]):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def _merge_with_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with defaults for missing keys."""
        result = cls.DEFAULT_CONFIG.copy()
        
        for section, values in config.items():
            if section in result and isinstance(values, dict):
                result[section].update(values)
            else:
                result[section] = values
        
        return result


def save_model_bundle(
    model: torch.nn.Module,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    name: str = "model"
) -> Dict[str, Path]:
    """
    Save complete model bundle including all components.
    
    Returns:
        Dict of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save main model
    model_path = output_dir / f"{name}_checkpoint.pt"
    ModelSerializer.save_model(
        model, None, config, metrics, str(model_path), save_optimizer=False
    )
    paths["model"] = model_path
    
    # Save config
    config_path = output_dir / f"{name}_config.json"
    ConfigManager.save(config, config_path)
    paths["config"] = config_path
    
    # Save metrics
    metrics_path = output_dir / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    paths["metrics"] = metrics_path
    
    return paths
