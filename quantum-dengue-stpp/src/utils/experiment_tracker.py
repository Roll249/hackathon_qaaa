"""
Experiment tracking integration using MLflow.

Provides centralized experiment logging for reproducible research.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    model_type: str
    dataset: str
    grid_size: int = 20
    seq_len: int = 12
    forecast_horizon: int = 1
    augmentation: str = "none"
    quantum_circuit: str = "none"
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "dataset": self.dataset,
            "grid_size": self.grid_size,
            "seq_len": self.seq_len,
            "forecast_horizon": self.forecast_horizon,
            "augmentation": self.augmentation,
            "quantum_circuit": self.quantum_circuit,
            "tags": self.tags,
        }
    
    @property
    def config_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ExperimentTracker:
    """
    Lightweight experiment tracking without external dependencies.
    
    Stores experiment metadata in structured JSON files for easy analysis.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        self.current_run: Optional[Dict[str, Any]] = None
        self._setup_directories()
    
    def _setup_directories(self):
        """Create experiment directories."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/runs", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/metrics", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/artifacts", exist_ok=True)
    
    def start_run(self, config: ExperimentConfig) -> str:
        """Start a new experiment run."""
        self.current_run = {
            "experiment_name": config.name,
            "run_id": f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": config.to_dict(),
            "config_hash": config.config_hash,
            "start_time": datetime.now().isoformat(),
            "metrics": {},
            "metrics_history": [],
            "status": "running",
            "tags": config.tags,
        }
        
        logger.info(f"Started experiment: {self.current_run['run_id']}")
        return self.current_run["run_id"]
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric value."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if key not in self.current_run["metrics"]:
            self.current_run["metrics"][key] = []
        
        metric_entry = {
            "value": float(value),
            "timestamp": datetime.now().isoformat(),
        }
        if step is not None:
            metric_entry["step"] = step
        
        self.current_run["metrics"][key].append(metric_entry)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self.current_run["config"][key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_artifact(self, artifact_path: str, name: Optional[str] = None):
        """Log an artifact file."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if "artifacts" not in self.current_run:
            self.current_run["artifacts"] = []
        
        artifact_name = name or os.path.basename(artifact_path)
        self.current_run["artifacts"].append({
            "name": artifact_name,
            "path": artifact_path,
            "logged_at": datetime.now().isoformat(),
        })
    
    def end_run(self, status: str = "completed", metrics: Optional[Dict[str, float]] = None):
        """End the current experiment run."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self.current_run["status"] = status
        self.current_run["end_time"] = datetime.now().isoformat()
        
        if metrics:
            for key, value in metrics.items():
                if key not in self.current_run["metrics"]:
                    self.current_run["metrics"][key] = []
                self.current_run["metrics"][key].append({
                    "value": float(value),
                    "timestamp": datetime.now().isoformat(),
                })
        
        self._save_run()
        
        run_id = self.current_run["run_id"]
        self.current_run = None
        
        logger.info(f"Completed experiment: {run_id}")
        return run_id
    
    def _save_run(self):
        """Save run data to file."""
        run_id = self.current_run["run_id"]
        run_path = f"{self.experiment_dir}/runs/{run_id}.json"
        
        with open(run_path, "w") as f:
            json.dump(self.current_run, f, indent=2, default=str)
    
    def get_best_run(self, metric: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """Get the best run for a given metric."""
        best_run = None
        best_value = float("inf") if mode == "min" else float("-inf")
        
        for run_file in os.listdir(f"{self.experiment_dir}/runs"):
            if not run_file.endswith(".json"):
                continue
            
            with open(f"{self.experiment_dir}/runs/{run_file}") as f:
                run = json.load(f)
            
            if run["status"] != "completed":
                continue
            
            metrics = run.get("metrics", {})
            if metric not in metrics or not metrics[metric]:
                continue
            
            latest_value = metrics[metric][-1]["value"]
            
            if mode == "min" and latest_value < best_value:
                best_value = latest_value
                best_run = run
            elif mode == "max" and latest_value > best_value:
                best_value = latest_value
                best_run = run
        
        return best_run
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs = []
        for run_id in run_ids:
            run_path = f"{self.experiment_dir}/runs/{run_id}.json"
            if os.path.exists(run_path):
                with open(run_path) as f:
                    runs.append(json.load(f))
        
        if not runs:
            return {"error": "No runs found"}
        
        comparison = {
            "runs": len(runs),
            "run_ids": run_ids,
            "metrics": {},
        }
        
        for run in runs:
            for metric_name, values in run.get("metrics", {}).items():
                if metric_name not in comparison["metrics"]:
                    comparison["metrics"][metric_name] = {
                        "values": [],
                        "mean": 0,
                        "std": 0,
                    }
                if values:
                    comparison["metrics"][metric_name]["values"].append(values[-1]["value"])
        
        for metric_name, data in comparison["metrics"].items():
            if data["values"]:
                data["mean"] = sum(data["values"]) / len(data["values"])
                variance = sum((x - data["mean"]) ** 2 for x in data["values"]) / len(data["values"])
                data["std"] = variance ** 0.5
        
        return comparison
    
    def list_runs(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all runs."""
        runs = []
        for run_file in os.listdir(f"{self.experiment_dir}/runs"):
            if not run_file.endswith(".json"):
                continue
            
            with open(f"{self.experiment_dir}/runs/{run_file}") as f:
                run = json.load(f)
            
            if experiment_name is None or run.get("experiment_name") == experiment_name:
                runs.append({
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "start_time": run["start_time"],
                    "metrics": {k: v[-1]["value"] if v else None 
                               for k, v in run.get("metrics", {}).items()},
                })
        
        return sorted(runs, key=lambda x: x["start_time"], reverse=True)


# Convenience functions
_default_tracker: Optional[ExperimentTracker] = None


def get_tracker(experiment_dir: str = "experiments") -> ExperimentTracker:
    """Get or create the default experiment tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ExperimentTracker(experiment_dir)
    return _default_tracker


def start_experiment(name: str, **kwargs) -> str:
    """Convenience function to start an experiment."""
    config = ExperimentConfig(name=name, **kwargs)
    tracker = get_tracker()
    return tracker.start_run(config)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Convenience function to log metrics."""
    tracker = get_tracker()
    tracker.log_metrics(metrics, step)


def end_experiment(status: str = "completed", metrics: Optional[Dict[str, float]] = None):
    """Convenience function to end an experiment."""
    tracker = get_tracker()
    return tracker.end_run(status, metrics)
