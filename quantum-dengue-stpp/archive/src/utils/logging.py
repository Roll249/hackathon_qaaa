"""
Logging configuration for Quantum Dengue STPP.

Provides structured logging with appropriate levels for different components.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "quantum_dengue",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Logging level (default INFO)
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


class TrainingLogger:
    """
    Specialized logger for training runs with epoch/metric tracking.
    """
    
    def __init__(self, name: str = "training"):
        self.logger = logging.getLogger(f"quantum_dengue.{name}")
        self.metrics_history = []
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics."""
        metric_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metric_str}")
        
        self.metrics_history.append({
            "epoch": epoch,
            **metrics
        })
    
    def log_best(self, metric_name: str, value: float):
        """Log best metric achieved."""
        self.logger.info(f"★ Best {metric_name}: {value:.4f}")
    
    def log_phase(self, phase: str, n_samples: int):
        """Log phase start/completion."""
        self.logger.info(f"[{phase}] Processing {n_samples} samples")


class DataLogger:
    """
    Specialized logger for data processing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("quantum_dengue.data")
    
    def log_split(self, name: str, n_samples: int, time_range: tuple):
        """Log data split information."""
        self.logger.info(f"Split '{name}': {n_samples} samples | "
                        f"Time: {time_range[0]} → {time_range[1]}")
    
    def log_preprocessing(self, step: str, n_removed: int = 0):
        """Log preprocessing step."""
        if n_removed > 0:
            self.logger.info(f"Preprocessing '{step}': removed {n_removed} samples")
        else:
            self.logger.info(f"Preprocessing '{step}': complete")
    
    def log_validation(self, passed: bool, details: str = ""):
        """Log validation result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        if details:
            self.logger.info(f"Validation {status}: {details}")
        else:
            self.logger.info(f"Validation {status}")


class QuantumLogger:
    """
    Specialized logger for quantum circuit operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("quantum_dengue.quantum")
    
    def log_circuit(self, n_qubits: int, n_layers: int, n_params: int):
        """Log circuit configuration."""
        self.logger.info(f"Circuit: {n_qubits} qubits, {n_layers} layers, "
                        f"{n_params} parameters")
    
    def log_training(self, epoch: int, loss: float, grad_norm: float = None):
        """Log quantum training progress."""
        if grad_norm is not None:
            self.logger.info(f"Q Epoch {epoch:3d} | Loss: {loss:.6f} | "
                           f"‖∇‖: {grad_norm:.4f}")
        else:
            self.logger.info(f"Q Epoch {epoch:3d} | Loss: {loss:.6f}")
    
    def log_generation(self, n_samples: int, quality_score: float):
        """Log synthetic data generation."""
        self.logger.info(f"Generated {n_samples} samples | "
                        f"Quality score: {quality_score:.4f}")


# Convenience loggers
training_logger = TrainingLogger()
data_logger = DataLogger()
quantum_logger = QuantumLogger()
