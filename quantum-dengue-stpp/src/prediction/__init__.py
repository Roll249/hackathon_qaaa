"""Prediction module for RAPID-DENGUE v17.

This module contains classification methods (classical + quantum) for the
downstream layer of the pipeline.
"""

from .quantum_knn import (
    grover_knn_predict,
    grover_knn_predict_proba,
    GroverKNNClassifier,
    estimate_grover_iterations,
)

__all__ = [
    "grover_knn_predict",
    "grover_knn_predict_proba",
    "GroverKNNClassifier",
    "estimate_grover_iterations",
]
