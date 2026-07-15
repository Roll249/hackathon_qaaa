"""
Zero-Inflated Negative Binomial (ZINB) Loss for Dengue Forecasting.

Canonical ZINB module. Core implementation lives in `physics_informed_zinb.py`
(controlled-noise regularizer). This file re-exports the canonical class and
provides auxiliary components used elsewhere in the codebase (hybrid quantum
head, spatial smoothness penalty, metric utilities).

Why this split?
    The pure-ZINB logic is small enough that one canonical class is enough.
    Auxiliary components (HybridQuantumZINB, SpatialZINBGridLoss,
    compute_zinb_metrics) live here so existing imports stay compatible
    after the consolidation.

References:
    - ZINB log-likelihood: P(Y=0) = π + (1-π)(1+μ/θ)^(-θ),
                           P(Y=k) = (1-π)·NB(k | μ, θ), k > 0
    - PhysicsInformedZINBLoss: classical noise injection as a Bayesian-style
      regularizer (see IMPROVEMENT_PROPOSAL.md for the rationale; the
      original "decoherence as regularizer" idea was rejected).
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical ZINB lives in physics_informed_zinb.py
from .physics_informed_zinb import PhysicsInformedZINBLoss


class ZeroInflatedNegativeBinomialLoss(PhysicsInformedZINBLoss):
    """
    Backwards-compatible alias for the canonical ZINB loss.

    Defaults are equivalent to a standard ZINB NLL (no noise injection,
    no spatial smoothness penalty) — which is the behaviour all existing
    tests and callers expect.

    For the new physics-informed variant with controlled noise injection,
    instantiate ``PhysicsInformedZINBLoss(noise_scale=...)`` directly.
    """

    def __init__(
        self,
        learn_theta: bool = True,
        theta_init: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            learn_theta=learn_theta,
            theta_init=theta_init,
            noise_scale=0.0,           # off — match classic ZINB behaviour
            spatial_smooth_weight=0.0, # off — match classic ZINB behaviour
            reduction=reduction,
        )


class HybridQuantumZINB(nn.Module):
    """
    Hybrid Quantum-Classical model with ZINB output for zero-inflated count data.

    Integrates with PennyLane quantum circuits to predict:
    - μ (mu): infection rate
    - π (pi): zero-inflation probability
    """

    def __init__(self, input_dim: int, n_qubits: int = 4, n_layers: int = 3,
                 hidden_dim: int = 64, grid_size: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.grid_size = grid_size

        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.mu_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, grid_size * grid_size),
        )
        self.pi_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, grid_size * grid_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_enc = self.classical_encoder(x)
        q_input = torch.pi * (x_enc - x_enc.min()) / (x_enc.max() - x_enc.min() + 1e-8)
        q_input = q_input[:, :self.n_qubits]

        q_out = self._quantum_circuit(q_input)
        pred_mu = self.mu_head(q_out)
        pred_pi = self.pi_head(q_out)
        return pred_mu, pred_pi

    def _quantum_circuit(self, inputs: torch.Tensor) -> torch.Tensor:
        import pennylane as qml
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        outputs = []
        for sample in inputs:
            out = circuit(sample, self.q_weights)
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)
            outputs.append(out)
        return torch.stack(outputs)


class SpatialZINBGridLoss(nn.Module):
    """ZINB loss for grid-based spatial prediction with optional TV smoothness."""

    def __init__(self, spatial_smooth_weight: float = 0.1,
                 learn_theta: bool = True):
        super().__init__()
        self.zinb_loss = ZeroInflatedNegativeBinomialLoss(
            learn_theta=learn_theta, reduction='none'
        )
        self.spatial_smooth_weight = spatial_smooth_weight

    def spatial_smoothness(self, pred: torch.Tensor, grid_size: int) -> torch.Tensor:
        pred_grid = pred.view(-1, grid_size, grid_size)
        h_diff = (pred_grid[:, :, 1:] - pred_grid[:, :, :-1]).pow(2)
        v_diff = (pred_grid[:, 1:, :] - pred_grid[:, :-1, :]).pow(2)
        return (h_diff.mean() + v_diff.mean()) / 2

    def forward(self, pred_mu: torch.Tensor, pred_pi: torch.Tensor,
                target: torch.Tensor, grid_size: int = 20) -> torch.Tensor:
        zinb_loss = self.zinb_loss(pred_mu, pred_pi, target)
        if self.spatial_smooth_weight > 0:
            smooth_loss = self.spatial_smoothness(pred_mu, grid_size)
            return zinb_loss + self.spatial_smooth_weight * smooth_loss
        return zinb_loss


def compute_zinb_metrics(pred_mu: torch.Tensor, pred_pi: torch.Tensor,
                         target: torch.Tensor, theta: float) -> dict:
    """Evaluation metrics for ZINB predictions."""
    mu = F.softplus(pred_mu).detach().cpu().numpy()
    pi = torch.sigmoid(pred_pi).detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    expected = (1 - pi) * mu

    mse = float(np.mean((expected - y) ** 2))
    mae = float(np.mean(np.abs(expected - y)))
    pred_zero = (expected < 0.5).astype(int)
    true_zero = (y == 0).astype(int)
    zero_acc = float(np.mean(pred_zero == true_zero))

    flat_expected = expected.flatten()
    flat_y = y.flatten()
    if flat_expected.std() > 0 and flat_y.std() > 0:
        corr = float(np.corrcoef(flat_expected, flat_y)[0, 1])
    else:
        corr = 0.0

    ss_res = float(np.sum((y - expected) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    return {
        'mse': mse,
        'mae': mae,
        'zero_accuracy': zero_acc,
        'correlation': corr,
        'r2': float(r2),
        'theta': float(theta),
        'mean_pi': float(pi.mean()),
        'mean_mu': float(mu.mean()),
    }


__all__ = [
    "PhysicsInformedZINBLoss",
    "ZeroInflatedNegativeBinomialLoss",
    "HybridQuantumZINB",
    "SpatialZINBGridLoss",
    "compute_zinb_metrics",
]