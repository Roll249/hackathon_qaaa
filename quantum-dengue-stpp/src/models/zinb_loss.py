"""
Zero-Inflated Negative Binomial (ZINB) Loss for Dengue Forecasting.

This module implements the ZINB distribution for modeling count data with
excess zeros — a critical characteristic of Southeast Asian dengue datasets
where many regions report zero cases in most months.

The ZINB model combines:
1. A point mass at zero (π) — structural zeros from no disease presence
2. A Negative Binomial distribution — overdispersed counts when disease is present

Mathematical formulation:
    P(Y=0) = π + (1-π) * (1 + μ*θ)^(-θ)
    P(Y=k) = (1-π) * Γ(k+θ) / (Γ(k+1)*Γ(θ)) * (θ/(θ+μ))^θ * (μ/(θ+μ))^k

where:
    π = probability of structural zero (zero-inflation)
    μ = mean count (rate)
    θ = dispersion parameter
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ZeroInflatedNegativeBinomialLoss(nn.Module):
    """
    ZINB Loss for spatio-temporal point process forecasting.

    Suitable for dengue data with:
    - High zero-inflation (>30% zeros in Vietnam regions)
    - Overdispersion (variance >> mean)
    - Spatial autocorrelation

    The model predicts two parameters:
    - mu: the rate parameter (must be > 0)
    - pi: the zero-inflation probability (must be in [0, 1])
    """

    def __init__(self, learn_theta: bool = True, theta_init: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.learn_theta = learn_theta
        self.reduction = reduction

        # Log-transformed dispersion parameter for stability
        if learn_theta:
            self.log_theta = nn.Parameter(torch.log(torch.tensor(theta_init)))
        else:
            self.register_buffer('log_theta', torch.log(torch.tensor(theta_init)))

    @property
    def theta(self) -> torch.Tensor:
        """Get the dispersion parameter (always positive)."""
        return torch.exp(self.log_theta)

    def zinb_log_prob(self, y: torch.Tensor, mu: torch.Tensor,
                      pi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under Zero-Inflated Negative Binomial.

        Args:
            y: observed counts (any shape)
            mu: rate parameter (same shape as y, > 0)
            pi: zero-inflation probability (same shape, in [0,1])
            theta: dispersion parameter (scalar, > 0)

        Returns:
            log probabilities with same shape as y
        """
        # Ensure numerical stability
        eps = 1e-8
        mu = mu.clamp(min=eps)
        pi = pi.clamp(min=eps, max=1 - eps)
        theta = theta.clamp(min=eps)

        # Case 1: y == 0
        case_zero = torch.log(
            pi + (1 - pi) * torch.pow(1 + mu / theta, -theta) + eps
        )

        # Case 2: y > 0
        # NB log probability: log Γ(y+θ) - log Γ(y+1) - log Γ(θ)
        #                     + y*log(θ/(θ+μ)) + θ*log(θ/(θ+μ))
        theta_over = theta / (theta + mu)
        log_nb = (
            torch.lgamma(y + theta) -
            torch.lgamma(y + 1) -
            torch.lgamma(theta) +
            y * torch.log(theta_over + eps) +
            theta * torch.log(theta_over + eps)
        )

        case_positive = torch.log(1 - pi + eps) + log_nb

        # Combine: if y == 0 use case_zero, else use case_positive
        mask_zero = (y == 0).float()
        log_prob = mask_zero * case_zero + (1 - mask_zero) * case_positive

        return log_prob

    def forward(self, pred_mu: torch.Tensor, pred_pi: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood ZINB loss.

        Args:
            pred_mu: predicted rate (any shape, will be exponentiated)
            pred_pi: predicted zero-inflation (will be sigmoid transformed)
            target: observed counts (same shape as predictions)

        Returns:
            Negative log-likelihood (scalar by default)
        """
        # Transform predictions to valid ranges
        mu = F.softplus(pred_mu)  # Ensure mu > 0
        pi = torch.sigmoid(pred_pi)  # Ensure 0 < pi < 1
        theta = self.theta

        # Compute log probabilities
        log_prob = self.zinb_log_prob(target, mu, pi, theta)

        # Negative log-likelihood
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class HybridQuantumZINB(nn.Module):
    """
    Hybrid Quantum-Classical model with ZINB output for zero-inflated count data.

    Integrates with PennyLane quantum circuits to predict:
    - μ (mu): infection rate
    - π (pi): zero-inflation probability

    Architecture:
        Classical Encoder → Quantum Feature Map → Classical Heads → ZINB Parameters
    """

    def __init__(self, input_dim: int, n_qubits: int = 4, n_layers: int = 3,
                 hidden_dim: int = 64, grid_size: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.grid_size = grid_size

        # Classical pre-processing
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Quantum circuit parameters (learnable)
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        # Classical post-processing heads for ZINB parameters
        # Mu head: predicts log(rate)
        self.mu_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, grid_size * grid_size),
        )

        # Pi head: predicts logit(zero-inflation probability)
        self.pi_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, grid_size * grid_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid quantum-classical model.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            pred_mu: predicted rates (batch, grid_h * grid_w)
            pred_pi: predicted zero-inflation (batch, grid_h * grid_w)
        """
        # Classical encoding
        x_enc = self.classical_encoder(x)

        # Prepare quantum inputs (normalize to [0, pi])
        q_input = torch.pi * (x_enc - x_enc.min()) / (x_enc.max() - x_enc.min() + 1e-8)
        q_input = q_input[:, :self.n_qubits]  # Match qubit count

        # Quantum processing using TorchInterface
        q_out = self._quantum_circuit(q_input)

        # Classical heads for ZINB parameters
        pred_mu = self.mu_head(q_out)   # Will be softplus'd in loss
        pred_pi = self.pi_head(q_out)    # Will be sigmoid'd in loss

        return pred_mu, pred_pi

    def _quantum_circuit(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute parameterized quantum circuit.

        Uses AngleEmbedding for data encoding and StronglyEntanglingLayers
        for variational optimization.
        """
        import pennylane as qml

        # Create quantum device
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Data encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')

            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Return expectation values as features
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Process batch
        outputs = []
        for sample in inputs:
            out = circuit(sample, self.q_weights)
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)
            outputs.append(out)

        return torch.stack(outputs)


class SpatialZINBGridLoss(nn.Module):
    """
    ZINB loss for grid-based spatial prediction.

    Combines per-cell ZINB losses with spatial regularization to encourage
    smooth spatial patterns consistent with disease transmission dynamics.
    """

    def __init__(self, spatial_smooth_weight: float = 0.1,
                 learn_theta: bool = True):
        super().__init__()
        self.zinb_loss = ZeroInflatedNegativeBinomialLoss(
            learn_theta=learn_theta, reduction='none'
        )
        self.spatial_smooth_weight = spatial_smooth_weight

    def spatial_smoothness(self, pred: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Compute spatial smoothness penalty using TV (Total Variation) norm.

        Encourages neighboring cells to have similar predictions.
        """
        pred_grid = pred.view(-1, grid_size, grid_size)

        # Horizontal differences
        h_diff = (pred_grid[:, :, 1:] - pred_grid[:, :, :-1]).pow(2)

        # Vertical differences
        v_diff = (pred_grid[:, 1:, :] - pred_grid[:, :-1, :]).pow(2)

        return (h_diff.mean() + v_diff.mean()) / 2

    def forward(self, pred_mu: torch.Tensor, pred_pi: torch.Tensor,
                target: torch.Tensor, grid_size: int = 20) -> torch.Tensor:
        """
        Compute combined ZINB + spatial smoothness loss.
        """
        # ZINB loss
        zinb_loss = self.zinb_loss(pred_mu, pred_pi, target)

        # Spatial smoothness
        if self.spatial_smooth_weight > 0:
            smooth_loss = self.spatial_smoothness(pred_mu, grid_size)
            total_loss = zinb_loss + self.spatial_smooth_weight * smooth_loss
        else:
            total_loss = zinb_loss

        return total_loss


def compute_zinb_metrics(pred_mu: torch.Tensor, pred_pi: torch.Tensor,
                          target: torch.Tensor, theta: float) -> dict:
    """
    Compute evaluation metrics for ZINB model.

    Args:
        pred_mu: predicted rates
        pred_pi: predicted zero-inflation probabilities
        target: observed counts
        theta: dispersion parameter

    Returns:
        Dictionary of metrics
    """
    import math

    # Transform predictions
    mu = F.softplus(pred_mu).detach().cpu().numpy()
    pi = torch.sigmoid(pred_pi).detach().cpu().numpy()
    y = target.detach().cpu().numpy()

    # Mean predictions
    expected = (1 - pi) * mu

    # Metrics
    mse = np.mean((expected - y) ** 2)
    mae = np.mean(np.abs(expected - y))

    # Zero-prediction accuracy
    pred_zero = (expected < 0.5).astype(int)
    true_zero = (y == 0).astype(int)
    zero_acc = np.mean(pred_zero == true_zero)

    # Correlation
    corr = np.corrcoef(expected.flatten(), y.flatten())[0, 1]

    # R-squared
    ss_res = np.sum((y - expected) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'zero_accuracy': float(zero_acc),
        'correlation': float(corr),
        'r2': float(r2),
        'theta': float(theta),
        'mean_pi': float(pi.mean()),
        'mean_mu': float(mu.mean()),
    }
