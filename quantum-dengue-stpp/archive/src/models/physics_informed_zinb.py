"""
Physics-Informed ZINB Loss with Controlled Noise Injection.

IMPORTANT — this is NOT the decoherence-as-regularizer proposal from improve.md.

The original proposal suggested using hardware decoherence (Lindblad-type) as a
regularizer. That idea has been REJECTED in the analysis (IMPROVEMENT_PROPOSAL.md)
for the following reasons:

1. Hardware noise is uncontrolled — not a useful prior.
2. Lindblad dissipation is Markovian and does not capture seasonal dynamics.
3. No theoretical justification for connecting quantum noise to ZINB likelihood.
4. References like Sweke et al. 2020 (Dissipative QNN) do not validate this use case.

This module instead implements a CLASSICAL analogue that achieves a similar
inductive bias goal: regularizing the count predictions via controlled noise
injection during training. This is well-established in:
- Bayesian deep learning (MC dropout, Gal & Ghahramani 2016)
- Variational inference for neural networks (Blundell et al. 2015)
- Diffusion-based regularization

The result is a noise-robust ZINB loss suitable for sparse count data.

Mathematical formulation:
    mu_noisy = mu + ε * noise_scale * mu,   where ε ~ N(0, 1)
    This adds proportional Gaussian noise to the rate parameter, which has
    similar effect to shot noise in quantum measurements but is fully
    reproducible and tunable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedZINBLoss(nn.Module):
    """
    ZINB loss with optional noise injection as classical regularizer.

    Args:
        learn_theta: if True, learn the dispersion parameter θ.
        theta_init: initial value of θ.
        noise_scale: scale of Gaussian noise added to mu during training (e.g. 0.05).
                     Set to 0 to disable noise (equivalent to standard ZINB).
        spatial_smooth_weight: weight for TV smoothness penalty.
        reduction: 'mean', 'sum', or 'none'.

    Reference for ZINB:
        P(Y=0) = π + (1-π)(1+μ/θ)^(-θ)
        P(Y=k) = (1-π)·NB(k | μ, θ),  k > 0

    NB log-probability:
        log P(k|μ,θ) = lgamma(k+θ) - lgamma(k+1) - lgamma(θ)
                        + k·log(θ/(θ+μ)) + θ·log(θ/(θ+μ))
    """

    def __init__(
        self,
        learn_theta: bool = True,
        theta_init: float = 1.0,
        noise_scale: float = 0.05,
        spatial_smooth_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.learn_theta = learn_theta
        self.noise_scale = noise_scale
        self.spatial_smooth_weight = spatial_smooth_weight
        self.reduction = reduction

        if learn_theta:
            self.log_theta = nn.Parameter(torch.log(torch.tensor(theta_init)))
        else:
            self.register_buffer("log_theta", torch.log(torch.tensor(theta_init)))

    @property
    def theta(self) -> torch.Tensor:
        return torch.exp(self.log_theta)

    def zinb_log_prob(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        pi: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability under ZINB distribution."""
        eps = 1e-8
        mu = mu.clamp(min=eps)
        pi = pi.clamp(min=eps, max=1 - eps)
        theta = theta.clamp(min=eps)

        # y == 0 case
        case_zero = torch.log(pi + (1 - pi) * torch.pow(1 + mu / theta, -theta) + eps)

        # y > 0 case
        theta_over = theta / (theta + mu)
        log_nb = (
            torch.lgamma(y + theta)
            - torch.lgamma(y + 1)
            - torch.lgamma(theta)
            + y * torch.log(theta_over + eps)
            + theta * torch.log(theta_over + eps)
        )
        case_positive = torch.log(1 - pi + eps) + log_nb

        mask_zero = (y == 0).float()
        log_prob = mask_zero * case_zero + (1 - mask_zero) * case_positive
        return log_prob

    def forward(
        self,
        pred_mu: torch.Tensor,
        pred_pi: torch.Tensor,
        target: torch.Tensor,
        grid_size: int | None = None,
    ) -> torch.Tensor:
        """
        Compute ZINB loss with optional noise injection.

        Args:
            pred_mu: (any) predicted log-rate.
            pred_pi: (any) predicted logit for zero-inflation probability.
            target: (any) observed counts.
            grid_size: if not None, apply spatial smoothness penalty assuming
                       prediction is reshaped to (batch, grid_size, grid_size).

        Returns:
            Scalar (or non-reduced) loss.
        """
        mu = F.softplus(pred_mu)
        pi = torch.sigmoid(pred_pi)

        # Controlled Gaussian noise injection — only during training
        if self.training and self.noise_scale > 0:
            mu = mu + torch.randn_like(mu) * self.noise_scale * mu

        log_prob = self.zinb_log_prob(target, mu, pi, self.theta)
        nll = -log_prob

        loss = nll
        if self.spatial_smooth_weight > 0 and grid_size is not None:
            pred_grid = mu.view(-1, grid_size, grid_size)
            h_diff = (pred_grid[:, :, 1:] - pred_grid[:, :, :-1]).pow(2).mean()
            v_diff = (pred_grid[:, 1:, :] - pred_grid[:, :-1, :]).pow(2).mean()
            tv = (h_diff + v_diff) / 2
            loss = nll + self.spatial_smooth_weight * tv

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def benchmark_zinb_with_noise(
    n_samples: int = 5000,
    epochs: int = 100,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Compare ZINB loss with and without noise injection on synthetic count data.

    Generates data from a true ZINB(μ=2.5, π=0.3, θ=1.5), then trains two models:
    1. Standard ZINB loss
    2. PhysicsInformedZINBLoss with noise_scale=0.05

    Measures final NLL on a held-out test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Synthetic data
    true_mu, true_pi, true_theta = 2.5, 0.3, 1.5
    n_total = n_samples + 1000

    u = torch.rand(n_total)
    is_zero = (u < true_pi).float()
    nb_samples = torch.distributions.NegativeBinomial(
        concentration=torch.tensor(true_theta),
        probs=torch.tensor(true_theta / (true_theta + true_mu)),
    ).sample()
    y = is_zero * 0.0 + (1 - is_zero) * nb_samples.float()
    y = y[:n_total]

    X = y.unsqueeze(1)  # trivial feature: just use y as input
    X_train, X_test = X[:n_samples], X[n_samples:]
    y_train, y_test = y[:n_samples], y[n_samples:]

    # Two models
    class ZINBModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_mu = nn.Linear(1, 1)
            self.fc_pi = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc_mu(x), self.fc_pi(x)

    def train_model(loss_fn, label):
        torch.manual_seed(seed)
        model = ZINBModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.05)
        for _ in range(epochs):
            opt.zero_grad()
            mu, pi = model(X_train)
            loss = loss_fn(mu, pi, y_train.unsqueeze(1))
            loss.backward()
            opt.step()
        with torch.no_grad():
            mu, pi = model(X_test)
            test_nll = loss_fn(mu, pi, y_test.unsqueeze(1)).item()
        return test_nll

    standard_loss = lambda m, p, t: -PhysicsInformedZINBLoss(
        noise_scale=0.0
    ).zinb_log_prob(t, F.softplus(m), torch.sigmoid(p), torch.tensor(true_theta)).mean()

    noisy_loss = PhysicsInformedZINBLoss(noise_scale=noise_scale)

    standard_nll = train_model(standard_loss, "standard")
    noisy_nll = train_model(noisy_loss, "noisy")

    return {
        "standard_test_nll": standard_nll,
        "noisy_test_nll": noisy_nll,
        "noise_scale": noise_scale,
        "improvement": standard_nll - noisy_nll,
    }


if __name__ == "__main__":
    print("Benchmarking ZINB with controlled noise injection...")
    result = benchmark_zinb_with_noise()
    for k, v in result.items():
        print(f"  {k}: {v}")
