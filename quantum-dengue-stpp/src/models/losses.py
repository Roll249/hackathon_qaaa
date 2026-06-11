"""
Custom loss functions for spatio-temporal forecasting.

Implements losses tailored for overdispersed count data common in dengue outbreaks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeBinomialLoss(nn.Module):
    """
    Negative Binomial Loss for overdispersed count data.

    Dengue case counts show extreme overdispersion (theta^-1 >> 1), making
    MSE suboptimal. NB loss models the variance explicitly:
        Var(Y) = mu + mu^2 / theta

    where theta is the dispersion parameter (learned or fixed).

    Reference: Cameron & Trivedi "Regression Analysis of Count Data" (1998)
    """

    def __init__(self, learn_dispersion=True, min_dispersion=1e-3):
        super().__init__()
        self.learn_dispersion = learn_dispersion
        self.min_dispersion = min_dispersion

        if learn_dispersion:
            self.log_r = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        """
        pred: predicted values (log of mu)
        target: actual counts
        """
        mu = pred.exp().clamp(min=1e-6)
        target = target.clamp(min=0)

        if self.learn_dispersion and hasattr(self, 'log_r'):
            r = self.log_r.exp().clamp(min=self.min_dispersion)
        else:
            r = 1.0

        eps = 1e-8

        # Negative Binomial NLL:
        # log L = log Gamma(y+r) - log Gamma(r) - log Gamma(y+1)
        #          + r * log(r/(r+mu)) + y * log(mu/(r+mu))
        loss = (
            torch.lgamma(target + r + eps)
            - torch.lgamma(r + eps)
            - torch.lgamma(target + 1 + eps)
            + r * torch.log(r / (r + mu) + eps)
            + target * torch.log(mu / (r + mu) + eps)
        )

        return -loss.mean()


class PoissonLoss(nn.Module):
    """
    Poisson loss for count data.

    Simpler than NB, assumes Var(Y) = E[Y] = mu.
    Good baseline for count regression.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mu = pred.exp().clamp(min=1e-6)
        target = target.clamp(min=0)

        loss = mu - target * torch.log(mu + 1e-6)
        return loss.mean()


class LogCauchyNBGaussLoss(nn.Module):
    """
    Multi-target loss combining NB and Gaussian on log-transformed counts.

    Benefits:
    - NB handles count prediction
    - Log-Gaussian handles outliers (extreme outbreaks)
    - Cauchy component provides robustness to extreme outliers

    Reference: Mixed loss approaches for count data with outliers
    """

    def __init__(self, nb_weight=0.5, gauss_weight=0.4, cauchy_weight=0.1):
        super().__init__()
        self.nb_weight = nb_weight
        self.gauss_weight = gauss_weight
        self.cauchy_weight = cauchy_weight

        self.nb_loss = NegativeBinomialLoss(learn_dispersion=True)
        self.eps = 1e-6

    def forward(self, pred, target):
        target = target.clamp(min=0)

        nb = self.nb_loss(pred, target)

        log_target = torch.log1p(target)
        log_pred = torch.log1p(pred.exp().clamp(min=0))
        gauss = F.mse_loss(log_pred, log_target)

        diff = pred.exp().clamp(min=0) - target
        cauchy = torch.log1p((diff / (self.eps + 1.0)) ** 2)

        total = (
            self.nb_weight * nb
            + self.gauss_weight * gauss
            + self.cauchy_weight * cauchy.mean()
        )

        return total


class TweedieLoss(nn.Module):
    """
    Tweedie loss for zero-inflated count data.

    Tweedie distribution with power p (1 < p < 2) is ideal for:
    - Zero-inflated data (common in dengue: 0-31% zeros)
    - Compound Poisson-Gamma structure
    - Variance = mu^power

    Reference: Tweedie "Index Codes for Multivariate Distributions" (1984)
    """

    def __init__(self, power=1.5, link="log"):
        super().__init__()
        self.power = power
        self.link = link

    def forward(self, pred, target):
        target = target.clamp(min=0)
        mu = pred.exp().clamp(min=1e-6)

        if self.power == 1:
            loss = mu - target * torch.log(mu)
        elif self.power == 2:
            loss = 1 / (self.power - 1) * (torch.pow(mu, 2 - self.power) - (2 - self.power) * target * torch.pow(mu, 1 - self.power))
        else:
            rho = 1 / (2 - self.power)
            loss = torch.pow(mu, 2 - self.power) / ((2 - self.power) * target + 1e-6) - (target * torch.pow(mu, 1 - self.power)) / (self.power - 1 + 1e-6)

        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.

    Provides uncertainty estimates alongside point predictions.
    Useful for generating prediction intervals.
    """

    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        target = target.clamp(min=0)
        error = target - pred

        loss = torch.max(
            self.quantile * error,
            (self.quantile - 1) * error
        )

        return loss.mean()


def get_loss_fn(loss_name, **kwargs):
    """
    Factory function to get loss by name.

    Args:
        loss_name: one of 'mse', 'mae', 'poisson', 'nb', 'tweedie', 'log_cauchy_nb_gauss', 'quantile'
        **kwargs: passed to loss constructor

    Returns:
        Loss function instance
    """
    loss_registry = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'poisson': PoissonLoss,
        'nb': NegativeBinomialLoss,
        'negative_binomial': NegativeBinomialLoss,
        'tweedie': TweedieLoss,
        'log_cauchy_nb_gauss': LogCauchyNBGaussLoss,
        'quantile': QuantileLoss,
    }

    if loss_name.lower() not in loss_registry:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_registry.keys())}")

    return loss_registry[loss_name.lower()](**kwargs)
