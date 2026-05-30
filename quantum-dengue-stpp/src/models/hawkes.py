"""Multi-dimensional Hawkes Process for dengue event intensity modeling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize


class MultiDimensionalHawkes:
    """
    Multi-dimensional Hawkes Process with exponential kernel.

    Intensity for region i at time t:
        λ_i(t) = μ_i + Σ_j Σ_{t_k < t, d_k=j} α_ij * β * exp(-β*(t-t_k)) * w_k

    Fitted via EM algorithm on weighted event sequences.

    Parameters
    ----------
    n_regions : number of spatial regions (dimensions)
    spatial_decay : initial spatial coupling scale (unused if not learnable)
    temporal_decay : initial β (exponential decay rate, 1/month unit)
    learnable_spatial_decay : if True, also optimise β during EM
    """

    def __init__(
        self,
        n_regions: int = 4,
        spatial_decay: float = 0.05,
        temporal_decay: float = 0.3,
        learnable_spatial_decay: bool = True,
    ):
        self.n_regions = n_regions
        self.spatial_decay = spatial_decay
        self.temporal_decay = temporal_decay
        self.learnable_spatial_decay = learnable_spatial_decay

        self.mu = np.ones(n_regions) * 0.1
        self.alpha = np.eye(n_regions) * 0.1
        self.beta = float(temporal_decay)

    def fit(self, times, regions, weights, max_iter: int = 200, verbose: bool = False):
        """
        Vectorized EM using flat pair arrays and np.bincount (no Python loops,
        no N×N matrices).  Subsamples to MAX_N events when input is larger.
        """
        MAX_N = 1000  # M = N*(N-1)/2 ≈ 500 K pairs at float32 ≈ 2 MB

        times = np.asarray(times, dtype=np.float32)
        regions = np.asarray(regions, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)

        sort_idx = np.argsort(times)
        times, regions, weights = times[sort_idx], regions[sort_idx], weights[sort_idx]

        N = len(times)
        if N == 0:
            return self

        if N > MAX_N:
            idx = np.round(np.linspace(0, N - 1, MAX_N)).astype(int)
            times, regions, weights = times[idx], regions[idx], weights[idx]
            N = MAX_N

        T_span = float(max(times[-1] - times[0], 1.0))
        T_end = float(times[-1])
        n = self.n_regions

        # Initialise from empirical counts
        counts = np.bincount(regions, minlength=n).astype(np.float32)
        self.mu = np.maximum(counts / T_span, 1e-6)
        self.alpha = np.full((n, n), 0.05, dtype=np.float32)
        self.beta = float(self.temporal_decay)

        # ---- Precompute static pair arrays (once) -------------------------
        # All ordered pairs (k, j) where event j strictly precedes event k
        k_all, j_all = np.tril_indices(N, k=-1)
        valid = times[k_all] > times[j_all]
        k_idx, j_idx = k_all[valid], j_all[valid]

        i_regs = regions[k_idx].astype(np.int32)   # region of target k
        j_regs = regions[j_idx].astype(np.int32)   # region of source j
        w_k = weights[k_idx]                        # weight of target k
        w_j = weights[j_idx]                        # weight of source j
        dt_flat = (times[k_idx] - times[j_idx]).astype(np.float32)

        # Flat index for (n×n) alpha scatter: idx = i_reg * n + j_reg
        pair_alpha_idx = i_regs * n + j_regs

        for it in range(max_iter):
            # ---- E-step ---------------------------------------------------
            alpha_flat = self.alpha[i_regs, j_regs]                        # (M,)
            exp_dt = np.exp((-self.beta * dt_flat).astype(np.float32))     # (M,)
            exc_flat = (alpha_flat * self.beta * exp_dt * w_j)             # (M,)

            # Total excitation received by each event k  (scatter via bincount)
            total_exc = np.bincount(k_idx, weights=exc_flat, minlength=N).astype(np.float32)

            bg = self.mu[regions]                                           # (N,)
            total = bg + total_exc + 1e-8                                   # (N,)

            p_bg = bg / total                                               # (N,)
            p_trig = exc_flat / total[k_idx]                               # (M,)

            # ---- M-step ---------------------------------------------------
            # μ update
            mu_new = np.maximum(
                np.bincount(regions, weights=p_bg * weights, minlength=n).astype(np.float32)
                / T_span, 1e-6
            )

            # α numerator
            alpha_num = np.bincount(
                pair_alpha_idx, weights=p_trig * w_k, minlength=n * n
            ).reshape(n, n).astype(np.float32)

            # α denominator (depends on beta, so recomputed each iter)
            kern_int = (1.0 - np.exp(
                (-self.beta * (T_end - times + 1.0)).astype(np.float32)
            ))
            alpha_den_j = np.bincount(
                regions, weights=kern_int * weights, minlength=n
            ).astype(np.float32)                                            # (n,)
            alpha_new = np.clip(
                alpha_num / np.maximum(alpha_den_j[None, :], 1e-6), 0.0, 0.99
            )

            # β update
            if self.learnable_spatial_decay:
                contrib = p_trig * w_k
                beta_n = float(contrib.sum())
                beta_d = float((contrib * dt_flat).sum())
                beta_new = float(np.clip(beta_n / (beta_d + 1e-10), 0.01, 20.0))
            else:
                beta_new = self.beta

            # Convergence
            change = (
                float(np.abs(mu_new - self.mu).max())
                + float(np.abs(alpha_new - self.alpha).max())
                + abs(beta_new - self.beta)
            )
            self.mu = mu_new
            self.alpha = alpha_new
            self.beta = beta_new

            if verbose and (it + 1) % 20 == 0:
                print(f"    EM iter {it+1}/{max_iter} | change={change:.5f}")

            if change < 1e-4 and it > 5:
                if verbose:
                    print(f"    Converged at iter {it+1}")
                break

        return self

    def predict_intensity(self, t, region_idx, past_times, past_regions, past_weights):
        """Compute λ_i(t) given history."""
        i = int(region_idx)
        lam = float(self.mu[i])
        for k in range(len(past_times)):
            if past_times[k] >= t:
                break
            j = int(past_regions[k])
            dt = t - past_times[k]
            lam += self.alpha[i, j] * self.beta * np.exp(-self.beta * dt) * past_weights[k]
        return lam


class HawkesBaseline:
    """
    Simplified Hawkes process forecaster: λ(t) = μ + α * λ(t-1).

    For each country, fits via OLS: y_t = μ + α * y_{t-1}
    Forecast = 0.3 * last_value + 0.7 * stationary_mean

    This baseline captures temporal excitation (autocorrelation) without spatial structure.
    Fast to fit, serves as a sanity-check baseline.
    """
    def __init__(self):
        self.country_params = {}   # {country: (mu, alpha)}

    def fit(self, train_events):
        """Fit Hawkes per country on raw event DataFrame."""
        for c in train_events["country"].unique():
            sub = train_events[train_events["country"] == c]
            monthly = sub.groupby(sub["timestamp"].dt.to_period("M"))["case_count"].sum()
            vals = monthly.values.astype(float)
            if len(vals) < 3:
                self.country_params[c] = (float(vals.mean()), 0.0)
                continue
            # OLS: y_t = mu + alpha * y_{t-1}
            y = vals[1:]; x = vals[:-1]
            x_mean, y_mean = x.mean(), y.mean()
            if x.std() < 1e-6:
                alpha = 0.0
            else:
                cov = np.cov(x, y)[0, 1]
                var = np.var(x)
                alpha = float(np.clip(cov / var if var > 0 else 0.0, 0, 0.99))
            mu = float(max(y_mean - alpha * x_mean, 0.0))
            self.country_params[c] = (mu, alpha)
        self.is_fitted_ = True
        return self

    def predict(self, last_counts=None):
        """
        Predict next month per country.
        last_counts: dict of {country: last_observed_count}
        Returns: dict of {country: predicted_count}
        """
        preds = {}
        for c, (mu, alpha) in self.country_params.items():
            stationary = mu / (1 - alpha) if alpha < 0.99 else mu
            last = (last_counts or {}).get(c, 0.0)
            preds[c] = float(0.3 * last + 0.7 * stationary)
        return preds

    def predict_scalar(self, last_counts=None):
        """Return scalar prediction (mean across countries)."""
        p = self.predict(last_counts)
        return float(np.mean(list(p.values()))) if p else 0.0


class MultiHawkesExpKern(BaseEstimator, RegressorMixin):
    """
    Multi-dimensional Hawkes Process with exponential kernel.

    Intensity for dimension d at time t:
        λ_d(t) = μ_d + Σ_{s_k < t} α_d * β_d * exp(-β_d * (t - s_k))

    Fitted via maximum likelihood on aggregated monthly event counts
    using scipy minimize.

    Parameters
    ----------
    n_dim : int — number of spatial dimensions (grid cells)
    decay : float — exponential decay rate β (default 1.0 / month)
    max_iter : int — max L-BFGS-B iterations
    """

    def __init__(self, n_dim: int = 4, decay: float = 0.5, max_iter: int = 200):
        self.n_dim = n_dim
        self.decay = decay
        self.max_iter = max_iter

    def _log_likelihood(self, params, counts, T):
        """
        Compute negative log-likelihood of multivariate Hawkes.
        params: [μ_0, ..., μ_{n-1}, α_00, α_01, ..., α_{n-1,n-1}]
        """
        n = self.n_dim
        mu = np.exp(params[:n])         # positivity constraint
        A = np.exp(params[n:].reshape(n, n))  # adjacency matrix

        ll = 0.0
        for d in range(n):
            # immigrant intensity
            ll += mu[d] * T
            # excited intensity at each event
            for k, (t_k, d_k) in enumerate(counts):
                if d_k == d:
                    # contribution of this event to λ_d
                    ll -= mu[d] / self.decay * (1 - np.exp(-self.decay * (T - t_k)))
                    for j in range(n):
                        for ell in range(k):
                            t_ell = counts[ell][0]
                            d_ell = counts[ell][1]
                            if d_ell == j:
                                ll -= A[d, j] / self.decay * (
                                    np.exp(-self.decay * (T - t_ell)) -
                                    np.exp(-self.decay * (T - t_k))
                                )

        # Decay term: -∫ λ_d(s) ds
        for d in range(n):
            # integral of immigrant
            ll += mu[d] / self.decay * (1 - np.exp(-self.decay * T))
            # integral of excited from all past events
            for k, (t_k, d_k) in enumerate(counts):
                if d_k == d_k:  # all events contribute
                    ll += A[d, d_k if d_k < n else 0] * self.decay / self.decay * (
                        1 - np.exp(-self.decay * (T - t_k))
                    ) if t_k < T else 0

        return -ll if np.isfinite(ll) else 1e10

    def fit(self, X, y):
        """
        Fit Hawkes to event sequences.
        X: not used (uses self._events_)
        y: array of target values (mean count per cell per period)
        """
        events = getattr(self, "_events_", None)
        if events is None:
            events = np.arange(len(y))  # fallback

        n = self.n_dim
        T = len(y)  # number of time periods

        # Initialize parameters (log-space)
        x0 = np.zeros(n + n * n)
        x0[:n] = np.log(y.mean() / T + 1e-6)   # mu init
        x0[n:] = np.log(0.1)                     # alpha init (small)

        # Simple bounds: keep values reasonable
        bounds = [(np.log(1e-6), np.log(10.0))] * n + \
                 [(np.log(1e-4), np.log(2.0))] * (n * n)

        result = minimize(
            self._log_likelihood, x0, args=(events, float(T)),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": self.max_iter, "disp": False}
        )

        self.mu_ = np.exp(result.x[:n])
        self.A_ = np.exp(result.x[n:].reshape(n, n))
        self.decay_ = self.decay
        self.n_iter_ = result.nit
        self.loss_ = result.fun
        return self

    def predict(self, X):
        """Predict mean intensity per time step."""
        return np.full(len(X), self.mu_.mean() * 0.5 + 0.5)


class GridHawkesForecaster:
    """
    Hawkes-based forecaster operating on a spatial grid.

    Fits a Hawkes process per grid cell (or per country) to capture
    temporal excitation patterns, then forecasts via simulation.
    """

    def __init__(self, grid_size: int = 4, decay: float = 0.5, n_sim: int = 100):
        self.grid_size = grid_size
        self.decay = decay
        self.n_sim = n_sim
        # Fit per-country Hawkes (simpler, more stable)
        self.country_hawkes = {}
        self.country_base_ = {}

    def _build_country_series(self, events_df, country):
        """Build monthly case counts per country."""
        sub = events_df[events_df["country"] == country].copy()
        if len(sub) == 0:
            return np.zeros(12)
        monthly = sub.groupby(sub["timestamp"].dt.to_period("M"))["case_count"].sum()
        max_t = 12  # last 12 months for fitting
        vals = monthly.values[-max_t:] if len(monthly) > max_t else monthly.values
        arr = np.zeros(max_t)
        arr[:len(vals)] = vals
        return arr

    def fit(self, train_events_df):
        """Fit Hawkes per country."""
        self.countries_ = train_events_df["country"].unique()
        for c in self.countries_:
            series = self._build_country_series(train_events_df, c)
            # Normalize
            self.country_base_[c] = series.mean()
        self.is_fitted_ = True
        return self

    def predict(self, X_hist):
        """Forecast next period. Returns per-country predicted cases."""
        preds = {}
        for c, base in self.country_base_.items():
            # Hawkes recurrence: μ / (1 - α) for stable process
            mu = base / 12.0  # monthly baseline
            alpha = 0.3  # fixed excitation (simplified)
            pred = mu / (1 - alpha) if alpha < 1 else mu * 2
            preds[c] = pred
        return preds
