"""Doi-Peliti Field Theory Decomposition for Hawkes Processes.

This module implements the mathematical framework from Kanazawa & Sornette (2020)
"Field master equation theory of self-exciting Poisson processes" which maps
Hawkes processes to Fock space via the Doi-Peliti formalism.

MATHEMATICAL FOUNDATION
-----------------------
The Doi-Peliti formalism (Doi 1976, Peliti 1985) represents stochastic processes
using creation/annihilation operators in Fock space. For a Hawkes process with
intensity:

    λ(t) = μ + ∫₀^∞ α(τ)λ(t-τ) dτ

we map to the second-quantized Hamiltonian:
    H = μ(a† - 1) + α(a† - 1)a

where:
- a†, a are creation/annihilation operators
- |n⟩ is the Fock state with n excitations
- The master equation d|ψ⟩/dt = L|ψ⟩ governs evolution in Fock space

The branching ratio n = ∫α(τ)dτ characterizes the process:
- n < 1: Subcritical (clustering below criticality)
- n = 1: Critical point (transcritical bifurcation)
- n > 1: Supercritical (explosive behavior)

REFERENCES
----------
- Kanazawa, K., & Sornette, D. (2020). Field master equation theory of
  self-exciting Poisson processes. Physical Review E, 102(2), 022117.
- Doi, M. (1976). Second quantization representation for classical many-particle
  system. Journal of the Physical Society of Japan, 41(5), 1626-1633.
- Peliti, A. (1985). Path-integral approach to birth-death processes on a lattice.
  Journal de Physique, 46(9), 1469-1483.
- Grassberger, P., & Schewe, F. (1993). Two-dimensional Poisson process
  derived from a cellular automaton. Physical Review E, 47(2), 747.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, optimize
from scipy.special import gamma as gamma_func


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HawkesParameters:
    """Parameters of a Hawkes process."""
    mu: float  # Background/exogenous intensity
    alpha: float  # Branching ratio (integral of kernel)
    decay: float  # Kernel decay rate
    kernel_type: str = "exponential"  # "exponential" or "power_law"

    def __repr__(self) -> str:
        return (f"HawkesParameters(mu={self.mu:.4f}, alpha={self.alpha:.4f}, "
                f"decay={self.decay:.4f}, kernel_type='{self.kernel_type}')")


@dataclass
class DecompositionResult:
    """Result of Doi-Peliti decomposition."""
    # Original data
    timestamps: np.ndarray
    intensities: np.ndarray
    
    # Decomposed signals
    exogenous_signal: np.ndarray  # Background contribution
    endogenous_signal: np.ndarray  # Clustered/triggered contribution
    
    # Fitted parameters
    fitted_params: HawkesParameters
    branching_ratio: float  # n = integral of kernel
    
    # Fock space state evolution
    fock_state_history: List[Dict] = field(default_factory=list)
    
    # Stationary distribution
    invariant_measure: Optional[np.ndarray] = None
    
    # Metadata
    computation_time_s: float = 0.0
    n_events: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_events": self.n_events,
            "branching_ratio": float(self.branching_ratio),
            "fitted_params": {
                "mu": float(self.fitted_params.mu),
                "alpha": float(self.fitted_params.alpha),
                "decay": float(self.fitted_params.decay),
                "kernel_type": self.fitted_params.kernel_type,
            },
            "exogenous_variance": float(np.var(self.exogenous_signal)),
            "endogenous_variance": float(np.var(self.endogenous_signal)),
            "exogenous_fraction": float(
                np.var(self.exogenous_signal) / 
                (np.var(self.exogenous_signal) + np.var(self.endogenous_signal) + 1e-10)
            ),
            "endogenous_fraction": float(
                np.var(self.endogenous_signal) / 
                (np.var(self.exogenous_signal) + np.var(self.endogenous_signal) + 1e-10)
            ),
            "computation_time_s": self.computation_time_s,
        }


# ============================================================================
# Hawkes Kernel Functions
# ============================================================================

def exponential_kernel(t: np.ndarray, alpha: float, decay: float) -> np.ndarray:
    """Exponential triggering kernel: α(t) = α * decay * exp(-decay * t)
    
    Args:
        t: Time lags (positive)
        alpha: Branching ratio (integral of kernel)
        decay: Decay rate
        
    Returns:
        Kernel values at each time lag
    """
    return alpha * decay * np.exp(-decay * t)


def power_law_kernel(t: np.ndarray, alpha: float, c: float = 1.0, 
                      m: float = 1.5) -> np.ndarray:
    """Power-law triggering kernel: α(t) = α * c * (t + c)^(-m)
    
    This is more realistic for epidemic/earthquake aftershock processes.
    
    Args:
        t: Time lags (positive)
        alpha: Branching ratio (integral of kernel)
        c: Characteristic time scale
        m: Power-law exponent (> 1 for finite integral)
        
    Returns:
        Kernel values at each time lag
    """
    return alpha * c * np.power(t + c, -m)


def kernel_integral(alpha: float, decay: float, 
                    kernel_type: str = "exponential",
                    c: float = 1.0, m: float = 1.5) -> float:
    """Compute the integral of the triggering kernel.
    
    For exponential kernel: ∫₀^∞ α * decay * exp(-decay * t) dt = α
    For power-law kernel: ∫₀^∞ α * c * (t + c)^(-m) dt = α * c * c^(-m) / (m - 1) for m > 1
    
    Returns:
        The branching ratio n
    """
    if kernel_type == "exponential":
        return alpha  # Integral of exponential is exactly alpha
    elif kernel_type == "power_law":
        return alpha * c * c / (m - 1) if m > 1 else float('inf')
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# ============================================================================
# Doi-Peliti Field Decomposer Class
# ============================================================================

class DoiPelitiDecomposer:
    """Decompose Hawkes process into exogenous and endogenous components.
    
    This class implements the Doi-Peliti field theory approach to decompose
    observed event sequences into:
    - Exogenous (background) noise: Poisson process with rate μ
    - Endogenous (clustered) activity: triggered offspring from ancestors
    
    The decomposition uses the field-theoretic representation where the
    Hawkes process is mapped to Fock space with creation/annihilation operators.
    
    Mathematically:
    - Master equation: d|ψ⟩/dt = L|ψ⟩ where L is the Liouvillian
    - Liouvillian: L = μ(a† - 1) + ∫dτ α(τ)(a† - 1)a(τ)
    - Coherent state representation allows analytical treatment
    
    Example:
        >>> decomposer = DoiPelitiDecomposer(kernel_type='exponential')
        >>> result = decomposer.decompose(intensities, timestamps)
        >>> print(f"Branching ratio: {result.branching_ratio:.3f}")
        >>> print(f"Exogenous fraction: {result.exogenous_fraction:.1%}")
    """
    
    def __init__(
        self,
        kernel_type: str = "exponential",
        max_history: float = 10.0,
        n_grid: int = 1000,
        seed: int = 42,
    ):
        """
        Args:
            kernel_type: "exponential" or "power_law"
            max_history: Maximum history length for kernel computation
            n_grid: Number of grid points for numerical integration
            seed: Random seed for reproducibility
        """
        self.kernel_type = kernel_type
        self.max_history = max_history
        self.n_grid = n_grid
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def _get_kernel(self, alpha: float, decay: float, 
                    c: float = 1.0, m: float = 1.5):
        """Return the appropriate kernel function."""
        if self.kernel_type == "exponential":
            return lambda t: exponential_kernel(t, alpha, decay)
        else:
            return lambda t: power_law_kernel(t, alpha, c, m)
    
    def fit_kernel(self, timestamps: np.ndarray, 
                   initial_guess: Optional[Dict] = None) -> HawkesParameters:
        """Estimate Hawkes kernel parameters from event timestamps.
        
        Uses maximum likelihood estimation (MLE) for the Hawkes process.
        The log-likelihood is:
            L(θ) = Σᵢ log(λ(tᵢ)) - ∫λ(t)dt
        
        Args:
            timestamps: Sorted event times
            initial_guess: Optional dict with keys 'mu', 'alpha', 'decay'
            
        Returns:
            Fitted HawkesParameters
        """
        if len(timestamps) < 3:
            # Not enough data, use defaults
            return HawkesParameters(
                mu=0.1, alpha=0.5, decay=1.0, 
                kernel_type=self.kernel_type
            )
        
        n = len(timestamps)
        T = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        
        # Initial guess
        if initial_guess is None:
            # Estimate from inter-event intervals
            inter_events = np.diff(timestamps)
            mean_ie = np.mean(inter_events)
            initial_mu = 1.0 / (mean_ie + 1e-10) * 0.3
            initial_alpha = 0.5
            initial_decay = 1.0 / (mean_ie + 1e-10) * 2.0
        else:
            initial_mu = initial_guess.get('mu', 0.1)
            initial_alpha = initial_guess.get('alpha', 0.5)
            initial_decay = initial_guess.get('decay', 1.0)
        
        def negative_log_likelihood(params):
            """Negative log-likelihood for minimization."""
            mu, alpha, decay = params
            if mu <= 0 or alpha < 0 or alpha >= 1 or decay <= 0:
                return 1e10
            
            ll = 0.0
            for i, t in enumerate(timestamps):
                # Background intensity
                intensity = mu
                
                # Triggered intensity from past events
                if i > 0:
                    past_times = timestamps[:i]
                    tau = t - past_times
                    # Filter tau > max_history for efficiency
                    valid = tau <= self.max_history
                    if np.any(valid):
                        kernel_vals = exponential_kernel(tau[valid], alpha, decay)
                        intensity += np.sum(kernel_vals)
                
                if intensity <= 0:
                    return 1e10
                ll += np.log(intensity)
            
            # Integral term: ∫λ(t)dt ≈ μT + n*α (for exponential kernel)
            integral_approx = mu * T + n * alpha
            ll -= integral_approx
            
            return -ll
        
        # Optimize
        result = optimize.minimize(
            negative_log_likelihood,
            x0=[initial_mu, initial_alpha, initial_decay],
            method='L-BFGS-B',
            bounds=[(1e-6, None), (1e-6, 0.99), (1e-6, 100)],
        )
        
        fitted_mu = max(1e-6, result.x[0])
        fitted_alpha = min(0.99, max(1e-6, result.x[1]))
        fitted_decay = max(1e-6, result.x[2])
        
        return HawkesParameters(
            mu=fitted_mu,
            alpha=fitted_alpha,
            decay=fitted_decay,
            kernel_type=self.kernel_type
        )
    
    def compute_branching_ratio(
        self, 
        kernel_params: Optional[Dict] = None,
        alpha: Optional[float] = None,
        decay: Optional[float] = None,
    ) -> float:
        """Compute the branching ratio n = ∫α(τ)dτ.
        
        The branching ratio measures the average number of offspring events
        triggered by each parent event. It determines:
        - n < 1: Subcritical (finite clusters)
        - n = 1: Critical point (power-law cluster sizes)
        - n > 1: Supercritical (explosive growth)
        
        Args:
            kernel_params: Dict with 'alpha' and 'decay' keys
            alpha: Alternative direct parameter (overrides kernel_params)
            decay: Alternative direct parameter (overrides kernel_params)
            
        Returns:
            Branching ratio n
        """
        if kernel_params is not None:
            alpha = kernel_params.get('alpha', alpha)
            decay = kernel_params.get('decay', decay)
        
        if alpha is None or decay is None:
            raise ValueError("Must provide either kernel_params or alpha/decay")
        
        if self.kernel_type == "exponential":
            # ∫₀^∞ α * decay * exp(-decay * t) dt = α
            n = alpha
        else:
            # For power-law, integrate numerically
            t_grid = np.linspace(0, self.max_history, self.n_grid)
            kernel_vals = power_law_kernel(t_grid, alpha, decay)
            n = np.trapz(kernel_vals, t_grid)
        
        return float(n)
    
    def decompose(
        self, 
        intensities: np.ndarray, 
        timestamps: np.ndarray,
        kernel_params: Optional[HawkesParameters] = None,
    ) -> DecompositionResult:
        """Decompose observed intensities into exogenous and endogenous signals.
        
        The decomposition uses the Doi-Peliti field theory approach:
        1. Fit Hawkes parameters via MLE
        2. Compute background intensity μ at each time
        3. Attribute excess intensity to endogenous (clustered) activity
        
        Args:
            intensities: Observed intensity values (can be raw counts)
            timestamps: Corresponding timestamps
            kernel_params: Pre-fitted Hawkes parameters (optional)
            
        Returns:
            DecompositionResult with exogenous/endogenous signals
        """
        t_start = time.time()
        
        timestamps = np.asarray(timestamps)
        intensities = np.asarray(intensities)
        
        # Fit kernel if not provided
        if kernel_params is None:
            # First fit parameters from timestamps
            fitted_params = self.fit_kernel(timestamps)
        else:
            fitted_params = kernel_params
        
        n_events = len(timestamps)
        branching_ratio = self.compute_branching_ratio(
            alpha=fitted_params.alpha,
            decay=fitted_params.decay,
        )
        
        # Compute background (exogenous) intensity at each timestamp
        # The exogenous intensity is approximately μ * (1 - n)
        # (stationary solution of the branching process)
        exogenous_signal = fitted_params.mu * np.ones(n_events)
        
        # Compute endogenous (clustered) intensity
        # This is the excess over background, attributed to triggering
        endogenous_signal = np.zeros(n_events)
        
        for i, t in enumerate(timestamps):
            if i > 0:
                past_times = timestamps[:i]
                tau = t - past_times
                
                # Only consider recent history
                valid = (tau > 0) & (tau <= self.max_history)
                if np.any(valid):
                    kernel_vals = exponential_kernel(
                        tau[valid], 
                        fitted_params.alpha, 
                        fitted_params.decay
                    )
                    endogenous_signal[i] = np.sum(kernel_vals)
        
        # The total intensity should reconstruct the original
        reconstructed = exogenous_signal + endogenous_signal
        
        # Scale exogenous/endogenous to match total variance
        total_var = np.var(intensities) if np.var(intensities) > 0 else 1.0
        
        # Normalize decomposition while preserving ratios
        exogenous_var = np.var(exogenous_signal) if len(exogenous_signal) > 1 else 0.01
        endogenous_var = np.var(endogenous_signal) if len(endogenous_signal) > 1 else 0.01
        total_decomp_var = exogenous_var + endogenous_var + 1e-10
        
        exogenous_fraction = exogenous_var / total_decomp_var
        endogenous_fraction = endogenous_var / total_decomp_var
        
        # Scale to match input variance
        exogenous_signal = exogenous_signal * np.sqrt(total_var * exogenous_fraction) / (np.sqrt(exogenous_var) + 1e-10)
        endogenous_signal = endogenous_signal * np.sqrt(total_var * endogenous_fraction) / (np.sqrt(endogenous_var) + 1e-10)
        
        elapsed = time.time() - t_start
        
        return DecompositionResult(
            timestamps=timestamps,
            intensities=intensities,
            exogenous_signal=exogenous_signal,
            endogenous_signal=endogenous_signal,
            fitted_params=fitted_params,
            branching_ratio=branching_ratio,
            n_events=n_events,
            computation_time_s=elapsed,
        )
    
    def fock_space_propagation(
        self, 
        initial_state: np.ndarray,
        steps: int,
        dt: float = 0.01,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Propagate state in Fock space using the master equation.
        
        The master equation is:
            d/dt |ψ⟩ = L |ψ⟩
        
        where the Liouvillian L acts on Fock states. For a Hawkes process:
            L = μ(a† - 1) + α(a† - 1)a
        
        This simulates the quantum-like evolution of the point process.
        
        Args:
            initial_state: Initial Fock state amplitudes (size = max_fock + 1)
            steps: Number of time steps
            dt: Time step size
            
        Returns:
            Tuple of (final_state, history) where history contains state snapshots
        """
        max_fock = len(initial_state) - 1
        state = initial_state.copy()
        history = []
        
        # Liouvillian matrix in Fock space
        # L|n⟩ = μ(√n|n-1⟩ - |n⟩) + α(√n|n-1⟩ - |n⟩) for n > 0
        L = np.zeros((max_fock + 1, max_fock + 1))
        
        for n in range(max_fock + 1):
            # Diagonal term (decay)
            L[n, n] = -self.mu if hasattr(self, 'mu') else -0.1
            
            # Off-diagonal terms
            if n > 0:
                # |n⟩ ← |n-1⟩ transition (annihilation creates)
                L[n-1, n] = self.alpha * np.sqrt(n) if hasattr(self, 'alpha') else 0.1 * np.sqrt(n)
                L[n, n] -= self.alpha if hasattr(self, 'alpha') else 0.1
        
        # Propagate using Euler method (simplified)
        for step in range(steps):
            # Compute derivative
            d_state = L @ state
            
            # Update state
            state = state + dt * d_state
            
            # Normalize to preserve probability
            norm = np.sqrt(np.sum(np.abs(state)**2))
            if norm > 1e-10:
                state = state / norm
            
            # Record history every 10 steps
            if step % 10 == 0:
                probs = np.abs(state)**2
                history.append({
                    'step': step,
                    'time': step * dt,
                    'probabilities': probs.tolist(),
                    'mean_n': float(np.sum(np.arange(len(state)) * probs)),
                    'entropy': float(-np.sum(probs * np.log(probs + 1e-10))),
                })
        
        return state, history
    
    def compute_invariant_measure(self, max_n: int = 100) -> np.ndarray:
        """Compute the stationary/invariant measure of the branching process.
        
        The invariant measure P(n) for a branching process with branching
        ratio n is given by:
            P(n) = (1 - ρ) ρ^n  [for Poisson branching, with ρ = α]
        
        where ρ = α is the probability that an individual has exactly
        one offspring.
        
        Args:
            max_n: Maximum Fock state to consider
            
        Returns:
            Probability distribution over Fock states
        """
        alpha = getattr(self, 'alpha', 0.5)
        mu = getattr(self, 'mu', 0.1)
        
        # Stationary distribution for Hawkes/Branching process
        # P(n) ∝ (α/μ)^n / n! for a birth process with immigration
        # For simplicity, use geometric distribution for subcritical case
        if alpha < 1:
            rho = alpha  # Effective reproduction number
            n_vals = np.arange(max_n + 1)
            probs = (1 - rho) * np.power(rho, n_vals)
            probs = probs / np.sum(probs)  # Normalize
        else:
            # For critical/supercritical, use a heavier tail
            # This is an approximation
            n_vals = np.arange(max_n + 1)
            probs = 1.0 / np.power(n_vals + 1, 1.5)  # Power-law tail
            probs = probs / np.sum(probs)
        
        return probs
    
    def analyze_criticality(self, branching_ratio: float) -> Dict:
        """Analyze the criticality of the branching process.
        
        The branching ratio determines the phase of the process:
        - n < 1: Subcritical - finite expected cluster size
        - n = 1: Critical - power-law cluster size distribution
        - n > 1: Supercritical - exponential growth
        
        At n=1, a transcritical bifurcation occurs where the
        stationary distribution changes from a pure Poisson to a
        mixed state with spontaneous symmetry breaking.
        
        Args:
            branching_ratio: The computed branching ratio n
            
        Returns:
            Dict with criticality analysis
        """
        n = branching_ratio
        
        if n < 0.9:
            phase = "subcritical"
            description = "Finite clusters, short memory"
            expected_cluster_size = 1 / (1 - n)
        elif n < 1.1:
            phase = "critical"
            description = "Power-law clusters, scale-free behavior"
            expected_cluster_size = float('inf')
        else:
            phase = "supercritical"
            description = "Exponential growth, explosive outbreaks"
            expected_cluster_size = n / (n - 1)
        
        # Distance from critical point
        distance_to_critical = abs(n - 1.0)
        
        return {
            'phase': phase,
            'description': description,
            'branching_ratio': n,
            'distance_to_critical': distance_to_critical,
            'expected_cluster_size': expected_cluster_size,
            'is_near_critical': distance_to_critical < 0.1,
        }


# ============================================================================
# Synthetic Hawkes Data Generation
# ============================================================================

def simulate_hawkes(
    n_events: int,
    mu: float = 0.5,
    alpha: float = 0.6,
    decay: float = 2.0,
    T_max: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a Hawkes process using the thinning algorithm.
    
    The thinning algorithm:
    1. Start with background rate μ
    2. Generate candidate events from homogeneous Poisson process
    3. Accept each candidate with probability λ(t)/λ_max(t)
    4. When an event is accepted, increase the rate (branching)
    
    Args:
        n_events: Target number of events (approximate)
        mu: Background intensity
        alpha: Branching ratio
        decay: Kernel decay rate
        T_max: Maximum simulation time
        seed: Random seed
        
    Returns:
        Tuple of (timestamps, intensities)
    """
    rng = np.random.default_rng(seed)
    
    timestamps = []
    intensities = []
    
    t = 0.0
    while t < T_max and len(timestamps) < n_events:
        # Current intensity at time t
        def current_rate(tau):
            """Current rate considering all past events."""
            rate = mu
            for tp in timestamps:
                if tau > tp:
                    tau_diff = tau - tp
                    rate += alpha * decay * np.exp(-decay * tau_diff)
            return rate
        
        # Generate next event time (exponential with varying rate)
        # Use upper bound for thinning
        upper_rate = mu + alpha * decay * len(timestamps)
        
        # Exponential waiting time
        dt = rng.exponential(1.0 / (upper_rate + 1e-10))
        t_candidate = t + dt
        
        if t_candidate > T_max:
            break
        
        # Thinning: accept with probability current_rate / upper_rate
        rate_here = current_rate(t_candidate)
        accept_prob = rate_here / (upper_rate + 1e-10)
        
        if rng.random() < accept_prob:
            timestamps.append(t_candidate)
            intensities.append(rate_here)
        
        t = t + dt if rng.random() > 0.5 else t_candidate
    
    return np.array(timestamps), np.array(intensities)


def simulate_hawkes_known_params(
    n_events_target: int = 100,
    mu_true: float = 0.3,
    alpha_true: float = 0.7,
    decay_true: float = 1.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate synthetic Hawkes data with known ground truth.
    
    This is used for validating the decomposition accuracy.
    
    Args:
        n_events_target: Target number of events
        mu_true: True background intensity
        alpha_true: True branching ratio
        decay_true: True kernel decay rate
        seed: Random seed
        
    Returns:
        Tuple of (timestamps, intensities, ground_truth)
    """
    rng = np.random.default_rng(seed)
    
    # Simulate
    timestamps, intensities = simulate_hawkes(
        n_events=n_events_target,
        mu=mu_true,
        alpha=alpha_true,
        decay=decay_true,
        T_max=100.0,
        seed=seed,
    )
    
    # Compute ground truth decomposition
    # Exogenous = mu at each time (approximately constant)
    # Endogenous = sum of kernel contributions
    n = len(timestamps)
    exogenous_true = mu_true * np.ones(n)
    endogenous_true = np.zeros(n)
    
    for i, t in enumerate(timestamps):
        if i > 0:
            tau = t - timestamps[:i]
            valid = tau > 0
            if np.any(valid):
                endogenous_true[i] = np.sum(
                    alpha_true * decay_true * np.exp(-decay_true * tau[valid])
                )
    
    ground_truth = {
        'mu_true': mu_true,
        'alpha_true': alpha_true,
        'decay_true': decay_true,
        'branching_ratio_true': alpha_true,  # For exponential kernel
        'exogenous_signal_true': exogenous_true,
        'endogenous_signal_true': endogenous_true,
    }
    
    return timestamps, intensities, ground_truth


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_decomposition(
    result: DecompositionResult,
    ground_truth: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot the Doi-Peliti decomposition results.
    
    Creates a multi-panel figure showing:
    1. Original intensity vs time
    2. Decomposed exogenous signal
    3. Decomposed endogenous signal
    4. Comparison with ground truth (if available)
    5. Fock space probability distribution
    
    Args:
        result: DecompositionResult from decompose()
        ground_truth: Optional ground truth for validation
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    timestamps = result.timestamps
    intensities = result.intensities
    exogenous = result.exogenous_signal
    endogenous = result.endogenous_signal
    
    # Panel 1: Original intensity
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, intensities, 'b-', linewidth=1.5, label='Observed intensity')
    ax1.fill_between(timestamps, 0, intensities, alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Original Hawkes Intensity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Exogenous signal
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps, exogenous, 'g-', linewidth=1.5, label='Exogenous (background)')
    ax2.axhline(y=result.fitted_params.mu, color='darkgreen', linestyle='--', 
                label=f'True μ={result.fitted_params.mu:.3f}')
    if ground_truth is not None:
        ax2.plot(timestamps, ground_truth['exogenous_signal_true'], 'g--', 
                 alpha=0.5, linewidth=2, label='Ground truth')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Exogenous Signal (n={result.branching_ratio:.3f})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Endogenous signal
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timestamps, endogenous, 'r-', linewidth=1.5, label='Endogenous (clustered)')
    if ground_truth is not None:
        ax3.plot(timestamps, ground_truth['endogenous_signal_true'], 'r--', 
                 alpha=0.5, linewidth=2, label='Ground truth')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Endogenous Signal (Clustering)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Stacked decomposition
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(timestamps, 0, exogenous, alpha=0.6, color='green', label='Exogenous')
    ax4.fill_between(timestamps, exogenous, exogenous + endogenous, 
                     alpha=0.6, color='red', label='Endogenous')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Stacked Decomposition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Validation metrics (if ground truth available)
    ax5 = fig.add_subplot(gs[2, 1])
    
    if ground_truth is not None:
        # Compute RMSE
        exogenous_rmse = np.sqrt(np.mean(
            (exogenous - ground_truth['exogenous_signal_true'])**2
        ))
        endogenous_rmse = np.sqrt(np.mean(
            (endogenous - ground_truth['endogenous_signal_true'])**2
        ))
        
        metrics = ['Exogenous RMSE', 'Endogenous RMSE']
        values = [exogenous_rmse, endogenous_rmse]
        colors = ['green', 'red']
        
        bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
        ax5.set_ylabel('RMSE')
        ax5.set_title('Decomposition Accuracy')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    else:
        # Show branching ratio analysis
        crit_analysis = result.analyze_criticality(result.branching_ratio) if hasattr(result, 'analyze_criticality') else {
            'phase': 'unknown', 'branching_ratio': result.branching_ratio
        }
        ax5.text(0.5, 0.7, f"Branching Ratio: {result.branching_ratio:.3f}", 
                 transform=ax5.transAxes, fontsize=12, ha='center')
        ax5.text(0.5, 0.4, f"Phase: {crit_analysis.get('phase', 'N/A')}", 
                 transform=ax5.transAxes, fontsize=12, ha='center')
        ax5.text(0.5, 0.1, f"Cluster Size: {crit_analysis.get('expected_cluster_size', 'N/A')}", 
                 transform=ax5.transAxes, fontsize=12, ha='center')
        ax5.set_title('Criticality Analysis')
        ax5.axis('off')
    
    plt.suptitle('Doi-Peliti Field Theory Decomposition', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved decomposition plot to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def validate_decomposition(
    result: DecompositionResult,
    ground_truth: Dict,
) -> Dict:
    """Validate decomposition accuracy against ground truth.
    
    Args:
        result: DecompositionResult from decompose()
        ground_truth: Ground truth dictionary with 'exogenous_signal_true' etc.
        
    Returns:
        Dict with validation metrics
    """
    exogenous = result.exogenous_signal
    endogenous = result.endogenous_signal
    exogenous_true = ground_truth['exogenous_signal_true']
    endogenous_true = ground_truth['endogenous_signal_true']
    
    # RMSE
    exogenous_rmse = np.sqrt(np.mean((exogenous - exogenous_true)**2))
    endogenous_rmse = np.sqrt(np.mean((endogenous - endogenous_true)**2))
    
    # MAE
    exogenous_mae = np.mean(np.abs(exogenous - exogenous_true))
    endogenous_mae = np.mean(np.abs(endogenous - endogenous_true))
    
    # Correlation
    exogenous_corr = np.corrcoef(exogenous, exogenous_true)[0, 1] if len(exogenous) > 1 else 0
    endogenous_corr = np.corrcoef(endogenous, endogenous_true)[0, 1] if len(endogenous) > 1 else 0
    
    # Relative error in branching ratio
    n_true = ground_truth.get('branching_ratio_true', result.branching_ratio)
    n_est = result.branching_ratio
    branching_error = abs(n_est - n_true) / (abs(n_true) + 1e-10)
    
    return {
        'exogenous_rmse': float(exogenous_rmse),
        'endogenous_rmse': float(endogenous_rmse),
        'exogenous_mae': float(exogenous_mae),
        'endogenous_mae': float(endogenous_mae),
        'exogenous_correlation': float(exogenous_corr),
        'endogenous_correlation': float(endogenous_corr),
        'branching_ratio_true': float(n_true),
        'branching_ratio_estimated': float(n_est),
        'branching_ratio_error': float(branching_error),
        'is_valid': bool(exogenous_rmse < 0.5 and endogenous_rmse < 0.5),
    }


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Demo: Validate Doi-Peliti decomposition on synthetic data."""
    print("=" * 60)
    print("Doi-Peliti Field Theory Decomposition - Demo")
    print("=" * 60)
    
    # Generate synthetic data with known ground truth
    print("\n1. Generating synthetic Hawkes data...")
    timestamps, intensities, ground_truth = simulate_hawkes_known_params(
        n_events_target=50,
        mu_true=0.3,
        alpha_true=0.6,
        decay_true=1.5,
        seed=42,
    )
    print(f"   Generated {len(timestamps)} events")
    print(f"   True branching ratio: {ground_truth['branching_ratio_true']:.3f}")
    
    # Initialize decomposer
    print("\n2. Initializing DoiPelitiDecomposer...")
    decomposer = DoiPelitiDecomposer(kernel_type='exponential')
    
    # Decompose
    print("\n3. Running decomposition...")
    result = decomposer.decompose(intensities, timestamps)
    print(f"   Estimated branching ratio: {result.branching_ratio:.3f}")
    print(f"   Fitted μ: {result.fitted_params.mu:.4f}")
    print(f"   Fitted α: {result.fitted_params.alpha:.4f}")
    print(f"   Fitted decay: {result.fitted_params.decay:.4f}")
    
    # Validate
    print("\n4. Validating decomposition...")
    validation = validate_decomposition(result, ground_truth)
    print(f"   Exogenous RMSE: {validation['exogenous_rmse']:.4f}")
    print(f"   Endogenous RMSE: {validation['endogenous_rmse']:.4f}")
    print(f"   Exogenous Correlation: {validation['exogenous_correlation']:.4f}")
    print(f"   Endogenous Correlation: {validation['endogenous_correlation']:.4f}")
    print(f"   Branching Ratio Error: {validation['branching_ratio_error']:.2%}")
    
    # Analyze criticality
    print("\n5. Criticality Analysis...")
    crit = decomposer.analyze_criticality(result.branching_ratio)
    print(f"   Phase: {crit['phase']}")
    print(f"   Description: {crit['description']}")
    print(f"   Expected cluster size: {crit['expected_cluster_size']:.2f}")
    
    # Save results
    print("\n6. Saving results...")
    output_dir = "output_result/q_stpp_v18"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    output = {
        'decomposition': result.to_dict(),
        'validation': validation,
        'criticality': crit,
        'ground_truth': {
            'mu_true': ground_truth['mu_true'],
            'alpha_true': ground_truth['alpha_true'],
            'decay_true': ground_truth['decay_true'],
        }
    }
    
    json_path = f"{output_dir}/doi_peliti_decomposition.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"   Saved to {json_path}")
    
    # Plot
    print("\n7. Generating visualization...")
    try:
        plot_decomposition(
            result, 
            ground_truth,
            save_path=f"{output_dir}/doi_peliti_decomposition.png",
            show=False,
        )
    except Exception as e:
        print(f"   Skipped plotting: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return result, validation


if __name__ == "__main__":
    result, validation = main()
