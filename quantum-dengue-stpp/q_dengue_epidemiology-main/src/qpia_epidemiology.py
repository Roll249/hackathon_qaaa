"""QPIA (Quantum Path Integral Approach) for Epidemiology.

Implementation based on Gautam & Ahn 2024 "Quantum Path Integral Approach for
Vehicle Routing Optimization With Limited Qubit" (IEEE TITS 2024).

Key insight from the paper:
- QPIA sums over ALL paths with phase e^(iS/ℏ) where S is the action
- Paths with lower action constructively interfere (Feynman path integral)
- This works for VRP because the problem = find optimal PATH (sequence of cities)

For epidemiology, we apply the same principle:
- Problem = find transmission chains that lead to hotspots
- State space = paths (sequences of communes) not individual nodes
- Action = sum of costs (distance × risk intensity along the path)
- Amplitude = e^(i * action) for each path

The QPIA approach differs fundamentally from coined quantum walk:
| Aspect         | Coined QW (arc-space)    | QPIA (path-space)           |
|----------------|--------------------------|----------------------------|
| State space    | Individual nodes         | Paths/sequences            |
| Evolution      | Walk on nodes            | Sum over all paths         |
| Interference   | Diffusion (probability)   | Phase from action          |
| Target         | Single marked node       | Path ending at target      |

Reference: Gautam & Ahn, IEEE TITS 2024
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_parent = _Path(__file__).parent.parent
if str(_parent) not in _sys.path:
    _sys.path.insert(0, str(_parent))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import math


# ═══════════════════════════════════════════════════════════════════════════════
# PATH INTEGRAL MATHEMATICS
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Feynman path integral states that the amplitude to go from initial state
# |ψ_i⟩ to final state |ψ_f⟩ is:
#
#   ⟨ψ_f|e^(-iHt/ℏ)|ψ_i⟩ = ∫ 𝒟[x] e^(iS[x]/ℏ)
#
# where S[x] = ∫ L(x, ẋ) dt is the action along path x.
#
# For discrete systems (our epidemiology graph):
#   - Time is discretized into T steps
#   - Each path is a sequence: (v₀ → v₁ → ... → v_T)
#   - Action: S[path] = Σ_t cost(v_t, v_{t+1})
#
# The amplitude for a specific path is:
#   A[path] = e^(i * S[path] / ℏ)
#
# For computational purposes, we set ℏ = 1 and use:
#   A[path] = exp(i * S[path] * action_scale)
#
# The probability of ending at node j is:
#   P(j) = |Σ_{paths ending at j} A[path]|^2 / Z
#
# where Z = Σ_{all paths} |A[path]|² is the normalization.
#


@dataclass
class PathState:
    """Represents a path (transmission chain) in the epidemiology graph.
    
    Attributes:
        nodes: Sequence of node indices forming the path
        action: The action S for this path (lower = lower cost)
        amplitude: Complex amplitude e^(i*S)
        probability: |amplitude|² (should be 1 if action is real)
    """
    nodes: tuple[int, ...]
    action: float
    amplitude: complex = field(init=False)
    
    def __post_init__(self):
        # Amplitude = e^(i * action) - complex phase from action
        self.amplitude = np.exp(1j * self.action)
    
    @property
    def probability(self) -> float:
        """Probability = |amplitude|² = 1 for real action."""
        return np.abs(self.amplitude) ** 2
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    @property
    def start(self) -> int:
        return self.nodes[0]
    
    @property
    def end(self) -> int:
        return self.nodes[-1]


class ActionFunction(Callable):
    """Base class for action functions in the path integral.
    
    The action S determines the phase e^(iS) for each path.
    Paths with lower action contribute more constructively.
    """
    
    def __call__(self, path: tuple[int, ...], 
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        raise NotImplementedError


class DistanceAction(ActionFunction):
    """Action based on total Euclidean distance of the path.
    
    S(path) = Σ_{t} distance(v_t, v_{t+1})
    
    This penalizes longer transmission chains.
    """
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            # Use adjacency weights as distances
            total += adjacency[path[i], path[i+1]]
        return total


class RiskWeightedAction(ActionFunction):
    """Action weighted by risk intensity at each node.
    
    S(path) = Σ_{t} (distance(v_t, v_{t+1}) + risk_weight * risk(v_t))
    
    This biases toward paths that traverse high-risk areas.
    """
    
    def __init__(self, risk_weight: float = 1.0):
        self.risk_weight = risk_weight
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return self.risk_weight * risk[path[0]] if path else 0.0
        
        total = 0.0
        for i in range(len(path)):
            # Add risk contribution at each node
            total += self.risk_weight * risk[path[i]]
            if i < len(path) - 1:
                # Add distance contribution
                total += adjacency[path[i], path[i+1]]
        return total


class EpidemicAction(ActionFunction):
    """Action based on epidemic transmission dynamics.
    
    S(path) = -Σ_{t} log(β * T(v_t, v_{t+1}) * risk(v_{t+1}))
    
    where T is transmission probability and β is transmission rate.
    This gives higher amplitude to more likely transmission chains.
    """
    
    def __init__(self, transmission_rate: float = 1.0):
        self.transmission_rate = transmission_rate
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            # Transmission probability from adjacency
            trans_prob = adjacency[path[i], path[i+1]]
            if trans_prob <= 0:
                trans_prob = 1e-10  # Small but non-zero
            
            # Risk factor at destination
            risk_factor = risk[path[i+1]]
            
            # Negative log of transmission probability
            # Lower S = higher transmission = more likely path
            contribution = -math.log(self.transmission_rate * trans_prob * (1 + risk_factor))
            total += contribution
        
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET-AWARE ACTION FUNCTIONS (for finding marked nodes)
# ═══════════════════════════════════════════════════════════════════════════════

class TargetAwareAction(ActionFunction):
    """Action that encodes the goal of reaching target nodes.
    
    S(path) = Σ_{t} distance(v_t, v_{t+1}) - reward * is_target(v_t+1)
    
    This makes paths to targets have LOWER action → HIGHER amplitude.
    Based on Gautam & Ahn 2024: lower action = constructive interference.
    """
    
    def __init__(self, reward: float = 2.0):
        self.reward = reward
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            # Distance cost
            dist = adjacency[path[i], path[i+1]]
            total += dist
        
        # Reward for ending at high-risk node
        total -= self.reward * risk[path[-1]]
        
        return total


class TargetPropagationAction(ActionFunction):
    """Action that propagates target information backward through the path.
    
    S(path) = Σ_{t} [distance(v_t, v_{t+1}) - α^t * risk(v_{t+1})]
    
    where α is a decay factor. This gives more weight to targets
    reached in fewer steps, modeling epidemic transmission distance.
    """
    
    def __init__(self, reward: float = 2.0, decay: float = 0.5):
        self.reward = reward
        self.decay = decay
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            # Distance cost
            dist = adjacency[path[i], path[i+1]]
            total += dist
            
            # Decayed reward at destination
            step_factor = self.decay ** i
            total -= self.reward * step_factor * risk[path[i+1]]
        
        return total


class HarmonicAction(ActionFunction):
    """Action based on harmonic oscillator potential toward targets.
    
    S(path) = Σ_{t} [distance(v_t, v_{t+1})² - λ * distance_to_target(v_t+1)]
    
    This creates a potential well toward target nodes, making paths
    ending at targets have lower action through constructive interference.
    """
    
    def __init__(self, target_position: np.ndarray, lambda_param: float = 2.0):
        self.target_position = target_position  # N x 2 array of positions
        self.lambda_param = lambda_param
    
    def __call__(self, path: tuple[int, ...],
                 adjacency: np.ndarray,
                 risk: np.ndarray) -> float:
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            # Distance cost (quadratic)
            dist = adjacency[path[i], path[i+1]]
            total += dist ** 2
            
            # Attraction to high-risk nodes
            if self.target_position is not None and len(self.target_position) > path[i+1]:
                # Attractive potential toward high-risk
                total -= self.lambda_param * risk[path[i+1]]
        
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# PATH ENUMERATION AND AMPLITUDE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_paths(
    adjacency: np.ndarray,
    start_nodes: list[int] | np.ndarray,
    max_length: int = 4,
    allow_revisit: bool = False,
) -> list[list[int]]:
    """Enumerate all paths up to max_length starting from start_nodes.
    
    Args:
        adjacency: N×N adjacency matrix (transmission strengths)
        start_nodes: List of starting node indices
        max_length: Maximum path length (number of edges)
        allow_revisit: If True, allow nodes to be revisited
    
    Returns:
        List of paths, each path is a list of node indices
    """
    paths = []
    n = adjacency.shape[0]
    
    def extend_path(path: list[int], length: int):
        if length >= max_length:
            return
        
        current = path[-1]
        neighbors = []
        for j in range(n):
            if adjacency[current, j] > 0 and j != current:
                neighbors.append(j)
        
        for neighbor in neighbors:
            new_path = path + [neighbor]
            paths.append(new_path)
            
            if allow_revisit or len(set(new_path)) == len(new_path):
                extend_path(new_path, length + 1)
    
    # Initialize paths from each start node
    for start in start_nodes:
        paths.append([start])
        if max_length > 1:
            extend_path([start], 1)
    
    return paths


def compute_path_amplitudes(
    paths: list[list[int]],
    adjacency: np.ndarray,
    risk: np.ndarray,
    action_func: ActionFunction,
    action_scale: float = 1.0,
) -> list[PathState]:
    """Compute complex amplitudes for each path using the path integral.
    
    For each path: amplitude = e^(i * S[path] * action_scale)
    
    Args:
        paths: List of paths (each path is list of node indices)
        adjacency: N×N adjacency matrix
        risk: N array of risk intensities
        action_func: Function to compute action S for a path
        action_scale: Scale factor for action (ℏ inverse)
    
    Returns:
        List of PathState objects with amplitudes
    """
    path_states = []
    
    for path in paths:
        path_tuple = tuple(path)
        action = action_func(path_tuple, adjacency, risk)
        scaled_action = action * action_scale
        
        state = PathState(nodes=path_tuple, action=scaled_action)
        path_states.append(state)
    
    return path_states


def aggregate_to_nodes(
    path_states: list[PathState],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate path amplitudes to node probabilities.
    
    For each node j:
        P(j) = |Σ_{paths ending at j} amplitude(path)|² / Z
    
    where Z = Σ_{all paths} |amplitude|² is normalization.
    
    Args:
        path_states: List of PathState objects
        n_nodes: Total number of nodes
    
    Returns:
        Tuple of (probabilities, amplitudes) arrays of shape (n_nodes,)
    """
    amplitudes = np.zeros(n_nodes, dtype=complex)
    
    for state in path_states:
        amplitudes[state.end] += state.amplitude
    
    # Normalize: P(j) = |A(j)|² / Σ_k |A(k)|²
    total_prob = np.sum(np.abs(amplitudes) ** 2)
    if total_prob > 0:
        probabilities = np.abs(amplitudes) ** 2 / total_prob
    else:
        probabilities = np.ones(n_nodes) / n_nodes
    
    return probabilities, amplitudes


# ═══════════════════════════════════════════════════════════════════════════════
# QPIA SEARCH ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QPIAResult:
    """Result from QPIA search."""
    node_probabilities: np.ndarray
    node_amplitudes: np.ndarray
    path_states: list[PathState]
    n_paths: int
    convergence_metric: float  # How much paths interfere (1 = no interference)
    
    @property
    def n_nodes(self) -> int:
        return len(self.node_probabilities)
    
    def top_k_nodes(self, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k nodes by probability."""
        indices = np.argsort(self.node_probabilities)[::-1][:k]
        return [(int(i), float(self.node_probabilities[i])) for i in indices]


def qpia_search(
    adjacency: np.ndarray,
    risk: np.ndarray,
    start_nodes: list[int] | None = None,
    max_path_length: int = 4,
    action_type: str = "risk_weighted",
    action_scale: float = 1.0,
    risk_weight: float = 2.0,
    transmission_rate: float = 1.0,
    allow_revisit: bool = False,
    seed: int = 42,
    # Target-aware parameters
    marked_nodes: list[int] | None = None,
    target_action: bool = False,
    target_reward: float = 3.0,
    backward_paths: bool = False,
) -> QPIAResult:
    """QPIA (Quantum Path Integral Approach) for epidemiology.
    
    This implements the path integral approach from Gautam & Ahn 2024:
    1. Enumerate all possible transmission paths
    2. Compute action S for each path
    3. Assign amplitude e^(iS) to each path
    4. Aggregate amplitudes to node probabilities via interference
    
    Args:
        adjacency: N×N transmission strength matrix
        risk: N array of risk intensities at each node
        start_nodes: Starting nodes for path enumeration (default: all nodes)
        max_path_length: Maximum path length (in edges)
        action_type: One of "distance", "risk_weighted", "epidemic", "target_aware"
        action_scale: Scale factor for action (higher = more oscillatory)
        risk_weight: Weight for risk in risk_weighted action
        transmission_rate: Transmission rate for epidemic action
        allow_revisit: Allow revisiting nodes in paths
        seed: Random seed
        marked_nodes: Target nodes for target-aware action
        target_action: If True, use target-aware action that rewards reaching marked nodes
        target_reward: Reward magnitude for reaching targets
        backward_paths: If True, enumerate paths BACKWARD from targets
    
    Returns:
        QPIAResult with node probabilities and analysis
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    
    # Select action function
    if action_type == "distance":
        action_func = DistanceAction()
    elif action_type == "risk_weighted":
        action_func = RiskWeightedAction(risk_weight=risk_weight)
    elif action_type == "epidemic":
        action_func = EpidemicAction(transmission_rate=transmission_rate)
    elif action_type == "target_aware":
        action_func = TargetAwareAction(reward=target_reward)
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    
    # Select start nodes
    if start_nodes is None:
        if backward_paths and marked_nodes:
            # Backward paths: start from targets (marked nodes)
            start_nodes = marked_nodes
        else:
            # Forward paths: start from all nodes uniformly
            start_nodes = list(range(n))
    elif isinstance(start_nodes, np.ndarray):
        start_nodes = start_nodes.tolist()
    
    # Enumerate paths
    if backward_paths and marked_nodes:
        # Enumerate paths ending at targets (for index case finding)
        # We use reverse adjacency for this
        reverse_adj = adjacency.T
        paths = enumerate_paths(
            adjacency=reverse_adj,
            start_nodes=start_nodes,
            max_length=max_path_length,
            allow_revisit=allow_revisit,
        )
        # Paths are in reverse order, need to flip them
        paths = [p[::-1] for p in paths]
    else:
        paths = enumerate_paths(
            adjacency=adjacency,
            start_nodes=start_nodes,
            max_length=max_path_length,
            allow_revisit=allow_revisit,
        )
    
    # Compute amplitudes for each path
    path_states = compute_path_amplitudes(
        paths=paths,
        adjacency=adjacency,
        risk=risk,
        action_func=action_func,
        action_scale=action_scale,
    )
    
    # Aggregate to nodes
    node_probs, node_amps = aggregate_to_nodes(path_states, n)
    
    # Compute convergence metric (interference measure)
    total_amp_squared = sum(abs(s.amplitude)**2 for s in path_states)
    interference_amp = sum(s.amplitude for s in path_states)
    interference_mag = abs(interference_amp)
    
    if total_amp_squared > 0:
        convergence = interference_mag**2 / total_amp_squared
    else:
        convergence = 0.0
    
    return QPIAResult(
        node_probabilities=node_probs,
        node_amplitudes=node_amps,
        path_states=path_states,
        n_paths=len(path_states),
        convergence_metric=float(convergence),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QPIA-GROVER HYBRID (Honest Quantum Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def qpia_grover_hybrid(
    adjacency: np.ndarray,
    risk: np.ndarray,
    marked: list[int],
    max_path_length: int = 3,
    n_grover_iterations: int | None = None,
    seed: int = 42,
) -> dict:
    """Hybrid QPIA-Grover for epidemiology.
    
    This combines:
    1. QPIA for computing path integral amplitudes (encodes problem structure)
    2. Grover oracle for marking target nodes (marks hotspots)
    3. Amplitude amplification for boosting marked states
    
    This is the honest quantum algorithm - uses both problem encoding (QPIA)
    and search amplification (Grover).
    
    Args:
        adjacency: N×N transmission matrix
        risk: N risk intensities
        marked: List of marked (target) node indices
        max_path_length: Max path length for QPIA
        n_grover_iterations: Number of Grover iterations (default: √N)
        seed: Random seed
    
    Returns:
        Dictionary with probabilities and metrics
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    dim = 2 ** int(np.ceil(np.log2(n)))
    
    # Step 1: QPIA path integral to get initial amplitude structure
    # This encodes the "physics" of the problem (transmission dynamics)
    result = qpia_search(
        adjacency=adjacency,
        risk=risk,
        start_nodes=list(range(n)),
        max_path_length=max_path_length,
        action_type="risk_weighted",
        action_scale=1.0,
        risk_weight=2.0,
    )
    
    # Use node amplitudes from QPIA as initial state
    # Scale to full Hilbert space
    psi = np.zeros(dim, dtype=complex)
    psi[:n] = result.node_amplitudes
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    if norm > 0:
        psi = psi / norm
    
    # Step 2: Grover iterations with oracle marking marked nodes
    # Number of iterations: π/4 * √(N/M)
    M = len(marked)
    if n_grover_iterations is None:
        theta = np.arcsin(np.sqrt(M / n))
        n_grover_iterations = max(1, int(np.round(np.pi / (4 * theta))))
    
    # Grover oracle: phase flip on marked states
    for iteration in range(n_grover_iterations):
        # Oracle: flip phase of marked states
        psi_marked = psi.copy()
        for m in marked:
            if m < len(psi):
                psi_marked[m] = -psi_marked[m]
        
        # Diffusion (inversion about mean) - only over real n states
        mean = np.mean(psi_marked[:n])
        psi_marked[:n] = 2 * mean - psi_marked[:n]
        
        psi = psi_marked / np.sqrt(np.sum(np.abs(psi_marked) ** 2))
    
    # Final probabilities
    probs = np.abs(psi[:n]) ** 2
    probs = probs / np.sum(probs)  # Renormalize
    
    p_marked = float(np.sum(probs[marked]))
    top5 = list(np.argsort(probs)[::-1][:5])
    
    return {
        "algorithm": "QPIA-Grover",
        "p_marked": p_marked,
        "n_grover_iterations": n_grover_iterations,
        "top_5": top5,
        "marked_in_top_5": sum(1 for m in marked if m in top5),
        "initial_p_marked": float(np.sum(result.node_probabilities[marked])),
    }


def qpia_index_case_finding(
    adjacency: np.ndarray,
    risk: np.ndarray,
    hotspots: list[int],
    max_path_length: int = 3,
    seed: int = 42,
) -> dict:
    """QPIA for index case finding (backward path search).
    
    Given hotspots, find likely index cases (transmission sources).
    
    This uses backward path enumeration to find paths that lead TO hotspots,
    then uses Grover to boost probability of likely index case nodes.
    
    Args:
        adjacency: N×N transmission matrix
        risk: N risk intensities
        hotspots: List of known hotspot indices
        max_path_length: Max path length for backward search
        seed: Random seed
    
    Returns:
        Dictionary with node probabilities and analysis
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    
    # Step 1: Backward QPIA to find paths leading to hotspots
    result = qpia_search(
        adjacency=adjacency,
        risk=risk,
        start_nodes=hotspots,
        max_path_length=max_path_length,
        action_type="risk_weighted",
        action_scale=1.0,
        backward_paths=True,  # This enumerates paths FROM hotspots
    )
    
    # The result.node_probabilities now reflect likelihood of being an index case
    # Higher prob = more paths from this node lead to hotspots
    
    # Step 2: Use Grover to amplify top-k (k = number of likely index cases)
    # Find nodes with highest probability of being index cases
    k_index = min(5, n)
    likely_indices = np.argsort(result.node_probabilities)[::-1][:k_index]
    
    probs = result.node_probabilities.copy()
    
    # Apply Grover-like amplification on likely index cases
    n_iterations = int(np.ceil(np.sqrt(n / k_index)))
    
    for _ in range(n_iterations):
        # Flip phase of likely index cases
        for idx in likely_indices:
            probs[idx] = -probs[idx]
        
        # Diffusion
        mean = np.mean(probs)
        probs = 2 * mean - probs
        probs = np.abs(probs)
    
    probs = probs / np.sum(probs)
    
    return {
        "algorithm": "QPIA-IndexCase",
        "p_hotspots_aggregate": float(np.sum(result.node_probabilities[hotspots])),
        "top_5_index_cases": list(np.argsort(probs)[::-1][:5]),
        "probs": probs,
        "initial_probs": result.node_probabilities,
        "n_paths_analyzed": result.n_paths,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BORN APPROXIMATION (SCATTERING FORMULATION)
# ═══════════════════════════════════════════════════════════════════════════════

def born_approximation_scattering(
    adjacency: np.ndarray,
    risk: np.ndarray,
    incident_nodes: list[int],
    scattering_centers: list[int],
    n_iterations: int = 3,
    action_scale: float = 2.0,
) -> np.ndarray:
    """Apply Born approximation for scattering from risk hotspots.
    
    The Born approximation treats high-risk nodes as scattering centers.
    The "incident wave" (uniform superposition) scatters off these centers,
    and the resulting amplitude distribution highlights likely transmission paths.
    
    Based on Gautam & Ahn 2024 Section III-B (First Born Approximation):
    - Incident wave: uniform superposition over all nodes
    - Scattered wave: sum over scattering from each hotspot
    - Total: incident + scattered
    
    Args:
        adjacency: N×N transmission matrix
        risk: N risk intensities
        incident_nodes: Nodes where incident wave originates
        scattering_centers: High-risk nodes acting as scatterers
        n_iterations: Number of scattering iterations
        action_scale: Action scaling factor
    
    Returns:
        Probability distribution over nodes
    """
    n = adjacency.shape[0]
    
    # Initialize incident wave (uniform superposition)
    incident_wave = np.ones(n, dtype=complex) / np.sqrt(n)
    
    # Current total wavefunction
    psi = incident_wave.copy()
    
    # Scattering potential from risk (higher risk = stronger scatterer)
    V = risk / (risk.max() + 1e-10)  # Normalized potential
    
    for iteration in range(n_iterations):
        # Scattered wave = potential × incident wave (Born approximation)
        scattered = V * psi * action_scale * np.exp(1j * action_scale * V)
        
        # Add scattered wave to total
        psi = incident_wave + scattered
        
        # Normalize
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))
    
    # Probability = |ψ|²
    probabilities = np.abs(psi) ** 2
    
    return probabilities


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HOTSPOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def qpia_multi_hotspot_detection(
    adjacency: np.ndarray,
    risk: np.ndarray,
    k_hotspots: int = 5,
    method: str = "path_integral",
    **qpia_kwargs,
) -> list[tuple[int, float]]:
    """Detect multiple hotspots using QPIA.
    
    Args:
        adjacency: N×N transmission matrix
        risk: N risk intensities
        k_hotspots: Number of hotspots to detect
        method: Either "path_integral" or "born_scattering"
        **qpia_kwargs: Additional arguments for qpia_search()
    
    Returns:
        List of (node_index, probability) tuples for top-k hotspots
    """
    if method == "path_integral":
        result = qpia_search(
            adjacency=adjacency,
            risk=risk,
            **qpia_kwargs,
        )
        return result.top_k_nodes(k=k_hotspots)
    
    elif method == "born_scattering":
        # Use Born approximation
        n = adjacency.shape[0]
        incident = list(range(n))  # All nodes as incident sources
        hotspots_idx = np.argsort(risk)[-k_hotspots:][::-1].tolist()
        
        probs = born_approximation_scattering(
            adjacency=adjacency,
            risk=risk,
            incident_nodes=incident,
            scattering_centers=hotspots_idx,
        )
        
        # Return top-k
        indices = np.argsort(probs)[::-1][:k_hotspots]
        return [(int(i), float(probs[i])) for i in indices]
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION AND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_interference_analysis(
    result: QPIAResult,
    adjacency: np.ndarray,
    risk: np.ndarray,
) -> dict:
    """Analyze interference patterns in QPIA results.
    
    Key insight from Gautam & Ahn: QPIA works because of interference.
    Constructive interference enhances paths with lower action (higher fitness).
    
    Returns:
        Dictionary with interference metrics
    """
    # Path statistics
    actions = np.array([s.action for s in result.path_states])
    amplitudes = np.array([s.amplitude for s in result.path_states])
    
    # Phase distribution (should be uniform if no structure)
    phases = np.angle(amplitudes)
    
    # Constructive vs destructive interference
    total_amplitude = np.sum(amplitudes)
    expected_if_uniform = np.sqrt(result.n_paths)  # If random phases
    
    interference_ratio = np.abs(total_amplitude) / (expected_if_uniform + 1e-10)
    
    # Group by end node
    end_nodes = np.array([s.end for s in result.path_states])
    unique_ends, counts = np.unique(end_nodes, return_counts=True)
    
    return {
        "n_paths": result.n_paths,
        "mean_action": float(np.mean(actions)),
        "std_action": float(np.std(actions)),
        "action_range": (float(np.min(actions)), float(np.max(actions))),
        "mean_phase": float(np.mean(phases)),
        "phase_coherence": float(interference_ratio),
        "paths_per_node": dict(zip(unique_ends.tolist(), counts.tolist())),
        "top_interfering_nodes": result.top_k_nodes(5),
        "convergence_metric": result.convergence_metric,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING AND BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════

def test_on_ring_graph(n: int = 10, seed: int = 42) -> dict:
    """Test QPIA on a ring graph (should show constructive interference).
    
    Ring graph: nodes 0-1-2-...-(n-1)-0
    Expected: Uniform probability (symmetric, no preferred path)
    """
    rng = np.random.default_rng(seed)
    
    # Build ring adjacency
    adjacency = np.zeros((n, n))
    for i in range(n):
        adjacency[i, (i + 1) % n] = 1.0
        adjacency[i, (i - 1) % n] = 1.0
    
    # Uniform risk
    risk = np.ones(n) * 0.5
    
    # Mark one node as hotspot
    hotspot = n // 4
    risk[hotspot] = 0.95
    
    result = qpia_search(
        adjacency=adjacency,
        risk=risk,
        start_nodes=[0],  # Start from node 0
        max_path_length=3,
        action_type="risk_weighted",
        risk_weight=3.0,
        action_scale=1.0,
    )
    
    analysis = compute_interference_analysis(result, adjacency, risk)
    
    return {
        "graph_type": "ring",
        "n": n,
        "hotspot": hotspot,
        "result": result,
        "analysis": analysis,
        "p_hotspot": float(result.node_probabilities[hotspot]),
        "p_hotspot_rank": int(np.argsort(result.node_probabilities)[::-1].tolist().index(hotspot)) + 1,
    }


def test_on_grid_graph(size: int = 3, seed: int = 42) -> dict:
    """Test QPIA on a grid graph.
    
    Grid graph: (size × size) lattice
    Expected: Paths toward hotspot should have higher probability
    """
    rng = np.random.default_rng(seed)
    n = size * size
    
    # Build grid adjacency
    adjacency = np.zeros((n, n))
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            # Right neighbor
            if j + 1 < size:
                adjacency[idx, idx + 1] = 1.0
            # Down neighbor
            if i + 1 < size:
                adjacency[idx, idx + size] = 1.0
    
    # Risk with one hotspot at center
    risk = np.ones(n) * 0.3
    center = n // 2
    risk[center] = 0.95
    
    result = qpia_search(
        adjacency=adjacency,
        risk=risk,
        start_nodes=[0],  # Start from corner
        max_path_length=4,
        action_type="risk_weighted",
        risk_weight=2.0,
    )
    
    analysis = compute_interference_analysis(result, adjacency, risk)
    
    return {
        "graph_type": "grid",
        "size": size,
        "hotspot": center,
        "result": result,
        "analysis": analysis,
        "p_hotspot": float(result.node_probabilities[center]),
        "top_5": result.top_k_nodes(5),
    }


def test_on_dien_bien(seed: int = 42) -> dict:
    """Test QPIA on Dien Bien graph."""
    from src.graph_dien_bien import build_synthetic_dien_bien
    
    graph = build_synthetic_dien_bien(seed=seed)
    
    # Find true hotspots (top risk)
    true_hotspots = np.argsort(graph.risk_intensity)[-5:][::-1]
    
    result = qpia_search(
        adjacency=graph.adjacency,
        risk=graph.risk_intensity,
        start_nodes=list(range(graph.n_communes)),
        max_path_length=3,
        action_type="risk_weighted",
        risk_weight=2.0,
        action_scale=0.5,
    )
    
    analysis = compute_interference_analysis(result, graph.adjacency, graph.risk_intensity)
    
    # Check if true hotspots are in top-k
    detected_top5 = [idx for idx, _ in result.top_k_nodes(5)]
    hits = sum(1 for h in true_hotspots if h in detected_top5)
    
    return {
        "graph_type": "dien_bien",
        "n_communes": graph.n_communes,
        "true_hotspots": true_hotspots.tolist(),
        "detected_top5": detected_top5,
        "hotspot_hits": hits,
        "result": result,
        "analysis": analysis,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("QPIA (Quantum Path Integral Approach) for Epidemiology")
    print("Based on Gautam & Ahn 2024, IEEE TITS")
    print("=" * 70)
    
    # Test on ring graph
    print("\n[Test 1] Ring Graph (n=10)")
    print("-" * 40)
    ring_result = test_on_ring_graph(n=10, seed=42)
    print(f"  Graph: ring with {ring_result['n']} nodes")
    print(f"  Hotspot: node {ring_result['hotspot']}")
    print(f"  P(hotspot): {ring_result['p_hotspot']:.4f}")
    print(f"  Hotspot rank: {ring_result['p_hotspot_rank']}")
    print(f"  Phase coherence: {ring_result['analysis']['phase_coherence']:.4f}")
    
    # Test on grid graph
    print("\n[Test 2] Grid Graph (3x3)")
    print("-" * 40)
    grid_result = test_on_grid_graph(size=3, seed=42)
    print(f"  Graph: 3x3 grid")
    print(f"  Hotspot: node {grid_result['hotspot']} (center)")
    print(f"  P(hotspot): {grid_result['p_hotspot']:.4f}")
    print(f"  Top 5 nodes: {grid_result['top_5']}")
    
    # Test on Dien Bien
    print("\n[Test 3] Dien Bien Realistic Graph")
    print("-" * 40)
    try:
        db_result = test_on_dien_bien(seed=42)
        print(f"  Communes: {db_result['n_communes']}")
        print(f"  True hotspots: {db_result['true_hotspots'][:3]}...")
        print(f"  Detected top 5: {db_result['detected_top5'][:3]}...")
        print(f"  Hotspot hits: {db_result['hotspot_hits']}/5")
        print(f"  Convergence: {db_result['analysis']['convergence_metric']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  - Phase coherence > 1 indicates constructive interference")
    print("  - Higher P(hotspot) and lower rank = better detection")
    print("  - QPIA works when problem structure aligns with path summation")
    print("=" * 70)
