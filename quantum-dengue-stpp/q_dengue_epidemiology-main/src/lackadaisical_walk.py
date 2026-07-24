"""Multi-hotspot detection — CORRECTED implementation.

Prior version had two bugs:
1. Length-bias: score_sum > best_score_sum always picks largest k_try
   because risk ≥ 0 → larger k always has larger sum.
2. Non-uniform Grover diffusion: applied mean-reflection on QPIE-encoded
   state, which distorts amplitudes (QPIE state is not uniform).

CORRECT approach:
- Grover amplification for top-K is mathematically valid ONLY when
  we know which K states are marked (oracle access).
- Without oracle access, we cannot do proper multi-target amplification.
- The honest thing: return top-K sorted by risk score directly.
  Grover adds nothing here without an oracle that distinguishes "hotspot".

If we want Grover-like amplification:
- Must use an oracle that says "is this index in the top-K?"
- That requires KNOWING the top-K in advance → circular.

Alternative: Use Grover for single-element search (Dürr-Høyer),
then iteratively remove found hotspots. This is the correct multi-hotspot
algorithm when K is unknown.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_parent = _Path(__file__).parent.parent
if str(_parent) not in _sys.path:
    _sys.path.insert(0, str(_parent))

import numpy as np


def multi_hotspot_detection(
    risk_scores: np.ndarray,
    adjacency: np.ndarray,
    k_max: int = 10,
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Return top-K hotspots sorted by risk score.

    This is the HONEST version: no fake amplification.
    
    If you want quantum advantage for multi-hotspot detection:
    1. Use Dürr-Høyer (single max) iteratively, removing found peaks
    2. Use Grover oracle with threshold, re-encode state each iteration
    3. Use QAOA on graph-structured Ising Hamiltonian

    For the dengue pipeline, top-K by risk score is what epidemiologists
    actually want. The QPIE encoding provides probability amplitudes
    proportional to sqrt(risk), which naturally favors hotspots.
    
    Returns:
        sorted list of (commune_index, risk_score) — top k_max by risk
    """
    n = len(risk_scores)
    k = min(k_max, n)
    
    # Sort indices by risk descending
    sorted_indices = np.argsort(risk_scores)[::-1]
    top_k = sorted_indices[:k]
    
    results = [(int(idx), float(risk_scores[idx])) for idx in top_k]
    
    # Already sorted by risk descending
    return results


def multi_hotspot_detection_iterative(
    risk_scores: np.ndarray,
    adjacency: np.ndarray,
    k_max: int = 10,
    seed: int = 42,
    use_grover: bool = True,
) -> list[tuple[int, float]]:
    """Iterative Dürr-Høyer for multi-hotspot detection.

    Algorithm:
    1. Run Dürr-Høyer to find current max
    2. Mask out found hotspot (set risk = -inf)
    3. Repeat until k hotspots found
    
    This uses ACTUAL Grover search with oracle counter at each step.
    Each iteration: O(√(N/m)) queries where m = remaining candidates.
    Total: O(k × √(N/k)) = O(√(Nk)) — still quadratic speedup over O(Nk).
    
    Returns:
        sorted list of (commune_index, risk_score) — top k_max by risk
    """
    from src.durr_hoyer_max import dur_hoyer_max_finding
    
    n = len(risk_scores)
    k = min(k_max, n)
    
    # Work on a copy
    remaining = risk_scores.copy()
    found = []
    rng = np.random.default_rng(seed)
    
    for _ in range(k):
        # Find max in remaining candidates
        max_idx, max_score = dur_hoyer_max_finding(
            remaining, seed=int(rng.integers(0, 2**31))
        )
        
        # Sanity check: if max is -inf, we're done
        if max_score < -1e9:
            break
            
        found.append((int(max_idx), float(max_score)))
        
        # Mask out: set to very negative so it won't be re-found
        remaining[max_idx] = -np.inf
    
    # Sort by risk descending
    found.sort(key=lambda x: -x[1])
    return found


if __name__ == "__main__":
    # Test on uniform risk with injected hotspots
    rng = np.random.default_rng(42)
    n = 16
    risk = rng.uniform(0.01, 0.3, n)
    risk[5] = 0.935   # top-1
    risk[9] = 0.913   # top-2
    risk[12] = 0.857  # top-3
    risk[3] = 0.017   # low
    risk[11] = 0.003  # very low
    
    # Build dummy adjacency
    adjacency = np.zeros((n, n))
    
    print("=" * 60)
    print("Test: uniform risk + 3 hotspots injected")
    print("=" * 60)
    print(f"True top-3: indices {[5, 9, 12]} risks {risk[5]:.3f}, {risk[9]:.3f}, {risk[12]:.3f}")
    print()
    
    # Honest top-k by risk
    result_simple = multi_hotspot_detection(risk, adjacency, k_max=3, seed=42)
    print(f"multi_hotspot_detection (honest top-k):")
    for idx, score in result_simple:
        print(f"  idx={idx}: risk={score:.3f}")
    
    print()
    
    # Iterative Grover
    result_iter = multi_hotspot_detection_iterative(risk, adjacency, k_max=3, seed=42)
    print(f"multi_hotspot_detection_iterative (Grover):")
    for idx, score in result_iter:
        print(f"  idx={idx}: risk={score:.3f}")
    
    print()
    
    # Check correctness
    simple_idx = [x[0] for x in result_simple]
    iter_idx = [x[0] for x in result_iter]
    true_top3 = [5, 9, 12]
    
    print(f"Simple match: {simple_idx == true_top3}")
    print(f"Iterative match: {iter_idx == true_top3}")
    
    print()
    print("=" * 60)
    print("CONCLUSION: Honest top-k always correct.")
    print("Iterative Grover may miss peaks (probabilistic search).")
    print("=" * 60)
