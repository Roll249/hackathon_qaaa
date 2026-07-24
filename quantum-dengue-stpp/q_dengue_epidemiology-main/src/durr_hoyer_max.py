"""Dürr-Høyer Maximum Finding — with REAL oracle + query counter.

This implements Tier 2: actual Grover search with oracle + counter.
Every "quantum query" to f(i) increments a real counter.
The number of oracle calls is MEASURED empirically, not hardcoded formula.

Algorithm (Dürr & Høyer 1996):
1. Start with random y = -inf, m = random
2. Repeat O(√N) times:
   a. Grover search for i where f(i) > y
   b. If found: y = f(i), m = i
   c. If not found: return m

Key insight: For MAXIMUM FINDING (unknown M>1), total queries = O(N).
The O(√N) bound applies to the ITERATION count, not the query count.
Each Grover iteration uses √(N/M) queries. With M potentially large,
this equals or exceeds classical O(N) scan.

Reference: Dürr & Høyer 1996, "A Quantum Algorithm for Finding the Minimum"
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_parent = _Path(__file__).parent.parent
if str(_parent) not in _sys.path:
    _sys.path.insert(0, str(_parent))

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Oracle with query counter
# ─────────────────────────────────────────────────────────────────────────────

class CountingOracle:
    """Oracle f(i) that counts every query to f(i)."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=float)
        self.count = 0

    def query(self, i: int) -> float:
        self.count += 1
        return float(self.values[i])

    def reset(self):
        self.count = 0

    @property
    def n_queries(self) -> int:
        return self.count


# ─────────────────────────────────────────────────────────────────────────────
# Grover diffusion
# ─────────────────────────────────────────────────────────────────────────────
# Grover search with oracle
# ─────────────────────────────────────────────────────────────────────────────

def grover_search(oracle: CountingOracle, threshold: float,
                 n_qubits: int, target_dim: int,
                 rng: np.random.Generator) -> tuple[int | None, float, int]:
    """Grover search for an index where f(i) > threshold.

    Returns: (best_idx, best_val, n_oracle_queries_used)
    """
    n = oracle.values.shape[0]
    # Determining the marked set (and M) requires reading every f(i) once —
    # charge it like any other classical oracle access instead of reading
    # oracle.values for free.
    marked = oracle.values > threshold
    oracle.count += n
    M = int(marked.sum())

    if M == 0:
        return None, 0.0, 0

    # Optimal iterations: π/4 × √(N/M) — each iteration = 1 oracle query
    theta = math.asin(math.sqrt(M / n))
    n_iter = max(1, int(round(math.pi / (4 * theta))))

    # Uniform superposition over all n states
    amp = np.zeros(target_dim, dtype=complex)
    amp[:n] = 1.0 / math.sqrt(n)
    amp[n:] = 0

    marked_padded = np.zeros(target_dim, dtype=bool)
    marked_padded[:n] = marked

    for _ in range(n_iter):
        # Phase flip on marked — 1 oracle query
        amp[marked_padded] = -amp[marked_padded]
        oracle.count += 1
        # Diffusion — classical unitary, no query
        # average only over the REAL n states, not the padded target_dim
        avg = np.mean(amp[:n])
        amp[:n] = 2 * avg - amp[:n]

    # Measure: a real quantum measurement samples an outcome with probability
    # |amplitude|^2 — it does NOT deterministically return the argmax, since
    # every marked state has (near-)identical amplitude by construction.
    probs = np.abs(amp[:n]) ** 2
    marked_probs = probs.copy()
    marked_probs[~marked] = 0

    total = marked_probs.sum()
    if total == 0:
        return None, 0.0, n_iter

    best_idx = int(rng.choice(n, p=marked_probs / total))
    return best_idx, float(oracle.values[best_idx]), n_iter


# ─────────────────────────────────────────────────────────────────────────────
# Dürr-Høyer max finding with real counter
# ─────────────────────────────────────────────────────────────────────────────

def dur_hoyer_max_finding(
    risk_scores: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[int, float]:
    """Find argmax of risk_scores using Dürr-Høyer with oracle counter.

    Returns:
        best_index, best_score

    The oracle is accessible via dur_hoyer_oracle after the call.
    """
    n = len(risk_scores)
    n_qubits = int(math.ceil(math.log2(n)))
    target_dim = 2 ** n_qubits

    global dur_hoyer_oracle
    dur_hoyer_oracle = CountingOracle(risk_scores)
    oracle = dur_hoyer_oracle
    oracle.count += 1  # initial read

    rng = np.random.default_rng(seed)
    best_index = int(rng.integers(0, n))
    best_score = float(risk_scores[best_index])

    if verbose:
        print(f"  Initial: idx={best_index}, score={best_score:.4f}")

    # Safety bound only — the real stopping condition is "no improvement
    # found" (M==0 or candidate not better), which triggers well before this
    # in practice. A cap of ceil(sqrt(n)) was too tight for a *probabilistic*
    # search (random-sampled measurement can miss the true max on a given
    # attempt) and could return early with the wrong index.
    max_iter = n
    total_grover_iters = 0

    for iteration in range(max_iter):
        candidate_idx, candidate_score, grover_queries = grover_search(
            oracle, best_score, n_qubits, target_dim, rng
        )
        total_grover_iters += grover_queries

        if candidate_idx is not None and candidate_score > best_score:
            best_index = candidate_idx
            best_score = candidate_score
            if verbose:
                print(f"  Iter {iteration}: improved → idx={best_index}, "
                      f"score={best_score:.4f}")
        else:
            if verbose:
                print(f"  Iter {iteration}: no improvement, stopping")
            break

    if verbose:
        print(f"  Total Grover iterations: {total_grover_iters}")
        print(f"  Total oracle queries: {oracle.n_queries}")
        print(f"  Classical scan: {n} queries")
        ratio = oracle.n_queries / n
        print(f"  Ratio DH/classical: {ratio:.2f}×")

    return best_index, best_score


# Module-level oracle for external access
dur_hoyer_oracle: CountingOracle | None = None


def get_oracle_queries() -> int:
    """Return oracle queries from last dur_hoyer_max_finding call."""
    global dur_hoyer_oracle
    return dur_hoyer_oracle.n_queries if dur_hoyer_oracle else 0


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────────────────────────────────────

def dur_hoyer_benchmark(n_target: int, seed: int) -> dict:
    """Run one benchmark trial. Returns measured query counts."""
    from src.graph_dien_bien import build_synthetic_dien_bien

    g = build_synthetic_dien_bien(seed=seed)
    risk = g.risk_intensity[:n_target]
    n = len(risk)

    classical_queries = n
    classical_idx = int(np.argmax(risk))

    oracle = CountingOracle(risk)
    oracle.count += 1  # initial read

    n_qubits = int(math.ceil(math.log2(n)))
    target_dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    best_index = int(rng.integers(0, n))
    best_score = float(risk[best_index])
    total_grover_iters = 0

    max_iter = n  # safety bound; real stopping condition is "no improvement"
    for _ in range(max_iter):
        candidate_idx, candidate_score, grover_queries = grover_search(
            oracle, best_score, n_qubits, target_dim, rng
        )
        total_grover_iters += grover_queries

        if candidate_idx is not None and candidate_score > best_score:
            best_index = candidate_idx
            best_score = candidate_score
        else:
            break

    dh_queries = oracle.n_queries
    theory_sqrt_n = int(math.ceil(math.sqrt(n)))

    return {
        "n": n,
        "classical_queries": classical_queries,
        "durr_hoyer_queries": dh_queries,
        "theory_sqrt_n": theory_sqrt_n,
        "total_grover_iters": total_grover_iters,
        "classical_idx": classical_idx,
        "durr_hoyer_idx": best_index,
        "match": bool(best_index == classical_idx),
        "ratio_vs_classical": dh_queries / classical_queries if classical_queries else float('inf'),
        "ratio_vs_theory": dh_queries / theory_sqrt_n if theory_sqrt_n else float('inf'),
    }


if __name__ == "__main__":
    print("=" * 75)
    print("Dürr-Høyer with REAL oracle counter")
    print("=" * 75)
    print(f"\n{'N':>4} | {'Class':>7} | {'DH':>7} | {'√N':>7} | "
          f"{'DH/Cl':>7} | {'DH/√N':>7} | {'iters':>6} | {'Match':>6}")
    print("-" * 75)

    for n_target in [16, 32, 64, 128, 130]:
        for seed in [42, 43, 44]:
            result = dur_hoyer_benchmark(n_target, seed)
            print(f"{result['n']:>4} | {result['classical_queries']:>7} | "
                  f"{result['durr_hoyer_queries']:>7} | {result['theory_sqrt_n']:>7} | "
                  f"{result['ratio_vs_classical']:>7.2f} | {result['ratio_vs_theory']:>7.2f} | "
                  f"{result['total_grover_iters']:>6} | "
                  f"{'✓' if result['match'] else '✗':>6}")

    print("\n" + "=" * 75)
    print("INTERPRETATION:")
    print("  DH/Class ≈ 1.0  → Dürr-Høyer matches classical scan")
    print("  DH/Class < 1.0  → Dürr-Høyer uses fewer oracle queries")
    print("  DH/Class > 1.0  → Dürr-Høyer uses MORE queries (expected for max-finding)")
    print("  DH/√N ≈ O(√N)  → Grover iterations scale as theory predicts")
    print("  DH/√N >> √N    → Overhead from large M (many above-threshold states)")
    print("")
    print("KEY POINT: The O(√N) in Dürr-Høyer is the ITERATION count, not query count.")
    print("Each iteration = √(N/M) queries. For M>1, total = O(N) — same as classical.")
    print("=" * 75)
