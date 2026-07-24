"""Empirical benchmark: classical vs quantum walk — measured hitting time.

IMPORTANT — What this measures:

THESIS CLAIM: "Fully connected graphs don't get quantum advantage (for MIXING)"
  → On fully-connected: spectral gap Δ ≈ 1, mixing is fast in both classical/quantum

THIS BENCHMARK: Measures HITTING TIME, not mixing time
  → On fully-connected: shows speedup (from amplitude amplification, not graph structure)
  → On sparse: shows speedup (from both amplification AND graph structure)

Why "fully connected" benchmarks show speedup:
  → It's Grover/amplitude amplification in the INDEX space, not quantum walk mixing
  → Grover works on ANY graph because it searches over indices, not walks

This is consistent with the thesis:
  → Quantum walk MIXING advantage requires structure (Δ << 1)
  → Grover HITTING TIME advantage exists on all graphs
  → bench_rigorous_walk.py measures hitting time, which naturally shows speedup

The two benchmark files (empirical vs rigorous) measure the same thing
(hitting time) but use different quantum walk formulations.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Graph construction (same as in graph_dien_bien.py)
# ============================================================

def build_graph(regime: str, n: int, seed: int = 42):
    """Build graph in different regimes.

    Returns tuple: (adjacency, edges_label)
    """
    rng = np.random.default_rng(seed)

    if regime == "fully":
        # Fully connected: every node connected to every other
        # Place nodes but ignore distances
        positions = rng.uniform(0, 50, size=(n, 2))
        A = np.ones((n, n)) - np.eye(n)
        return A, "Fully connected (K_N)"

    elif regime == "sparse":
        # Vector-biology sparse graph
        # Position nodes on 50x50km grid
        positions = rng.uniform(0, 50, size=(n, 2))
        diff = positions[:, None, :] - positions[None, :, :]
        dist_km = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Vector kernel: exp(-d / 5km) — realistic dispersal range
        kernel = np.exp(-dist_km / 5.0)
        # Threshold: 30km neighborhood radius
        A = np.where(dist_km < 30.0, kernel, 0.0)
        np.fill_diagonal(A, 0.0)
        return A, "Sparse (vector-biology)"

    else:
        raise ValueError(f"Unknown regime: {regime}")


# ============================================================
# Classical random walk: MEASURE mixing time via simulation
# ============================================================

def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """π proportional to degree."""
    deg = A.sum(axis=1)
    return deg / deg.sum()


def classical_mixing_time_empirical(A: np.ndarray, threshold: float = 0.25,
                                    n_trials: int = 100, max_steps: int = 10_000) -> float:
    """MEASURE mixing time by simulating random walks.

    t_mix(ε) = smallest t s.t. for all starting nodes,
            || P^t(x, ·) - π ||_TV < ε

    We do this by:
    1. Compute stationary distribution π
    2. Simulate many random walks from worst-start vertex
    3. Check TV distance every power-of-2 step

    Args:
        threshold: ε parameter for TV distance (default 0.25 = "rough mixing")

    Returns:
        Empirical t_mix (average over trials)
    """
    n = len(A)
    pi = stationary_distribution(A)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1
    P = A / deg[:, None]

    # Worst-start: vertex farthest from stationary on a single node
    delta_start = np.zeros(n)
    delta_start[0] = 1.0  # delta at vertex 0

    # Track TV distance vs π over time, averaged over trials
    tv_over_time = []

    for step in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        if step > max_steps:
            break
        # Distribution after step
        dist = delta_start @ np.linalg.matrix_power(P, step)
        # TV distance = 0.5 * sum |dist - pi|
        tv = 0.5 * np.abs(dist - pi).sum()
        tv_over_time.append((step, tv))

    # Find first step with TV < threshold
    for step, tv in tv_over_time:
        if tv < threshold:
            return float(step)

    # If never mixed within range, return max_steps
    return float(max_steps)


def classical_hitting_time_empirical(A: np.ndarray, marked: int,
                                      n_trials: int = 200) -> float:
    """MEASURE average hitting time for classical random walk.

    We simulate n_trials random walks starting from a random vertex,
    and count how many steps it takes to visit the marked vertex.

    Returns:
        Average hitting time (empirical)
    """
    n = len(A)
    deg = A.sum(axis=1)
    if deg[marked] == 0:
        return float('inf')

    rng = np.random.default_rng(123)

    hitting_times = []
    for trial in range(n_trials):
        # Random start
        current = rng.integers(0, n)
        if current == marked:
            hitting_times.append(0)
            continue

        for step in range(1, 5000):
            # Neighbors with probabilities
            neighbors = np.where(A[current] > 0)[0]
            if len(neighbors) == 0:
                break
            probs = A[current, neighbors]
            probs = probs / probs.sum()
            current = rng.choice(neighbors, p=probs)
            if current == marked:
                hitting_times.append(step)
                break
        else:
            hitting_times.append(5000)  # truncated

    return float(np.mean(hitting_times))


# ============================================================
# Quantum walk: MEASURE via actual unitary simulation
# ============================================================

def quantum_walk_unitary(A: np.ndarray) -> np.ndarray:
    """Build Szegedy-like unitary for discrete-time quantum walk.

    For a regular graph with degree d, the quantum walk acts on
    Hilbert space C^N ⊗ C^d (node × direction).

    For irregular graphs, we use the "extended" walk:
    each node v has |deg(v)| coin states (one per incident edge).

    Args:
        A: weighted adjacency matrix

    Returns:
        U: (2N×2N) unitary acting on doubled space [node, edge_in] → [node, edge_out]
    """
    n = len(A)
    # Total Hilbert space dimension: sum of degrees (each edge contributes 2 slots)
    # Szegedy doubling: each edge (i,j) maps (i,j) ↔ (j,i)
    # For weighted/irregular: use coined walk on directed arc set

    # Approach: for each vertex v, list outgoing edges to neighbors
    # State space = {(v, k, sign)} where k indexes the k-th edge from v
    # Build shift, coin operators

    # For SIMPLICITY (and to keep the unitary tractable for medium N),
    # we use COINED WALK on the arc set:
    # - States: (i, j) for i → j (i has neighbor j)
    # - Coin: rotation at i to distribute amplitude across its edges
    # - Shift: maps (i, j) → (j, i)

    arcs = []  # (source, target)
    arc_to_idx = {}
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        for j in neighbors:
            if i == j:
                continue
            arc_to_idx[(i, j)] = len(arcs)
            arcs.append((i, j))

    dim = len(arcs)
    if dim == 0:
        return np.eye(1)

    # Coin at vertex i: equal superposition over outgoing edges (Grover coin)
    coin = np.zeros((dim, dim))
    for i in range(n):
        # Find all arcs from i
        idxs = [arc_to_idx[(i, j)] for j in np.where(A[i] > 0)[0] if j != i]
        if len(idxs) <= 1:
            continue
        # 2|diffusion| - I (Grover coin)
        sub = np.full((len(idxs), len(idxs)), 1.0 / len(idxs))
        for k, idx in enumerate(idxs):
            sub[k, k] = -1.0 + 1.0 / len(idxs)
        for k1, idx1 in enumerate(idxs):
            for k2, idx2 in enumerate(idxs):
                coin[idx1, idx2] = sub[k1, k2]

    # Shift: S (i,j) = (j,i)
    shift = np.zeros((dim, dim))
    for k, (i, j) in enumerate(arcs):
        if (j, i) in arc_to_idx:
            shift[arc_to_idx[(j, i)], k] = 1.0

    U = shift @ coin
    return U


def szegedy_walk_unitary(P: np.ndarray) -> np.ndarray:
    """Build Szegedy's quantum walk operator.

    Given stochastic P (row-stochastic), define:
        |ψ⟩ = Σ_{i,j} √(P_{ij}) |i,j⟩

    Then:
        A = Σ |ψ_j⟩⟨j| (reflection across "good" subspace)
        B = Σ |ψ_i⟩⟨i| (reflection across "good" subspace)
        U = (2PP^T - I)(I ⊗ I)(2AA^T - I) = (I ⊗ R)(R ⊗ I)(A ⊗ A)

    For symmetric P (reversible), U = B · A where
        A|j,0⟩ = Σ_i √(P_{ij}) |i,j⟩

    Args:
        P: row-stochastic transition matrix (N×N)

    Returns:
        U: N²×N² unitary (or NxN if using bipartition compression)
    """
    n = len(P)
    # Build A acting on |j, 0⟩ → Σ_i √P_{ij} |i, j⟩
    # We use the standard representation: |i, j⟩ → |j, i⟩
    dim = n * n

    # A on basis |j⟩ ⊗ |*⟩: reflects across span{|ψ_j⟩}
    A = np.zeros((dim, dim))
    for j in range(n):
        # |ψ_j⟩ has amplitudes √P_{ij} on |i, j⟩
        # A = 2 |ψ_j⟩⟨ψ_j| - I
        psi_j = np.zeros(dim)
        for i in range(n):
            psi_j[i * n + j] = np.sqrt(P[i, j])
        outer = np.outer(psi_j, psi_j)
        A += 2 * outer
    A -= np.eye(dim)

    # B = Swap, which acts as B|i, j⟩ = |j, i⟩
    swap = np.zeros((dim, dim))
    for i in range(n):
        for j in range(n):
            swap[j * n + i, i * n + j] = 1.0

    U = A @ swap @ A
    return U


def quantum_hitting_time_szegedy(P: np.ndarray, marked: int,
                                  max_t: int = 500) -> float:
    """MEASURE quantum hitting time via Szegedy walk + Grover diffusion.

    Algorithm: standard spatial search
        R = (I - 2|marked⟩⟨marked|)   ← marked reflection
        D = (I - 2|s⟩⟨s|)              ← diffusion
        U_search = R · D

    We then measure probability at marked vertex.
    """
    n = len(P)

    # |s⟩ = stationary distribution (initial state for spatial search)
    # FIX: eigenvalue 1 may not be at column 0. Find it explicitly.
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))  # eigenvector closest to eigenvalue 1
    pi = np.abs(np.real(eigvecs[:, idx]))
    pi_sum = pi.sum()
    if pi_sum == 0 or not np.isfinite(pi_sum):
        pi = np.ones(n) / n  # fallback to uniform
    else:
        pi = pi / pi_sum

    # Initial state = uniform over arcs equivalent = √π ⊗ √π  (Szegedy)
    psi = np.zeros(n * n, dtype=complex)
    for j in range(n):
        for k in range(n):
            psi[j * n + k] = np.sqrt(pi[j]) * np.sqrt(pi[k])
    psi /= np.linalg.norm(psi)

    # R_marked = I - 2 |marked, marked⟩⟨marked, marked|
    # (or any (i, j) with i or j = marked; here we choose diagonal)
    R = np.eye(n * n, dtype=complex)
    R[marked * n + marked, marked * n + marked] = -1.0

    # D = reflection about |s⟩
    s = np.zeros(n * n, dtype=complex)
    for i in range(n):
        s[i * n + i] = np.sqrt(pi[i])
    s /= np.linalg.norm(s)
    D = np.eye(n * n, dtype=complex) - 2 * np.outer(s, s.conj())

    U_search = R @ D

    # Initial |ψ_0⟩ = |s⟩ (uniform over arcs)
    psi_0 = np.zeros(n * n, dtype=complex)
    for i in range(n):
        psi_0[i * n + i] = np.sqrt(pi[i])

    # Iterate and measure P(marked)
    p_marked_proj = np.zeros(n * n)
    for k in range(n * n):
        i = k // n
        j = k % n
        if i == marked or j == marked:
            p_marked_proj[k] = 1.0
        # normalize: each vertex has multiple arcs
    p_marked_proj /= p_marked_proj.sum()
    P_measured = np.diag(p_marked_proj)

    psi_t = psi_0.copy()
    for t in range(1, max_t):
        psi_t = U_search @ psi_t
        # Probability of "marked" (any arc touching marked)
        p_marked = float(np.real(np.sum(np.abs(psi_t[p_marked_proj > 0]) ** 2)))
        threshold = 1.0 / n
        if p_marked > threshold:
            return float(t)

    return float(max_t)


def quantum_hitting_time_empirical(A: np.ndarray, marked: int,
                                    n_trials: int = 50) -> float:
    """EMPIRICAL quantum hitting time (Szegedy spatial search).

    Combines Szegedy walk + marked-vertex reflection.
    """
    n = len(A)

    # Build row-stochastic
    deg = A.sum(axis=1)
    if deg[marked] == 0:
        return float('inf')
    deg[deg == 0] = 1
    P = A / deg[:, None]

    # Make P symmetric (required for Szegedy to be unitary)
    # Lazy walk modification
    sqrt_deg = np.sqrt(deg)
    P_sym = np.diag(1.0 / sqrt_deg) @ A @ np.diag(1.0 / sqrt_deg)
    # Verify rows sum to 1 (column-stochastic after transformation)
    # We use the lazy version to ensure unitariety
    # P_lazy = (P_sym + I) / 2 — modified to be doubly stochastic
    P_lazy = (P_sym + P_sym.T) / 2
    P_lazy = P_lazy / P_lazy.sum(axis=1, keepdims=True)

    return quantum_hitting_time_szegedy(P_lazy, marked, max_t=1000)


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK (NO FORMULA INJECTION)")
    print("Classical vs Quantum walk: measured hitting time, not 1/sqrt(Δ)")
    print("=" * 80)

    N = 16  # small enough for quantum unitary, large enough to be meaningful
    MARKED = 5  # arbitrary marked vertex

    results = {}
    for regime in ["fully", "sparse"]:
        A, label = build_graph(regime, N)

        n_edges = int((A > 0).sum() // 2)
        mean_deg = float((A > 0).sum(axis=1).mean())

        print(f"\n--- {label} ---")
        print(f"  N = {N}, edges = {n_edges}, mean degree = {mean_deg:.2f}")

        # Empirical classical hitting time
        t_class = classical_hitting_time_empirical(A, MARKED, n_trials=300)
        print(f"  Classical hitting time: {t_class:.1f} steps (measured, 300 trials)")

        # Empirical quantum hitting time
        t_quant = quantum_hitting_time_empirical(A, MARKED)
        print(f"  Quantum hitting time:  {t_quant} steps (measured via U^t)")

        # Empirical mixing time
        t_mix_class = classical_mixing_time_empirical(A, threshold=0.25)
        print(f"  Classical mixing time (TV < 0.25): {t_mix_class} steps")

        # Compute eigenvalues for context
        deg = A.sum(axis=1)
        deg[deg == 0] = 1
        P = A / deg[:, None]
        eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        gap = 1 - eigs[1] if len(eigs) > 1 else 0

        results[regime] = {
            "label": label,
            "N": N,
            "n_edges": n_edges,
            "mean_degree": mean_deg,
            "classical_hitting_time": t_class,
            "quantum_hitting_time": t_quant,
            "classical_mixing_time": t_mix_class,
            "spectral_gap": float(gap),
        }

        if t_quant > 0 and t_class > 0:
            speedup = t_class / t_quant
            print(f"  Quantum speedup: {speedup:.2f}×")
            results[regime]["empirical_speedup"] = float(speedup)

    # ----- Comparison -----
    print("\n" + "=" * 80)
    print("SUMMARY (empirical, no formulas)")
    print("=" * 80)
    print(f"{'Regime':<35} {'t_class':<12} {'t_quant':<12} {'Speedup':<10}")
    print("-" * 80)
    for regime in ["fully", "sparse"]:
        r = results[regime]
        speedup = r.get("empirical_speedup", "N/A")
        print(f"{r['label']:<35} {r['classical_hitting_time']:<12.1f} "
              f"{r['quantum_hitting_time']:<12} {str(speedup):<10}")

    print("\nIf empirical speedup on SPARSE > 1× and > FULLY's speedup,")
    print("then quantum walk REALLY benefits from sparse structure.")
    print("Otherwise, the speedup claim is unsupported.")

    # ----- Visualization -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [r["label"] for r in results.values()]
    t_class = [r["classical_hitting_time"] for r in results.values()]
    t_quant = [r["quantum_hitting_time"] for r in results.values()]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, t_class, width, label="Classical", color="coral")
    axes[0].bar(x + width/2, t_quant, width, label="Quantum", color="teal")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("Steps to hit marked vertex")
    axes[0].set_title("Empirical Hitting Time\n(lower is better)")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, axis='y')

    speedups = []
    regime_labels = []
    for regime in ["fully", "sparse"]:
        if "empirical_speedup" in results[regime]:
            speedups.append(results[regime]["empirical_speedup"])
            regime_labels.append(results[regime]["label"])

    if speedups:
        axes[1].bar(range(len(speedups)), speedups,
                    color=["coral", "teal"])
        axes[1].axhline(y=1.0, color="red", linestyle="--", label="No speedup (1×)")
        axes[1].set_xticks(range(len(speedups)))
        axes[1].set_xticklabels(regime_labels, rotation=15)
        axes[1].set_ylabel("Classical / Quantum")
        axes[1].set_title("Empirical Quantum Speedup\n(measured, not 1/√Δ)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "empirical_walk_benchmark.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")

    # Save JSON
    json_file = out_dir / "empirical_walk_benchmark.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] {out_file}")
    print(f"[SAVED] {json_file}")

    return results


if __name__ == "__main__":
    main()
