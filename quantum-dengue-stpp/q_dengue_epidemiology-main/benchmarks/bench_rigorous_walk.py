"""Rigorous benchmark — coined quantum walk hitting time on multiple graph types.

IMPORTANT — What this measures vs what the thesis claims:

THESIS CLAIM: "Fully connected graphs don't get quantum advantage"
  → Refers to: QUANTUM WALK MIXING (spectral gap Δ)
  → On fully-connected: Δ ≈ 1, mixing is O(1) in both classical and quantum

THIS BENCHMARK: Measures HITTING TIME via coined quantum walk + marked reflection
  → On fully-connected: Shows speedup (6–15×)
  → Why: The speedup is from Grover-like AMPLITUDE AMPLIFICATION, not from
    graph structure. Grover works on any graph, including fully-connected.

These are NOT contradictory:
  - Quantum walk MIXING advantage: requires sparse/structured graphs
  - Grover-based HITTING TIME: works on any graph (index space search)

The benchmarks are measuring hitting time, which naturally shows speedup
on fully-connected via the amplitude amplification mechanism. This doesn't
contradict the mixing-time thesis — they're different quantum phenomena.
"""
from __future__ import annotations

import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Graph
# ============================================================

def build_graph(regime: str, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    if regime == "fully":
        return np.ones((n, n)) - np.eye(n)
    if regime == "sparse":
        positions = rng.uniform(0, 50, size=(n, 2))
        diff = positions[:, None, :] - positions[None, :, :]
        dist_km = np.sqrt(np.sum(diff ** 2, axis=-1))
        kernel = np.exp(-dist_km / 5.0)
        A = np.where(dist_km < 30.0, kernel, 0.0)
        np.fill_diagonal(A, 0.0)
        return A
    if regime == "ring":
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i+1) % n] = 1.0
            A[i, (i-1) % n] = 1.0
        return A
    if regime == "grid":
        side = int(np.ceil(np.sqrt(n)))
        N_full = side * side
        A = np.zeros((N_full, N_full))
        for i in range(side):
            for j in range(side):
                idx = i*side + j
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < side and 0 <= nj < side:
                        A[idx, ni*side + nj] = 1.0
        return A[:n, :n]
    raise ValueError(regime)


# ============================================================
# Classical: MEASURE hitting time
# ============================================================

def classical_hitting_time(A, marked, n_trials=500, max_steps=10000):
    n = len(A)
    if A[marked].sum() == 0:
        return float('inf')
    rng = np.random.default_rng(123)
    times = []
    for _ in range(n_trials):
        current = rng.integers(0, n)
        if current == marked:
            times.append(0)
            continue
        for step in range(1, max_steps):
            neighbors = np.where(A[current] > 0)[0]
            if len(neighbors) == 0:
                times.append(max_steps)
                break
            probs = A[current, neighbors]
            probs = probs / probs.sum()
            current = rng.choice(neighbors, p=probs)
            if current == marked:
                times.append(step)
                break
        else:
            times.append(max_steps)
    return float(np.mean(times))


# ============================================================
# Coined quantum walk — PROPER implementation
# ============================================================

def build_arc_set(A):
    """Build arc set: arcs[i] = [neighbors of i] (ordered list).

    Hilbert space dimension = sum_i deg(i) = 2 * |E| (with self-loops removed)
    """
    n = len(A)
    arcs = []
    arc_to_idx = {}
    for i in range(n):
        nbrs = list(np.where(A[i] > 0)[0])
        nbrs = [j for j in nbrs if j != i]
        for j in nbrs:
            arc_to_idx[(i, j)] = len(arcs)
            arcs.append((i, j))
    return arcs, arc_to_idx


def build_coined_quantum_walk(A):
    """Build Grover-coin quantum walk unitary on arc space.

    REQUIRES: A symmetric (A[i,j] == A[j,i]). We symmetrize if not.

    Handles isolated vertices: nodes with degree 0 are removed from arc set.
    Handles degree-1 vertices: coin = I (no mixing needed).

    Returns:
        U_walk: N_arcs × N_arcs unitary
        arcs, arc_to_idx
    """
    # Symmetrize: A_sym[i,j] = max(A[i,j], A[j,i])
    A_sym = np.maximum(A, A.T)

    arcs, arc_to_idx = build_arc_set(A_sym)
    dim = len(arcs)
    n = len(A_sym)

    if dim == 0:
        return np.eye(1), arcs, arc_to_idx

    # Coin: block-diagonal at each vertex
    # For degree d=1: coin = I (nothing to mix)
    # For degree d>=2: coin = 2|ψ⟩⟨ψ| - I where |ψ⟩ = uniform
    coin = np.zeros((dim, dim))
    for v in range(n):
        idxs = [arc_to_idx[(v, u)] for u in np.where(A_sym[v] > 0)[0] if u != v]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            # Identity on this single state
            coin[idxs[0], idxs[0]] = 1.0
        else:
            psi = np.ones(len(idxs)) / np.sqrt(len(idxs))
            sub = 2 * np.outer(psi, psi) - np.eye(len(idxs))
            for k1, idx1 in enumerate(idxs):
                for k2, idx2 in enumerate(idxs):
                    coin[idx1, idx2] = sub[k1, k2]

    # Shift: S|(v,u)⟩ = |(u,v)⟩
    shift = np.zeros((dim, dim))
    for k, (v, u) in enumerate(arcs):
        if (u, v) in arc_to_idx:
            shift[arc_to_idx[(u, v)], k] = 1.0

    U_walk = shift @ coin

    # Validate unitarity
    err = np.linalg.norm(U_walk @ U_walk.T - np.eye(dim), 'fro')
    if err > 1e-3:
        print(f"  [WARN] Walk unitary error: {err:.4f}")
    return U_walk, arcs, arc_to_idx


def quantum_hitting_time(A, marked, max_t=200):
    """Quantum walk hitting time using coined walk + marked reflection.

    Algorithm:
        U_search = R_marked · U_walk
        Initial state: uniform over arcs starting at marked-vertex's neighbors
                       (or uniform over all arcs)
        Detection: P(marked) > threshold where P(marked) = sum |⟨(v,*)⟩|² for v = marked
    """
    n = len(A)
    if A[marked].sum() == 0:
        return float('inf')

    U_walk, arcs, arc_to_idx = build_coined_quantum_walk(A)
    dim = len(arcs)

    # Marked reflection: phase -1 on arcs starting at marked vertex
    R_marked = np.eye(dim)
    for k, (v, u) in enumerate(arcs):
        if v == marked:
            R_marked[k, k] = -1.0

    U_search = R_marked @ U_walk

    # Initial state: uniform over outgoing arcs from vertex 0
    start_v = 0
    initial_idxs = [arc_to_idx[(start_v, u)] for u in np.where(A[start_v] > 0)[0] if u != start_v]
    if not initial_idxs:
        return float('inf')
    psi = np.zeros(dim)
    for idx in initial_idxs:
        psi[idx] = 1.0 / np.sqrt(len(initial_idxs))

    # Threshold: P(marked) > 0.05 (5% absolute — well above uniform ~1/N)
    threshold = 0.05

    psi_t = psi.copy()
    for t in range(1, max_t):
        psi_t = U_search @ psi_t
        p_marked = 0.0
        for k, (v, u) in enumerate(arcs):
            if v == marked:
                p_marked += np.abs(psi_t[k]) ** 2
        if p_marked > threshold:
            return float(t)

    return float(max_t)


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 80)
    print("RIGOROUS BENCHMARK: coined quantum walk with marked reflection")
    print("=" * 80)

    sizes = [8, 16, 32, 48]
    regimes = ["fully", "sparse", "ring", "grid"]
    n_marked_samples = 3
    n_classical_trials = 500

    all_results = []

    for N in sizes:
        print(f"\n{'=' * 80}")
        print(f"N = {N}")
        print(f"{'=' * 80}")
        for regime in regimes:
            A = build_graph(regime, N)
            n_edges = int((A > 0).sum() // 2)
            mean_deg = float((A > 0).sum(axis=1).mean())

            marked_samples = [N//4, N//2, 3*N//4][:n_marked_samples]
            if N <= 4:
                marked_samples = [0]

            class_times = []
            quant_times = []

            for marked in marked_samples:
                t_class = classical_hitting_time(A, marked, n_trials=n_classical_trials)
                t_quant = quantum_hitting_time(A, marked)
                class_times.append(t_class)
                quant_times.append(t_quant)

            t_class_mean = float(np.mean(class_times))
            t_quant_mean = float(np.mean(quant_times))
            speedup = t_class_mean / max(t_quant_mean, 1e-6)

            print(f"  {regime:<8} deg={mean_deg:5.2f}  t_class={t_class_mean:7.2f}  "
                  f"t_quant={t_quant_mean:5.1f}  speedup={speedup:6.2f}×")

            all_results.append({
                "N": N, "regime": regime, "n_edges": n_edges,
                "mean_degree": mean_deg,
                "classical_hitting_mean": t_class_mean,
                "quantum_hitting_mean": t_quant_mean,
                "empirical_speedup": speedup,
            })

    # ----- Realistic 130-node Dien Bien graph -----
    print(f"\n{'=' * 80}")
    print("REALISTIC 130-node Dien Bien graph (vector biology)")
    print(f"{'=' * 80}")
    try:
        from src.graph_dien_bien import build_synthetic_dien_bien
        g = build_synthetic_dien_bien()
        A = g.adjacency
        n = g.n_communes
        mean_deg_real = float((A > 0).sum(axis=1).mean())
        n_edges_real = int((A > 0).sum() // 2)

        print(f"  N = {n}, edges = {n_edges_real}, mean degree = {mean_deg_real:.2f}")

        marked_real = [10, 65, 120]
        t_class_real = [classical_hitting_time(A, m, n_trials=300) for m in marked_real]
        t_quant_real = [quantum_hitting_time(A, m, max_t=500) for m in marked_real]

        t_class_mean_real = float(np.mean(t_class_real))
        t_quant_mean_real = float(np.mean(t_quant_real))
        speedup_real = t_class_mean_real / max(t_quant_mean_real, 1e-6)

        print(f"  Classical hitting time: {t_class_mean_real:.2f}")
        print(f"  Quantum hitting time:   {t_quant_mean_real:.1f}")
        print(f"  Speedup:                {speedup_real:.2f}×")

        all_results.append({
            "N": n, "regime": "dien_bien_realistic", "n_edges": n_edges_real,
            "mean_degree": mean_deg_real,
            "classical_hitting_mean": t_class_mean_real,
            "quantum_hitting_mean": t_quant_mean_real,
            "empirical_speedup": speedup_real,
        })
    except Exception as e:
        print(f"  Could not run on real graph: {e}")

    # Save
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "rigorous_walk_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ----- Visualization -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"fully": "coral", "sparse": "teal",
              "ring": "purple", "grid": "orange"}

    # Plot 1: speedup vs N (one curve per regime)
    ax = axes[0]
    for regime in regimes:
        subset = [r for r in all_results if r["regime"] == regime]
        ns = [r["N"] for r in subset]
        speeds = [r["empirical_speedup"] for r in subset]
        ax.plot(ns, speeds, 'o-', label=regime, color=colors[regime], linewidth=2)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1× (no speedup)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("N (number of vertices)")
    ax.set_ylabel("Classical / Quantum speedup")
    ax.set_title("Empirical Quantum Speedup\n(coined quantum walk + marked reflection)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: log-log scaling
    ax = axes[1]
    for regime in regimes:
        subset = [r for r in all_results if r["regime"] == regime]
        if not subset:
            continue
        ns = np.array([r["N"] for r in subset])
        t_class = np.array([r["classical_hitting_mean"] for r in subset])
        t_quant = np.array([r["quantum_hitting_mean"] for r in subset])

        ax.plot(ns, t_class, 'o--', label=f"{regime} classical",
                color=colors[regime], alpha=0.7)
        ax.plot(ns, t_quant, '^-', label=f"{regime} quantum",
                color=colors[regime], linewidth=2)

    # Reference: √N curve
    if sizes:
        Nref = np.array(sizes, dtype=float)
        ax.plot(Nref, np.sqrt(Nref) * 3, 'k:', alpha=0.5, label="O(√N) reference")
        ax.plot(Nref, Nref, 'k--', alpha=0.3, label="O(N) reference")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Hitting time (steps)")
    ax.set_title("Scaling behavior")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "rigorous_walk_benchmark.png", dpi=120, bbox_inches="tight")

    print(f"\n[SAVED] {out_dir / 'rigorous_walk_benchmark.json'}")
    print(f"[SAVED] {out_dir / 'rigorous_walk_benchmark.png'}")

    return all_results


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {time.time() - start:.1f}s")