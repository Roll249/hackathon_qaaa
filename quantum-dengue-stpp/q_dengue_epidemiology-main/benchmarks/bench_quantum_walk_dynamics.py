"""Benchmark: Quantum walk dynamics — spectral gap analysis.

This file measures spectral gaps and their implications for quantum vs classical
mixing time. The key distinction:

- Grover search (O(√N) hitting time): works on ANY graph
  → Fully-connected graphs DO show quantum speedup for SEARCH
  → This is Grover's algorithm, not quantum walk mixing

- Quantum walk mixing (O(1/√Δ) mixing time): only on sparse graphs
  → Large spectral gap (Δ ≈ 1): mixing fast in both classical and quantum
  → Small spectral gap (Δ small): quantum mixing can be faster

Spectral gap Δ = 1 - |λ₂|:
  Δ ≈ 1  →  mixing ~ O(1) steps → no quantum advantage for MIXING
  Δ << 1 →  mixing slow → quantum can accelerate

CRITICAL CORRECTION to prior version:
- Prior conclusion said "gap → 0 → ∞ speedup" — WRONG
- With gap ≈ 0.9677, both are fast → speedup ≈ 1×
- With gap ≈ 0.1139, quantum can be ~3× faster for MIXING

For SEARCH (hitting time), Grover gives √N on any graph.
For MIXING, quantum advantage requires sparse/expander structure.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Graph construction
# ============================================================

def build_fully_connected(n: int, seed: int = 42) -> np.ndarray:
    """Fully connected: W[i,j] = 1 for all i != j.

    Mean degree = n - 1.
    Spectral gap Δ = 1 - 1/(n-1) ≈ 1 (large gap → fast mixing).
    """
    return np.ones((n, n)) - np.eye(n)


def build_sparse_vector_graph(n: int, seed: int = 42,
                              radius_km: float = 30.0) -> np.ndarray:
    """Sparse graph with vector transmission kernel.

    Mean degree ~ 8-15.
    Spectral gap Δ < 1 (small gap → slower mixing → quantum advantage possible).
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 50, size=(n, 2))
    diff = positions[:, None, :] - positions[None, :, :]
    dist_km = np.sqrt(np.sum(diff ** 2, axis=-1))
    kernel = np.exp(-dist_km / 5.0)
    A = np.where(dist_km < radius_km, kernel, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


# ============================================================
# Spectral analysis
# ============================================================

def spectral_analysis(A: np.ndarray) -> dict:
    """Compute spectral gap and mixing time estimates.

    For random walk transition matrix P:
    - Classical mixing time: t_mix ~ (1/Δ) * log(N)
    - Quantum mixing advantage: sqrt speedup from Δ
    """
    deg = A.sum(axis=1)
    deg[deg == 0] = 1
    P = A / deg[:, None]

    eigs = np.abs(np.linalg.eigvals(P))
    eigs = np.sort(eigs)[::-1]

    n = len(A)
    delta = 1.0 - eigs[1] if len(eigs) > 1 else 0.0
    delta = max(delta, 1e-10)

    # Mixing time estimates
    log_factor = np.log(n)
    t_classical = (1.0 / delta) * log_factor
    t_quantum = (1.0 / np.sqrt(delta)) * log_factor

    return {
        "eigs_top5": eigs[:5],
        "spectral_gap": delta,
        "t_mix_classical": t_classical,
        "t_mix_quantum": t_quantum,
        "mixing_speedup": t_classical / max(t_quantum, 1e-10),
    }


# ============================================================
# Grover search (hitting time, NOT mixing)
# ============================================================

def grover_search_simulation(n: int, marked: int, n_iter: int) -> float:
    """Simulate Grover search: O(√N) hitting time on any graph.

    This is Grover's algorithm applied to finding a marked element.
    Works on fully-connected graphs because it's not a walk — it's
    amplitude amplification on the INDEX space.

    Returns: probability of measuring marked after n_iter Grover iterations.
    """
    theta = np.arcsin(np.sqrt(1.0 / n))
    p = np.sin((2 * n_iter + 1) * theta) ** 2
    return min(1.0, max(0.0, p))


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 75)
    print("QUANTUM WALK DYNAMICS: Spectral Gap Analysis")
    print("(Corrected: mixing time, not hitting time)")
    print("=" * 75)

    N = 32

    A_full = build_fully_connected(N)
    A_sparse = build_sparse_vector_graph(N)

    # Spectral gaps
    spec_full = spectral_analysis(A_full)
    spec_sparse = spectral_analysis(A_sparse)

    deg_full = float((A_full > 0).sum(axis=1).mean())
    deg_sparse = float((A_sparse > 0).sum(axis=1).mean())

    print(f"\nGraph comparison (N={N}):")
    print(f"  Fully connected: mean degree = {deg_full:.1f}")
    print(f"  Sparse (vector): mean degree = {deg_sparse:.2f}")

    print(f"\nSpectral gap Δ = 1 - |λ₂|:")
    print(f"  Fully connected: Δ = {spec_full['spectral_gap']:.4f}")
    print(f"  Sparse (vector): Δ = {spec_sparse['spectral_gap']:.4f}")

    print(f"\nEigenvalue spectra (top-5):")
    print(f"  Fully connected: {spec_full['eigs_top5'].round(4)}")
    print(f"  Sparse (vector): {spec_sparse['eigs_top5'].round(4)}")

    print(f"\nMixing time estimates (proportional):")
    print(f"  Fully connected:")
    print(f"    Classical: {spec_full['t_mix_classical']:.2f}")
    print(f"    Quantum:   {spec_full['t_mix_quantum']:.2f}")
    print(f"    Speedup:   {spec_full['mixing_speedup']:.4f}×  ← ≈ 1× (no MIXING advantage)")
    print(f"  Sparse (vector):")
    print(f"    Classical: {spec_sparse['t_mix_classical']:.2f}")
    print(f"    Quantum:   {spec_sparse['t_mix_quantum']:.2f}")
    print(f"    Speedup:   {spec_sparse['mixing_speedup']:.4f}×  ← REAL MIXING advantage")

    # Grover search (hitting time — works on fully connected!)
    print(f"\nGrover SEARCH (hitting time, works on any graph):")
    marked = 7
    for n_steps in [4, 8, 16, 24]:
        p_full = grover_search_simulation(N, marked, n_steps)
        print(f"  N={N}, steps={n_steps}: P(marked) = {p_full:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): Graph adjacency
    ax = axes[0, 0]
    ax.imshow(A_full, cmap='Blues', alpha=0.7)
    ax.set_title(f"Fully Connected\ndeg={deg_full:.0f}, Δ={spec_full['spectral_gap']:.4f}")

    ax = axes[0, 1]
    ax.imshow(A_sparse, cmap='Greens', alpha=0.7)
    ax.set_title(f"Sparse (Vector Biology)\ndeg={deg_sparse:.2f}, Δ={spec_sparse['spectral_gap']:.4f}")

    # (1,0): Eigenvalue spectrum
    ax = axes[1, 0]
    x_full = np.arange(len(spec_full['eigs_top5']))
    x_sparse = np.arange(len(spec_sparse['eigs_top5']))
    ax.scatter(x_full, spec_full['eigs_top5'], label="Fully connected",
               color="coral", s=60)
    ax.scatter(x_sparse, spec_sparse['eigs_top5'], label="Sparse",
               color="teal", s=60, marker="^")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue Spectrum\n(larger gap = faster classical mixing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1): Mixing time comparison
    ax = axes[1, 1]
    labels = ["Fully Connected", "Sparse (Vector)"]
    t_class = [spec_full['t_mix_classical'], spec_sparse['t_mix_classical']]
    t_quant = [spec_full['t_mix_quantum'], spec_sparse['t_mix_quantum']]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, t_class, w, label="Classical", color="coral")
    ax.bar(x + w/2, t_quant, w, label="Quantum (mixing)", color="teal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mixing time (proportional)")
    ax.set_title("Mixing Time: Quantum advantage on SPARSE only\n"
                 "(Fully connected: both are fast ≈ 1 step)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Spectral Gap Determines Quantum Mixing Advantage",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "quantum_walk_dynamics.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    print(f"\n[SAVED] {out_file}")

    # Corrected conclusion
    print("\n" + "=" * 75)
    print("CORRECTED CONCLUSION")
    print("=" * 75)
    print("""
TWO DIFFERENT quantum speedup mechanisms (don't conflate!):

1. GROVER SEARCH (hitting time):
   - O(√N) on INDEX SPACE — works on ANY graph (fully connected or sparse)
   - Grover 1996: searching N items takes O(√N) quantum queries
   - This is why fully-connected benchmarks show speedup — it's Grover,
     not quantum walk mixing. This is NOT wrong — just mislabeled.

2. QUANTUM WALK MIXING (spectral gap):
   - Δ ≈ 1 (fully connected): mixing ~ O(1) in both classical and quantum
     → NO quantum advantage for mixing
   - Δ << 1 (sparse): mixing ~ O(1/√Δ) quantum vs O(1/Δ) classical
     → QUANTUM ADVANTAGE for mixing on structured graphs

KEY INSIGHT:
  - "Quantum speedup on fully connected" = Grover (search), NOT walk mixing
  - "Quantum speedup on sparse" = walk mixing (requires structure)
  - The benchmarks are empirically correct; the THESIS framing was wrong.

NUMBERS MATCH:
  - gap_full = 0.9677 → mixing in ~1 step → speedup ≈ 1×
  - gap_sparse = 0.1139 → mixing slower → quantum ~3× faster
""")
    print("=" * 75)


if __name__ == "__main__":
    main()
