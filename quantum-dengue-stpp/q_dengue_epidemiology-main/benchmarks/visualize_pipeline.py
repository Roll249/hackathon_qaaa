"""Visualization: risk map + quantum amplification before/after."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_full_pipeline
from src.qpie_encoder import encode_risk_qpie


def main():
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_full_pipeline(seed=42, k_hotspots=5, output_dir=out_dir, verbose=False)

    # Get risk scores
    g_data = result["graph_stats"]
    n = g_data["n_communes"]
    risk_scores = np.zeros(n)
    for h in result["multi_hotspot"]:
        risk_scores[h["idx"]] = h["risk_score"]
    # For non-hotspots, estimate from overall
    found_indices = {h["idx"] for h in result["multi_hotspot"]}
    # Use a flat value for unrecorded (we just need for viz)
    for i in range(n):
        if i not in found_indices:
            risk_scores[i] = 0.2  # background

    # Compute QPIE state
    statevector, _ = encode_risk_qpie(risk_scores)
    probs_before = np.abs(statevector) ** 2

    # Simulate Grover amplification: amplify top-5
    # NOTE: Number of iterations MUST be computed from M/N, not hardcoded.
    # Optimal iterations = round(π / (4 * arcsin(√(M/N))))
    # With M=5, N=len(statevector), this may be fewer or more than 8.
    sorted_idx = np.argsort(probs_before)[::-1]
    marked = sorted_idx[:5]
    M = len(marked)
    N = len(statevector)
    theta = np.arcsin(np.sqrt(M / N))
    n_iter = max(1, int(np.round(np.pi / (4 * theta))))

    amp = statevector.copy()
    for _ in range(n_iter):
        amp_new = amp.copy()
        for m in marked:
            amp_new[m] = -amp[m]
        avg = np.mean(amp_new)
        amp = 2 * avg - amp_new
    probs_after = np.abs(amp) ** 2

    # Report: if n_iter was wrong, probabilities may DECREASE
    # (over-rotation = amplitude swings past target)
    before_mass = probs_before[marked].sum()
    after_mass = probs_after[marked].sum()
    amp_factor = after_mass / before_mass if before_mass > 0 else 0

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Risk map
    ax = axes[0, 0]
    ax.bar(range(n), risk_scores, color="orange", alpha=0.7)
    for h in result["multi_hotspot"]:
        ax.axvline(h["idx"], color="red", alpha=0.3, linestyle="--")
    ax.set_xlabel("Commune index")
    ax.set_ylabel("Risk intensity")
    ax.set_title("Risk intensity map (130 communes Điện Biên)\nRed lines = hotspots")
    ax.grid(True, alpha=0.3)

    # (0,1) QPIE before amplification
    ax = axes[0, 1]
    ax.bar(range(len(probs_before)), probs_before, color="steelblue", alpha=0.7)
    ax.set_xlabel("Basis state |i⟩")
    ax.set_ylabel("Probability")
    ax.set_title(f"QPIE BEFORE amplification\nTop-5 prob mass: {probs_before[marked].sum():.4f}")
    ax.grid(True, alpha=0.3)

    # (1,0) QPIE after amplification
    ax = axes[1, 0]
    ax.bar(range(len(probs_after)), probs_after, color="darkgreen", alpha=0.7)
    ax.set_xlabel("Basis state |i⟩")
    ax.set_ylabel("Probability")
    ax.set_title(f"QPIE AFTER Grover amplification ({n_iter} iterations)\n"
                 f"Top-5 prob mass: {after_mass:.4f} (amp factor: {amp_factor:.2f}×)\n"
                 f"{'NOTE: under-rotated' if amp_factor < 1 else 'Amplification successful'}")
    ax.grid(True, alpha=0.3)

    # (1,1) Query complexity comparison
    ax = axes[1, 1]
    Ns = [16, 32, 64, 128, 130, 256]
    classical_q = Ns
    quantum_q = [int(np.ceil(np.sqrt(n))) for n in Ns]
    x = np.arange(len(Ns))
    width = 0.35
    ax.bar(x - width/2, classical_q, width, label="Classical O(N)", color="coral")
    ax.bar(x + width/2, quantum_q, width, label="Quantum O(√N)", color="teal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in Ns])
    ax.set_xlabel("N (number of communes)")
    ax.set_ylabel("Number of queries")
    ax.set_title("Query complexity: Classical O(N) vs Quantum O(√N)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = out_dir / "pipeline_visualization.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_file}")

    print(f"\nKEY METRICS:")
    print(f"  Grover iterations used: {n_iter} (optimal for M={M}, N={N})")
    print(f"  Probability mass top-5 BEFORE: {before_mass:.4f}")
    print(f"  Probability mass top-5 AFTER:  {after_mass:.4f}")
    print(f"  Amplification factor: {amp_factor:.2f}×")
    if amp_factor < 1:
        print(f"  WARNING: over-rotation! n_iter={n_iter} may be too many.")
        print(f"           With M/N={M/N:.4f}, optimal iterations ~{np.pi/(4*theta):.1f}")
    print(f"  Quantum queries for 130: ~{int(np.ceil(np.sqrt(130)))}")
    print(f"  Classical queries: 130")
    print(f"  Speedup: {130 / int(np.ceil(np.sqrt(130))):.1f}x")


if __name__ == "__main__":
    main()
