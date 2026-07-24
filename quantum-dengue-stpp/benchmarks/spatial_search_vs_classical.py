#!/usr/bin/env python3
"""Benchmark: Quantum Spatial Search vs Classical Brute Force.

PURPOSE
-------
This benchmark compares Grover's quantum spatial search against classical
brute-force search for dengue hotspot detection.

WHAT IS QUANTUM
---------------
- Grover's algorithm for spatial search
- O(√N) oracle evaluations vs O(N) for classical

WHAT IS CLASSICAL
----------------
- Brute-force scan of all grid cells
- O(N) oracle evaluations

HONEST CLAIMS
------------
1. √N speedup in ORACLE QUERIES (not wall-clock time)
2. Wall-clock advantage requires:
   - Fast quantum gates (<< classical oracle)
   - Low error rates
   - Large grid (N >> 1000)

THEORETICAL REFERENCE
--------------------
Figgatt et al. 2017, "Complete 3-qubit Grover search on a quantum computer",
Nature Communications 8, 1918 (2017).

Usage:
    python benchmarks/spatial_search_vs_classical.py \
        --grid-sizes 30 50 100 \
        --seeds 1 2 3 \
        --top-k 5 \
        --output-dir output_result/q_stpp_v18
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

# Matplotlib config
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(
        tempfile.gettempdir(),
        f"mplconfig-{os.getuid() if hasattr(os, 'getuid') else 'user'}",
    ),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.quantum.quantum_spatial_search import (
    SpatialGrid,
    RiskMap,
    run_grover_search,
    classical_spatial_search,
    compare_spatial_search,
)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark: Quantum vs Classical Spatial Search for Dengue Hotspots"
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="Grid sizes (n x n). Default: 30 50 100",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Random seeds. Default: 1 2 3",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hotspots to find. Default: 5",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Quantum circuit shots. Default: 1024",
    )
    parser.add_argument(
        "--output-dir",
        default="output_result/q_stpp_v18",
        help="Output directory. Default: output_result/q_stpp_v18",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generate Realistic Dengue Risk Map
# ---------------------------------------------------------------------------


def generate_realistic_risk_map(
    nx: int, ny: int, seed: int = 42
) -> RiskMap:
    """Generate a realistic dengue risk map with clustered hotspots.

    The map includes:
    - Background risk (low, uniform)
    - 3-8 clustered hotspots (realistic dengue outbreaks)
    - Gaussian falloff from hotspot centers

    Args:
        nx, ny: Grid dimensions.
        seed: Random seed.

    Returns:
        RiskMap with realistic dengue risk values.
    """

    rng = np.random.default_rng(seed)
    grid = SpatialGrid(nx=nx, ny=ny)

    # Background risk (low)
    risk = rng.uniform(0.1, 0.3, size=(nx, ny))

    # Add clustered hotspots
    n_hotspots = rng.integers(3, 8)
    for _ in range(n_hotspots):
        # Random hotspot center
        cx = rng.uniform(0, nx - 1)
        cy = rng.uniform(0, ny - 1)
        radius = rng.uniform(2, 6)
        intensity = rng.uniform(0.7, 1.0)

        # Gaussian falloff
        for i in range(nx):
            for j in range(ny):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist < radius:
                    # Smooth falloff
                    risk[i, j] += intensity * math.exp(-(dist**2) / (2 * (radius / 3) ** 2))

    return RiskMap(grid=grid, values=risk)


# ---------------------------------------------------------------------------
# Run Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    grid_sizes: list,
    seeds: list,
    top_k: int = 5,
    shots: int = 1024,
    verbose: bool = False,
) -> dict:
    """Run the full benchmark suite.

    Args:
        grid_sizes: List of grid sizes (n x n).
        seeds: List of random seeds.
        top_k: Number of hotspots to find.
        shots: Quantum circuit shots.
        verbose: Print detailed progress.

    Returns:
        dict with all results.
    """

    results = {
        "config": {
            "grid_sizes": grid_sizes,
            "seeds": seeds,
            "top_k": top_k,
            "shots": shots,
        },
        "rows": [],
        "summary": {},
    }

    # Theoretical summary
    theory_rows = []
    for n in [s * s for s in grid_sizes]:
        classical = n
        grover = int(math.pi / 4 * math.sqrt(n))
        speedup = classical / grover
        theory_rows.append({
            "grid_size": n,
            "classical_oracle_calls": classical,
            "grover_iterations": grover,
            "theoretical_speedup": speedup,
        })
    results["theory"] = theory_rows

    # Run experiments
    for grid_n in grid_sizes:
        for seed in seeds:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  Grid: {grid_n}×{grid_n} = {grid_n**2} cells | Seed: {seed}")
                print(f"{'=' * 60}")

            # Generate risk map
            risk_map = generate_realistic_risk_map(grid_n, grid_n, seed=seed)
            threshold = float(np.percentile(risk_map.values, 90))

            # Classical search
            classical_result = classical_spatial_search(
                risk_map, top_k=top_k, threshold=threshold, verbose=verbose
            )

            # Quantum search
            quantum_result = run_grover_search(
                risk_map,
                n_iterations=None,  # Use optimal
                threshold=threshold,
                top_k=top_k,
                shots=shots,
                seed=seed,
                verbose=verbose,
            )

            # Calculate accuracy
            true_top = set(risk_map.get_top_k_indices(top_k))
            quantum_top = set(quantum_result.top_measured[:top_k])

            accuracy_top1 = (
                1.0 if quantum_result.top_measured[0] in true_top else 0.0
            )
            accuracy_topk = len(quantum_top & true_top) / min(
                top_k, len(true_top)
            )

            # Speedup
            speedup = (
                classical_result["oracle_calls"] / quantum_result.n_iterations
                if quantum_result.n_iterations > 0
                else float("inf")
            )

            row = {
                "grid_size_n": grid_n,
                "total_cells": grid_n**2,
                "seed": seed,
                "n_qubits": risk_map.grid.n_qubits,
                "top_k": top_k,
                "threshold": threshold,
                # Classical
                "classical_oracle_calls": classical_result["oracle_calls"],
                "classical_top_indices": classical_result["top_indices"],
                "classical_time_s": classical_result["total_time_s"],
                # Quantum
                "quantum_iterations": quantum_result.n_iterations,
                "quantum_top_indices": quantum_result.top_measured[:top_k],
                "quantum_time_s": quantum_result.total_time_s,
                "quantum_build_time_s": quantum_result.circuit_build_time_s,
                "quantum_run_time_s": quantum_result.circuit_run_time_s,
                # Accuracy
                "accuracy_top1": accuracy_top1,
                f"accuracy_top{top_k}": accuracy_topk,
                # Speedup
                "speedup_oracle_queries": speedup,
                "theoretical_speedup": grid_n**2 / quantum_result.n_iterations,
            }

            results["rows"].append(row)

            if verbose:
                print(f"\n  Results:")
                print(f"    Classical oracle calls: {classical_result['oracle_calls']}")
                print(f"    Quantum iterations: {quantum_result.n_iterations}")
                print(f"    Oracle query speedup: {speedup:.1f}×")
                print(f"    Accuracy @1: {accuracy_top1:.2%}")
                print(f"    Accuracy @{top_k}: {accuracy_topk:.2%}")

    # Aggregate results
    results["aggregate"] = aggregate_results(results["rows"], grid_sizes)

    return results


def aggregate_results(rows: list, grid_sizes: list) -> dict:
    """Aggregate results by grid size."""

    agg = {}
    for grid_n in grid_sizes:
        subset = [r for r in rows if r["grid_size_n"] == grid_n]
        if not subset:
            continue

        agg[f"{grid_n}x{grid_n}"] = {
            "total_cells": grid_n**2,
            "n_trials": len(subset),
            "classical_oracle_calls": {
                "mean": float(np.mean([r["classical_oracle_calls"] for r in subset])),
                "std": float(np.std([r["classical_oracle_calls"] for r in subset])),
            },
            "quantum_iterations": {
                "mean": float(np.mean([r["quantum_iterations"] for r in subset])),
                "std": float(np.std([r["quantum_iterations"] for r in subset])),
            },
            "speedup_oracle_queries": {
                "mean": float(np.mean([r["speedup_oracle_queries"] for r in subset])),
                "std": float(np.std([r["speedup_oracle_queries"] for r in subset])),
            },
            "accuracy_top1": {
                "mean": float(np.mean([r["accuracy_top1"] for r in subset])),
                "std": float(np.std([r["accuracy_top1"] for r in subset])),
            },
            "theoretical_speedup": {
                "mean": float(np.mean([r["theoretical_speedup"] for r in subset])),
                "std": float(np.std([r["theoretical_speedup"] for r in subset])),
            },
        }

    return agg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_results(results: dict, output_dir: str):
    """Generate comparison plots."""

    rows = results["rows"]
    if not rows:
        return

    # Extract data
    grid_sizes = sorted(set(r["total_cells"] for r in rows))
    grid_labels = [f"{int(math.sqrt(n))}×{int(math.sqrt(n))}" for n in grid_sizes]

    # Compute means
    classical_calls = [
        np.mean([r["classical_oracle_calls"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]
    quantum_iters = [
        np.mean([r["quantum_iterations"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]
    speedups = [
        np.mean([r["speedup_oracle_queries"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]
    accuracies = [
        np.mean([r["accuracy_top1"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Oracle Queries Comparison
    ax = axes[0, 0]
    x = np.arange(len(grid_sizes))
    width = 0.35
    ax.bar(x - width / 2, classical_calls, width, label="Classical O(N)", color="steelblue")
    ax.bar(x + width / 2, quantum_iters, width, label="Grover O(√N)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels)
    ax.set_ylabel("Number of Oracle Calls / Iterations")
    ax.set_title("Oracle Query Complexity")
    ax.legend()
    ax.set_yscale("log")

    # Plot 2: Speedup Factor
    ax = axes[0, 1]
    ax.bar(x, speedups, color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels)
    ax.set_ylabel("Speedup Factor (×)")
    ax.set_title("Oracle Query Speedup (Classical / Quantum)")
    # Add theoretical line
    theoretical = [math.sqrt(n) * math.pi / 4 for n in grid_sizes]
    ax.plot(x, theoretical, "r--", label="Theoretical √N")
    ax.legend()

    # Plot 3: Accuracy
    ax = axes[1, 0]
    ax.bar(x, accuracies, color="purple", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Grover Search Accuracy @ Top-1")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()

    # Plot 4: Wall-clock Time (Simulator)
    ax = axes[1, 1]
    classical_times = [
        np.mean([r["classical_time_s"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]
    quantum_times = [
        np.mean([r["quantum_time_s"] for r in rows if r["total_cells"] == n])
        for n in grid_sizes
    ]
    ax.bar(x - width / 2, classical_times, width, label="Classical", color="steelblue")
    ax.bar(x + width / 2, quantum_times, width, label="Quantum (sim)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_labels)
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_title("Wall-clock Time (Simulator)")
    ax.legend()
    ax.set_yscale("log")

    fig.suptitle(
        "Quantum Spatial Search vs Classical Brute Force\n"
        "Dengue Hotspot Detection on Spatial Grid\n"
        "Speedup is in ORACLE QUERIES, not wall-clock (simulator overhead dominates)",
        fontsize=11,
    )
    fig.tight_layout()

    # Save
    output_path = os.path.join(output_dir, "spatial_search_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Saved plot: {output_path}")

    # Additional: Scaling plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    ns = grid_sizes
    classical = [n for n in ns]
    grover = [int(math.pi / 4 * math.sqrt(n)) for n in ns]
    measured_speedup = speedups

    ax2.plot(ns, classical, "b-", linewidth=2, marker="o", label="Classical O(N)")
    ax2.plot(ns, grover, "r-", linewidth=2, marker="s", label="Grover O(√N)")
    ax2.plot(ns, measured_speedup, "g--", linewidth=2, marker="^", label="Measured Speedup")
    ax2.plot(ns, [math.sqrt(n) for n in ns], "k--", alpha=0.5, label="Theoretical √N")

    ax2.set_xlabel("Grid Size N (cells)")
    ax2.set_ylabel("Number of Oracle Queries")
    ax2.set_title("Complexity Scaling: Classical vs Grover")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    output_path2 = os.path.join(output_dir, "spatial_search_scaling.png")
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[done] Saved plot: {output_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Quantum Spatial Search Benchmark")
    print(f"  Grid sizes: {args.grid_sizes}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Shots: {args.shots}")
    print("=" * 70)

    # Theoretical summary
    print("\n[THEORY] Theoretical Complexity")
    print("-" * 70)
    print(f"{'Grid':>12} | {'Cells':>8} | {'Classical O(N)':>15} | {'Grover O(√N)':>14} | {'Speedup':>10}")
    print("-" * 70)
    for n in args.grid_sizes:
        cells = n**2
        classical = cells
        grover = int(math.pi / 4 * math.sqrt(cells))
        speedup = classical / grover
        print(f"{n}×{n:>3}={n**2:>6} | {cells:>8} | {classical:>15} | {grover:>14} | {speedup:>9.1f}×")

    # Run benchmark
    print("\n[RUNNING] Experiments...")
    results = run_benchmark(
        grid_sizes=args.grid_sizes,
        seeds=args.seeds,
        top_k=args.top_k,
        shots=args.shots,
        verbose=args.verbose,
    )

    # Save JSON
    output_json = os.path.join(output_dir, "spatial_search_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[demo] Saved: {output_json}")

    # Plot
    plot_results(results, output_dir)

    # Summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    for grid_key, agg in results.get("aggregate", {}).items():
        print(f"\n  Grid {grid_key} ({agg['total_cells']} cells):")
        print(f"    Classical oracle calls: {agg['classical_oracle_calls']['mean']:.0f}")
        print(f"    Quantum iterations: {agg['quantum_iterations']['mean']:.1f} ± {agg['quantum_iterations']['std']:.1f}")
        print(f"    Speedup (oracle queries): {agg['speedup_oracle_queries']['mean']:.1f}×")
        print(f"    Accuracy @1: {agg['accuracy_top1']['mean']:.1%}")
        print(f"    Theoretical speedup: {agg['theoretical_speedup']['mean']:.1f}×")

    print("\n" + "=" * 70)
    print("  HONEST ASSESSMENT")
    print("=" * 70)
    print("""
  The speedup is in ORACLE QUERIES, not wall-clock time.

  On a real quantum computer, √N speedup translates to wall-clock advantage only when:
  1. Gate times << classical oracle evaluation time
  2. Error rates are low enough for deep circuits
  3. Grid size N >> 1000

  On a simulator (this benchmark), wall-clock shows NO quantum advantage
  because the statevector simulation cost dominates.
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
