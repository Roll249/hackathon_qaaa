#!/usr/bin/env python3
"""Benchmark: Real Quantum Circuits vs Classical for Dengue Hotspot Detection.

PURPOSE
-------
This benchmark compares:
1. Real Qiskit quantum circuits (Aer simulator)
2. Aer with realistic noise models (NISQ simulation)
3. Classical brute-force search

The key insight is that Grover's algorithm provides √N speedup in ORACLE QUERIES,
but this doesn't necessarily translate to wall-clock advantage on simulators
due to classical overhead.

THEORETICAL BACKGROUND
----------------------
- Classical search: O(N) oracle evaluations
- Grover quantum: O(√N) oracle evaluations
- Speedup: √N factor (when M << N targets)

REFERENCE
---------
Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search."
Proceedings of 28th ACM STOC.

USAGE
-----
    python benchmarks/benchmark_real_quantum.py \
        --grid-sizes 8 16 32 \
        --top-k 5 \
        --shots 1024 \
        --seeds 42 43 44 \
        --compare-noise \
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

from src.quantum.quantum_real import (
    SpatialGrid,
    RiskMap,
    build_grover_circuit,
    run_real_grover_search,
    classical_spatial_search,
    analyze_circuit,
    compare_real_vs_simulated,
    ExecutionBackend,
    get_available_backends,
)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark: Real Quantum Circuits vs Classical for Dengue Hotspots"
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Grid sizes (n x n). Default: 8 16 32",
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
        help="Number of measurement shots. Default: 1024",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds. Default: 42 43 44",
    )
    parser.add_argument(
        "--compare-noise",
        action="store_true",
        help="Compare ideal vs noisy simulation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Grover iterations (uses optimal if not specified)",
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
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available quantum backends and exit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Risk Map Generation
# ---------------------------------------------------------------------------


def generate_realistic_risk_map(
    nx: int, ny: int, seed: int = 42
) -> RiskMap:
    """Generate a realistic dengue risk map with clustered hotspots.

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

    # Add clustered hotspots (realistic dengue pattern)
    n_hotspots = rng.integers(3, 8)
    for _ in range(n_hotspots):
        cx = rng.uniform(0, nx - 1)
        cy = rng.uniform(0, ny - 1)
        radius = rng.uniform(2, 6)
        intensity = rng.uniform(0.7, 1.0)

        for i in range(nx):
            for j in range(ny):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist < radius:
                    risk[i, j] += intensity * math.exp(-(dist**2) / (2 * (radius / 3) ** 2))

    return RiskMap(grid=grid, values=risk)


# ---------------------------------------------------------------------------
# Benchmark Runners
# ---------------------------------------------------------------------------


def run_circuit_analysis(risk_map: RiskMap, verbose: bool = False) -> dict:
    """Analyze circuit properties for different grid sizes."""
    results = {
        "grid_size": risk_map.grid.total_cells,
        "n_qubits": risk_map.grid.n_qubits,
        "circuits": {},
    }

    for top_k in [1, 5]:
        for n_iters in [1, 2, 3]:
            if n_iters > 5 and risk_map.grid.n_qubits > 6:
                continue

            circuit, targets = build_grover_circuit(
                risk_map,
                top_k=top_k,
                n_iterations=n_iters,
            )

            stats = analyze_circuit(circuit)
            key = f"k{top_k}_i{n_iters}"
            results["circuits"][key] = {
                "num_qubits": stats["num_qubits"],
                "depth": stats["depth"],
                "size": stats["size"],
                "gate_counts": stats["gate_counts"],
                "targets": targets,
            }

            if verbose:
                print(f"  Circuit k={top_k}, i={n_iters}: depth={stats['depth']}, size={stats['size']}")

    return results


def run_quantum_benchmark(
    risk_map: RiskMap,
    top_k: int = 5,
    shots: int = 1024,
    seeds: list = [42],
    use_noise: bool = False,
    n_iterations: int = None,
    verbose: bool = False,
) -> list:
    """Run quantum benchmark across seeds."""
    results = []

    backend_name = "aer_simulator_noise_model" if use_noise else "aer_simulator"

    for seed in seeds:
        if verbose:
            print(f"\n  Running {backend_name} with seed={seed}...")

        result = run_real_grover_search(
            risk_map=risk_map,
            top_k=top_k,
            shots=shots,
            seed=seed,
            backend_name=backend_name,
            use_noise=use_noise,
            n_iterations=n_iterations,
            verbose=False,
        )

        results.append({
            "seed": seed,
            "success_probability": result.success_probability,
            "top_measured": result.top_measured[:top_k],
            "counts": result.counts,
            "circuit_depth": result.n_iterations,
            "execution_time_s": result.circuit_execute_time_s,
            "build_time_s": result.circuit_build_time_s,
        })

    return results


def run_full_benchmark(
    grid_sizes: list,
    top_k: int = 5,
    shots: int = 1024,
    seeds: list = [42],
    compare_noise: bool = True,
    n_iterations: int = None,
    verbose: bool = False,
) -> dict:
    """Run complete benchmark across grid sizes."""
    results = {
        "config": {
            "grid_sizes": grid_sizes,
            "top_k": top_k,
            "shots": shots,
            "seeds": seeds,
            "compare_noise": compare_noise,
        },
        "grid_results": [],
        "theory": [],
    }

    # Theoretical summary
    for n in [s * s for s in grid_sizes]:
        n_qubits = math.ceil(math.log2(n))
        classical = n
        grover = int(math.pi / 4 * math.sqrt(n))
        speedup = classical / grover
        results["theory"].append({
            "cells": n,
            "qubits": n_qubits,
            "classical_oracle_calls": classical,
            "grover_iterations": grover,
            "theoretical_speedup": speedup,
        })

    # Run benchmarks
    for grid_n in grid_sizes:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Grid: {grid_n}×{grid_n} = {grid_n**2} cells")
            print(f"{'=' * 60}")

        # Generate risk map
        risk_map = generate_realistic_risk_map(grid_n, grid_n, seed=seeds[0])
        target_indices = risk_map.get_top_k_indices(top_k)

        grid_result = {
            "grid_n": grid_n,
            "total_cells": grid_n**2,
            "n_qubits": risk_map.grid.n_qubits,
            "target_indices": target_indices,
        }

        # Circuit analysis
        if verbose:
            print("\n[Circuit Analysis]")
        grid_result["circuit_analysis"] = run_circuit_analysis(risk_map, verbose=verbose)

        # Classical baseline
        classical_result = classical_spatial_search(risk_map, top_k=top_k)
        grid_result["classical"] = classical_result

        # Quantum: Ideal
        if verbose:
            print("\n[Ideal Simulator]")
        ideal_results = run_quantum_benchmark(
            risk_map=risk_map,
            top_k=top_k,
            shots=shots,
            seeds=seeds,
            use_noise=False,
            n_iterations=n_iterations,
            verbose=verbose,
        )
        grid_result["quantum_ideal"] = ideal_results

        # Quantum: Noisy
        if compare_noise:
            if verbose:
                print("\n[Noisy Simulator]")
            noisy_results = run_quantum_benchmark(
                risk_map=risk_map,
                top_k=top_k,
                shots=shots,
                seeds=seeds,
                use_noise=True,
                n_iterations=n_iterations,
                verbose=verbose,
            )
            grid_result["quantum_noisy"] = noisy_results

        # Calculate accuracy
        true_top = set(target_indices)

        # Ideal accuracy
        ideal_accuracies = []
        for r in ideal_results:
            measured_top = set(r["top_measured"])
            acc = len(measured_top & true_top) / min(top_k, len(true_top))
            ideal_accuracies.append(acc)

        grid_result["accuracy_ideal"] = {
            "mean": float(np.mean(ideal_accuracies)),
            "std": float(np.std(ideal_accuracies)),
        }

        if compare_noise:
            noisy_accuracies = []
            for r in noisy_results:
                measured_top = set(r["top_measured"])
                acc = len(measured_top & true_top) / min(top_k, len(true_top))
                noisy_accuracies.append(acc)

            grid_result["accuracy_noisy"] = {
                "mean": float(np.mean(noisy_accuracies)),
                "std": float(np.std(noisy_accuracies)),
            }

        # Success probability
        grid_result["success_prob_ideal"] = {
            "mean": float(np.mean([r["success_probability"] for r in ideal_results])),
            "std": float(np.std([r["success_probability"] for r in ideal_results])),
        }

        if compare_noise:
            grid_result["success_prob_noisy"] = {
                "mean": float(np.mean([r["success_probability"] for r in noisy_results])),
                "std": float(np.std([r["success_probability"] for r in noisy_results])),
            }

        results["grid_results"].append(grid_result)

        if verbose:
            print(f"\n  Results Summary:")
            print(f"    Classical top: {classical_result['top_indices'][:5]}")
            print(f"    True targets:  {target_indices[:5]}")
            print(f"    Ideal accuracy: {grid_result['accuracy_ideal']['mean']:.2%}")
            if compare_noise:
                print(f"    Noisy accuracy:  {grid_result['accuracy_noisy']['mean']:.2%}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_benchmark_results(results: dict, output_dir: str):
    """Generate visualization plots for benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    grid_results = results["grid_results"]
    if not grid_results:
        return

    # Extract data
    grid_ns = [r["grid_n"] for r in grid_results]
    total_cells = [r["total_cells"] for r in grid_results]
    n_qubits = [r["n_qubits"] for r in grid_results]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Circuit Depth vs Grid Size
    ax = axes[0, 0]
    depths_ideal = []
    for r in grid_results:
        avg_depth = np.mean([rr["circuit_depth"] for rr in r["quantum_ideal"]])
        depths_ideal.append(avg_depth)
    ax.plot(grid_ns, depths_ideal, "bo-", linewidth=2, markersize=8, label="Circuit Iterations")
    ax.set_xlabel("Grid Size (n × n)")
    ax.set_ylabel("Grover Iterations")
    ax.set_title("Grover Iterations vs Grid Size")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add secondary axis for theoretical
    ax2 = ax.twinx()
    grover_iters = [int(math.pi / 4 * math.sqrt(n)) for n in total_cells]
    ax2.plot(grid_ns, grover_iters, "r--", linewidth=1.5, alpha=0.7, label="Theoretical O(√N)")
    ax2.set_ylabel("Theoretical Grover Iterations", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="lower right")

    # Plot 2: Success Probability (Ideal vs Noisy)
    ax = axes[0, 1]
    x = np.arange(len(grid_ns))
    width = 0.35

    success_ideal = [r["success_prob_ideal"]["mean"] for r in grid_results]
    err_ideal = [r["success_prob_ideal"]["std"] for r in grid_results]

    ax.bar(x - width/2, success_ideal, width, yerr=err_ideal,
           label="Ideal Sim", color="steelblue", capsize=3)

    if "success_prob_noisy" in grid_results[0]:
        success_noisy = [r["success_prob_noisy"]["mean"] for r in grid_results]
        err_noisy = [r["success_prob_noisy"]["std"] for r in grid_results]
        ax.bar(x + width/2, success_noisy, width, yerr=err_noisy,
               label="Noisy Sim", color="coral", capsize=3)

    ax.set_xlabel("Grid Size (n × n)")
    ax.set_ylabel("Success Probability")
    ax.set_title("Grover Success Probability (Top-K Target)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}×{n}" for n in grid_ns])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Accuracy (Ideal vs Noisy)
    ax = axes[1, 0]
    acc_ideal = [r["accuracy_ideal"]["mean"] for r in grid_results]
    err_acc_ideal = [r["accuracy_ideal"]["std"] for r in grid_results]

    ax.bar(x - width/2, acc_ideal, width, yerr=err_acc_ideal,
           label="Ideal Sim", color="green", alpha=0.7, capsize=3)

    if "accuracy_noisy" in grid_results[0]:
        acc_noisy = [r["accuracy_noisy"]["mean"] for r in grid_results]
        err_acc_noisy = [r["accuracy_noisy"]["std"] for r in grid_results]
        ax.bar(x + width/2, acc_noisy, width, yerr=err_acc_noisy,
               label="Noisy Sim", color="orange", alpha=0.7, capsize=3)

    ax.axhline(y=1.0 / grid_results[0]["total_cells"],
               color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Grid Size (n × n)")
    ax.set_ylabel("Top-K Accuracy")
    ax.set_title("Hotspot Detection Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}×{n}" for n in grid_ns])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.2)

    # Plot 4: Execution Time
    ax = axes[1, 1]
    time_ideal = [np.mean([rr["execution_time_s"] for rr in r["quantum_ideal"]]) for r in grid_results]
    ax.plot(grid_ns, time_ideal, "bo-", linewidth=2, markersize=8, label="Ideal Sim")

    if "quantum_noisy" in grid_results[0]:
        time_noisy = [np.mean([rr["execution_time_s"] for rr in r["quantum_noisy"]]) for r in grid_results]
        ax.plot(grid_ns, time_noisy, "rs-", linewidth=2, markersize=8, label="Noisy Sim")

    ax.set_xlabel("Grid Size (n × n)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Circuit Execution Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle(
        "Real Quantum Circuits for Dengue Hotspot Detection\n"
        "Grover Search with Qiskit Aer Simulator",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    # Save
    output_path = os.path.join(output_dir, "real_quantum_benchmark.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")

    # Plot 2: Theoretical Scaling
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    ns = total_cells
    classical = ns  # O(N)
    grover = [int(math.pi / 4 * math.sqrt(n)) for n in ns]  # O(√N)

    ax2.plot(ns, classical, "b-", linewidth=2, marker="o", label="Classical O(N)")
    ax2.plot(ns, grover, "r-", linewidth=2, marker="s", label="Grover O(√N)")
    ax2.plot(ns, [math.sqrt(n) for n in ns], "k--", alpha=0.5, label="Theoretical √N")

    ax2.set_xlabel("Number of Grid Cells (N)", fontsize=12)
    ax2.set_ylabel("Oracle Evaluations", fontsize=12)
    ax2.set_title("Theoretical Complexity: Classical vs Grover Search", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Add speedup annotation
    speedups = [n / g for n, g in zip(ns, grover)]
    for i, (n, s) in enumerate(zip(ns, speedups)):
        ax2.annotate(f"{s:.1f}×", (n, grover[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="red")

    output_path2 = os.path.join(output_dir, "theoretical_scaling.png")
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] {output_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # List backends if requested
    if args.list_backends:
        print("Available Quantum Backends:")
        print("-" * 60)
        backends = get_available_backends()
        for backend in backends:
            hw_tag = "[HARDWARE]" if backend.is_real_hardware else "[SIMULATOR]"
            print(f"  {backend.name} {hw_tag}")
            print(f"    Description: {backend.description}")
            print(f"    Min qubits: {backend.min_qubits}, Max shots: {backend.max_shots}")
            print()
        return 0

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Real Quantum Circuit Benchmark")
    print("  Grover Search for Dengue Hotspot Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid sizes: {args.grid_sizes}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Shots: {args.shots}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Compare noise: {args.compare_noise}")
    print(f"  Output: {output_dir}")

    # Theoretical summary
    print("\n" + "-" * 70)
    print("  THEORETICAL COMPLEXITY")
    print("-" * 70)
    print(f"{'Grid':>10} | {'Cells':>8} | {'Qubits':>7} | {'Classical':>12} | {'Grover':>10} | {'Speedup':>10}")
    print("-" * 70)
    for n in args.grid_sizes:
        cells = n**2
        qubits = math.ceil(math.log2(cells))
        classical = cells
        grover = int(math.pi / 4 * math.sqrt(cells))
        speedup = classical / grover
        print(f"{n}×{n:<5} | {cells:>8} | {qubits:>7} | {classical:>12} | {grover:>10} | {speedup:>9.1f}×")

    # Run benchmark
    print("\n" + "-" * 70)
    print("  RUNNING BENCHMARK")
    print("-" * 70)

    results = run_full_benchmark(
        grid_sizes=args.grid_sizes,
        top_k=args.top_k,
        shots=args.shots,
        seeds=args.seeds,
        compare_noise=args.compare_noise,
        n_iterations=args.iterations,
        verbose=args.verbose,
    )

    # Save JSON
    output_json = os.path.join(output_dir, "real_quantum_benchmark_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[Saved] {output_json}")

    # Generate plots
    print("\n[Generating plots...]")
    plot_benchmark_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    for r in results["grid_results"]:
        print(f"\n  Grid {r['grid_n']}×{r['grid_n']} ({r['total_cells']} cells, {r['n_qubits']} qubits):")
        print(f"    Classical top indices: {r['classical']['top_indices'][:5]}")
        print(f"    Target indices:        {r['target_indices'][:5]}")
        print(f"    Ideal accuracy:        {r['accuracy_ideal']['mean']:.2%} ± {r['accuracy_ideal']['std']:.2%}")
        if "accuracy_noisy" in r:
            print(f"    Noisy accuracy:        {r['accuracy_noisy']['mean']:.2%} ± {r['accuracy_noisy']['std']:.2%}")
        print(f"    Ideal success prob:    {r['success_prob_ideal']['mean']:.3f}")
        if "success_prob_noisy" in r:
            print(f"    Noisy success prob:    {r['success_prob_noisy']['mean']:.3f}")

    print("\n" + "=" * 70)
    print("  HONEST ASSESSMENT")
    print("=" * 70)
    print("""
  This benchmark uses REAL quantum circuits (Qiskit) with:
  1. Actual gate sequences for oracle and diffusion
  2. Shot-based measurement (not statevector)
  3. Realistic noise models (for NISQ comparison)

  Key observations:
  1. Grover provides √N speedup in ORACLE QUERIES
  2. On simulators, wall-clock time is dominated by classical overhead
  3. On real hardware, √N speedup requires:
     - Fast quantum gates (<< classical oracle time)
     - Low error rates for deep circuits
     - Large problem sizes (N >> 1000)

  The success probability degrades with noise, but even noisy
  circuits can achieve reasonable accuracy for small grids.
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
