#!/usr/bin/env python3
"""
Benchmark v1 (4 qubits, 2 layers) vs v2 (8 qubits, 3 layers) for QRC.

This script compares the performance of Quantum Reservoir Computing v1 and v2
on synthetic dengue-like time series data.

Key differences:
- v1: 4 qubits, 2 layers, 10 internal units, leaky=0.3
- v2: 8 qubits, 3 layers, 30 internal units, leaky=0.2, multi-horizon

Author: QC4SG 2026
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum.quantum_reservoir import qrc_predict, QuantumReservoir, QRCOutputLayer
from quantum.quantum_reservoir_v2 import (
    ImprovedQRCConfig,
    ImprovedQuantumReservoir,
    direct_multihorizon_predict,
    build_enhanced_features,
    normalize_features,
    QRCOutputLayer as QRCOutputLayerV2,
)


# =============================================================================
# Synthetic Dengue Data Generation
# =============================================================================


def generate_dengue_time_series(
    n_weeks: int = 200,
    seed: int = 42,
    outbreak_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate synthetic dengue-like epidemic time series.

    The time series includes:
    - Seasonal pattern (annual cycle)
    - Semi-annual pattern (secondary peak)
    - Long-term trend
    - Outbreak spikes
    - Random noise

    Args:
        n_weeks: Number of weeks to generate
        seed: Random seed
        outbreak_scale: Scale of outbreak spikes

    Returns:
        Tuple of (cases array, climate data dict)
    """
    np.random.seed(seed)
    t = np.arange(n_weeks)

    # Seasonal component (annual, peak in rainy season ~October)
    seasonal = 50 * np.sin(2 * np.pi * (t - 20) / 52)

    # Semi-annual component (smaller secondary peak)
    semi_annual = 20 * np.sin(4 * np.pi * t / 52)

    # Multi-year trend (slight increase over time)
    trend = 0.05 * t

    # Outbreak spikes (random epidemics)
    outbreak = np.zeros(n_weeks)
    if outbreak_scale > 0:
        n_outbreaks = max(1, n_weeks // 50)
        for _ in range(n_outbreaks):
            start = np.random.randint(10, n_weeks - 20)
            duration = np.random.randint(4, 12)
            magnitude = np.random.uniform(50, 150) * outbreak_scale
            for i in range(duration):
                if start + i < n_weeks:
                    outbreak[start + i] += magnitude * np.exp(-i / 3)

    # Random noise
    noise = 8 * np.random.randn(n_weeks)

    # Combine components
    cases = 80 + seasonal + semi_annual + trend + outbreak + noise
    cases = np.maximum(cases, 0)  # Ensure non-negative

    # Generate synthetic climate data
    climate_data = {
        'temperature': 28 + 4 * np.sin(2 * np.pi * t / 52 + 0.5) + 2 * np.random.randn(n_weeks),
        'humidity': 75 + 12 * np.sin(2 * np.pi * t / 52 + 1.2) + 5 * np.random.randn(n_weeks),
        'rainfall': np.maximum(5 + 3 * np.random.randn(n_weeks) + 10 * np.sin(2 * np.pi * t / 52), 0),
    }

    return cases, climate_data


def generate_multiple_series(
    n_series: int = 5,
    n_weeks: int = 200,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[np.ndarray, Dict[str, np.ndarray], str]]:
    """Generate multiple synthetic time series for robustness testing.

    Args:
        n_series: Number of series to generate
        n_weeks: Length of each series
        seeds: Random seeds (auto-generated if None)

    Returns:
        List of (cases, climate_data, name) tuples
    """
    if seeds is None:
        seeds = [42 + i * 17 for i in range(n_series)]

    series_list = []
    for i, seed in enumerate(seeds[:n_series]):
        cases, climate = generate_dengue_time_series(
            n_weeks=n_weeks,
            seed=seed,
            outbreak_scale=0.8 + 0.4 * (i % 3),
        )
        series_list.append((cases, climate, f"Synthetic_{i+1}"))
    
    return series_list


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_v1(
    cases: np.ndarray,
    seed: int = 42,
    n_qubits: int = 4,
    n_internal: int = 10,
) -> Dict:
    """Run QRC v1 benchmark.

    Args:
        cases: Time series data
        seed: Random seed
        n_qubits: Number of qubits
        n_internal: Internal dimension

    Returns:
        Results dictionary
    """
    t_start = time.time()
    
    result = qrc_predict(
        timeseries=cases,
        n_qubits=n_qubits,
        n_internal=n_internal,
        seed=seed,
        verbose=False,
    )
    
    train_time = result.train_time_s
    predict_time = result.predict_time_s
    total_time = time.time() - t_start

    return {
        "mse": result.mse,
        "nmse": result.nmse,
        "n_params": result.n_params,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,
        "n_qubits": n_qubits,
        "n_internal": n_internal,
        "seed": seed,
    }


def benchmark_v2(
    cases: np.ndarray,
    climate_data: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 42,
    n_qubits: int = 8,
    n_layers: int = 3,
    n_internal: int = 30,
    max_horizon: int = 4,
) -> Dict:
    """Run QRC v2 (Improved) benchmark.

    Args:
        cases: Time series data
        climate_data: Optional climate features
        seed: Random seed
        n_qubits: Number of qubits
        n_layers: Number of circuit layers
        n_internal: Internal dimension
        max_horizon: Maximum prediction horizon

    Returns:
        Results dictionary
    """
    config = ImprovedQRCConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_internal=n_internal,
        max_horizon=max_horizon,
        leaky=0.2,
        use_climate_features=climate_data is not None,
        seed=seed,
    )

    t_start = time.time()
    
    result = direct_multihorizon_predict(
        cases=cases,
        climate_data=climate_data,
        config=config,
        train_fraction=0.7,
    )
    
    train_time = result.train_time_s
    predict_time = result.predict_time_s
    total_time = time.time() - t_start

    return {
        "mse_by_horizon": result.mse_by_horizon,
        "nmse_by_horizon": result.nmse_by_horizon,
        "mae_by_horizon": result.mae_by_horizon,
        "n_params": result.n_params,
        "n_features": result.n_features,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_internal": n_internal,
        "max_horizon": max_horizon,
        "seed": seed,
    }


def run_comparison_benchmark(
    cases_list: List[Tuple[np.ndarray, Dict[str, np.ndarray], str]],
    seeds: List[int] = [42, 43, 44],
    output_dir: str = "output_result/q_stpp_v18",
) -> Dict:
    """Run full comparison benchmark across multiple series and seeds.

    Args:
        cases_list: List of (cases, climate_data, name) tuples
        seeds: Random seeds for averaging
        output_dir: Output directory

    Returns:
        Aggregated results
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "config": {
            "v1": {"n_qubits": 4, "n_layers": 2, "n_internal": 10, "leaky": 0.3},
            "v2": {"n_qubits": 8, "n_layers": 3, "n_internal": 30, "leaky": 0.2},
            "seeds": seeds,
            "n_series": len(cases_list),
        },
        "v1_results": [],
        "v2_results": [],
        "per_series_comparison": {},
    }

    print("=" * 70)
    print("QRC v1 vs v2 Benchmark")
    print("=" * 70)

    for series_idx, (cases, climate_data, series_name) in enumerate(cases_list):
        print(f"\n--- Series {series_idx + 1}: {series_name} ---")
        print(f"  Data length: {len(cases)} weeks")
        print(f"  Cases range: [{cases.min():.1f}, {cases.max():.1f}]")

        series_v1 = []
        series_v2 = []

        for seed in seeds:
            # V1 Benchmark
            print(f"  Running v1 (seed={seed})...", end=" ", flush=True)
            v1_res = benchmark_v1(cases, seed=seed)
            series_v1.append(v1_res)
            results["v1_results"].append({
                "series": series_name,
                "series_idx": series_idx,
                **v1_res,
            })
            print(f"MSE={v1_res['mse']:.4f}, time={v1_res['total_time_s']:.2f}s")

            # V2 Benchmark
            print(f"  Running v2 (seed={seed})...", end=" ", flush=True)
            v2_res = benchmark_v2(cases, climate_data=climate_data, seed=seed)
            series_v2.append(v2_res)
            results["v2_results"].append({
                "series": series_name,
                "series_idx": series_idx,
                **v2_res,
            })
            print(f"MSE_h1={v2_res['mse_by_horizon'][1]:.4f}, time={v2_res['total_time_s']:.2f}s")

        # Per-series summary
        v1_mses = [r["mse"] for r in series_v1]
        v2_mses_h1 = [r["mse_by_horizon"][1] for r in series_v2]

        results["per_series_comparison"][series_name] = {
            "v1_mse_mean": float(np.mean(v1_mses)),
            "v1_mse_std": float(np.std(v1_mses)),
            "v2_mse_h1_mean": float(np.mean(v2_mses_h1)),
            "v2_mse_h1_std": float(np.std(v2_mses_h1)),
            "improvement_pct": float(100 * (1 - np.mean(v2_mses_h1) / max(np.mean(v1_mses), 1e-8))),
        }

    # Overall summary
    all_v1_mse = [r["mse"] for r in results["v1_results"]]
    all_v2_mse_h1 = [r["mse_by_horizon"][1] for r in results["v2_results"]]

    results["summary"] = {
        "v1_mse_overall_mean": float(np.mean(all_v1_mse)),
        "v1_mse_overall_std": float(np.std(all_v1_mse)),
        "v2_mse_h1_overall_mean": float(np.mean(all_v2_mse_h1)),
        "v2_mse_h1_overall_std": float(np.std(all_v2_mse_h1)),
        "overall_improvement_pct": float(
            100 * (1 - np.mean(all_v2_mse_h1) / max(np.mean(all_v1_mse), 1e-8))
        ),
        "v1_avg_train_time": float(np.mean([r["train_time_s"] for r in results["v1_results"]])),
        "v2_avg_train_time": float(np.mean([r["train_time_s"] for r in results["v2_results"]])),
        "v1_avg_params": int(np.mean([r["n_params"] for r in results["v1_results"]])),
        "v2_avg_params": int(np.mean([r["n_params"] for r in results["v2_results"]])),
    }

    # Multi-horizon summary for v2
    for h in [1, 2, 3, 4]:
        h_mses = [r["mse_by_horizon"][h] for r in results["v2_results"] if h in r["mse_by_horizon"]]
        if h_mses:
            results["summary"][f"v2_mse_h{h}_overall_mean"] = float(np.mean(h_mses))
            results["summary"][f"v2_mse_h{h}_overall_std"] = float(np.std(h_mses))

    # Save results
    output_file = os.path.join(output_dir, "v1_vs_v2_benchmark.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[Saved] {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"V1 Overall MSE: {results['summary']['v1_mse_overall_mean']:.4f} ± {results['summary']['v1_mse_overall_std']:.4f}")
    print(f"V2 Overall MSE (h=1): {results['summary']['v2_mse_h1_overall_mean']:.4f} ± {results['summary']['v2_mse_h1_overall_std']:.4f}")
    print(f"Improvement: {results['summary']['overall_improvement_pct']:.1f}%")
    print(f"V1 avg params: {results['summary']['v1_avg_params']}")
    print(f"V2 avg params: {results['summary']['v2_avg_params']}")

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_v1_vs_v2_comparison(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate comparison plots for v1 vs v2.

    Args:
        results: Benchmark results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QRC v1 vs v2 Benchmark Results', fontsize=16, fontweight='bold')

    # Plot 1: Bar chart comparison (MSE)
    ax1 = axes[0, 0]
    series_names = list(results["per_series_comparison"].keys())
    v1_mses = [results["per_series_comparison"][s]["v1_mse_mean"] for s in series_names]
    v2_mses = [results["per_series_comparison"][s]["v2_mse_h1_mean"] for s in series_names]

    x = np.arange(len(series_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, v1_mses, width, label='V1 (4 qubits)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, v2_mses, width, label='V2 (8 qubits)', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Synthetic Data Series')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE Comparison by Series')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('Synthetic_', 'S') for s in series_names], rotation=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Plot 2: Multi-horizon performance (V2 only)
    ax2 = axes[0, 1]
    horizons = [1, 2, 3, 4]
    h_mses = [results["summary"].get(f"v2_mse_h{h}_overall_mean", 0) for h in horizons]
    h_stds = [results["summary"].get(f"v2_mse_h{h}_overall_std", 0) for h in horizons]

    ax2.bar(horizons, h_mses, yerr=h_stds, capsize=5, color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Prediction Horizon (weeks)')
    ax2.set_ylabel('MSE')
    ax2.set_title('V2 Multi-Horizon Performance')
    ax2.set_xticks(horizons)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Training time comparison
    ax3 = axes[1, 0]
    categories = ['V1', 'V2']
    times = [results['summary']['v1_avg_train_time'], results['summary']['v2_avg_train_time']]
    colors = ['#3498db', '#e74c3c']

    bars = ax3.bar(categories, times, color=colors, alpha=0.8)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Average Training Time')
    ax3.grid(axis='y', alpha=0.3)

    for bar, time_val in zip(bars, times):
        ax3.annotate(f'{time_val:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Plot 4: Parameter count comparison
    ax4 = axes[1, 1]
    params = [results['summary']['v1_avg_params'], results['summary']['v2_avg_params']]
    bars = ax4.bar(categories, params, color=colors, alpha=0.8)
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Complexity')
    ax4.grid(axis='y', alpha=0.3)

    for bar, param in zip(bars, params):
        ax4.annotate(f'{param}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "v1_vs_v2_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_horizon_degradation(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Plot how performance degrades with longer prediction horizons.

    Args:
        results: Benchmark results
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = [1, 2, 3, 4]
    h_mses = [results["summary"].get(f"v2_mse_h{h}_overall_mean", 0) for h in horizons]
    h_stds = [results["summary"].get(f"v2_mse_h{h}_overall_std", 0) for h in horizons]

    ax.errorbar(horizons, h_mses, yerr=h_stds, fmt='o-', capsize=5, 
                capthick=2, linewidth=2, markersize=10, color='#e74c3c')
    ax.fill_between(horizons, 
                    [m - s for m, s in zip(h_mses, h_stds)],
                    [m + s for m, s in zip(h_mses, h_stds)],
                    alpha=0.2, color='#e74c3c')

    ax.set_xlabel('Prediction Horizon (weeks ahead)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('V2 Performance Degradation Over Multi-Horizon Prediction', fontsize=14, fontweight='bold')
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)

    # Add annotation for degradation rate
    if h_mses[0] > 0:
        degradation = (h_mses[-1] - h_mses[0]) / h_mses[0] * 100
        ax.annotate(f'Total degradation: {degradation:.1f}%',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(output_dir, "horizon_degradation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the full benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QRC v1 vs v2 Benchmark')
    parser.add_argument('--output-dir', type=str, default='output_result/q_stpp_v18',
                       help='Output directory')
    parser.add_argument('--n-series', type=int, default=5,
                       help='Number of synthetic series to test')
    parser.add_argument('--n-weeks', type=int, default=200,
                       help='Length of each series (weeks)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Random seeds')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    args = parser.parse_args()

    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)

    # Generate synthetic data
    print("Generating synthetic dengue time series...")
    series_list = generate_multiple_series(
        n_series=args.n_series,
        n_weeks=args.n_weeks,
    )

    # Run benchmark
    results = run_comparison_benchmark(
        cases_list=series_list,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )

    # Generate plots
    if not args.skip_plots:
        print("\nGenerating visualizations...")
        plot_v1_vs_v2_comparison(results, args.output_dir)
        plot_horizon_degradation(results, args.output_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
