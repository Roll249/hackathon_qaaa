#!/usr/bin/env python3
"""
Hyperparameter Tuning for QRC v2.

This script performs grid search over hyperparameter combinations to find
optimal QRC v2 settings for dengue forecasting.

Parameters tuned:
- n_qubits: [4, 8, 12]
- n_layers: [2, 3, 4]
- n_internal: [20, 30, 50]
- leaky: [0.1, 0.2, 0.3]
- spectral_radius: [0.8, 0.9, 0.95, 1.0]

Uses cross-validation to evaluate each configuration.

Author: QC4SG 2026
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum.quantum_reservoir_v2 import (
    ImprovedQRCConfig,
    direct_multihorizon_predict,
)


# =============================================================================
# Configuration Space
# =============================================================================


@dataclass
class TuningSpace:
    """Hyperparameter search space definition."""
    n_qubits: List[int] = field(default_factory=lambda: [4, 8, 12])
    n_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    n_internal: List[int] = field(default_factory=lambda: [20, 30, 50])
    leaky: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    spectral_radius: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95, 1.0])

    def get_grid_size(self) -> int:
        """Get total number of configurations."""
        return (
            len(self.n_qubits) *
            len(self.n_layers) *
            len(self.n_internal) *
            len(self.leaky) *
            len(self.spectral_radius)
        )

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all configuration combinations."""
        configs = []
        for nq, nl, ni, lk, sr in itertools.product(
            self.n_qubits,
            self.n_layers,
            self.n_internal,
            self.leaky,
            self.spectral_radius,
        ):
            configs.append({
                "n_qubits": nq,
                "n_layers": nl,
                "n_internal": ni,
                "leaky": lk,
                "spectral_radius": sr,
            })
        return configs


@dataclass
class TuningResult:
    """Result of a single tuning experiment."""
    config: Dict[str, Any]
    mse_h1: float
    mse_h2: float
    mse_h4: float
    nmse_h1: float
    train_time_s: float
    n_params: int
    fold_results: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Data Generation
# =============================================================================


def generate_tuning_data(
    n_series: int = 3,
    n_weeks: int = 200,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Generate synthetic data for hyperparameter tuning.

    Args:
        n_series: Number of synthetic series
        n_weeks: Length of each series
        seeds: Random seeds

    Returns:
        List of (cases, climate_data) tuples
    """
    if seeds is None:
        seeds = [42, 123, 456]

    data_list = []
    for seed in seeds[:n_series]:
        np.random.seed(seed)
        t = np.arange(n_weeks)

        # Complex dengue-like pattern
        seasonal = 50 * np.sin(2 * np.pi * (t - 20) / 52)
        semi_annual = 20 * np.sin(4 * np.pi * t / 52)
        trend = 0.05 * t
        outbreak = np.zeros(n_weeks)
        for _ in range(n_weeks // 50):
            start = np.random.randint(10, n_weeks - 20)
            for i in range(np.random.randint(4, 12)):
                if start + i < n_weeks:
                    outbreak[start + i] += np.random.uniform(50, 150) * np.exp(-i / 3)
        noise = 8 * np.random.randn(n_weeks)

        cases = 80 + seasonal + semi_annual + trend + outbreak + noise
        cases = np.maximum(cases, 0)

        climate_data = {
            'temperature': 28 + 4 * np.sin(2 * np.pi * t / 52 + 0.5) + 2 * np.random.randn(n_weeks),
            'humidity': 75 + 12 * np.sin(2 * np.pi * t / 52 + 1.2) + 5 * np.random.randn(n_weeks),
            'rainfall': np.maximum(5 + 3 * np.random.randn(n_weeks) + 10 * np.sin(2 * np.pi * t / 52), 0),
        }

        data_list.append((cases, climate_data))

    return data_list


# =============================================================================
# Cross-Validation
# =============================================================================


def cross_validate_config(
    config_dict: Dict[str, Any],
    data_list: List[Tuple[np.ndarray, Dict[str, np.ndarray]]],
    n_folds: int = 3,
    max_horizon: int = 4,
    seed: int = 42,
) -> TuningResult:
    """Evaluate a configuration using cross-validation.

    Args:
        config_dict: QRC configuration as dict
        data_list: List of (cases, climate_data) tuples
        n_folds: Number of CV folds
        max_horizon: Maximum horizon to predict
        seed: Random seed

    Returns:
        TuningResult with averaged metrics
    """
    fold_mses_h1 = []
    fold_mses_h2 = []
    fold_mses_h4 = []
    fold_nmses_h1 = []
    fold_times = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold_seed = seed + fold_idx * 1000
        fold_mses = {1: [], 2: [], 4: []}

        for cases, climate_data in data_list:
            try:
                config = ImprovedQRCConfig(
                    n_qubits=config_dict["n_qubits"],
                    n_layers=config_dict["n_layers"],
                    n_internal=config_dict["n_internal"],
                    leaky=config_dict["leaky"],
                    spectral_radius=config_dict["spectral_radius"],
                    max_horizon=max_horizon,
                    use_climate_features=True,
                    seed=fold_seed,
                )

                result = direct_multihorizon_predict(
                    cases=cases,
                    climate_data=climate_data,
                    config=config,
                    train_fraction=0.7,
                )

                for h in [1, 2, 4]:
                    if h in result.mse_by_horizon:
                        fold_mses[h].append(result.mse_by_horizon[h])

                fold_times.append(result.train_time_s)

            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                continue

        # Average over all series for this fold
        if fold_mses[1]:
            fold_mses_h1.append(np.mean(fold_mses[1]))
            fold_mses_h2.append(np.mean(fold_mses[2]) if fold_mses[2] else np.nan)
            fold_mses_h4.append(np.mean(fold_mses[4]) if fold_mses[4] else np.nan)
            # Skip NMSE calculation for now (simplified)
            fold_nmses_h1.append(0.0)

            fold_results.append({
                "fold": fold_idx,
                "mse_h1": np.mean(fold_mses[1]),
                "mse_h2": np.mean(fold_mses[2]) if fold_mses[2] else np.nan,
                "mse_h4": np.mean(fold_mses[4]) if fold_mses[4] else np.nan,
            })

    return TuningResult(
        config=config_dict,
        mse_h1=float(np.nanmean(fold_mses_h1)) if fold_mses_h1 else float('inf'),
        mse_h2=float(np.nanmean(fold_mses_h2)) if fold_mses_h2 else float('inf'),
        mse_h4=float(np.nanmean(fold_mses_h4)) if fold_mses_h4 else float('inf'),
        nmse_h1=float(np.nanmean(fold_nmses_h1)) if fold_nmses_h1 else float('inf'),
        train_time_s=float(np.mean(fold_times)) if fold_times else 0,
        n_params=config_dict["n_internal"] ** 2 + config_dict["n_internal"] * config_dict["n_qubits"],
        fold_results=fold_results,
    )


# =============================================================================
# Grid Search
# =============================================================================


def run_grid_search(
    tuning_space: TuningSpace,
    data_list: List[Tuple[np.ndarray, Dict[str, np.ndarray]]],
    output_dir: str = "output_result/q_stpp_v18",
    n_folds: int = 3,
    max_horizon: int = 4,
    seed: int = 42,
    max_configs: Optional[int] = None,
) -> Dict:
    """Run grid search over hyperparameter space.

    Args:
        tuning_space: TuningSpace definition
        data_list: Data for evaluation
        output_dir: Output directory
        n_folds: CV folds
        max_horizon: Maximum horizon
        seed: Random seed
        max_configs: Limit number of configs (for quick testing)

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = tuning_space.generate_configs()
    if max_configs:
        configs = configs[:max_configs]

    total_configs = len(configs)
    print(f"Running grid search over {total_configs} configurations...")

    results = {
        "config_space": {
            "n_qubits": tuning_space.n_qubits,
            "n_layers": tuning_space.n_layers,
            "n_internal": tuning_space.n_internal,
            "leaky": tuning_space.leaky,
            "spectral_radius": tuning_space.spectral_radius,
        },
        "settings": {
            "n_folds": n_folds,
            "max_horizon": max_horizon,
            "n_series": len(data_list),
            "seed": seed,
        },
        "results": [],
        "best_config": None,
        "summary": {},
    }

    start_time = time.time()

    for i, config_dict in enumerate(configs):
        print(f"[{i+1}/{total_configs}] {config_dict}...", end=" ", flush=True)

        t0 = time.time()
        result = cross_validate_config(
            config_dict,
            data_list,
            n_folds=n_folds,
            max_horizon=max_horizon,
            seed=seed,
        )
        elapsed = time.time() - t0

        results["results"].append({
            **config_dict,
            "mse_h1": result.mse_h1,
            "mse_h2": result.mse_h2,
            "mse_h4": result.mse_h4,
            "nmse_h1": result.nmse_h1,
            "train_time_s": result.train_time_s,
            "n_params": result.n_params,
            "elapsed_s": elapsed,
        })

        print(f"MSE_h1={result.mse_h1:.2f}, time={elapsed:.1f}s")

    # Find best configuration
    valid_results = [r for r in results["results"] if not np.isinf(r["mse_h1"])]
    if valid_results:
        best = min(valid_results, key=lambda x: x["mse_h1"])
        results["best_config"] = best

        # Summary statistics
        all_mses = [r["mse_h1"] for r in valid_results]
        results["summary"] = {
            "total_configs": total_configs,
            "valid_configs": len(valid_results),
            "best_mse_h1": best["mse_h1"],
            "best_config": {k: v for k, v in best.items() if k not in ["mse_h1", "mse_h2", "mse_h4", "nmse_h1", "train_time_s", "n_params", "elapsed_s"]},
            "mse_h1_mean": float(np.mean(all_mses)),
            "mse_h1_std": float(np.std(all_mses)),
            "mse_h1_min": float(np.min(all_mses)),
            "mse_h1_max": float(np.max(all_mses)),
            "total_time_s": time.time() - start_time,
        }

    # Save results
    output_file = os.path.join(output_dir, "hp_tuning_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[Saved] {output_file}")

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_tuning_curves(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate hyperparameter tuning curves.

    Args:
        results: Tuning results
        output_dir: Output directory
    """
    if not results.get("results"):
        print("No tuning results to plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results["results"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QRC v2 Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

    params = ['n_qubits', 'n_layers', 'n_internal', 'leaky', 'spectral_radius']
    titles = ['Number of Qubits', 'Number of Layers', 'Internal Dimension',
              'Leakage Rate', 'Spectral Radius']

    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx // 3, idx % 3]

        # Group by parameter and compute mean MSE
        grouped = df_results.groupby(param)['mse_h1'].agg(['mean', 'std']).reset_index()

        ax.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'],
                    fmt='o-', capsize=5, linewidth=2, markersize=8)

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('MSE (h=1)', fontsize=11)
        ax.set_title(f'MSE vs {title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 6: Best config comparison
    ax = axes[1, 2]
    if results.get("best_config"):
        best = results["best_config"]
        params_to_show = ['n_qubits', 'n_layers', 'n_internal', 'leaky', 'spectral_radius']
        best_values = [best.get(p, 'N/A') for p in params_to_show]

        # Create a simple table-like visualization
        ax.axis('off')
        ax.set_title('Best Configuration', fontsize=12, fontweight='bold')

        table_data = [[p.upper(), str(v)] for p, v in zip(params_to_show, best_values)]
        table_data.append(['MSE (h=1)', f'{best["mse_h1"]:.4f}'])
        table_data.append(['Runtime', f'{best["elapsed_s"]:.1f}s'])

        table = ax.table(
            cellText=table_data,
            colLabels=['Parameter', 'Value'],
            loc='center',
            cellLoc='left',
            colWidths=[0.4, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "hp_tuning_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_tuning_heatmap(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate heatmaps for hyperparameter interactions.

    Args:
        results: Tuning results
        output_dir: Output directory
    """
    if not results.get("results"):
        return

    df_results = pd.DataFrame(results["results"])

    # Create heatmaps for key parameter pairs
    param_pairs = [
        ('n_qubits', 'n_layers'),
        ('n_qubits', 'n_internal'),
        ('n_layers', 'n_internal'),
        ('leaky', 'spectral_radius'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('QRC v2 Hyperparameter Interaction Heatmaps', fontsize=14, fontweight='bold')

    for idx, (p1, p2) in enumerate(param_pairs):
        ax = axes[idx // 2, idx % 2]

        # Create pivot table
        pivot = df_results.pivot_table(
            values='mse_h1',
            index=p1,
            columns=p2,
            aggfunc='mean'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'MSE'}
        )
        ax.set_title(f'{p1} vs {p2}', fontsize=12, fontweight='bold')
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "hp_tuning_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_training_convergence(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate training convergence plot.

    Args:
        results: Tuning results
        output_dir: Output directory
    """
    if not results.get("results"):
        return

    df_results = pd.DataFrame(results["results"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: MSE vs Training Time
    ax1 = axes[0]
    ax1.scatter(df_results['train_time_s'], df_results['mse_h1'],
               alpha=0.6, c=df_results['n_qubits'], cmap='viridis')
    ax1.set_xlabel('Training Time (s)', fontsize=11)
    ax1.set_ylabel('MSE (h=1)', fontsize=11)
    ax1.set_title('MSE vs Training Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    sc = plt.scatter([], [], c=[], cmap='viridis')
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('n_qubits')

    # Plot 2: MSE vs Parameters
    ax2 = axes[1]
    ax2.scatter(df_results['n_params'], df_results['mse_h1'],
               alpha=0.6, c=df_results['n_layers'], cmap='plasma')
    ax2.set_xlabel('Number of Parameters', fontsize=11)
    ax2.set_ylabel('MSE (h=1)', fontsize=11)
    ax2.set_title('MSE vs Model Complexity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight best
    if results.get("best_config"):
        best = results["best_config"]
        ax2.scatter([best['n_params']], [best['mse_h1']],
                   s=200, c='red', marker='*', label='Best', zorder=5)
        ax2.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_convergence.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='QRC v2 Hyperparameter Tuning')
    parser.add_argument('--output-dir', type=str, default='output_result/q_stpp_v18',
                       help='Output directory')
    parser.add_argument('--n-series', type=int, default=3,
                       help='Number of synthetic series')
    parser.add_argument('--n-weeks', type=int, default=200,
                       help='Length of each series')
    parser.add_argument('--n-folds', type=int, default=3,
                       help='Cross-validation folds')
    parser.add_argument('--max-horizon', type=int, default=4,
                       help='Maximum prediction horizon')
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Limit configs for quick testing')
    parser.add_argument('--reduced-space', action='store_true',
                       help='Use reduced parameter space for faster tuning')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    args = parser.parse_args()

    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)

    # Define tuning space
    if args.reduced_space:
        tuning_space = TuningSpace(
            n_qubits=[4, 8],
            n_layers=[2, 3],
            n_internal=[20, 30],
            leaky=[0.2, 0.3],
            spectral_radius=[0.9, 0.95],
        )
    else:
        tuning_space = TuningSpace()

    print("=" * 70)
    print("QRC v2 Hyperparameter Tuning")
    print("=" * 70)
    print(f"Grid size: {tuning_space.get_grid_size()} configurations")
    print(f"Using reduced space: {args.reduced_space}")

    # Generate data
    print("\nGenerating synthetic data...")
    data_list = generate_tuning_data(
        n_series=args.n_series,
        n_weeks=args.n_weeks,
    )
    print(f"Generated {len(data_list)} series")

    # Run grid search
    print("\n" + "=" * 70)
    print("Running Grid Search")
    print("=" * 70)
    results = run_grid_search(
        tuning_space=tuning_space,
        data_list=data_list,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        max_horizon=args.max_horizon,
        max_configs=args.max_configs,
    )

    # Print best configuration
    if results.get("best_config"):
        print("\n" + "=" * 70)
        print("Best Configuration Found")
        print("=" * 70)
        best = results["best_config"]
        print(f"  n_qubits: {best['n_qubits']}")
        print(f"  n_layers: {best['n_layers']}")
        print(f"  n_internal: {best['n_internal']}")
        print(f"  leaky: {best['leaky']}")
        print(f"  spectral_radius: {best['spectral_radius']}")
        print(f"  MSE (h=1): {best['mse_h1']:.4f}")
        print(f"  MSE (h=2): {best['mse_h2']:.4f}")
        print(f"  MSE (h=4): {best['mse_h4']:.4f}")
        print(f"  Training time: {best['train_time_s']:.2f}s")

    # Generate plots
    if not args.skip_plots:
        print("\nGenerating visualizations...")
        plot_tuning_curves(results, args.output_dir)
        plot_tuning_heatmap(results, args.output_dir)
        plot_training_convergence(results, args.output_dir)

    print("\n" + "=" * 70)
    print("Tuning complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
