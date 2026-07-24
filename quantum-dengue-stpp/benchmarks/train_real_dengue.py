#!/usr/bin/env python3
"""
Train QRC v2 on Real Dengue Data.

This script trains the improved Quantum Reservoir Computing (v2) model
on real dengue case data from Vietnam provinces.

Features:
- Loads real dengue data from CSV files
- Trains per-province models
- Evaluates multi-horizon predictions
- Generates province-level performance metrics

Author: QC4SG 2026
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    ImprovedQuantumReservoir,
    direct_multihorizon_predict,
    build_enhanced_features,
    normalize_features,
    QRCOutputLayer as QRCOutputLayerV2,
)


# =============================================================================
# Data Loading
# =============================================================================


def load_vietnam_dengue_data(
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load Vietnam dengue data from CSV.

    Args:
        data_path: Path to the data file. If None, searches in standard locations.

    Returns:
        DataFrame with columns: province, year, month, dengue_cases, temperature, humidity, precipitation
    """
    if data_path is None:
        # Search in standard locations
        possible_paths = [
            "dengue_dataset/vietnam_enhanced_full.csv",
            "dengue_dataset/vietnam_enhanced.csv",
            "dengue_dataset/vietnam_final_features.csv",
            "../dengue_dataset/vietnam_enhanced_full.csv",
            "../dengue_dataset/vietnam_final_features.csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError(
                f"Could not find Vietnam dengue data. Searched: {possible_paths}"
            )

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Rename to standard names - simple approach
    df = df.rename(columns={
        'province_normalized': 'province',
        'dengue_total': 'cases',
    })

    # Ensure required columns exist
    required_cols = ['province', 'year', 'month', 'cases']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill missing climate data with defaults
    if 'temperature' not in df.columns:
        df['temperature'] = 28.0  # Default tropical temperature
    if 'humidity' not in df.columns:
        df['humidity'] = 75.0
    if 'rainfall' not in df.columns:
        df['rainfall'] = 100.0

    df['cases'] = df['cases'].fillna(0)
    df['temperature'] = df['temperature'].fillna(28.0)
    df['humidity'] = df['humidity'].fillna(75.0)
    df['rainfall'] = df['rainfall'].fillna(100.0)

    # Count provinces using basic pandas
    province_series = df['province']
    if hasattr(province_series, 'unique'):
        n_provinces = len(province_series.unique())
    else:
        n_provinces = 1
        
    print(f"Loaded {len(df)} records from {n_provinces} provinces")
    return df


def prepare_province_data(
    df: pd.DataFrame,
    province: str,
    min_cases: int = 100,
    min_years: int = 3,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Prepare time series data for a specific province.

    Args:
        df: Full dengue DataFrame
        province: Province name
        min_cases: Minimum total cases required
        min_years: Minimum number of years required

    Returns:
        Tuple of (cases array, climate data dict) or None if insufficient data
    """
    prov_df = df[df['province'] == province].sort_values(['year', 'month'])

    if len(prov_df) < 12:  # At least 1 year of data
        return None

    if prov_df['cases'].sum() < min_cases:
        return None

    if prov_df['year'].nunique() < min_years:
        return None

    # Convert to time series
    cases = prov_df['cases'].values.astype(float)

    climate_data = {
        'temperature': prov_df['temperature'].values.astype(float),
        'humidity': prov_df['humidity'].values.astype(float),
        'rainfall': prov_df['rainfall'].values.astype(float),
    }

    return cases, climate_data


def get_provinces_by_region(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group provinces by region based on zone column.

    Args:
        df: Dengue DataFrame

    Returns:
        Dict mapping region to list of provinces
    """
    if 'zone' in df.columns:
        regions = {}
        for zone in df['zone'].unique():
            provinces = df[df['zone'] == zone]['province'].unique().tolist()
            if provinces:
                regions[zone] = provinces
        return regions

    # Fallback: group by first letter or alphabetical
    provinces = df['province'].unique()
    regions = {}
    for prov in provinces:
        if len(prov) > 0:
            first_char = prov[0].upper()
            if first_char not in regions:
                regions[first_char] = []
            regions[first_char].append(prov)
    return regions


# =============================================================================
# Training
# =============================================================================


def train_province_model(
    cases: np.ndarray,
    climate_data: Dict[str, np.ndarray],
    province: str,
    config: Optional[ImprovedQRCConfig] = None,
    train_fraction: float = 0.7,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train QRC v2 model for a specific province.

    Args:
        cases: Time series of dengue cases
        climate_data: Climate features
        province: Province name
        config: QRC configuration
        train_fraction: Fraction for training

    Returns:
        Tuple of (results dict, predictions, actuals)
    """
    if config is None:
        config = ImprovedQRCConfig(
            n_qubits=8,
            n_layers=3,
            n_internal=30,
            max_horizon=4,
            leaky=0.2,
            use_climate_features=True,
            seed=42,
        )

    t_start = time.time()

    result = direct_multihorizon_predict(
        cases=cases,
        climate_data=climate_data,
        config=config,
        train_fraction=train_fraction,
    )

    train_time = time.time() - t_start

    return {
        "province": province,
        "mse_by_horizon": result.mse_by_horizon,
        "nmse_by_horizon": result.nmse_by_horizon,
        "mae_by_horizon": result.mae_by_horizon,
        "n_params": result.n_params,
        "n_timesteps": result.n_timesteps,
        "n_features": result.n_features,
        "train_time_s": train_time,
        "config": result.config,
    }, result.horizon_mses, []


def train_all_provinces(
    df: pd.DataFrame,
    config: Optional[ImprovedQRCConfig] = None,
    provinces: Optional[List[str]] = None,
    min_cases: int = 100,
    output_dir: str = "output_result/q_stpp_v18",
) -> Dict:
    """Train QRC v2 models for all provinces.

    Args:
        df: Dengue DataFrame
        config: QRC configuration
        provinces: Specific provinces to train (None = all)
        min_cases: Minimum cases for inclusion
        output_dir: Output directory

    Returns:
        Training results
    """
    os.makedirs(output_dir, exist_ok=True)

    if provinces is None:
        provinces = df['province'].unique().tolist()

    print(f"Training on {len(provinces)} provinces...")

    results = {
        "config": {
            "n_qubits": config.n_qubits if config else 8,
            "n_layers": config.n_layers if config else 3,
            "n_internal": config.n_internal if config else 30,
            "max_horizon": config.max_horizon if config else 4,
        },
        "province_results": [],
        "skipped_provinces": [],
        "summary": {},
    }

    for i, province in enumerate(provinces):
        print(f"[{i+1}/{len(provinces)}] Training {province}...", end=" ", flush=True)

        prov_data = prepare_province_data(df, province, min_cases=min_cases)
        if prov_data is None:
            results["skipped_provinces"].append(province)
            print("skipped (insufficient data)")
            continue

        cases, climate = prov_data

        try:
            prov_result, horizon_mses, _ = train_province_model(
                cases, climate, province, config
            )
            results["province_results"].append(prov_result)
            mse_h1 = prov_result["mse_by_horizon"][1]
            print(f"MSE_h1={mse_h1:.2f}")
        except Exception as e:
            print(f"error: {e}")
            results["skipped_provinces"].append(province)

    # Compute summary statistics
    if results["province_results"]:
        all_mses_h1 = [r["mse_by_horizon"][1] for r in results["province_results"]]
        all_mses_h2 = [r["mse_by_horizon"][2] for r in results["province_results"] if 2 in r["mse_by_horizon"]]
        all_mses_h4 = [r["mse_by_horizon"][4] for r in results["province_results"] if 4 in r["mse_by_horizon"]]

        results["summary"] = {
            "n_provinces_trained": len(results["province_results"]),
            "n_provinces_skipped": len(results["skipped_provinces"]),
            "mse_h1_mean": float(np.mean(all_mses_h1)),
            "mse_h1_std": float(np.std(all_mses_h1)),
            "mse_h1_min": float(np.min(all_mses_h1)),
            "mse_h1_max": float(np.max(all_mses_h1)),
            "mse_h2_mean": float(np.mean(all_mses_h2)) if all_mses_h2 else None,
            "mse_h4_mean": float(np.mean(all_mses_h4)) if all_mses_h4 else None,
            "best_province": min(results["province_results"], key=lambda x: x["mse_by_horizon"][1])["province"],
            "worst_province": max(results["province_results"], key=lambda x: x["mse_by_horizon"][1])["province"],
        }

    # Save results
    output_file = os.path.join(output_dir, "train_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[Saved] {output_file}")

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_per_province_results(
    results: Dict,
    df: Optional[pd.DataFrame] = None,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate per-province performance heatmap.

    Args:
        results: Training results
        df: Optional DataFrame for zone information
        output_dir: Output directory
    """
    if not results.get("province_results"):
        print("No province results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(results["province_results"]) * 0.4)))

    province_names = [r["province"] for r in results["province_results"]]
    horizons = [1, 2, 3, 4]

    # Create MSE matrix
    mse_matrix = []
    for r in results["province_results"]:
        row = [r["mse_by_horizon"].get(h, np.nan) for h in horizons]
        mse_matrix.append(row)
    mse_matrix = np.array(mse_matrix)

    # Heatmap of MSE by province and horizon
    ax1 = axes[0]
    sns.heatmap(
        mse_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=[f'h={h}' for h in horizons],
        yticklabels=province_names,
        ax=ax1,
        cbar_kws={'label': 'MSE'}
    )
    ax1.set_title('MSE by Province and Horizon', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Prediction Horizon')
    ax1.set_ylabel('Province')

    # Bar chart of MSE h=1 by province
    ax2 = axes[1]
    mse_h1 = [r["mse_by_horizon"][1] for r in results["province_results"]]
    
    # Sort by MSE
    sorted_indices = np.argsort(mse_h1)
    sorted_names = [province_names[i] for i in sorted_indices]
    sorted_mses = [mse_h1[i] for i in sorted_indices]

    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = ax2.barh(range(len(sorted_names)), sorted_mses, color=colors)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names)
    ax2.set_xlabel('MSE (h=1)')
    ax2.set_title('Provinces Ranked by MSE (h=1)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, mse) in enumerate(zip(bars, sorted_mses)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{mse:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "per_province_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


def plot_province_summary_bar(
    results: Dict,
    output_dir: str = "output_result/q_stpp_v18",
) -> None:
    """Generate summary bar chart for top/bottom provinces.

    Args:
        results: Training results
        output_dir: Output directory
    """
    if not results.get("province_results"):
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by MSE
    sorted_results = sorted(results["province_results"], key=lambda x: x["mse_by_horizon"][1])

    provinces = [r["province"] for r in sorted_results]
    mses_h1 = [r["mse_by_horizon"][1] for r in sorted_results]
    mses_h2 = [r["mse_by_horizon"].get(2, mses_h1[i]) for i, r in enumerate(sorted_results)]
    mses_h4 = [r["mse_by_horizon"].get(4, mses_h2[i]) for i, r in enumerate(sorted_results)]

    x = np.arange(len(provinces))
    width = 0.25

    ax.bar(x - width, mses_h1, width, label='h=1', color='#2ecc71', alpha=0.8)
    ax.bar(x, mses_h2, width, label='h=2', color='#f39c12', alpha=0.8)
    ax.bar(x + width, mses_h4, width, label='h=4', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Province', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('QRC v2 Performance by Province (All Horizons)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(provinces, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "province_summary_bar.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train QRC v2 on Real Dengue Data')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to dengue data CSV')
    parser.add_argument('--output-dir', type=str, default='output_result/q_stpp_v18',
                       help='Output directory')
    parser.add_argument('--provinces', type=str, nargs='+', default=None,
                       help='Specific provinces to train (default: all)')
    parser.add_argument('--n-qubits', type=int, default=8,
                       help='Number of qubits')
    parser.add_argument('--n-layers', type=int, default=3,
                       help='Number of circuit layers')
    parser.add_argument('--n-internal', type=int, default=30,
                       help='Internal dimension')
    parser.add_argument('--max-horizon', type=int, default=4,
                       help='Maximum prediction horizon')
    parser.add_argument('--min-cases', type=int, default=100,
                       help='Minimum cases for province inclusion')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    args = parser.parse_args()

    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)

    # Create config
    config = ImprovedQRCConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_internal=args.n_internal,
        max_horizon=args.max_horizon,
        leaky=0.2,
        use_climate_features=True,
        seed=42,
    )

    # Load data
    print("=" * 70)
    print("Loading Vietnam Dengue Data")
    print("=" * 70)
    df = load_vietnam_dengue_data(args.data_path)

    # Train models
    print("\n" + "=" * 70)
    print("Training QRC v2 on Real Data")
    print("=" * 70)
    results = train_all_provinces(
        df=df,
        config=config,
        provinces=args.provinces,
        min_cases=args.min_cases,
        output_dir=args.output_dir,
    )

    # Print summary
    if results["summary"]:
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"Provinces trained: {results['summary']['n_provinces_trained']}")
        print(f"Provinces skipped: {results['summary']['n_provinces_skipped']}")
        print(f"MSE (h=1) mean: {results['summary']['mse_h1_mean']:.2f} ± {results['summary']['mse_h1_std']:.2f}")
        print(f"Best province: {results['summary']['best_province']}")
        print(f"Worst province: {results['summary']['worst_province']}")

    # Generate plots
    if not args.skip_plots:
        print("\nGenerating visualizations...")
        plot_per_province_results(results, df, args.output_dir)
        plot_province_summary_bar(results, args.output_dir)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
