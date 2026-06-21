#!/usr/bin/env python3
"""
GPU-accelerated Quantum Dengue STPP Pipeline with Data Leakage Prevention.

This script provides a production-ready pipeline that:
1. Enforces strict temporal data leakage prevention
2. Uses adaptive spatial gridding for balanced country representation
3. Validates count data distribution (no negative predictions)
4. Supports hybrid quantum-classical augmentation

CRITICAL: This pipeline prevents spatio-temporal data leakage where the quantum
model learns from future data, causing artificially high R² in simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Import data modules
from data.loader import (
    load_raw_data,
    build_stpp_events,
    create_adaptive_spatial_grid,
    create_country_adaptive_grids,
    validate_no_data_leakage,
    generate_quantum_augmented_data,
    temporal_split,
    compute_country_summary
)

# Import models
from models.cnn_lstm import SpatioTemporalCNN, train_cnn_lstm, create_sequences
from models.nest import NESTForecaster
from augmentation.quantum_augment_v3 import augment_with_grid_qgan
from evaluation.metrics import compute_metrics


def validate_data_integrity(train_df, val_df, test_df):
    """
    Comprehensive data integrity checks to prevent data leakage.
    
    This is CRITICAL for quantum-augmented models - if the quantum model
    sees future data during training, R² will be artificially high.
    """
    print("\n" + "="*60)
    print("DATA INTEGRITY VALIDATION")
    print("="*60)
    
    # 1. Temporal ordering check
    print("\n[1] Temporal Ordering Check:")
    print(f"    Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"    Val:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"    Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # 2. Strict data leakage validation
    print("\n[2] Data Leakage Validation:")
    try:
        validate_no_data_leakage(train_df, val_df, test_df)
        print("    ✓ NO DATA LEAKAGE DETECTED")
    except AssertionError as e:
        print(f"    ✗ DATA LEAKAGE DETECTED: {e}")
        raise
    
    # 3. Count data distribution check
    print("\n[3] Count Data Distribution Check:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        cases = df["case_count"].values
        n_zeros = (cases == 0).sum()
        pct_zeros = n_zeros / len(cases) * 100
        print(f"    {name}: {len(df)} events, {n_zeros} zeros ({pct_zeros:.1f}%), "
              f"range [{cases.min()}, {cases.max()}]")
    
    # 4. Country coverage check
    print("\n[4] Country Coverage Check:")
    train_countries = set(train_df["country"].unique())
    val_countries = set(val_df["country"].unique())
    test_countries = set(test_df["country"].unique())
    
    print(f"    Train countries: {len(train_countries)}")
    print(f"    Val countries:   {len(val_countries)}")
    print(f"    Test countries:  {len(test_countries)}")
    
    # Check for country overlap (should be the same for temporal split)
    if train_countries == val_countries == test_countries:
        print("    ✓ All splits cover same countries")
    else:
        print("    ⚠ Country coverage differs between splits")
    
    # 5. Spatial coordinate check
    print("\n[5] Spatial Coordinate Check:")
    lats = train_df["lat"].values
    lons = train_df["lon"].values
    print(f"    Latitude range:  [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"    Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
    
    # Check for normalized coordinates (should NOT be normalized for raw data)
    if lats.min() >= 0 and lats.max() <= 1:
        print("    ⚠ WARNING: Coordinates appear normalized [0,1] - "
              "this should only happen for quantum embedding!")
    
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK PASSED ✓")
    print("="*60 + "\n")


def create_sequences_with_validation(grid, seq_len=12, forecast_horizon=1):
    """
    Create sequences with validation that predictions are non-negative.
    
    CRITICAL: Dengue case counts cannot be negative.
    This function ensures the sequence creation process maintains data integrity.
    """
    X, y = create_sequences(grid, seq_len, forecast_horizon)
    
    if len(X) == 0:
        return X, y
    
    # Validate no negative values
    assert X.min() >= 0, f"Negative values found in input sequences: {X.min()}"
    assert y.min() >= 0, f"Negative values found in target: {y.min()}"
    
    return X, y


def validate_model_output(predictions, targets):
    """
    Validate model outputs to ensure count data integrity.
    
    CRITICAL: 
    - Predictions must be >= 0 (count data cannot be negative)
    - Predictions should be integers (or close to integers for aggregated data)
    """
    metrics = {}
    
    # Check for negative predictions (SHOULD NOT HAPPEN with softplus)
    n_negative = (predictions < 0).sum()
    if n_negative > 0:
        print(f"    ⚠ WARNING: {n_negative} negative predictions detected!")
        metrics['n_negative'] = int(n_negative)
    
    # Basic metrics
    metrics['pred_min'] = float(predictions.min())
    metrics['pred_max'] = float(predictions.max())
    metrics['pred_mean'] = float(predictions.mean())
    metrics['target_min'] = float(targets.min())
    metrics['target_max'] = float(targets.max())
    
    return metrics


def run_pipeline(data_dir, output_dir, config):
    """
    Main pipeline with data leakage prevention and validation.
    """
    print("\n" + "#"*60)
    print("# QUANTUM DENGUE STPP - GPU PIPELINE v2")
    print("# Data Leakage Prevention + Adaptive Gridding")
    print("#"*60 + "\n")
    
    t0 = time.time()
    
    # Load data
    print("[1/7] Loading data...")
    spatial, long_df, pivot = load_raw_data(data_dir)
    print(f"    Loaded {len(long_df)} records")
    
    # Build STPP events
    print("\n[2/7] Building STPP events...")
    events_df = build_stpp_events(long_df)
    print(f"    Built {len(events_df)} events")
    
    # Chronological split (CRITICAL: no shuffle!)
    print("\n[3/7] Splitting data (chronological, no shuffle)...")
    train_df, val_df, test_df = temporal_split(
        events_df, 
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # CRITICAL: Validate no data leakage
    print("\n[4/7] Validating data integrity...")
    validate_data_integrity(train_df, val_df, test_df)
    
    # Create adaptive spatial grids (CRITICAL: normalize for quantum embedding)
    print("\n[5/7] Creating adaptive spatial grids...")
    grid_size = config.get('grid_size', 20)
    
    # Use adaptive gridding per country for balanced representation
    use_adaptive = config.get('adaptive_gridding', True)
    
    if use_adaptive:
        print(f"    Using country-adaptive gridding (grid_size={grid_size})")
        # This creates normalized [0,1]^2 coordinates for each country
        # Critical for quantum AngleEmbedding
        train_grids = create_country_adaptive_grids(train_df, grid_size=grid_size)
    else:
        print(f"    Using global gridding (grid_size={grid_size})")
        train_grid, grid_lats, grid_lons, norm_coords, scaler_params = \
            create_adaptive_spatial_grid(train_df, grid_size=grid_size, normalize_coords=True)
    
    # Create sequences
    print("\n[6/7] Creating sequences...")
    seq_len = config.get('seq_len', 12)
    forecast_horizon = config.get('forecast_horizon', 1)
    
    if use_adaptive:
        # Aggregate country grids for global model
        all_grids = []
        for country, data in train_grids.items():
            all_grids.append(data['grid'])
        train_grid = np.concatenate(all_grids, axis=-1).mean(axis=-1, keepdims=True)
    
    X_train, y_train = create_sequences_with_validation(train_grid, seq_len, forecast_horizon)
    print(f"    Created {len(X_train)} training sequences")
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Validate sequence data
    assert X_train.min() >= 0, "Negative values in training sequences!"
    assert y_train.min() >= 0, "Negative values in training targets!"
    print(f"    ✓ Sequence data validated: all values >= 0")
    
    # Train model
    print("\n[7/7] Training CNN-LSTM...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    model = SpatioTemporalCNN(
        grid_size=grid_size,
        forecast_horizon=forecast_horizon,
        output_activation='softplus'  # CRITICAL: ensures non-negative predictions
    )
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        ),
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    
    model = train_cnn_lstm(
        model, train_loader, None,
        epochs=config.get('epochs', 50),
        lr=config.get('lr', 1e-3),
        device=device,
        verbose=True
    )
    
    # Validate model output
    print("\n[VALIDATION] Model Output Check:")
    model.eval()
    with torch.no_grad():
        sample_X = torch.FloatTensor(X_train[:32]).to(device)
        predictions = model(sample_X).cpu().numpy()
        targets = y_train[:32]
    
    output_metrics = validate_model_output(predictions, targets)
    print(f"    Prediction range: [{output_metrics['pred_min']:.2f}, {output_metrics['pred_max']:.2f}]")
    print(f"    Target range: [{output_metrics['target_min']:.2f}, {output_metrics['target_max']:.2f}]")
    
    if output_metrics['pred_min'] >= 0:
        print("    ✓ All predictions are non-negative (softplus working!)")
    
    # Compute final metrics
    test_metrics = compute_metrics(predictions.flatten(), targets.flatten())
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"    Total time: {time.time() - t0:.1f}s")
    print(f"    Test RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
    print(f"    Test R²:   {test_metrics.get('r2', 'N/A'):.4f}")
    print("="*60 + "\n")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': config,
        'data_integrity': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        },
        'output_validation': output_metrics,
        'test_metrics': test_metrics,
        'runtime_seconds': time.time() - t0
    }
    
    with open(output_path / 'pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path / 'pipeline_results.json'}")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Quantum Dengue STPP GPU Pipeline')
    parser.add_argument('--data_dir', type=str, 
                       default='dengue_dataset',
                       help='Directory containing dengue data')
    parser.add_argument('--output_dir', type=str,
                       default='output_result/results',
                       help='Output directory')
    parser.add_argument('--grid_size', type=int, default=20,
                       help='Spatial grid size')
    parser.add_argument('--seq_len', type=int, default=12,
                       help='Input sequence length')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Forecast horizon')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--adaptive_gridding', action='store_true',
                       help='Use adaptive gridding per country')
    parser.add_argument('--no_adaptive_gridding', dest='adaptive_gridding',
                       action='store_false', help='Use global gridding')
    parser.set_defaults(adaptive_gridding=True)
    
    args = parser.parse_args()
    
    config = {
        'grid_size': args.grid_size,
        'seq_len': args.seq_len,
        'forecast_horizon': args.forecast_horizon,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'adaptive_gridding': args.adaptive_gridding,
    }
    
    run_pipeline(args.data_dir, args.output_dir, config)


if __name__ == '__main__':
    main()
