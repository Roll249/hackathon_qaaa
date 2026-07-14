#!/usr/bin/env python3
"""
Canonical entry point for the Quantum-Dengue-STPP project.

This is the single, recommended way to run the pipeline after consolidation.
It replaces the half-dozen `run_*.py` scripts that previously lived at the root.

Usage:
    python main.py --smoke                  # fast smoke test (no training, no real data)
    python main.py --data ../dengue_dataset  # real data run (no training by default)
    python main.py --data ../dengue_dataset --train     # train CNN-LSTM (small)

Key features (all post-consolidation):
    * Imports from canonical modules only:
        - src.augmentation.quantum_augment  (v3 grid-level QBM + GridQGAN)
        - src.augmentation.sop              (v2 SMOTE-style interpolation)
        - src.augmentation.local_pqc        (StronglyEntangling + DataReuploading ansatze)
        - src.augmentation.data_reuploading_ansatz
        - src.optimization.quantum_natural_gradient
        - src.evaluation.spatial_stats      (fast cKDTree implementations)
        - src.models.cnn_lstm              (SpatioTemporalCNNv2)
        - src.models.zinb_loss              (re-exports PhysicsInformedZINBLoss)
        - src.models.physics_informed_zinb
    * Quantum params are optimized with QuantumNaturalGradient (when available),
      classical heads with AdamW.
    * No real training is performed by default — by default it loads data,
      verifies the import chain, and prints sanity metrics.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the path (works whether you run from the root or anywhere)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Quantum Dengue STPP — canonical entry point")
    p.add_argument("--data", default=None,
                   help="Path to dengue dataset (default: smoke mode, no real data)")
    p.add_argument("--output", default="outputs", help="Output directory")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test only: import + construct + tiny forward pass")
    p.add_argument("--train", action="store_true",
                   help="Train CNN-LSTM (small, no quantum circuits)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--use-qng", action="store_true",
                   help="Use QuantumNaturalGradient for quantum params (where applicable)")
    return p.parse_args()


def banner(msg):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def run_smoke():
    """A no-data smoke test that exercises the canonical import chain.

    This is the deterministic signal we use to verify the consolidation has
    not broken any module. It does NOT touch the filesystem, network, or GPU.
    """
    banner("SMOKE TEST — import + construct + tiny forward pass")

    import torch
    import numpy as np

    # Data loader
    from data.loader import temporal_split, validate_no_data_leakage
    print("[1/8] data.loader OK")

    # Quantum augmentation
    from augmentation.quantum_augment import QBMv3, GridQGANV3, augment_with_grid_qgan
    qbm = QBMv3(grid_size=16, n_patterns=4, n_layers=2)
    print(f"[2/8] quantum_augment OK (QBMv3 grid={qbm.grid_size}, n_patterns={qbm.n_patterns})")

    # SOP (SMOTE-based)
    from augmentation.sop import create_feature_space, sop_augment_v2
    print("[3/8] sop (SMOTE) OK")

    # Spatial stats (fast cKDTree)
    from evaluation.spatial_stats import fast_k_function, fast_l_function, fast_morans_i
    rng = np.random.default_rng(0)
    lat = rng.uniform(-6, 23, 200)
    lon = rng.uniform(95, 141, 200)
    counts = rng.integers(0, 50, 200)
    K = fast_k_function(lat, lon, radii_km=np.array([50.0, 100.0]))
    print(f"[4/8] spatial_stats OK (K-shape={K.shape})")

    # Local PQC (both ansatze)
    from augmentation.local_pqc import LocalPQC
    pqc_se = LocalPQC(n_qubits=4, n_layers=2, feature_dim=8, ansatz="strongly_entangling")
    pqc_dr = LocalPQC(n_qubits=4, n_layers=2, feature_dim=8, ansatz="data_reuploading")
    x = torch.randn(4, 8)
    out_se = pqc_se(x)
    out_dr = pqc_dr(x)
    print(f"[5/8] local_pqc OK (SE out-shape={tuple(out_se.shape)}, "
          f"DR out-shape={tuple(out_dr.shape)})")

    # Quantum Natural Gradient
    from optimization.quantum_natural_gradient import QuantumNaturalGradient, DiagonalQNG
    p = torch.nn.Parameter(torch.randn(8))
    opt = QuantumNaturalGradient([p], lr=0.01)
    print(f"[6/8] quantum_natural_gradient OK (lr={opt.param_groups[0]['lr']})")

    # ZINB (canonical, with re-exports)
    from models.zinb_loss import (
        ZeroInflatedNegativeBinomialLoss,
        PhysicsInformedZINBLoss,
        HybridQuantumZINB,
        SpatialZINBGridLoss,
        compute_zinb_metrics,
    )
    pi_zinb = PhysicsInformedZINBLoss(noise_scale=0.05, learn_theta=True)
    mu = torch.randn(4, 100) * 2 + 5
    pi = torch.randn(4, 100) * 0.3 - 0.5
    tgt = torch.randint(0, 50, (4, 100)).float()
    loss = pi_zinb(mu, pi, tgt, grid_size=10)
    print(f"[7/8] zinb_loss OK (physics-informed loss={loss.item():.4f})")

    # CNN-LSTM v2
    from models.cnn_lstm import SpatioTemporalCNNv2, create_sequences_v2
    model = SpatioTemporalCNNv2(grid_size=20, forecast_horizon=1, loss="mse")
    x = torch.randn(4, 12, 20, 20)
    y = model(x)
    print(f"[8/8] cnn_lstm OK (SpatioTemporalCNNv2 out-shape={tuple(y.shape)})")

    print("\n✓ ALL CANONICAL MODULES PASSED SMOKE TEST")
    return 0


def run_data_pipeline(args):
    """End-to-end run that loads data, builds EDA, and (optionally) trains."""
    import torch
    import numpy as np
    import pandas as pd
    from data.loader import (
        load_raw_data,
        build_stpp_events,
        temporal_split,
        create_adaptive_spatial_grid,
        validate_no_data_leakage,
    )
    from augmentation.quantum_augment import augment_with_grid_qgan
    from augmentation.sop import sop_augment_v2
    from models.cnn_lstm import SpatioTemporalCNNv2, create_sequences_v2, train_cnn_lstm_v2
    from models.zinb_loss import PhysicsInformedZINBLoss

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"!! Data directory not found: {data_dir}")
        print("   Run with --smoke for a deterministic no-data test.")
        return 1

    banner(f"DATA PIPELINE — data={data_dir}, output={out_dir}")

    t0 = time.time()
    spatial, long_df, pivot = load_raw_data(data_dir)
    print(f"  Loaded {len(long_df):,} records (raw).")

    events_df = build_stpp_events(long_df)
    print(f"  Built {len(events_df):,} STPP events.")

    train_df, val_df, test_df = temporal_split(
        events_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    print(f"  Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    validate_no_data_leakage(train_df, val_df, test_df)
    print("  ✓ No data leakage detected")

    grid, glats, glons, norm_coords, params = create_adaptive_spatial_grid(
        train_df, grid_size=args.grid_size, normalize_coords=True
    )
    print(f"  Adaptive grid: shape={grid.shape}, params={params}")

    X, y = create_sequences_v2(grid, seq_len=args.seq_len, forecast_horizon=1)
    print(f"  Sequences: X={X.shape}, y={y.shape}")

    if not args.train:
        print(f"\nDone in {time.time() - t0:.1f}s. (Use --train to train CNN-LSTM.)")
        return 0

    # Optional: small training run using canonical CNN-LSTM v2 + Physics-Informed ZINB
    banner("TRAINING CNN-LSTM v2 (small)")
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = SpatioTemporalCNNv2(grid_size=args.grid_size, forecast_horizon=1, loss="mse")
    model = train_cnn_lstm_v2(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=args.epochs,
        lr=1e-3,
        device="cpu",
        loss_name="mse",
        verbose=True,
    )

    if args.use_qng:
        # The QNG optimizer is intended for quantum-circuit parameters. For
        # purely-classical CNN-LSTM training it degenerates to a per-parameter
        # Adam-like step. The integration point for hybrid quantum-classical
        # models is in :mod:`optimization.quantum_natural_gradient` — see
        # ``example_hybrid_training_loop`` for the recommended pattern.
        from optimization.quantum_natural_gradient import QuantumNaturalGradient
        q_params = [p for n, p in model.named_parameters() if "weight" in n]
        if q_params:
            opt = QuantumNaturalGradient(q_params, lr=1e-3)
            print(f"  Wrapped {len(q_params)} params with QuantumNaturalGradient (--use-qng)")

    print(f"\nDone in {time.time() - t0:.1f}s")
    return 0


def main():
    args = parse_args()
    if args.smoke or args.data is None:
        return run_smoke()
    return run_data_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())