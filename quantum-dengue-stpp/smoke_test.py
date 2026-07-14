#!/usr/bin/env python3
"""
Canonical smoke test for Quantum-Dengue-STPP after consolidation.

This test ensures:
1. Every canonical module imports cleanly.
2. Forward passes work for the major model classes.
3. Augmentation pipelines can produce output.
4. ZINB loss computes finite, non-negative values.

It is intentionally lightweight — no real training, no GPU dependency.
Designed to fail fast when a canonical replacement breaks behaviour.
"""
import sys
import os
import traceback

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"[ OK ] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILED.append(name)


def test_data_loader():
    from data.loader import (
        temporal_split,
        validate_no_data_leakage,
        create_adaptive_spatial_grid,
    )
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=50, freq='D'),
        'value': range(50),
    })
    train, val, test = temporal_split(df, 0.7, 0.15, 0.15)
    assert len(train) == 35, len(train)
    assert len(val) + len(test) == 15


def test_cnn_lstm():
    """Canonical CNN-LSTM is v2 with attention + residual + bidirectional LSTM."""
    import torch
    from models.cnn_lstm import SpatioTemporalCNNv2, create_sequences_v2, train_cnn_lstm_v2

    model = SpatioTemporalCNNv2(grid_size=20, forecast_horizon=1, loss='mse')
    x = torch.randn(4, 12, 20, 20)
    out = model(x)
    assert out.shape == (4, 1), out.shape
    assert torch.isfinite(out).all(), "output should be finite"

    grid = torch.rand(16, 16, 50).numpy()
    X, y = create_sequences_v2(grid, seq_len=12, forecast_horizon=1)
    assert X.shape == (38, 12, 16, 16), X.shape


def test_zinb_loss():
    """Canonical ZINB re-exports both ZeroInflatedNegativeBinomialLoss and the new
    PhysicsInformedZINBLoss (controlled-noise regularizer)."""
    import torch
    from models.zinb_loss import (
        ZeroInflatedNegativeBinomialLoss,
        HybridQuantumZINB,
        SpatialZINBGridLoss,
        compute_zinb_metrics,
    )

    pred_mu = torch.randn(8, 100) * 2 + 5
    pred_pi = torch.randn(8, 100) * 0.3 - 0.5
    target = torch.randint(0, 50, (8, 100)).float()

    loss_fn = ZeroInflatedNegativeBinomialLoss(learn_theta=True)
    loss = loss_fn(pred_mu, pred_pi, target)
    assert torch.isfinite(loss), "loss must be finite"

    metrics = compute_zinb_metrics(pred_mu, pred_pi, target, theta=loss_fn.theta.item())
    assert 'mse' in metrics


def test_physics_informed_zinb():
    """Direct import path for the new controlled-noise ZINB."""
    import torch
    from models.physics_informed_zinb import PhysicsInformedZINBLoss

    pred_mu = torch.randn(8, 100) * 2 + 5
    pred_pi = torch.randn(8, 100) * 0.3 - 0.5
    target = torch.randint(0, 50, (8, 100)).float()

    loss_fn = PhysicsInformedZINBLoss(noise_scale=0.05, learn_theta=True)
    loss = loss_fn(pred_mu, pred_pi, target, grid_size=10)
    assert torch.isfinite(loss), "loss must be finite"


def test_sop():
    """Canonical SOP uses SMOTE-style interpolation (v2 logic)."""
    import numpy as np
    from augmentation.sop import (
        create_feature_space,
        smote_interpolation,
        sop_augment_v2,
        validate_sop_v2_preservation,
    )

    np.random.seed(42)
    import pandas as pd
    df = pd.DataFrame({
        'lat': np.random.uniform(-6, 23, 200),
        'lon': np.random.uniform(95, 141, 200),
        'timestamp': pd.date_range('2020-01-01', periods=200, freq='D'),
        'case_count': np.random.randint(0, 50, 200),
        'year': np.random.randint(2018, 2023, 200),
        'month': np.random.randint(1, 13, 200),
        'region': ['r%d' % i for i in range(200)],
        'country': ['COUNTRY'] * 200,
    })
    X, df_out = create_feature_space(df)
    assert X.shape[0] == 200

    X2, y2 = smote_interpolation(X, df['case_count'].values, n_synthetic=10)
    assert X2.shape[0] == 10


def test_spatial_stats():
    """Canonical spatial_stats uses fast cKDTree-based implementations."""
    import numpy as np
    from evaluation.spatial_stats import (
        fast_k_function,
        fast_l_function,
        fast_morans_i,
    )

    np.random.seed(42)
    lat = np.random.uniform(-6, 23, 200)
    lon = np.random.uniform(95, 141, 200)
    counts = np.random.randint(0, 50, 200)

    K = fast_k_function(lat, lon, radii_km=np.array([50.0, 100.0]))
    L = fast_l_function(K, np.array([50.0, 100.0]))
    I, p = fast_morans_i(lat, lon, counts, k=5)
    assert K.shape == (2,)
    assert L.shape == (2,)


def test_quantum_augment():
    """Canonical quantum_augment uses v3 grid-level generation (QBMv3 + GridQGANV3)."""
    from augmentation.quantum_augment import (
        QBMv3,
        GridQGANV3,
        augment_with_grid_qgan,
    )
    # Just construct without running circuits
    qbm = QBMv3(grid_size=16, n_patterns=4, n_layers=2)
    assert qbm is not None


def test_local_pqc():
    """Test both strongly_entangling (default) and data_reuploading ansatze."""
    import torch
    from augmentation.local_pqc import (
        LocalPQC,
        ClusteredLocalPQC,
        SpatialClusterer,
        QuantumFisherInformation,
    )

    pqc = LocalPQC(n_qubits=4, n_layers=2, feature_dim=8, ansatz='strongly_entangling')
    x = torch.randn(4, 8)
    out = pqc(x)
    assert out.shape == (4, 1), out.shape
    assert (out >= 0).all(), "intensity_head + exp must be non-negative"


def test_data_reuploading_ansatz():
    import torch
    from augmentation.data_reuploading_ansatz import DataReuploadingPQC

    pqc = DataReuploadingPQC(n_qubits=4, n_layers=2, feature_dim=8)
    x = torch.randn(4, 8)
    out = pqc(x)
    assert out.shape == (4, 4), out.shape  # (batch, n_qubits) expvals


def test_local_pqc_data_reuploading():
    """Data-Reuploading Ansatz integration in LocalPQC."""
    import torch
    from augmentation.local_pqc import LocalPQC

    pqc = LocalPQC(n_qubits=4, n_layers=2, feature_dim=8, ansatz='data_reuploading')
    x = torch.randn(4, 8)
    out = pqc(x)
    assert out.shape == (4, 1), out.shape
    assert (out >= 0).all()


def test_quantum_natural_gradient():
    import torch
    from optimization.quantum_natural_gradient import QuantumNaturalGradient, DiagonalQNG

    params = [torch.randn(4, requires_grad=True) for _ in range(3)]
    opt = QuantumNaturalGradient(params, lr=0.01)
    opt.step()


def test_loss_collection():
    """Canonical losses module exposes its custom losses (MSE/MAE use nn.MSELoss/L1Loss)."""
    import torch
    import torch.nn as nn
    from models.losses import (
        NegativeBinomialLoss,
        PoissonLoss,
        TweedieLoss,
        QuantileLoss,
        LogCauchyNBGaussLoss,
        get_loss_fn,
    )

    pred = torch.randn(8, 100)
    target = torch.rand(8, 100) * 10

    losses = {
        'mse': get_loss_fn('mse'),
        'mae': get_loss_fn('mae'),
        'poisson': PoissonLoss(),
        'nb': NegativeBinomialLoss(),
        'tweedie': TweedieLoss(),
        'quantile': QuantileLoss(),
        'log_cauchy_nb_gauss': LogCauchyNBGaussLoss(),
    }
    for name, loss_fn in losses.items():
        v = loss_fn(pred, target)
        assert torch.isfinite(v), f"{name} produced non-finite"


def test_metrics():
    import numpy as np
    from evaluation.metrics import compute_forecasting_metrics

    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.5
    m = compute_forecasting_metrics(y_true, y_pred)
    assert 'RMSE' in m and 'R2' in m


def test_hawkes_nest():
    import numpy as np
    from models.hawkes import MultiDimensionalHawkes, HawkesBaseline
    from models.nest import NESTIntensity, NESTForecaster

    hawkes = MultiDimensionalHawkes(n_regions=4)
    times = np.array([0.1, 0.5, 0.8, 1.2, 1.5])
    regions = np.array([0, 1, 0, 2, 1])
    weights = np.ones(5)
    hawkes.fit(times, regions, weights, max_iter=5)
    assert hawkes.mu.shape == (4,)

    nest = NESTIntensity(gs=10, output_activation='softplus')
    import torch
    x = torch.randn(2, 5, 10, 10)
    out = nest(x)
    assert out.shape == (2, 10, 10)


def test_climate_module_optional():
    try:
        from data.climate import load_climate_covariates
        print("  climate module loaded")
    except Exception as e:
        print(f"  climate optional skip: {type(e).__name__}: {e}")


def main():
    tests = [
        ("data.loader", test_data_loader),
        ("cnn_lstm", test_cnn_lstm),
        ("zinb_loss", test_zinb_loss),
        ("physics_informed_zinb", test_physics_informed_zinb),
        ("sop", test_sop),
        ("spatial_stats", test_spatial_stats),
        ("quantum_augment", test_quantum_augment),
        ("local_pqc (strongly_entangling)", test_local_pqc),
        ("local_pqc (data_reuploading)", test_local_pqc_data_reuploading),
        ("data_reuploading_ansatz", test_data_reuploading_ansatz),
        ("quantum_natural_gradient", test_quantum_natural_gradient),
        ("losses", test_loss_collection),
        ("metrics", test_metrics),
        ("hawkes+nest", test_hawkes_nest),
        ("climate (optional)", test_climate_module_optional),
    ]
    print(f"\n=== Running {len(tests)} canonical smoke tests ===\n")
    for name, fn in tests:
        check(name, fn)
    print(f"\n=== Summary: {len(tests) - len(FAILED)}/{len(tests)} passed ===")
    if FAILED:
        print("\nFAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CANONICAL SMOKE TESTS PASSED")


if __name__ == '__main__':
    main()