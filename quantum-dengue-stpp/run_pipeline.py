#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main quantum dengue STPP pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings("ignore")

import torch

from src.data.loader import (
    load_raw_data, build_stpp_events, create_spatial_grid,
    temporal_split, compute_country_summary
)
from src.data.coordinates import get_all_coords
from src.evaluation.spatial_stats import (
    compute_k_function, compute_l_function, compute_pc_function,
    spatial_autocorrelation, temporal_autocorrelation,
    seasonal_decomposition, zero_inflation_ratio, compute_overdispersion
)
from src.evaluation.metrics import (
    compute_forecasting_metrics, compute_classification_metrics,
    compute_point_process_metrics, comprehensive_evaluation
)
from src.models.hawkes import MultiDimensionalHawkes
from src.models.cnn_lstm import SpatioTemporalCNN, create_sequences, train_cnn_lstm
from src.augmentation.sop import (
    sop_augment, sop_augment_spatial_clusters, validate_sop_preservation,
    compute_k_function_empirical, compute_l_function_from_k
)
from src.augmentation.quantum_augment import (
    QuantumBornMachine, HybridStyleQGAN, QuantumAugmentationPipeline
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("../dengue_dataset")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ═══ PHASE 1: DATA LOADING ═══
print("\n" + "="*70)
print("PHASE 1: DATA LOADING & PREPROCESSING")
print("="*70)

spatial_df, long_df, pivot_df = load_raw_data(DATA_DIR)
print(f"Raw data: spatial={spatial_df.shape}, long={long_df.shape}, pivot={pivot_df.shape}")

events_df = build_stpp_events(long_df, min_cases=0, remove_sparse=True, zero_threshold=0.95)
print(f"STPP events: {events_df.shape}")
print(f"  Date range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
print(f"  Countries: {events_df['country'].nunique()}, Regions: {events_df['region'].nunique()}")
print(f"  Total cases: {events_df['case_count'].sum():,.0f}")

coords = get_all_coords()
matched = events_df["region"].apply(lambda r: r in coords).mean()
print(f"  Coordinate match rate: {matched:.1%}")

grid, grid_lats, grid_lons = create_spatial_grid(events_df, grid_size=16)
print(f"  Spatial grid: {grid.shape}")

train_events, val_events, test_events = temporal_split(
    events_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
)
print(f"  Train: {len(train_events):,}, Val: {len(val_events):,}, Test: {len(test_events):,}")

country_summary = compute_country_summary(events_df)
country_summary.to_csv(OUTPUT_DIR / "country_summary.csv", index=False)

# ═══ PHASE 2: EDA ═══
print("\n" + "="*70)
print("PHASE 2: EXTENDED EDA")
print("="*70)

print("\n  Zero-inflation and overdispersion:")
for country in ["VIET NAM", "THAILAND", "INDONESIA", "MALAYSIA", "CAMBODIA"]:
    sub = events_df[events_df["country"] == country]
    zi = zero_inflation_ratio(sub["case_count"].values)
    od = compute_overdispersion(sub["case_count"].values)
    print(f"    {country}: ZI={zi:.1%}, OD={od:.1f}")

print("\n  Spatial autocorrelation (Moran's I):")
for country in ["VIET NAM", "THAILAND", "INDONESIA", "MALAYSIA"]:
    sub = events_df[events_df["country"] == country]
    if len(sub) > 20:
        I, p = spatial_autocorrelation(sub["lat"].values, sub["lon"].values, sub["case_count"].values)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"    {country}: I={I:.4f}, p={p:.4f} {sig}")

print("\n  K/L-function analysis:")
radii = np.array([0.5, 1.0, 2.0, 5.0])
K_by_country, L_by_country = {}, {}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
countries_list = ["VIET NAM", "THAILAND", "INDONESIA", "MALAYSIA",
                   "CAMBODIA", "SINGAPORE", "PHILIPPINES", "LAO PEOPLE'S DEMOCRATIC REPUBLIC"]
countries_list = [c for c in countries_list if c in events_df["country"].unique()]

for idx, country in enumerate(countries_list):
    ax = axes.flat[idx]
    sub = events_df[events_df["country"] == country]
    if len(sub) > 20:
        K, _, _, _ = compute_k_function_empirical(sub["lat"].values, sub["lon"].values, radii, n_permutations=19)
        L = compute_l_function_from_k(K, radii) if K is not None else None
        K_by_country[country] = K
        L_by_country[country] = L
        if L is not None:
            ax.plot(radii, L, "o-", color="steelblue")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            interp = "Clustering" if np.mean(L) > 0 else "Regularity" if np.mean(L) < 0 else "CSR"
            ax.set_title(f"{country[:15]} ({interp})")
        else:
            ax.set_title(f"{country[:15]} (insufficient data)")
        ax.set_xlabel("r (deg)"); ax.set_ylabel("L(r)")
    else:
        ax.set_title(f"{country[:15]} (n<20)")

plt.suptitle("L-function by Country: Clustering vs Regularity", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_l_functions_by_country.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR / 'eda_l_functions_by_country.png'}")

# Monthly patterns
fig, ax = plt.subplots(figsize=(14, 6))
monthly = events_df.groupby(["country", "month"])["case_count"].sum().unstack(level=0)
monthly.plot(ax=ax, linewidth=2, colormap="tab10")
ax.set_xlabel("Month"); ax.set_ylabel("Total Cases")
ax.set_title("Monthly Dengue Seasonality by Country (1993-2022)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_monthly_patterns.png", dpi=150, bbox_inches="tight")
plt.close()

# Yearly trends
fig, ax = plt.subplots(figsize=(14, 6))
yearly = events_df.groupby(["country", "year"])["case_count"].sum().unstack(level=0)
yearly.plot(ax=ax, linewidth=2, colormap="tab10")
ax.set_xlabel("Year"); ax.set_ylabel("Total Cases")
ax.set_title("Yearly Dengue Outbreaks by Country (1993-2022)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_yearly_trends.png", dpi=150, bbox_inches="tight")
plt.close()

# Geographic scatter
fig, ax = plt.subplots(figsize=(14, 10))
total_by_region = events_df.groupby(["lat", "lon"])["case_count"].sum().reset_index()
scatter = ax.scatter(total_by_region["lon"], total_by_region["lat"],
    c=np.log1p(total_by_region["case_count"]), cmap="YlOrRd", alpha=0.6, s=30)
plt.colorbar(scatter, label="log(1 + total cases)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Dengue Cases (SE Asia)")
ax.set_xlim(90, 145); ax.set_ylim(-12, 25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_geographic_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved all EDA plots.")

# ═══ PHASE 3: BASELINE MODELS ═══
print("\n" + "="*70)
print("PHASE 3: BASELINE MODELS")
print("="*70)

# Hawkes
print("\n[3a] Hawkes Process...")
hawkes_events = train_events[train_events["case_count"] > 0].copy()
if len(hawkes_events) > 50:
    hawkes_events["t_month"] = (hawkes_events["timestamp"] - hawkes_events["timestamp"].min()).dt.days / 30.0
    unique_regions = hawkes_events["country"].unique()
    region_to_idx = {r: i for i, r in enumerate(unique_regions)}
    hawkes_events["region_idx"] = hawkes_events["country"].map(region_to_idx)

    hawkes_model = MultiDimensionalHawkes(
        n_regions=len(unique_regions), spatial_decay=0.05,
        temporal_decay=0.3, learnable_spatial_decay=True,
    )
    hawkes_model.fit(
        hawkes_events["t_month"].values, hawkes_events["region_idx"].values,
        hawkes_events["case_count"].values, max_iter=200, verbose=False,
    )
    print(f"  Fitted. mu (first 3): {hawkes_model.mu[:3].round(4)}")
    print(f"  alpha diagonal (first 3): {np.diag(hawkes_model.alpha)[:3].round(4)}")
    print(f"  beta: {hawkes_model.beta:.4f}")
    hawkes_trained = hawkes_model
else:
    hawkes_trained = None
    print("  Insufficient data.")

# CNN-LSTM
print("\n[3b] CNN-LSTM...")
try:
    train_grid, _, _ = create_spatial_grid(train_events, grid_size=16)
    val_grid, _, _ = create_spatial_grid(val_events, grid_size=16)
    X_train, y_train = create_sequences(train_grid, seq_len=12, forecast_horizon=1)
    X_val, y_val = create_sequences(val_grid, seq_len=12, forecast_horizon=1)
    test_grid, _, _ = create_spatial_grid(test_events, grid_size=16)
    X_test, y_test = create_sequences(test_grid, seq_len=12, forecast_horizon=1)
    print(f"  Sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if len(X_train) > 50 and len(X_val) > 10:
        train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=32)

        cnn_lstm_base = SpatioTemporalCNN(input_channels=1, conv_channels=[32, 64],
            lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16, forecast_horizon=1)
        cnn_lstm_base = train_cnn_lstm(cnn_lstm_base, train_ld, val_ld,
            epochs=50, lr=1e-3, device=DEVICE, patience=10, seed=SEED, verbose=True)

        with torch.no_grad():
            y_pred_base = cnn_lstm_base(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy().flatten()
            base_metrics = compute_forecasting_metrics(y_val.flatten(), y_pred_base)
        print(f"  Baseline: RMSE={base_metrics['RMSE']:.2f}, MAE={base_metrics['MAE']:.2f}, R2={base_metrics['R2']:.3f}")
    else:
        cnn_lstm_base = None
        base_metrics = {}
        print("  Insufficient sequences.")
except Exception as e:
    cnn_lstm_base = None
    base_metrics = {}
    print(f"  Failed: {e}")

# ═══ PHASE 4: SOP AUGMENTATION ═══
print("\n" + "="*70)
print("PHASE 4: SOP AUGMENTATION")
print("="*70)

print("\n[4a] SOP augmentation (cluster-based)...")
sop_aug_dfs = sop_augment_spatial_clusters(
    train_events, n_augment=2, n_clusters=5, window_months=3,
    preserve_case_distribution=True, random_state=SEED,
)
print(f"  Generated {len(sop_aug_dfs)} augmented datasets")

print("\n[4b] Validating K/L preservation...")
sop_valid = validate_sop_preservation(train_events, sop_aug_dfs, radii=radii, n_permutations=19)
if sop_valid.get("k_function_mae"):
    valid_k = [v for v in sop_valid["k_function_mae"] if v is not None]
    valid_l = [v for v in sop_valid["l_function_mae"] if v is not None]
    valid_wd = [v for v in sop_valid["case_dist_wasserstein"] if v is not None]
    if valid_k: print(f"  K-func MAE: {np.mean(valid_k):.4f}")
    if valid_l: print(f"  L-func MAE: {np.mean(valid_l):.4f}")
    if valid_wd: print(f"  Wasserstein dist: {np.mean(valid_wd):.2f}")

sop_combined = pd.concat([train_events] + sop_aug_dfs, ignore_index=True)
print(f"  Combined: {len(sop_combined):,} events")

print("\n[4c] Retraining CNN-LSTM with SOP...")
try:
    sop_grid, _, _ = create_spatial_grid(sop_combined, grid_size=16)
    X_sop, y_sop = create_sequences(sop_grid, seq_len=12, forecast_horizon=1)
    if len(X_sop) > 50 and len(X_val) > 10:
        sop_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_sop), torch.FloatTensor(y_sop))
        sop_ld = torch.utils.data.DataLoader(sop_ds, batch_size=32, shuffle=True)
        cnn_lstm_sop = SpatioTemporalCNN(input_channels=1, conv_channels=[32, 64],
            lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16, forecast_horizon=1)
        cnn_lstm_sop = train_cnn_lstm(cnn_lstm_sop, sop_ld, val_ld,
            epochs=50, lr=1e-3, device=DEVICE, patience=10, seed=SEED, verbose=True)

        with torch.no_grad():
            y_pred_sop = cnn_lstm_sop(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy().flatten()
            sop_metrics = compute_forecasting_metrics(y_val.flatten(), y_pred_sop)
        print(f"  SOP: RMSE={sop_metrics['RMSE']:.2f}, MAE={sop_metrics['MAE']:.2f}, R2={sop_metrics['R2']:.3f}")
    else:
        cnn_lstm_sop = None
        sop_metrics = {}
except Exception as e:
    cnn_lstm_sop = None
    sop_metrics = {}
    print(f"  Failed: {e}")

# ═══ PHASE 5: QUANTUM AUGMENTATION ═══
print("\n" + "="*70)
print("PHASE 5: QUANTUM AUGMENTATION")
print("="*70)

print("\n[5a] Training Quantum Born Machine...")
qbm_pipeline, qbm_fitted = None, False
try:
    # QBM uses numpy-based optimisation internally — no CUDA needed
    qbm_pipeline = QuantumAugmentationPipeline(model_type="qbm", n_qubits=8, n_layers=4,
        augmentation_ratio=2, seed=SEED)
    qbm_pipeline.fit(train_events, epochs=150, verbose=True)
    qbm_fitted = True
    print("  QBM training complete.")
except Exception as e:
    print(f"  QBM failed: {e}")

print("\n[5b] Training Hybrid Style-Based QGAN...")
qgan_pipeline, qgan_fitted = None, False
try:
    # Classical components (Autoencoder, Discriminator) run on DEVICE;
    # quantum circuit stays on CPU automatically
    qgan_pipeline = QuantumAugmentationPipeline(model_type="qgan", n_qubits=8, n_layers=4,
        latent_dim=16, augmentation_ratio=3, seed=SEED, torch_device=DEVICE)
    qgan_pipeline.fit(train_events, epochs=150, verbose=True)
    qgan_fitted = True
    print("  QGAN training complete.")
except Exception as e:
    print(f"  QGAN failed: {e}")

print("\n[5c] Generating synthetic events...")
synth_events = []
if qbm_fitted and qbm_pipeline:
    try:
        qbm_samples = qbm_pipeline.generate(n_samples=len(train_events) * 2)
        print(f"  QBM generated {len(qbm_samples)} binary samples")
    except Exception as e:
        print(f"  QBM generation failed: {e}")

if qgan_fitted and qgan_pipeline:
    try:
        qgan_samples = qgan_pipeline.generate(n_samples=len(train_events) * 3)
        print(f"  QGAN generated {len(qgan_samples)} samples")
        if len(qgan_samples) > 0:
            base_ts = train_events["timestamp"].min()
            for i, s in enumerate(qgan_samples):
                case_val = max(1, int(abs(s[0]) * 50 + 1)) if len(s) > 0 else 5
                lat_val = (abs(s[1]) if len(s) > 1 else 0.5) * 15 + 2
                lon_val = (abs(s[2]) if len(s) > 2 else 0.5) * 50 + 95
                ts_offset = pd.Timedelta(days=(i * 7) % 365)
                synth_events.append({
                    "event_id": train_events["event_id"].max() + i + 1,
                    "lat": float(lat_val), "lon": float(lon_val),
                    "timestamp": base_ts + ts_offset,
                    "case_count": case_val,
                    "region": "SYNTHETIC_QGAN",
                    "country": "SYNTHETIC",
                    "year": (base_ts + ts_offset).year,
                    "month": (base_ts + ts_offset).month,
                    "augmented": True, "aug_method": "qgan",
                })
            print(f"  Converted {len(synth_events)} synthetic events")
    except Exception as e:
        print(f"  QGAN generation failed: {e}")

if synth_events:
    synth_df = pd.DataFrame(synth_events)
    quantum_combined = pd.concat([train_events, synth_df], ignore_index=True)
else:
    quantum_combined = train_events.copy()
    synth_df = pd.DataFrame()
print(f"  Combined quantum dataset: {len(quantum_combined):,} events")

print("\n[5d] Retraining CNN-LSTM with quantum augmentation...")
try:
    q_grid, _, _ = create_spatial_grid(quantum_combined, grid_size=16)
    X_q, y_q = create_sequences(q_grid, seq_len=12, forecast_horizon=1)
    if len(X_q) > 50 and len(X_val) > 10:
        q_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_q), torch.FloatTensor(y_q))
        q_ld = torch.utils.data.DataLoader(q_ds, batch_size=32, shuffle=True)
        cnn_lstm_q = SpatioTemporalCNN(input_channels=1, conv_channels=[32, 64],
            lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16, forecast_horizon=1)
        cnn_lstm_q = train_cnn_lstm(cnn_lstm_q, q_ld, val_ld,
            epochs=50, lr=1e-3, device=DEVICE, patience=10, seed=SEED, verbose=True)

        with torch.no_grad():
            y_pred_q = cnn_lstm_q(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy().flatten()
            q_metrics = compute_forecasting_metrics(y_val.flatten(), y_pred_q)
        print(f"  Quantum: RMSE={q_metrics['RMSE']:.2f}, MAE={q_metrics['MAE']:.2f}, R2={q_metrics['R2']:.3f}")
    else:
        cnn_lstm_q = None
        q_metrics = {}
except Exception as e:
    cnn_lstm_q = None
    q_metrics = {}
    print(f"  Failed: {e}")

# ═══ PHASE 6: EVALUATION ═══
print("\n" + "="*70)
print("PHASE 6: COMPREHENSIVE EVALUATION")
print("="*70)

all_methods = {
    "no_augmentation": base_metrics,
    "sop_augmentation": sop_metrics,
    "quantum_augmentation": q_metrics,
}

print("\n┌────────────────────────┬──────────┬──────────┬──────────┐")
print("│ Method                 │ RMSE     │ MAE      │ R2       │")
print("├────────────────────────┼──────────┼──────────┼──────────┤")
for method, m in all_methods.items():
    rmse = m.get("RMSE", float("nan"))
    mae = m.get("MAE", float("nan"))
    r2 = m.get("R2", float("nan"))
    print(f"│ {method:<22} │ {rmse:>8.2f} │ {mae:>8.2f} │ {r2:>8.3f} │")
print("└────────────────────────┴──────────┴──────────┴──────────┘")

# Test set
print("\nTest Set Results:")
if cnn_lstm_base and len(X_test) > 10:
    with torch.no_grad():
        y_t = torch.FloatTensor(X_test).to(DEVICE)
        t_base = compute_forecasting_metrics(y_test.flatten(),
            cnn_lstm_base(y_t).cpu().numpy().flatten())
        print(f"  No aug: RMSE={t_base['RMSE']:.2f}")
if cnn_lstm_sop and len(X_test) > 10:
    with torch.no_grad():
        t_sop = compute_forecasting_metrics(y_test.flatten(),
            cnn_lstm_sop(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten())
        print(f"  SOP: RMSE={t_sop['RMSE']:.2f}")
if cnn_lstm_q and len(X_test) > 10:
    with torch.no_grad():
        t_q = compute_forecasting_metrics(y_test.flatten(),
            cnn_lstm_q(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten())
        print(f"  Quantum: RMSE={t_q['RMSE']:.2f}")

# Comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Methods comparison
ax = axes[0]
methods = list(all_methods.keys())
rmses = [all_methods[m].get("RMSE", 0) for m in methods]
maes = [all_methods[m].get("MAE", 0) for m in methods]
x = np.arange(len(methods))
ax.bar(x - 0.2, rmses, 0.4, label="RMSE", color="steelblue")
ax.bar(x + 0.2, maes, 0.4, label="MAE", color="coral")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
ax.set_ylabel("Error")
ax.set_title("Forecasting Error Comparison")
ax.legend()

# L-function comparison
ax = axes[1]
for label, df_data in [("Original", train_events), ("SOP", sop_combined.iloc[len(train_events):] if len(sop_combined) > len(train_events) else train_events)]:
    if len(df_data) > 20:
        lats, lons = df_data["lat"].values, df_data["lon"].values
        K, _, _, _ = compute_k_function_empirical(lats, lons, radii, n_permutations=9)
        L = compute_l_function_from_k(K, radii) if K is not None else None
        if L is not None:
            ax.plot(radii, L, "o-", label=label)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("r (degrees)"); ax.set_ylabel("L(r)")
ax.set_title("L-function: Original vs SOP")
ax.legend()

# K-function comparison
ax = axes[2]
if len(train_events) > 20:
    lats_o, lons_o = train_events["lat"].values, train_events["lon"].values
    K_o, _, _, _ = compute_k_function_empirical(lats_o, lons_o, radii, n_permutations=9)
    if K_o is not None:
        ax.plot(radii, K_o, "o-", label="Original", color="steelblue")
if len(synth_df) > 20:
    lats_s, lons_s = synth_df["lat"].values, synth_df["lon"].values
    K_s, _, _, _ = compute_k_function_empirical(lats_s, lons_s, radii, n_permutations=9)
    if K_s is not None:
        ax.plot(radii, K_s, "s--", label="Quantum Generated", color="coral")
ax.set_xlabel("r (degrees)"); ax.set_ylabel("K(r)")
ax.set_title("K-function: Original vs Quantum Generated")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'evaluation_comparison.png'}")

# ═══ SAVE RESULTS ═══
results_summary = {
    "validation_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_methods.items()},
    "quantum_methods": {"qbm_fitted": qbm_fitted, "qgan_fitted": qgan_fitted},
    "n_synthetic": len(synth_events),
    "dataset": {
        "n_train": int(len(train_events)), "n_val": int(len(val_events)),
        "n_test": int(len(test_events)), "n_total": int(len(events_df)),
        "total_cases": int(events_df["case_count"].sum()),
        "n_countries": int(events_df["country"].nunique()),
        "n_regions": int(events_df["region"].nunique()),
    },
    "eda": {
        "zero_inflation_by_country": {c: float(zero_inflation_ratio(
            events_df[events_df["country"]==c]["case_count"].values))
            for c in events_df["country"].unique()},
    }
}

with open(OUTPUT_DIR / "results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'results_summary.json'}")
print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
