#!/usr/bin/env python3
"""Fast pipeline — optimized spatial statistics + all models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, time

import torch

from src.data.loader import (load_raw_data, build_stpp_events, create_spatial_grid,
                               temporal_split, compute_country_summary)
from src.evaluation.spatial_stats_fast import (
    fast_k_function as compute_k_function, fast_l_function as compute_l_function,
    fast_pair_correlation as compute_pc_function, fast_morans_i as spatial_autocorrelation,
    zero_inflation_ratio, compute_overdispersion
)
from src.augmentation.sop import compute_k_function_empirical, compute_l_function_from_k
from src.evaluation.metrics import compute_forecasting_metrics
from src.models.hawkes import MultiDimensionalHawkes
from src.models.cnn_lstm import SpatioTemporalCNN, create_sequences, train_cnn_lstm
from src.augmentation.quantum_augment import QuantumAugmentationPipeline

OUTPUT = Path("outputs"); OUTPUT.mkdir(exist_ok=True)
DATA = Path("../dengue_dataset")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cpu"

t0 = time.time()

# ═══ PHASE 1 ═══
print("="*60)
print("PHASE 1: DATA")
print("="*60)
spatial_df, long_df, pivot_df = load_raw_data(DATA)
print(f"  Raw: spatial={spatial_df.shape}, long={long_df.shape}")

events_df = build_stpp_events(long_df, min_cases=0, remove_sparse=True, zero_threshold=0.95)
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
print(f"  STPP: {len(events_df):,} events, {events_df['country'].nunique()} countries, {events_df['region'].nunique()} regions")
print(f"  Total cases: {events_df['case_count'].sum():,.0f}")
print(f"  Period: {events_df['timestamp'].min()} → {events_df['timestamp'].max()}")

grid, grid_lats, grid_lons = create_spatial_grid(events_df, grid_size=16)
print(f"  Grid: {grid.shape}")

train_ev, val_ev, test_ev = temporal_split(events_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
print(f"  Split: train={len(train_ev):,}, val={len(val_ev):,}, test={len(test_ev):,}")

country_summary = compute_country_summary(events_df)
country_summary.to_csv(OUTPUT / "country_summary.csv", index=False)
print(f"  Phase 1 done ({time.time()-t0:.1f}s)")

# ═══ PHASE 2: EDA ═══
print("\n" + "="*60)
print("PHASE 2: EXTENDED EDA")
print("="*60)

t_eda = time.time()

print("\n  [Zero-inflation & Overdispersion]")
for c in ["VIET NAM","THAILAND","INDONESIA","MALAYSIA","CAMBODIA","PHILIPPINES","SINGAPORE"]:
    sub = events_df[events_df["country"]==c]
    if len(sub) > 0:
        zi = zero_inflation_ratio(sub["case_count"].values)
        od = compute_overdispersion(sub["case_count"].values)
        print(f"    {c:<15} ZI={zi:>6.1%}  OD={od:>8.1f}  n={len(sub):>5,}")

print("\n  [Moran's I — Spatial Autocorrelation]")
for c in ["VIET NAM","THAILAND","INDONESIA","MALAYSIA","CAMBODIA","PHILIPPINES"]:
    sub = events_df[events_df["country"]==c]
    if len(sub) > 20:
        vals = sub.groupby(["lat","lon"])["case_count"].mean().values
        lats = sub.groupby(["lat","lon"])["lat"].first().values
        lons = sub.groupby(["lat","lon"])["lon"].first().values
        if len(vals) > 10:
            I, p = spatial_autocorrelation(lats, lons, vals, k=5)
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "   "
            print(f"    {c:<15} I={I:>7.4f}  p={p:.4f} {sig}")

print("\n  [K/L-function — Spatial Clustering]")
radii = np.array([50.0, 100.0, 200.0, 500.0])  # km
country_K = {}
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
ax_list = axes.flat
countries_eda = [c for c in ["VIET NAM","THAILAND","INDONESIA","MALAYSIA","CAMBODIA","SINGAPORE"]
                if c in events_df["country"].unique()]

for idx, c in enumerate(countries_eda):
    sub = events_df[events_df["country"]==c]
    if len(sub) > 30:
        lats, lons = sub["lat"].values, sub["lon"].values
        K = compute_k_function(lats, lons, radii, max_n=300)
        L = compute_l_function(K, radii)
        country_K[c] = {"K": K, "L": L}
        ax = ax_list[idx]
        ax.plot(radii, L, "o-", color="steelblue", lw=2, ms=6)
        ax.axhline(0, color="gray", ls="--", alpha=0.6)
        interp = "CLUSTERED" if np.mean(L) > 5 else "REGULAR" if np.mean(L) < -5 else "RANDOM"
        ax.set_title(f"{c} ({interp})", fontsize=10)
        ax.set_xlabel("r (km)"); ax.set_ylabel("L(r)")
        ax.grid(True, alpha=0.3)
        print(f"    {c:<15} L mean={np.mean(L):>7.2f}  K[200km]={K[2]:>8.2f}  -> {interp}")

plt.suptitle("L-function Analysis: Clustering vs Regularity in SE Asia Dengue", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_l_functions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  K/L plots saved ({time.time()-t_eda:.1f}s)")

# Monthly patterns
fig, ax = plt.subplots(figsize=(14, 5))
monthly = events_df.groupby([events_df["country"], events_df["timestamp"].dt.month])["case_count"].sum().unstack(level=0)
monthly.plot(ax=ax, lw=1.5, colormap="tab10")
ax.set_xlabel("Month"); ax.set_ylabel("Total Cases")
ax.set_title("Monthly Dengue Seasonality by Country (1993-2022)")
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_monthly.png", dpi=150, bbox_inches="tight"); plt.close()

# Yearly trends
fig, ax = plt.subplots(figsize=(14, 5))
yearly = events_df.groupby([events_df["country"], events_df["timestamp"].dt.year])["case_count"].sum().unstack(level=0)
yearly.plot(ax=ax, lw=1.5, colormap="tab10")
ax.set_xlabel("Year"); ax.set_ylabel("Total Cases")
ax.set_title("Yearly Dengue Outbreaks (1993-2022)")
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_yearly.png", dpi=150, bbox_inches="tight"); plt.close()

# Geographic scatter
fig, ax = plt.subplots(figsize=(12, 8))
tot = events_df.groupby(["lat","lon"])["case_count"].sum().reset_index()
sc = ax.scatter(tot["lon"], tot["lat"], c=np.log1p(tot["case_count"]),
                cmap="YlOrRd", alpha=0.7, s=25, edgecolors="none")
plt.colorbar(sc, label="log(1+cases)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Dengue Hotspots SE Asia (log scale)")
ax.set_xlim(95, 140); ax.set_ylim(-12, 23)
plt.tight_layout()
plt.savefig(OUTPUT / "eda_geographic.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"  Phase 2 done ({time.time()-t_eda:.1f}s)")

# ═══ PHASE 3: BASELINES ═══
print("\n" + "="*60)
print("PHASE 3: BASELINE MODELS")
print("="*60)

# Hawkes
print("\n  [Hawkes Process]")
t_hw = time.time()
hw_ev = train_ev[train_ev["case_count"]>0].copy()
# Use monthly aggregates for speed
hw_agg = hw_ev.groupby([hw_ev["timestamp"].dt.to_period("M"), "country"])["case_count"].sum().reset_index()
hw_agg["t"] = np.arange(len(hw_agg))
hw_agg = hw_agg.sort_values("t")

# Simple scalar Hawkes: total cases per time step
hw_times = hw_agg["t"].values.astype(float)
hw_counts = hw_agg["case_count"].values.astype(float)

# Normalize
hw_times_norm = hw_times / (hw_times.max() + 1e-9)
T = hw_times_norm.max()

hw_mu = float(hw_counts.mean())
# Estimate alpha from autocorrelation
ac = np.corrcoef(hw_counts[:-1], hw_counts[1:])[0,1]
hw_alpha = max(0.1, min(0.9, abs(ac)))
hw_beta = 0.3  # fixed decay rate
print(f"    Scalar Hawkes (approx): mu={hw_mu:.2f}, alpha={hw_alpha:.3f}, beta={hw_beta:.3f}")
print(f"    Time: {time.time()-t_hw:.1f}s")

# CNN-LSTM
print("\n  [CNN-LSTM]")
t_cnn = time.time()
# Compute shared grid from all data, then slice per split
def _make_grid(ev_df, lat_min, lat_max, lon_min, lon_max, global_min_t, global_max_t, grid_size=16):
    """Build grid using shared global bounds so all splits have same T dimension."""
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)

    ev_df = ev_df.copy()
    ev_df["lat_bin"] = pd.cut(ev_df["lat"], lat_edges, labels=False, include_lowest=True)
    ev_df["lon_bin"] = pd.cut(ev_df["lon"], lon_edges, labels=False, include_lowest=True)
    ev_df["t_idx"] = ((ev_df["timestamp"] - global_min_t).dt.days / 30.44).astype(int).clip(0)

    total_t = int((global_max_t - global_min_t).days / 30.44) + 1
    grid = np.zeros((grid_size, grid_size, total_t), dtype=np.float32)
    valid = (ev_df["lat_bin"] >= 0) & (ev_df["lat_bin"] < grid_size) & \
            (ev_df["lon_bin"] >= 0) & (ev_df["lon_bin"] < grid_size) & \
            (ev_df["t_idx"] >= 0) & (ev_df["t_idx"] < total_t)
    for _, row in ev_df[valid].iterrows():
        grid[int(row["lat_bin"]), int(row["lon_bin"]), int(row["t_idx"])] += row["case_count"]
    return grid, total_t

all_lat_min, all_lat_max = events_df["lat"].min(), events_df["lat"].max()
all_lon_min, all_lon_max = events_df["lon"].min(), events_df["lon"].max()
global_min_t = events_df["timestamp"].min()
global_max_t = events_df["timestamp"].max()

train_grid, T_train = _make_grid(train_ev, all_lat_min, all_lat_max, all_lon_min, all_lon_max, global_min_t, global_max_t, 16)
val_grid,   T_val   = _make_grid(val_ev,   all_lat_min, all_lat_max, all_lon_min, all_lon_max, global_min_t, global_max_t, 16)
test_grid,  T_test  = _make_grid(test_ev,  all_lat_min, all_lat_max, all_lon_min, all_lon_max, global_min_t, global_max_t, 16)
print(f"    Grids: train={train_grid.shape}, val={val_grid.shape}, test={test_grid.shape}")
# Grid shape is (16, 16, 360) = (H, W, T)
X_tr, y_tr = create_sequences(train_grid, seq_len=12, forecast_horizon=1)
X_vl, y_vl = create_sequences(val_grid, seq_len=12, forecast_horizon=1)
X_ts, y_ts = create_sequences(test_grid, seq_len=12, forecast_horizon=1)
print(f"    Sequences: train={len(X_tr)}, val={len(X_vl)}, test={len(X_ts)}")

if len(X_tr) > 50:
    tr_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    vl_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))
    cnn = SpatioTemporalCNN(input_channels=1, conv_channels=[32,64],
                            lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16)
    cnn = train_cnn_lstm(cnn, tr_ds, vl_ds, epochs=50, lr=1e-3,
                          device=DEVICE, patience=10, seed=SEED, verbose=True)
    with torch.no_grad():
        p = cnn(torch.FloatTensor(X_vl)).numpy().flatten()
        m_base = compute_forecasting_metrics(y_vl.flatten(), p)
    print(f"    NoAug: RMSE={m_base.get('RMSE',0):.2f}  MAE={m_base.get('MAE',0):.2f}  R2={m_base.get('R2',0):.3f}")
    print(f"    CNN-LSTM done ({time.time()-t_cnn:.1f}s)")
else:
    cnn = None; m_base = {}
    print("    Too few sequences.")

# ═══ PHASE 4: SOP ═══
print("\n" + "="*60)
print("PHASE 4: SOP AUGMENTATION")
print("="*60)
t_sop = time.time()

from src.augmentation.sop import sop_augment_spatial_clusters, validate_sop_preservation
sop_dfs = sop_augment_spatial_clusters(train_ev, n_augment=2, n_clusters=5,
                                       window_months=3, random_state=SEED)
print(f"  Generated {len(sop_dfs)} SOP datasets")
# Fix: reset index before concat to avoid duplicate index issue
train_ev_r = train_ev.reset_index(drop=True)
for i, df in enumerate(sop_dfs):
    df = df.copy().reset_index(drop=True)
    df["augmented"] = True
    df["aug_method"] = "sop"
    sop_dfs[i] = df
sop_combined = pd.concat([train_ev_r] + sop_dfs, ignore_index=True)
print(f"  Combined: {len(sop_combined):,} events")

sop_v = validate_sop_preservation(train_ev, sop_dfs, radii=radii, n_permutations=9)
k_mae = np.nanmean([v for v in sop_v.get("k_function_mae",[]) if v is not None])
l_mae = np.nanmean([v for v in sop_v.get("l_function_mae",[]) if v is not None])
wd = np.nanmean([v for v in sop_v.get("case_dist_wasserstein",[]) if v is not None])
print(f"  Preservation: K-MAE={k_mae:.2f}  L-MAE={l_mae:.2f}  Wass={wd:.2f}")
print(f"  SOP done ({time.time()-t_sop:.1f}s)")

# Retrain CNN-LSTM with SOP
if cnn and len(X_tr) > 50:
    sop_grid, T_sp = _make_grid(sop_combined, all_lat_min, all_lat_max, all_lon_min, all_lon_max, global_min_t, global_max_t, 16)
    X_sp, y_sp = create_sequences(sop_grid, seq_len=12, forecast_horizon=1)
    if len(X_sp) > 50:
        sp_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_sp), torch.FloatTensor(y_sp))
        cnn_sop = SpatioTemporalCNN(input_channels=1, conv_channels=[32,64],
                                   lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16)
        cnn_sop = train_cnn_lstm(cnn_sop, sp_ds, vl_ds, epochs=50, lr=1e-3,
                                 device=DEVICE, patience=10, seed=SEED, verbose=True)
        with torch.no_grad():
            p = cnn_sop(torch.FloatTensor(X_vl)).numpy().flatten()
            m_sop = compute_forecasting_metrics(y_vl.flatten(), p)
        print(f"  SOP-CNN: RMSE={m_sop.get('RMSE',0):.2f}  MAE={m_sop.get('MAE',0):.2f}  R2={m_sop.get('R2',0):.3f}")
    else:
        m_sop = {}
else:
    m_sop = {}

# ═══ PHASE 5: QUANTUM ═══
print("\n" + "="*60)
print("PHASE 5: QUANTUM AUGMENTATION")
print("="*60)
t_q = time.time()

qbm_fit = qgan_fit = False
qbm_pipe = qgan_pipe = None
synth_dfs = []

# QBM
print("  [QBM training...]")
try:
    qbm_pipe = QuantumAugmentationPipeline(model_type="qbm", n_qubits=8, n_layers=4,
                                           augmentation_ratio=2, seed=SEED)
    qbm_pipe.fit(train_ev, epochs=100, verbose=True)
    qbm_fit = True
    print("  QBM fitted.")
except Exception as e:
    print(f"  QBM failed: {e}")

# QGAN
print("  [QGAN training...]")
try:
    qgan_pipe = QuantumAugmentationPipeline(model_type="qgan", n_qubits=8, n_layers=4,
                                            latent_dim=16, augmentation_ratio=3, seed=SEED)
    qgan_pipe.fit(train_ev, epochs=100, verbose=True)
    qgan_fit = True
    print("  QGAN fitted.")
except Exception as e:
    print(f"  QGAN failed: {e}")

# Generate
if qgan_fit and qgan_pipe:
    try:
        gens = qgan_pipe.generate(n_samples=len(train_ev)*3)
        print(f"  Generated {len(gens)} samples")
        if len(gens) > 0:
            base_ts = train_ev["timestamp"].min()
            rows = []
            for i, s in enumerate(gens):
                case_val = max(1, int(abs(s[0])*30+1)) if len(s)>0 else 5
                lat_v = float(10 + abs(s[1]%1)*10) if len(s)>1 else 12.0
                lon_v = float(100 + abs(s[2]%1)*20) if len(s)>2 else 105.0
                days = int(i % 365)
                ts = base_ts + pd.Timedelta(days=days)
                rows.append({"event_id": train_ev["event_id"].max()+i+1,
                             "lat": lat_v, "lon": lon_v, "timestamp": ts,
                             "case_count": case_val, "region":"SYNTHETIC",
                             "country":"SYNTHETIC","year":ts.year,"month":ts.month,
                             "augmented":True,"aug_method":"qgan"})
            synth_dfs.append(pd.DataFrame(rows))
    except Exception as e:
        print(f"  Generation failed: {e}")

if synth_dfs:
    synth_df = pd.concat(synth_dfs, ignore_index=True)
    q_combined = pd.concat([train_ev, synth_df], ignore_index=True)
else:
    q_combined = train_ev.copy()
    synth_df = pd.DataFrame()
print(f"  Quantum combined: {len(q_combined):,} events ({len(synth_df)} synthetic)")

# Retrain with quantum
if cnn and len(X_tr) > 50 and len(synth_df) > 100:
    q_grid, T_q = _make_grid(q_combined, all_lat_min, all_lat_max, all_lon_min, all_lon_max, global_min_t, global_max_t, 16)
    X_q, y_q = create_sequences(q_grid, seq_len=12, forecast_horizon=1)
    if len(X_q) > 50:
        q_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_q), torch.FloatTensor(y_q))
        cnn_q = SpatioTemporalCNN(input_channels=1, conv_channels=[32,64],
                                  lstm_hidden=64, lstm_layers=2, dropout=0.3, grid_size=16)
        cnn_q = train_cnn_lstm(cnn_q, q_ds, vl_ds, epochs=50, lr=1e-3,
                                device=DEVICE, patience=10, seed=SEED, verbose=True)
        with torch.no_grad():
            p = cnn_q(torch.FloatTensor(X_vl)).numpy().flatten()
            m_q = compute_forecasting_metrics(y_vl.flatten(), p)
        print(f"  Quantum-CNN: RMSE={m_q.get('RMSE',0):.2f}  MAE={m_q.get('MAE',0):.2f}  R2={m_q.get('R2',0):.3f}")
    else:
        m_q = {}
elif cnn:
    m_q = {}
else:
    m_q = {}
print(f"  Quantum done ({time.time()-t_q:.1f}s)")

# ═══ PHASE 6: EVALUATION ═══
print("\n" + "="*60)
print("PHASE 6: COMPREHENSIVE EVALUATION")
print("="*60)

results = {
    "No Augmentation": m_base,
    "SOP Augmentation": m_sop,
    "Quantum Augmentation": m_q,
}

print("\n  Validation Set Results:")
print("  " + "-"*55)
print(f"  {'Method':<22} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("  " + "-"*55)
best_rmse = ("N/A", float("inf"))
for method, m in results.items():
    rmse = m.get("RMSE", float("nan"))
    mae = m.get("MAE", float("nan"))
    r2 = m.get("R2", float("nan"))
    print(f"  {method:<22} {rmse:>8.2f} {mae:>8.2f} {r2:>8.3f}")
    if not np.isnan(rmse) and rmse < best_rmse[1]:
        best_rmse = (method, rmse)
print("  " + "-"*55)
print(f"  Best: {best_rmse[0]} (RMSE={best_rmse[1]:.2f})")

# Comparison plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Bar chart
ax = axes[0]
methods = list(results.keys())
rmses = [results[m].get("RMSE", 0) for m in methods]
maes = [results[m].get("MAE", 0) for m in methods]
colors = ["#3498db", "#e74c3c", "#2ecc71"]
x = np.arange(len(methods))
bars1 = ax.bar(x - 0.2, rmses, 0.35, label="RMSE", color=[c+"99" for c in colors])
bars2 = ax.bar(x + 0.2, maes, 0.35, label="MAE", color=colors)
ax.set_xticks(x); ax.set_xticklabels([m.replace(" ","\n") for m in methods], fontsize=8)
ax.set_ylabel("Error"); ax.set_title("Forecasting Error Comparison")
ax.legend()
for bar, val in zip(bars1, rmses):
    if val > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# L-function
ax = axes[1]
for label, sub_ev in [("Original", train_ev)]:
    if len(sub_ev) > 30:
        K = compute_k_function(sub_ev["lat"].values, sub_ev["lon"].values, radii, max_n=300)
        L = compute_l_function(K, radii)
        ax.plot(radii, L, "o-", label=label, lw=2, color="steelblue")
if len(sop_dfs) > 0:
    K_sop = compute_k_function(sop_dfs[0]["lat"].values, sop_dfs[0]["lon"].values, radii, max_n=300)
    L_sop = compute_l_function(K_sop, radii)
    ax.plot(radii, L_sop, "s--", label="SOP Aug", lw=2, color="coral")
ax.axhline(0, color="gray", ls="--", alpha=0.6)
ax.set_xlabel("r (km)"); ax.set_ylabel("L(r)")
ax.set_title("L-function: Structural Preservation")
ax.legend(); ax.grid(True, alpha=0.3)

# Case distribution
ax = axes[2]
cases_orig = train_ev["case_count"].values
cases_sop = sop_combined["case_count"].values[len(train_ev):] if len(sop_combined) > len(train_ev) else np.array([])
ax.hist(np.log1p(cases_orig), bins=50, alpha=0.6, label="Original", density=True, color="steelblue")
if len(cases_sop) > 0:
    ax.hist(np.log1p(cases_sop), bins=50, alpha=0.6, label="SOP Augmented", density=True, color="coral")
ax.set_xlabel("log(1 + cases)"); ax.set_ylabel("Density")
ax.set_title("Case Count Distribution")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT / "evaluation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: evaluation_comparison.png")

# Save JSON
summary = {
    "validation": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    "quantum": {"qbm_fitted": qbm_fit, "qgan_fitted": qgan_fit, "n_synthetic": len(synth_df)},
    "dataset": {"n_train": int(len(train_ev)), "n_val": int(len(val_ev)),
                "n_test": int(len(test_ev)), "total_cases": int(events_df["case_count"].sum()),
                "n_countries": int(events_df["country"].nunique()), "n_regions": int(events_df["region"].nunique())},
    "spatial": {"morans_I": {c: float(spatial_autocorrelation(
        events_df[events_df["country"]==c].groupby(["lat","lon"])["lat"].first().values,
        events_df[events_df["country"]==c].groupby(["lat","lon"])["lon"].first().values,
        events_df[events_df["country"]==c].groupby(["lat","lon"])["case_count"].mean().values)[0])
                   for c in ["VIET NAM","THAILAND","INDONESIA","MALAYSIA"]}},
    "timing": {"total_s": round(time.time()-t0, 1)}
}
with open(OUTPUT / "results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  Total runtime: {time.time()-t0:.1f}s")
print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
