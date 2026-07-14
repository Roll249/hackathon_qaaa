#!/usr/bin/env python3
"""
Optimized Full Pipeline — maximizes laptop performance.
Strategy:
  1. Load data ONCE, build grids in-place
  2. Parallel CNN-LSTM training via joblib (NoAug + SOP + Quantum simultaneously)
  3. Memory-efficient tensor operations
  4. Optimal batch size for CPU throughput
  5. Chunk-based spatial statistics
  6. GC aggressively between stages
"""
import sys, os, time, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
import multiprocessing as mp

# ── CPU config ──────────────────────────────────────────────────────────────
CPU_COUNT = os.cpu_count() or 8
N_WORKERS = CPU_COUNT          # joblib threads for parallel model training
torch_threads = min(CPU_COUNT, 8)  # PyTorch intra-op threads
os.environ["OMP_NUM_THREADS"] = str(torch_threads)
os.environ["MKL_NUM_THREADS"] = str(torch_threads)

import torch
torch.set_num_threads(torch_threads)
DEVICE = "cpu"

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT = Path("outputs"); OUTPUT.mkdir(exist_ok=True)
DATA   = Path("../dengue_dataset")
SEED   = 42
np.random.seed(SEED); torch.manual_seed(SEED)

t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA LOADING (single pass)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STAGE 1: DATA LOADING")
print("=" * 65)

from src.data.loader import load_raw_data, build_stpp_events, temporal_split

t1 = time.time()
spatial_df, long_df, pivot_df = load_raw_data(DATA)
events_df = build_stpp_events(long_df, min_cases=0, remove_sparse=True, zero_threshold=0.95)
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
train_ev, val_ev, test_ev = temporal_split(events_df, 0.70, 0.15, 0.15)

# Global extents for gridding
LAT_MIN, LAT_MAX = events_df["lat"].min(), events_df["lat"].max()
LON_MIN, LON_MAX = events_df["lon"].min(), events_df["lon"].max()
T_MIN,   T_MAX   = events_df["timestamp"].min(), events_df["timestamp"].max()
TOTAL_T  = int((T_MAX - T_MIN).days / 30.44) + 1

# Grid parameters — balance resolution vs sequence count
GS = 20          # 20×20 grid (was 16×16)
SEQ_LEN = 10      # 10-month lookback (was 12)
FH = 1            # 1-month forecast

print(f"  Events: {len(events_df):,} | Cases: {events_df['case_count'].sum():,.0f}")
print(f"  Grid: {GS}×{GS}×{TOTAL_T} ({TOTAL_T} months)")
print(f"  Split: train={len(train_ev):,} val={len(val_ev):,} test={len(test_ev):,}")
print(f"  Loaded in {time.time()-t1:.1f}s")


def _build_grid(ev_df, gs=GS):
    """Build (gs × gs × T) case-count grid from event DataFrame."""
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, gs+1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, gs+1)
    ev = ev_df.copy()
    ev["lat_bin"] = pd.cut(ev["lat"], lat_edges, labels=False, include_lowest=True)
    ev["lon_bin"] = pd.cut(ev["lon"], lon_edges, labels=False, include_lowest=True)
    ev["t_idx"] = ((ev["timestamp"] - T_MIN).dt.days / 30.44).astype(int).clip(0, TOTAL_T-1)
    grid = np.zeros((gs, gs, TOTAL_T), dtype=np.float32)
    valid = (ev["lat_bin"]>=0)&(ev["lat_bin"]<gs)&(ev["lon_bin"]>=0)&(ev["lon_bin"]<gs)
    for _, row in ev[valid].iterrows():
        grid[int(row["lat_bin"]), int(row["lon_bin"]), int(row["t_idx"])] += row["case_count"]
    return grid


def _sequences(grid, seq_len=SEQ_LEN, fh=FH):
    """Create (X, y) sequences from grid. X: (N, seq_len, H, W), y: (N,)"""
    N = grid.shape[2] - seq_len - fh + 1
    if N <= 0:
        return np.zeros((0, seq_len, grid.shape[0], grid.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X, y = [], []
    for t in range(N):
        X.append(grid[:, :, t:t+seq_len].transpose(2, 0, 1))
        y.append(grid[:, :, t+seq_len+fh-1].mean())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# Build grids ONCE
print("\n  Building grids...")
t_grid = time.time()
train_grid = _build_grid(train_ev)
val_grid   = _build_grid(val_ev)
test_grid  = _build_grid(test_ev)
print(f"  Grids built in {time.time()-t_grid:.1f}s")

X_tr, y_tr = _sequences(train_grid)
X_vl, y_vl = _sequences(val_grid)
X_ts, y_ts = _sequences(test_grid)
print(f"  Sequences: train={len(X_tr)}, val={len(X_vl)}, test={len(X_ts)}")

# Free raw event DataFrames
del train_ev, val_ev, test_ev
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: EDA — spatial statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 2: EDA — SPATIAL STATISTICS")
print("=" * 65)

from src.evaluation.spatial_stats_fast import (
    fast_k_function, fast_l_function,
    fast_morans_i, zero_inflation_ratio, compute_overdispersion
)

RADII = np.array([50.0, 100.0, 200.0, 500.0], dtype=np.float32)

def _eda_country(name, sub):
    n = len(sub)
    if n < 5:
        return None
    zi = zero_inflation_ratio(sub["case_count"].values)
    od = compute_overdispersion(sub["case_count"].values)
    I, p = fast_morans_i(sub["lat"].values, sub["lon"].values, sub["case_count"].values, k=5)
    K = fast_k_function(sub["lat"].values, sub["lon"].values, RADII, max_n=300)
    L = fast_l_function(K, RADII)
    interp = "CLUSTERED" if np.mean(L) > 5 else "REGULAR" if np.mean(L) < -5 else "RANDOM"
    return {"ZI": zi, "OD": od, "n": n, "morans_I": float(I), "morans_p": float(p),
            "K": K.tolist(), "L": L.tolist(), "interpretation": interp}

COUNTRIES = ["VIET NAM","THAILAND","INDONESIA","MALAYSIA","CAMBODIA","SINGAPORE","PHILIPPINES","LAO PEOPLE'S DEM. REPUBLIC"]
eda_results = {}
print(f"  Computing per-country spatial stats ({CPU_COUNT} CPUs available)...")
for c in COUNTRIES:
    sub = events_df[events_df["country"] == c]
    r = _eda_country(c, sub)
    if r:
        eda_results[c] = r
        sig = "***" if r["morans_p"]<0.001 else "**" if r["morans_p"]<0.01 else "*" if r["morans_p"]<0.05 else ""
        print(f"    {c:<38} n={r['n']:>5}  ZI={r['ZI']:>6.1%}  I={r['morans_I']:>+6.3f}{sig}  L@200km={r['L'][2]:>+7.1f}  -> {r['interpretation']}")

# Plot EDA figures
print("  Generating EDA plots...")

# Geographic + trend
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
colors_c = plt.cm.tab10(np.linspace(0, 1, len(eda_results)))
for (c, r), col in zip(eda_results.items(), colors_c):
    monthly = events_df[events_df["country"]==c].groupby(
        events_df[events_df["country"]==c]["timestamp"].dt.to_period("M"))["case_count"].sum()
    if len(monthly) > 0:
        ax.plot(monthly.index.astype(str), monthly.values, alpha=0.5, lw=1, color=col)
        ma = pd.Series(monthly.values).rolling(12, min_periods=1).mean()
        ax.plot(range(len(monthly)), ma, lw=2, color=col, label=c)
ax.set_xlabel("Month"); ax.set_ylabel("Cases"); ax.set_title("Monthly Cases by Country (12-mo MA)")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

ax = axes[1]
country_cases = events_df.groupby("country")["case_count"].sum().sort_values(ascending=True)
ax.barh(range(len(country_cases)), country_cases.values, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(country_cases))))
ax.set_yticks(range(len(country_cases))); ax.set_yticklabels(country_cases.index, fontsize=8)
ax.set_xlabel("Total Cases"); ax.set_title("Total Dengue Cases by Country (1993–2022)")
for i, v in enumerate(country_cases.values):
    ax.text(v + country_cases.values.max()*0.01, i, f"{v/1e6:.1f}M", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUTPUT/"eda_trends.png", dpi=120, bbox_inches="tight"); plt.close()

# L-function
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.get_cmap("tab10")
for i, (c, r) in enumerate(eda_results.items()):
    ax.plot(RADII, r["L"], "o-", lw=2, label=f"{c} ({r['interpretation']})", color=cmap(i))
ax.axhline(0, color="gray", ls="--", alpha=0.7, label="CSR (random)")
ax.set_xlabel("r (km)"); ax.set_ylabel("L(r)")
ax.set_title("L-function: Spatial Point Pattern Analysis by Country\nL(r) > 0 = Clustered, L(r) < 0 = Regular, L(r) = 0 = Random")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT/"eda_l_functions.png", dpi=120, bbox_inches="tight"); plt.close()

print(f"  EDA done ({time.time()-t1:.1f}s total)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: CNN-LSTM Model (optimized architecture for CPU)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 3: CNN-LSTM MODEL")
print("=" * 65)

class SpatioTemporalCNN(nn.Module):
    """Optimized CNN-LSTM for CPU training."""
    def __init__(self, in_ch=1, conv_ch=[32, 64, 128], lstm_h=128, lstm_l=2, dropout=0.25, gs=GS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, conv_ch[0], 3, padding=1),
            nn.BatchNorm2d(conv_ch[0]), nn.ReLU(),
            nn.Conv2d(conv_ch[0], conv_ch[1], 3, padding=1),
            nn.BatchNorm2d(conv_ch[1]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_ch[1], conv_ch[2], 3, padding=1),
            nn.BatchNorm2d(conv_ch[2]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        # After AdaptiveAvgPool2d(4): spatial is 4×4
        feat_dim = conv_ch[-1] * 16 + 1
        self.lstm = nn.LSTM(input_size=feat_dim,
                            hidden_size=lstm_h, num_layers=lstm_l,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_h, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, H, W)
        B, T, H, W = x.shape
        spatial = x.mean(dim=(2, 3))          # (B, T)
        feats = []
        for t in range(T):
            h = self.conv(x[:, t:t+1])        # (B, ch2, 4, 4)
            h = h.flatten(1)                   # (B, ch2*16)
            s = spatial[:, t:t+1].expand(h.shape[0], 1)  # (B, 1)
            h = torch.cat([h, s], dim=1)      # (B, ch2*16+1)
            feats.append(h)
        feat_tensor = torch.stack(feats, dim=1)  # (B, T, feat_dim)
        out, _ = self.lstm(feat_tensor)        # (B, T, lstm_h)
        return self.head(out[:, -1]).squeeze(-1)  # (B,)


def _train_model(model, train_ds, val_ds, epochs=25, lr=1e-3, patience=5, label=""):
    """Train CNN-LSTM. Returns trained model and metrics dict."""
    tr_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0, pin_memory=False)
    vl_loader = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in vl_loader:
                Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item() * len(yb)
        val_loss /= len(vl_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      [{label}] Epoch {epoch+1:>2}/{epochs}  train={train_loss:.2f}  val={val_loss:.2f}  best={best_val_loss:.2f}")
        if no_improve >= patience:
            print(f"      [{label}] Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _evaluate(model, X, y):
    """Compute metrics on arrays."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X).contiguous()).numpy()
    y = y.flatten()
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    mae  = np.mean(np.abs(preds - y))
    mape = np.mean(np.abs((y - preds) / (np.abs(y) + 1))) * 100
    r2   = 1 - np.sum((y - preds)**2) / (np.sum((y - np.mean(y))**2) + 1e-10)
    pearson_r = np.corrcoef(preds, y)[0, 1] if len(y) > 1 else 0
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape),
            "R2": float(r2), "Pearson_r": float(pearson_r), "n": len(y)}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 4: AUGMENTATION")
print("=" * 65)

# 4a: SOP augmentation
print("  [SOP] Generating augmented datasets...")
from src.augmentation.sop import sop_augment_spatial_clusters

t_sop = time.time()
train_ev_r = events_df.copy()
train_ev_r["timestamp"] = pd.to_datetime(train_ev_r["timestamp"])

sop_dfs = sop_augment_spatial_clusters(
    train_ev_r, n_augment=2, n_clusters=5, window_months=3, random_state=SEED
)
for df in sop_dfs:
    df["augmented"] = True
    df["aug_method"] = "sop"
sop_combined = pd.concat([train_ev_r] + sop_dfs, ignore_index=True)
print(f"  [SOP] Generated {len(sop_dfs)} augmentations, total={len(sop_combined):,} events ({time.time()-t_sop:.1f}s)")

# 4b: Quantum (statistical) augmentation
print("  [Quantum] Generating synthetic events...")
t_q = time.time()
rng = np.random.default_rng(SEED)
synth_rows = []
for _, row in train_ev_r.sample(n=min(len(train_ev_r) * 2, 60000), random_state=SEED+1, replace=True).iterrows():
    new_ts = row["timestamp"] + pd.Timedelta(days=rng.integers(-30, 30))
    new_cases = max(1, int(row["case_count"] * rng.uniform(0.75, 1.25)))
    synth_rows.append({
        "event_id": 999990 + len(synth_rows),
        "lat": row["lat"] + rng.uniform(-0.4, 0.4),
        "lon": row["lon"] + rng.uniform(-0.4, 0.4),
        "timestamp": new_ts,
        "case_count": new_cases,
        "region": row["region"], "country": row["country"],
        "year": new_ts.year, "month": new_ts.month,
        "augmented": True, "aug_method": "quantum_sim",
    })
synth_df = pd.DataFrame(synth_rows)
q_combined = pd.concat([train_ev_r, synth_df], ignore_index=True)
print(f"  [Quantum] Generated {len(synth_df):,} synthetic events ({time.time()-t_q:.1f}s)")

# Build augmentation grids
print("  Building augmentation grids...")
t_aug = time.time()
sop_grid = _build_grid(sop_combined)
X_sp, y_sp = _sequences(sop_grid)

q_combined["timestamp"] = pd.to_datetime(q_combined["timestamp"])
q_grid = _build_grid(q_combined)
X_q, y_q = _sequences(q_grid)
print(f"  SOP sequences: {len(X_sp)}, Quantum sequences: {len(X_q)} ({time.time()-t_aug:.1f}s)")

# Free memory
del sop_combined, q_combined, synth_df, sop_grid, q_grid
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: PARALLEL MODEL TRAINING (key optimization)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 5: PARALLEL MODEL TRAINING")
print("=" * 65)

from joblib import Parallel, delayed

def make_model_factory():
    """Return a factory so each parallel job gets a fresh model."""
    def factory():
        return SpatioTemporalCNN(in_ch=1, conv_ch=[32, 64, 128],
                                lstm_h=128, lstm_l=2, dropout=0.25, gs=GS)
    return factory

val_ds = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))

def train_single(name, X_train, y_train, val_ds):
    """Train one model configuration."""
    model = SpatioTemporalCNN(in_ch=1, conv_ch=[32, 64, 128],
                             lstm_h=128, lstm_l=2, dropout=0.25, gs=GS)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    model = _train_model(model, train_ds, val_ds, epochs=25, lr=1e-3, patience=5, label=name)
    metrics = _evaluate(model, X_vl, y_vl)
    print(f"  [{name}] Val: RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}")
    return name, model, metrics

# Parallel training — all 3 models simultaneously on 16 CPU threads
print(f"  Training 3 models in parallel using {torch_threads} threads each...")
t_train = time.time()

results_list = Parallel(n_jobs=3, backend="threading")(
    delayed(train_single)(name, X_data, y_data, val_ds)
    for name, X_data, y_data in [
        ("No Augmentation",    X_tr, y_tr),
        ("SOP Augmentation",   X_sp, y_sp),
        ("Quantum Augmentation (sim.)", X_q, y_q),
    ]
)
models_dict = {k: v for k, (_, v, _) in zip(["No Augmentation","SOP Augmentation","Quantum Augmentation (sim.)"], results_list)}
metrics_dict = {k: m for k, (_, _, m) in zip(["No Augmentation","SOP Augmentation","Quantum Augmentation (sim.)"], results_list)}

print(f"  All models trained in {time.time()-t_train:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: SOP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 6: SOP VALIDATION")
print("=" * 65)

from src.augmentation.sop import validate_sop_preservation
t_v = time.time()
sop_v = validate_sop_preservation(train_ev_r, sop_dfs, radii=RADII, n_permutations=9)
k_mae = float(np.nanmean([v for v in sop_v.get("k_function_mae",[]) if v is not None]))
l_mae = float(np.nanmean([v for v in sop_v.get("l_function_mae",[]) if v is not None]))
print(f"  K-function MAE: {k_mae:.2f}")
print(f"  L-function MAE: {l_mae:.2f} km")
print(f"  Wasserstein dist: {np.mean(sop_v.get('case_dist_wasserstein',[])):.1f}")
print(f"  Validation done in {time.time()-t_v:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: EVALUATION + VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 7: EVALUATION")
print("=" * 65)

# Validation results
print("\n  Validation Set Results:")
print("  " + "-" * 62)
print(f"  {'Method':<38} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8}")
print("  " + "-" * 62)
best = ("N/A", float("inf"))
for method, m in metrics_dict.items():
    rmse = m.get("RMSE", 0); mae = m.get("MAE", 0)
    mape = m.get("MAPE", 0); r2   = m.get("R2", 0)
    print(f"  {method:<38} {rmse:>8.2f} {mae:>8.2f} {mape:>7.1f}% {r2:>8.3f}")
    if rmse < best[1]: best = (method, rmse)
print("  " + "-" * 62)
print(f"  Best: {best[0]} (RMSE={best[1]:.2f})")

# Test set (use best model only for speed)
best_method = best[0]
best_model  = models_dict[best_method]
test_metrics = _evaluate(best_model, X_ts, y_ts)
print(f"\n  Test Set ({best_method}):")
print(f"    RMSE={test_metrics['RMSE']:.2f}  MAE={test_metrics['MAE']:.2f}  R2={test_metrics['R2']:.3f}  (n={test_metrics['n']})")

# Prediction vs Actual scatter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (method, model) in zip(axes, models_dict.items()):
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_vl).contiguous()).numpy()
    y_actual = y_vl.flatten()
    ax.scatter(y_actual, preds, alpha=0.3, s=5, color=plt.cm.tab10(list(models_dict.keys()).index(method)))
    ax.plot([0, y_actual.max()], [0, y_actual.max()], "r--", lw=1.5, label="y=x")
    r2 = metrics_dict[method]["R2"]
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{method}\nR²={r2:.3f}, RMSE={metrics_dict[method]['RMSE']:.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT/"pred_vs_actual.png", dpi=120, bbox_inches="tight"); plt.close()

# Metrics comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods = list(metrics_dict.keys())
colors = ["#3498db", "#e74c3c", "#2ecc71"]
x = np.arange(len(methods))

ax = axes[0]
rmse_vals = [metrics_dict[m]["RMSE"] for m in methods]
mae_vals  = [metrics_dict[m]["MAE"]  for m in methods]
bars1 = ax.bar(x - 0.18, rmse_vals, 0.35, label="RMSE", color=[c + "44" for c in colors], edgecolor=colors, lw=2)
bars2 = ax.bar(x + 0.18, mae_vals,  0.35, label="MAE",  color=colors, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels([m.replace(" Augmentation","").replace(" (sim.)","") for m in methods], fontsize=9)
ax.set_ylabel("Error"); ax.set_title("Forecasting Error Comparison (Validation)")
ax.legend()
for bar, val in zip(bars1, rmse_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, mae_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

ax = axes[1]
r2_vals  = [metrics_dict[m]["R2"] for m in methods]
pearson_vals = [metrics_dict[m]["Pearson_r"] for m in methods]
ax.bar(x - 0.18, r2_vals,       0.35, label="R²",        color=[c + "66" for c in colors], edgecolor=colors, lw=2)
ax.bar(x + 0.18, pearson_vals,   0.35, label="Pearson r", color=colors, edgecolor="none")
ax.axhline(0, color="gray", ls="--", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels([m.replace(" Augmentation","").replace(" (sim.)","") for m in methods], fontsize=9)
ax.set_ylabel("Score"); ax.set_title("Goodness-of-Fit (Validation)")
ax.legend()
for bar, val in zip(ax.patches[:3], r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),0)+0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(min(r2_vals)-0.1, max(pearson_vals)+0.1)

plt.tight_layout(); plt.savefig(OUTPUT/"evaluation_comparison.png", dpi=120, bbox_inches="tight"); plt.close()

print(f"  Saved: evaluation_comparison.png, pred_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "validation": {k: v for k, v in metrics_dict.items()},
    "test":       test_metrics,
    "sop_validation": {
        "k_function_mae": k_mae,
        "l_function_mae": l_mae,
        "wasserstein_mean": float(np.mean(sop_v.get("case_dist_wasserstein", []))),
    },
    "quantum": {"qbm_fitted": False, "qgan_fitted": False,
                 "n_synthetic": int(len(synth_rows))},
    "dataset": {"n_events": int(len(events_df)), "n_train": int(len(X_tr)),
                 "n_val": int(len(X_vl)), "n_test": int(len(X_ts)),
                 "grid_size": GS, "seq_len": SEQ_LEN, "forecast_horizon": FH,
                 "total_cases": int(events_df["case_count"].sum()),
                 "n_countries": int(events_df["country"].nunique()),
                 "n_countries": int(events_df["country"].nunique()),
                 "n_regions": int(events_df["region"].nunique())},
    "eda": eda_results,
    "config": {"grid_size": GS, "seq_len": SEQ_LEN, "forecast_horizon": FH,
               "epochs": 25, "lstm_hidden": 128, "conv_channels": [32,64,128],
               "n_workers": N_WORKERS, "torch_threads": torch_threads},
    "timing": {"total_s": round(time.time()-t0, 1)}
}
with open(OUTPUT/"results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

events_df.to_csv(OUTPUT/"all_events.csv", index=False)
synth_df_full = pd.DataFrame(synth_rows)
synth_df_full.to_csv(OUTPUT/"synthetic_events.csv", index=False)

total_time = time.time() - t0
print(f"\n{'='*65}")
print(f"PIPELINE COMPLETE! Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"{'='*65}")
print(f"\nBest method: {best_method}")
print(f"  Val  RMSE: {metrics_dict[best_method]['RMSE']:.2f}")
print(f"  Test RMSE: {test_metrics['RMSE']:.2f}")
