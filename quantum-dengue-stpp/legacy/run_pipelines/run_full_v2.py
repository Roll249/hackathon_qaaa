#!/usr/bin/env python3
"""
Full Pipeline v2 — 5 models + 32x32 grid + Hawkes + NEST.
Maximizes laptop performance with parallel training.
"""
import sys, os, time, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── CPU config ──────────────────────────────────────────────────────────────
CPU_COUNT  = os.cpu_count() or 8
T_THEADS   = min(CPU_COUNT, 8)
os.environ["OMP_NUM_THREADS"] = str(T_THEADS)
os.environ["MKL_NUM_THREADS"]  = str(T_THEADS)

import torch
torch.set_num_threads(T_THEADS)
DEVICE = "cpu"

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT  = Path("outputs"); OUTPUT.mkdir(exist_ok=True)
DATA    = Path("../dengue_dataset")
SEED    = 42
GS      = 32          # grid size: 32×32 (was 20×20)
SEQ_LEN = 8           # lookback months
FH      = 1           # forecast horizon
EPOCHS  = 25
LR      = 1e-3
PATIENCE = 5

np.random.seed(SEED); torch.manual_seed(SEED)
t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STAGE 1: DATA LOADING  (grid: {}x{})".format(GS, GS))
print("=" * 65)

from src.data.loader import load_raw_data, build_stpp_events, temporal_split

spatial_df, long_df, pivot_df = load_raw_data(DATA)
events_df = build_stpp_events(long_df, min_cases=0, remove_sparse=True, zero_threshold=0.95)
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
train_ev, val_ev, test_ev = temporal_split(events_df, 0.70, 0.15, 0.15)

LAT_MIN, LAT_MAX = events_df["lat"].min(), events_df["lat"].max()
LON_MIN, LON_MAX = events_df["lon"].min(), events_df["lon"].max()
T_MIN,   T_MAX   = events_df["timestamp"].min(), events_df["timestamp"].max()
TOTAL_T  = int((T_MAX - T_MIN).days / 30.44) + 1

print(f"  Events: {len(events_df):,} | Cases: {events_df['case_count'].sum():,.0f}")
print(f"  Grid: {GS}×{GS}×{TOTAL_T}  |  Split: train={len(train_ev):,} val={len(val_ev):,} test={len(test_ev):,}")


def _build_grid(ev_df, gs=GS):
    """Build (gs × gs × T) grid from event DataFrame."""
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, gs+1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, gs+1)
    ev = ev_df.copy()
    ev["lat_bin"] = pd.cut(ev["lat"], lat_edges, labels=False, include_lowest=True)
    ev["lon_bin"] = pd.cut(ev["lon"], lon_edges, labels=False, include_lowest=True)
    ev["t_idx"]   = ((ev["timestamp"] - T_MIN).dt.days / 30.44).astype(int).clip(0, TOTAL_T-1)
    grid = np.zeros((gs, gs, TOTAL_T), dtype=np.float32)
    valid = (ev["lat_bin"]>=0)&(ev["lat_bin"]<gs)&(ev["lon_bin"]>=0)&(ev["lon_bin"]<gs)
    for _, row in ev[valid].iterrows():
        grid[int(row["lat_bin"]), int(row["lon_bin"]), int(row["t_idx"])] += row["case_count"]
    return grid


def _sequences(grid, seq_len=SEQ_LEN, fh=FH):
    """Create (X, y) sequences. X: (N, seq_len, H, W), y: (N,) mean count."""
    N = grid.shape[2] - seq_len - fh + 1
    if N <= 0:
        return np.zeros((0, seq_len, grid.shape[0], grid.shape[1]), np.float32), np.zeros(0, np.float32)
    X, y = [], []
    for t in range(N):
        X.append(grid[:, :, t:t+seq_len].transpose(2, 0, 1))
        y.append(grid[:, :, t+seq_len+fh-1].mean())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


print("\n  Building grids...")
tg = time.time()
train_grid = _build_grid(train_ev)
val_grid   = _build_grid(val_ev)
test_grid  = _build_grid(test_ev)
X_tr, y_tr = _sequences(train_grid)
X_vl, y_vl = _sequences(val_grid)
X_ts, y_ts = _sequences(test_grid)
print(f"  Grid built in {time.time()-tg:.1f}s | Sequences: train={len(X_tr)} val={len(X_vl)} test={len(X_ts)}")

# Keep train_ev for augmentation (free val/test)
del train_ev, val_ev, test_ev
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: EDA — spatial statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 2: EDA — SPATIAL STATISTICS")
print("=" * 65)

from src.evaluation.spatial_stats_fast import (
    fast_k_function, fast_l_function, fast_morans_i,
    zero_inflation_ratio, compute_overdispersion
)

RADII = np.array([50.0, 100.0, 200.0, 500.0], np.float32)

COUNTRIES = [
    "VIET NAM","THAILAND","INDONESIA","MALAYSIA","CAMBODIA",
    "SINGAPORE","PHILIPPINES","LAO PEOPLE'S DEM. REPUBLIC"
]
eda_results = {}
for c in COUNTRIES:
    sub = events_df[events_df["country"] == c]
    n = len(sub)
    if n < 5:
        continue
    zi = zero_inflation_ratio(sub["case_count"].values)
    od = compute_overdispersion(sub["case_count"].values)
    I, p = fast_morans_i(sub["lat"].values, sub["lon"].values, sub["case_count"].values, k=5)
    K = fast_k_function(sub["lat"].values, sub["lon"].values, RADII, max_n=300)
    L = fast_l_function(K, RADII)
    interp = "CLUSTERED" if np.mean(L) > 5 else "REGULAR" if np.mean(L) < -5 else "RANDOM"
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    eda_results[c] = {"ZI": float(zi),"OD": float(od),"n": int(n),
                       "morans_I": float(I),"morans_p": float(p),
                       "K": K.tolist(),"L": L.tolist(),"interpretation": interp}
    print(f"    {c:<38} n={n:>5}  ZI={zi:>6.1%}  I={I:>+6.3f}{sig}  L@200km={L[2]:>+7.1f}  -> {interp}")

# EDA plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
cmap = plt.cm.tab10
for i, (c, r) in enumerate(eda_results.items()):
    monthly = events_df[events_df["country"]==c].groupby(
        events_df[events_df["country"]==c]["timestamp"].dt.to_period("M"))["case_count"].sum()
    if len(monthly) > 0:
        ma = pd.Series(monthly.values).rolling(12, min_periods=1).mean()
        ax.plot(range(len(monthly)), ma, lw=2, color=cmap(i / max(len(eda_results)-1, 1)), label=f"{c} ({r['interpretation']})")
ax.set_xlabel("Month index"); ax.set_ylabel("Cases (12-mo MA)")
ax.set_title("Dengue Cases by Country — 12-Month Moving Average")
ax.legend(fontsize=7, loc="upper left", ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
cc = events_df.groupby("country")["case_count"].sum().sort_values(ascending=True)
bars = ax.barh(range(len(cc)), cc.values,
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(cc))))
ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc.index, fontsize=8)
ax.set_xlabel("Total Cases"); ax.set_title("Total Dengue Cases by Country (1993–2022)")
for i, (v, b) in enumerate(zip(cc.values, bars)):
    ax.text(v + cc.values.max()*0.01, i, f"{v/1e6:.1f}M", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUTPUT/"eda_trends.png", dpi=120, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for i, (c, r) in enumerate(eda_results.items()):
    ax.plot(RADII, r["L"], "o-", lw=2, label=f"{c} ({r['interpretation']})", color=cmap(i / max(len(eda_results)-1, 1)))
ax.axhline(0, color="gray", ls="--", alpha=0.7, label="CSR (random)")
ax.set_xlabel("r (km)"); ax.set_ylabel("L(r)")
ax.set_title("L-function: Spatial Point Pattern\nL>0 = Clustered, L<0 = Regular, L=0 = Random")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT/"eda_l_functions.png", dpi=120, bbox_inches="tight"); plt.close()

print(f"  EDA done ({time.time()-t0:.0f}s total)")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 3: MODELS (CNN-LSTM, NEST, Hawkes)")
print("=" * 65)

# ── 3a: CNN-LSTM ─────────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, gs=GS, conv_ch=[32, 64, 128], lstm_h=128, lstm_l=2, dropout=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_ch[0], 3, padding=1), nn.BatchNorm2d(conv_ch[0]), nn.ReLU(),
            nn.Conv2d(conv_ch[0], conv_ch[1], 3, padding=1), nn.BatchNorm2d(conv_ch[1]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_ch[1], conv_ch[2], 3, padding=1), nn.BatchNorm2d(conv_ch[2]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        feat_dim = conv_ch[-1] * 16 + 1
        self.lstm = nn.LSTM(feat_dim, lstm_h, lstm_l, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(lstm_h, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        B, T, H, W = x.shape
        spatial = x.mean(dim=(2, 3))
        feats = []
        for t in range(T):
            h = self.conv(x[:, t:t+1]).flatten(1)
            s = spatial[:, t:t+1].expand(h.shape[0], 1)
            feats.append(torch.cat([h, s], dim=1))
        out, _ = self.lstm(torch.stack(feats, dim=1))
        return self.head(out[:, -1]).squeeze(-1)


def _train_cnn_lstm(name, X_train, y_train, val_ds, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """Train CNN-LSTM. Returns (name, model, metrics)."""
    model = CNNLSTM(gs=GS).to(DEVICE)
    tr_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0)
    vl_ld = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.MSELoss()
    best_loss = float("inf"); best_state = None; ni = 0
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(Xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl_loss = 0
        with torch.no_grad():
            for Xb, yb in vl_ld:
                Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
                vl_loss += crit(model(Xb), yb).item() * len(yb)
        vl_loss /= len(vl_ld.dataset); sch.step(vl_loss)
        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
        if (ep+1) % 5 == 0:
            print(f"      [{name}] Ep {ep+1:>2}/{epochs}  val={vl_loss:.2f}  best={best_loss:.2f}")
        if ni >= patience:
            print(f"      [{name}] Early stop ep {ep+1}")
            break
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return name, model


# ── 3b: NEST ───────────────────────────────────────────────────────────────
class NESTSpatial(nn.Module):
    """NEST: CNN encoder → LSTM → scalar mean-count prediction."""
    def __init__(self, gs=GS, hidden=64, t_hid=64, t_lay=2, dropout=0.2):
        super().__init__()
        self.gs = gs
        self.enc = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 2, padding=0), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        enc_dim = hidden * 16
        self.lstm = nn.LSTM(enc_dim, t_hid, t_lay, batch_first=True, dropout=dropout if t_lay > 1 else 0)
        self.head = nn.Sequential(nn.Linear(t_hid, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        B, T, H, W = x.shape
        embs = [self.enc(x[:, t:t+1]).flatten(1) for t in range(T)]
        _, (h, _) = self.lstm(torch.stack(embs, dim=1))
        return self.head(h[-1]).squeeze(-1)


def _train_nest(name, X_train, y_train, val_ds, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """Train NEST (scalar output). Returns (name, model, metrics)."""
    model = NESTSpatial(gs=GS).to(DEVICE)
    tr_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0)
    vl_ld = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.MSELoss()
    best_loss = float("inf"); best_state = None; ni = 0
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr_ld:
            Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(Xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl_loss = 0
        with torch.no_grad():
            for Xb, yb in vl_ld:
                Xb, yb = Xb.contiguous().to(DEVICE), yb.to(DEVICE)
                vl_loss += crit(model(Xb), yb).item() * len(yb)
        vl_loss /= len(vl_ld.dataset); sch.step(vl_loss)
        if vl_loss < best_loss:
            best_loss = vl_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
        if (ep+1) % 5 == 0:
            print(f"      [{name}] Ep {ep+1:>2}/{epochs}  val={vl_loss:.2f}  best={best_loss:.2f}")
        if ni >= patience:
            print(f"      [{name}] Early stop ep {ep+1}"); break
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return name, model


from src.models.hawkes import HawkesBaseline


def _evaluate(name, model_or_preds, X, y, model_type="cnn_lstm"):
    """Compute metrics. model_type: 'cnn_lstm' | 'nest' | 'hawkes'"""
    if model_type == "hawkes":
        preds = model_or_preds  # dict
    else:
        model = model_or_preds
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X).contiguous().to(DEVICE)).cpu().numpy().flatten()
    y_flat = y.flatten()
    rmse = np.sqrt(np.mean((preds - y_flat)**2))
    mae  = np.mean(np.abs(preds - y_flat))
    mape = np.mean(np.abs((y_flat - preds) / (np.abs(y_flat) + 1))) * 100
    r2   = 1 - np.sum((y_flat - preds)**2) / (np.sum((y_flat - np.mean(y_flat))**2) + 1e-10)
    pr   = float(np.corrcoef(preds, y_flat)[0, 1]) if (len(y_flat) > 1 and not np.any(np.isnan(preds))) else float("nan")
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape),
            "R2": float(r2), "Pearson_r": float(pr), "n": int(len(y_flat))}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 4: AUGMENTATION (SOP + Quantum)")
print("=" * 65)

from src.augmentation.sop import sop_augment_spatial_clusters

# Reload train events
train_ev_r = events_df.copy()
train_ev_r["timestamp"] = pd.to_datetime(train_ev_r["timestamp"])

# SOP
t_sop = time.time()
sop_dfs = sop_augment_spatial_clusters(train_ev_r, n_augment=2, n_clusters=5, window_months=3, random_state=SEED)
for df in sop_dfs:
    df["augmented"] = True; df["aug_method"] = "sop"
sop_combined = pd.concat([train_ev_r] + sop_dfs, ignore_index=True)
print(f"  [SOP] {len(sop_dfs)} augs, total={len(sop_combined):,} ({time.time()-t_sop:.1f}s)")

# Quantum (statistical resampling)
t_q = time.time()
rng = np.random.default_rng(SEED)
synth_rows = []
for _, row in train_ev_r.sample(n=min(len(train_ev_r)*2, 60000), random_state=SEED+1, replace=True).iterrows():
    new_ts = row["timestamp"] + pd.Timedelta(days=rng.integers(-30, 30))
    synth_rows.append({
        "event_id": 999990 + len(synth_rows),
        "lat": row["lat"] + rng.uniform(-0.4, 0.4),
        "lon": row["lon"] + rng.uniform(-0.4, 0.4),
        "timestamp": new_ts,
        "case_count": max(1, int(row["case_count"] * rng.uniform(0.75, 1.25))),
        "region": row["region"], "country": row["country"],
        "year": new_ts.year, "month": new_ts.month,
        "augmented": True, "aug_method": "quantum_sim",
    })
synth_df = pd.DataFrame(synth_rows)
q_combined = pd.concat([train_ev_r, synth_df], ignore_index=True)
print(f"  [Quantum] {len(synth_df):,} synth events ({time.time()-t_q:.1f}s)")

# Build grids for augmented data
print("  Building augmentation grids...")
t_ag = time.time()
sop_grid = _build_grid(sop_combined)
X_sp, y_sp = _sequences(sop_grid)
q_combined["timestamp"] = pd.to_datetime(q_combined["timestamp"])
q_grid = _build_grid(q_combined)
X_q, y_q = _sequences(q_grid)
print(f"  SOP: {len(X_sp)} seqs, Quantum: {len(X_q)} seqs ({time.time()-t_ag:.1f}s)")
del sop_combined, q_combined, synth_df, sop_grid, q_grid; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: PARALLEL TRAINING — 5 models simultaneously
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 5: PARALLEL TRAINING (5 models × 3 configs = 5 training jobs)")
print("=" * 65)

from joblib import Parallel, delayed

val_ds = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))
models_dict, metrics_dict = {}, {}

# ── Training jobs ─────────────────────────────────────────────────────────
def cnn_job(name, X_data, y_data):
    n, m = _train_cnn_lstm(name, X_data, y_data, val_ds)
    metrics = _evaluate(n, m, X_vl, y_vl, "cnn_lstm")
    return n, m, metrics

def nest_job(name, X_data, y_data):
    n, m = _train_nest(name, X_data, y_data, val_ds)
    metrics = _evaluate(n, m, X_vl, y_vl, "cnn_lstm")
    return n, m, metrics

# ── Hawkes (no grid needed) ────────────────────────────────────────────────
print("  [Hawkes] Fitting on train events...")
t_h = time.time()
hawkes = HawkesBaseline()
hawkes.fit(train_ev_r)

# Hawkes predicts a scalar (stationary mean per country averaged)
# For fair evaluation: use Hawkes stationary mean as constant prediction
# and compute metrics against actual grid-level means
country_preds = hawkes.predict()
hawkes_scalar = float(np.mean(list(country_preds.values())))
hawkes_preds = np.full(len(y_vl), hawkes_scalar)

# Evaluate: constant prediction vs actual
# Note: country-level scalar (~2000) vs grid-level cell mean (~10) — R² will be very poor
y_flat = y_vl.flatten()
rmse_h = float(np.sqrt(np.mean((hawkes_preds - y_flat)**2)))
mae_h  = float(np.mean(np.abs(hawkes_preds - y_flat)))
mape_h = float(np.mean(np.abs((y_flat - hawkes_preds) / (np.abs(y_flat) + 1))) * 100)
r2_h   = float("nan")   # N/A: cross-scale (country vs cell)
pr_h   = float("nan")   # N/A

hawkes_metrics = {"RMSE": rmse_h, "MAE": mae_h, "MAPE": mape_h,
                   "R2": r2_h, "Pearson_r": pr_h, "n": len(y_flat),
                   "note": "cross-scale: country-level scalar vs grid cell mean"}
print(f"  [Hawkes] pred={hawkes_scalar:.0f}  RMSE={rmse_h:.2f}  note: cross-scale ({time.time()-t_h:.1f}s)")
models_dict["Hawkes Process"] = hawkes_preds
metrics_dict["Hawkes Process"] = hawkes_metrics

# ── CNN-LSTM + NEST (parallel) ─────────────────────────────────────────────
print(f"\n  Training 4 deep models in parallel ({T_THEADS} threads each)...")
t_tr = time.time()

# CNN-LSTM jobs (3 configs)
cnn_jobs = [
    ("CNN-LSTM (No Aug)",    X_tr, y_tr),
    ("CNN-LSTM + SOP",       X_sp, y_sp),
    ("CNN-LSTM + Quantum",   X_q,  y_q),
]
# NEST jobs (2 configs)
nest_jobs = [
    ("NEST (No Aug)",        X_tr, y_tr),
    ("NEST + SOP",           X_sp, y_sp),
]

all_jobs = (
    [("CNN-LSTM (No Aug)",  "cnn", X_tr, y_tr),
     ("CNN-LSTM + SOP",     "cnn", X_sp, y_sp),
     ("CNN-LSTM + Quantum", "cnn", X_q,  y_q),
     ("NEST (No Aug)",      "nest", X_tr, y_tr),
     ("NEST + SOP",         "nest", X_sp, y_sp),
    ]
)

def run_job(name, jtype, X_data, y_data):
    if jtype == "cnn":
        n, m, met = cnn_job(name, X_data, y_data)
    else:
        n, m, met = nest_job(name, X_data, y_data)
    return (n, m, met)

results = Parallel(n_jobs=min(5, CPU_COUNT), backend="threading")(
    delayed(run_job)(name, jtype, X_data, y_data)
    for name, jtype, X_data, y_data in all_jobs
)

for name, model, m in results:
    models_dict[name] = model
    metrics_dict[name] = m
    print(f"  [{name}] Val: RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R2={m['R2']:.3f}")

print(f"  All models trained in {time.time()-t_tr:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: SOP VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 6: SOP VALIDATION")
print("=" * 65)

from src.augmentation.sop import validate_sop_preservation
t_v = time.time()
sop_v = validate_sop_preservation(train_ev_r, sop_dfs, radii=RADII, n_permutations=9)
k_mae = float(np.nanmean([v for v in sop_v.get("k_function_mae", []) if v is not None]))
l_mae = float(np.nanmean([v for v in sop_v.get("l_function_mae", []) if v is not None]))
w_dist = float(np.nanmean(sop_v.get("case_dist_wasserstein", [])))
print(f"  K-MAE: {k_mae:.0f}  L-MAE: {l_mae:.2f} km  Wasserstein: {w_dist:.1f}")
print(f"  Done in {time.time()-t_v:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 7: EVALUATION")
print("=" * 65)

# Print results table
print(f"\n  {'Method':<30} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R2':>8} {'Pearson':>8}")
print(f"  {'-'*76}")
best = ("N/A", float("inf"))
for name in ["Hawkes Process","CNN-LSTM (No Aug)","CNN-LSTM + SOP","CNN-LSTM + Quantum",
             "NEST (No Aug)","NEST + SOP"]:
    m = metrics_dict.get(name, {})
    rmse = m.get("RMSE", float("nan"))
    mae  = m.get("MAE", float("nan"))
    mape = m.get("MAPE", float("nan"))
    r2   = m.get("R2", float("nan"))
    pr   = m.get("Pearson_r", float("nan"))
    print(f"  {name:<30} {rmse:>8.2f} {mae:>8.2f} {mape:>7.1f}% {r2:>8.3f} {pr:>8.3f}")
    if rmse < best[1]: best = (name, rmse)
print(f"  {'-'*76}")
print(f"  Best: {best[0]} (RMSE={best[1]:.2f})")

# Test set — best model only
best_name = best[0]
best_model = models_dict[best_name]
if best_name == "Hawkes Process":
    test_preds = best_model  # already scalar predictions
    test_m = _evaluate(best_name, test_preds, y_ts, y_ts, "hawkes")
else:
    if best_name.startswith("NEST"):
        test_m = _evaluate(best_name, best_model, X_ts, y_ts, "cnn_lstm")
    else:
        test_m = _evaluate(best_name, best_model, X_ts, y_ts, "cnn_lstm")

print(f"\n  Test Set ({best_name}):")
print(f"    RMSE={test_m['RMSE']:.2f}  MAE={test_m['MAE']:.2f}  R2={test_m['R2']:.3f}  (n={test_m['n']})")

# ── Plots ──────────────────────────────────────────────────────────────────
# Metrics comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
names_order = ["Hawkes Process","CNN-LSTM (No Aug)","CNN-LSTM + SOP","CNN-LSTM + Quantum","NEST (No Aug)","NEST + SOP"]
valid_names = [n for n in names_order if n in metrics_dict]
cmap_vals = plt.cm.tab10(np.linspace(0, 0.9, len(valid_names)))
x = np.arange(len(valid_names))

ax = axes[0]
rmse_v = [metrics_dict[n]["RMSE"] for n in valid_names]
mae_v  = [metrics_dict[n]["MAE"]  for n in valid_names]
fill_cols = [(c[0], c[1], c[2], 0.45) for c in cmap_vals]
edge_cols = cmap_vals
ax.bar(x - 0.18, rmse_v, 0.35, label="RMSE", color=fill_cols, edgecolor=edge_cols, lw=1.5)
ax.bar(x + 0.18, mae_v,  0.35, label="MAE",  color=cmap_vals, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (","\n(") for n in valid_names], fontsize=8)
ax.set_ylabel("Error"); ax.set_title("Forecasting Error (Validation)")
ax.legend()
for bar, val in zip(ax.patches[:len(valid_names)], rmse_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

ax = axes[1]
r2_v  = [metrics_dict[n]["R2"]  for n in valid_names]
pr_v  = [metrics_dict[n]["Pearson_r"] for n in valid_names]
r2_v  = [float("nan") if np.isnan(v) else v for v in r2_v]
pr_v  = [float("nan") if np.isnan(v) else v for v in pr_v]
ax.bar(x - 0.18, r2_v, 0.35, label="R²",        color=fill_cols, edgecolor=edge_cols, lw=1.5)
ax.axhline(0, color="gray", ls="--", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (","\n(") for n in valid_names], fontsize=8)
ax.set_ylabel("Score"); ax.set_title("Goodness-of-Fit (Validation)")
ax.set_ylim(min(min([v for v in r2_v if np.isfinite(v)] + [0]), 0) - 0.1,
              max([v for v in pr_v if np.isfinite(v)] + [0]) + 0.1)
plt.tight_layout(); plt.savefig(OUTPUT/"evaluation_comparison.png", dpi=120, bbox_inches="tight"); plt.close()

# Predicted vs Actual (deep models only)
deep_names = [n for n in valid_names if n != "Hawkes Process"]
fig, axes = plt.subplots(1, len(deep_names), figsize=(5*len(deep_names), 4))
if len(deep_names) == 1: axes = [axes]
for ax, name in zip(axes, deep_names):
    model = models_dict[name]
    if name.startswith("NEST"):
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_vl).contiguous().to(DEVICE)).cpu().numpy().flatten()
    else:
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_vl).contiguous().to(DEVICE)).cpu().numpy().flatten()
    y_actual = y_vl.flatten()
    ax.scatter(y_actual, preds, alpha=0.3, s=8, color="steelblue")
    ax.plot([0, y_actual.max()], [0, y_actual.max()], "r--", lw=1.5, label="y=x")
    r2 = metrics_dict[name]["R2"]
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{name}\nR²={r2:.3f}  RMSE={metrics_dict[name]['RMSE']:.2f}")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT/"pred_vs_actual.png", dpi=120, bbox_inches="tight"); plt.close()

print(f"\n  Saved: evaluation_comparison.png, pred_vs_actual.png")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "validation": {k: {kk: (float(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else str(vv))
                      for kk, vv in v.items()} for k, v in metrics_dict.items()},
    "test":       {k: (float(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else str(vv))
                   for k, vv in test_m.items()},
    "best_method": best_name,
    "sop_validation": {"k_function_mae": k_mae, "l_function_mae": l_mae, "wasserstein": w_dist},
    "quantum": {"qbm_fitted": False, "qgan_fitted": False, "n_synthetic": int(len(synth_rows))},
    "dataset": {"n_events": int(len(events_df)), "total_cases": int(events_df["case_count"].sum()),
                 "n_countries": int(events_df["country"].nunique()),
                 "n_regions": int(events_df["region"].nunique()),
                 "grid_size": GS, "seq_len": SEQ_LEN, "forecast_horizon": FH,
                 "n_train_seq": int(len(X_tr)), "n_val_seq": int(len(X_vl)), "n_test_seq": int(len(X_ts))},
    "eda": eda_results,
    "config": {"grid_size": GS, "seq_len": SEQ_LEN, "epochs": EPOCHS,
               "lstm_hidden": 128, "conv_channels": [32, 64, 128],
               "nest_hidden": 64, "nest_temporal_hidden": 64,
               "torch_threads": T_THEADS},
    "timing": {"total_s": round(time.time()-t0, 1)}
}
with open(OUTPUT/"results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

synth_df_full = pd.DataFrame(synth_rows)
synth_df_full.to_csv(OUTPUT/"synthetic_events.csv", index=False)

total_time = time.time() - t0
print(f"\n{'='*65}")
print(f"PIPELINE COMPLETE! Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"{'='*65}")
print(f"\n  Best: {best_name}  (Val RMSE={best[1]:.2f})")
print(f"  Test: RMSE={test_m['RMSE']:.2f}  R2={test_m['R2']:.3f}")
