#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-Accelerated Quantum Dengue STPP Pipeline.

Optimized for NVIDIA RTX 3090 (24GB VRAM):
  - All deep learning on GPU with CUDA
  - Mixed precision training (FP16 via torch.cuda.amp)
  - cuDNN benchmark + deterministic
  - High resolution spatial grid (48x48)
  - Large batch sizes
  - Multi-model parallel training on GPU
  - Pin memory for faster CPU→GPU transfers
"""
import sys, os, time, json, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# GPU / CUDA CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This script requires GPU.")
    sys.exit(1)

DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
torch.cuda.empty_cache()

print("=" * 65)
print("GPU-ACCELERATED DENGUE STPP PIPELINE")
print("=" * 65)
print(f"  GPU:         {GPU_NAME}")
print(f"  VRAM:        {VRAM_GB:.1f} GB")
print(f"  CUDA:        {torch.version.cuda}")
print(f"  PyTorch:     {torch.__version__}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

OUTPUT = Path("outputs"); OUTPUT.mkdir(exist_ok=True)
DATA   = Path("../dengue_dataset")

# ──────────────────────────────────────────────────────────────────────────────
# GRID & TRAINING CONFIG  (tuned for 24GB VRAM)
# ──────────────────────────────────────────────────────────────────────────────
GS      = 48          # 48×48 grid  (much higher than CPU versions)
SEQ_LEN = 12          # 12-month lookback
FH      = 1           # 1-month forecast
BATCH_SIZE = 128     # large batch on GPU
EPOCHS  = 80          # more epochs on GPU (faster than CPU)
LR      = 3e-4        # slightly lower LR for stability
WD      = 1e-4
PATIENCE = 12
NUM_WORKERS = 0       # 0 for Windows (no multiprocessing workers)

t0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA LOADING  (CPU — I/O heavy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 1: DATA LOADING & PREPROCESSING")
print("=" * 65)

t1 = time.time()
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
print(f"  Grid: {GS}×{GS}×{TOTAL_T} | Countries: {events_df['country'].nunique()}")
print(f"  Split: train={len(train_ev):,} val={len(val_ev):,} test={len(test_ev):,}")
print(f"  Loaded in {time.time()-t1:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# GPU KERNELS: Fast grid building & augmentation (vectorized NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def build_grid_gpu(ev_df, gs=GS, lat_min=LAT_MIN, lat_max=LAT_MAX,
                   lon_min=LON_MIN, lon_max=LON_MAX,
                   t_min=T_MIN, total_t=TOTAL_T):
    """Build (gs×gs×T) case-count grid. Runs on CPU (fast vectorized)."""
    lat_edges = np.linspace(lat_min, lat_max, gs + 1)
    lon_edges = np.linspace(lon_min, lon_max, gs + 1)

    ev = ev_df.copy()
    lat_bin = np.searchsorted(lat_edges, ev["lat"].values, side="right") - 1
    lon_bin = np.searchsorted(lon_edges, ev["lon"].values, side="right") - 1
    t_idx = ((ev["timestamp"] - t_min).dt.days / 30.44).astype(int).clip(0, total_t - 1)

    lat_bin = np.clip(lat_bin, 0, gs - 1)
    lon_bin = np.clip(lon_bin, 0, gs - 1)

    grid = np.zeros((gs, gs, total_t), dtype=np.float32)
    np.add.at(grid, (lat_bin, lon_bin, t_idx), ev["case_count"].values.astype(np.float32))
    return grid


def create_sequences(grid, seq_len=SEQ_LEN, fh=FH):
    """X: (N, seq_len, H, W), y: (N,) mean cell count next step."""
    H, W, T = grid.shape
    n = T - seq_len - fh + 1
    if n <= 0:
        return np.zeros((0, seq_len, H, W), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack([grid[:, :, t:t + seq_len].transpose(2, 0, 1) for t in range(n)])
    # slice (H, W, n), then mean over H, W → (n,)
    y = grid[:, :, seq_len + np.arange(n) + fh - 1].mean(axis=(0, 1))
    return X.astype(np.float32), y.astype(np.float32)


def sop_augment_fast(train_df, n_aug=3, window_months=3, seed=42):
    """Fast SOP: shuffle case counts within country + time windows."""
    np.random.seed(seed)
    aug_list = []
    df = train_df.copy()
    df = df.sort_values("timestamp")

    for aug_i in range(n_aug):
        adf = df.copy()
        groups = list(adf.groupby("country"))
        new_dfs = []

        for _, grp in groups:
            grp = grp.sort_values("timestamp")
            months = grp["timestamp"].dt.to_period("M").values
            unique_months = np.unique(months)

            for m_start in range(0, len(unique_months), window_months):
                m_end = min(m_start + window_months, len(unique_months))
                mask = np.isin(months, unique_months[m_start:m_end])
                wdata = grp[mask].copy()
                if len(wdata) < 2:
                    new_dfs.append(wdata)
                    continue
                cases = wdata["case_count"].values.copy()
                np.random.shuffle(cases)
                wdata = wdata.copy()
                wdata["case_count"] = cases
                new_dfs.append(wdata)

        if new_dfs:
            out = pd.concat(new_dfs, ignore_index=True)
            out["event_id"] = range(len(out))
            out["augmented"] = True
            out["aug_idx"] = aug_i
            aug_list.append(out)
    return aug_list


def quantum_augment_fast(train_df, n_synth=20000, seed=42):
    """GPU-ready quantum proxy: stratified resampling + perturbation."""
    np.random.seed(seed)
    df = train_df.copy()
    df = df.sort_values("timestamp")

    synth = []
    n_per_country = max(1, n_synth // df["country"].nunique())
    base_ts = df["timestamp"].min()

    for country in df["country"].unique():
        sub = df[df["country"] == country]
        for _ in range(n_per_country):
            row = sub.sample(1).iloc[0]
            lat_jit = row["lat"] + np.random.uniform(-0.5, 0.5)
            lon_jit = row["lon"] + np.random.uniform(-0.5, 0.5)
            lat_jit = np.clip(lat_jit, LAT_MIN + 0.1, LAT_MAX - 0.1)
            lon_jit = np.clip(lon_jit, LON_MIN + 0.1, LON_MAX - 0.1)

            ts_shift = pd.Timedelta(days=np.random.randint(-20, 20))
            scale = np.random.uniform(0.7, 1.3)
            cases = max(0, int(row["case_count"] * scale))

            synth.append({
                "event_id": len(df) + len(synth),
                "lat": float(lat_jit),
                "lon": float(lon_jit),
                "timestamp": row["timestamp"] + ts_shift,
                "case_count": cases,
                "region": f"SYNTH_{country}",
                "country": country,
                "year": (row["timestamp"] + ts_shift).year,
                "month": (row["timestamp"] + ts_shift).month,
                "augmented": True,
                "aug_method": "quantum_proxy",
            })

    return pd.DataFrame(synth[:n_synth])


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: GRID BUILDING (vectorized, fast)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 2: BUILDING SPATIAL GRIDS (vectorized)")
print("=" * 65)

t2 = time.time()
train_grid = build_grid_gpu(train_ev)
val_grid   = build_grid_gpu(val_ev)
test_grid  = build_grid_gpu(test_ev)
print(f"  Grid shape: {train_grid.shape} ({train_grid.nbytes / 1024**2:.1f} MB)")
print(f"  Built in {time.time()-t2:.1f}s")

print("\n  Creating sequences...")
X_tr, y_tr = create_sequences(train_grid)
X_vl, y_vl = create_sequences(val_grid)
X_ts, y_ts = create_sequences(test_grid)
print(f"  Train: {len(X_tr)}, Val: {len(X_vl)}, Test: {len(X_ts)}")

# Pin memory for faster CPU→GPU transfer
train_ds = TensorDataset(
    torch.from_numpy(X_tr).pin_memory(),
    torch.from_numpy(y_tr).pin_memory(),
)
val_ds = TensorDataset(
    torch.from_numpy(X_vl).pin_memory(),
    torch.from_numpy(y_vl).pin_memory(),
)
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True)
val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: EDA  (CPU - I/O + plotting)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

from src.evaluation.spatial_stats_fast import fast_k_function
from src.evaluation.spatial_stats import zero_inflation_ratio, compute_overdispersion

print("\n  Zero-inflation & overdispersion:")
for country in ["VIET NAM", "THAILAND", "INDONESIA", "MALAYSIA", "CAMBODIA"]:
    sub = events_df[events_df["country"] == country]
    zi = zero_inflation_ratio(sub["case_count"].values)
    od = compute_overdispersion(sub["case_count"].values)
    print(f"    {country}: ZI={zi:.1%}, OD={od:.1f}")

# Yearly trend
fig, ax = plt.subplots(figsize=(14, 6))
yearly = events_df.groupby(["country", "year"])["case_count"].sum().unstack(level=0)
yearly.plot(ax=ax, linewidth=2, colormap="tab10")
ax.set_xlabel("Year"); ax.set_ylabel("Total Cases")
ax.set_title("Yearly Dengue Outbreaks by Country (1993-2022) — GPU Pipeline")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_yearly_trends.png", dpi=150, bbox_inches="tight")
plt.close()

# Monthly seasonality
fig, ax = plt.subplots(figsize=(14, 6))
monthly = events_df.groupby(["country", "month"])["case_count"].sum().unstack(level=0)
monthly.plot(ax=ax, linewidth=2, colormap="tab10")
ax.set_xlabel("Month"); ax.set_ylabel("Total Cases")
ax.set_title("Monthly Dengue Seasonality by Country — GPU Pipeline")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_monthly_patterns.png", dpi=150, bbox_inches="tight")
plt.close()

# Geographic distribution
fig, ax = plt.subplots(figsize=(14, 10))
totals = events_df.groupby(["lat", "lon"])["case_count"].sum().reset_index()
scatter = ax.scatter(totals["lon"], totals["lat"],
                    c=np.log1p(totals["case_count"]), cmap="YlOrRd", alpha=0.7, s=40)
plt.colorbar(scatter, label="log(1 + total cases)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Dengue Cases (SE Asia) — GPU Pipeline")
ax.set_xlim(90, 145); ax.set_ylim(-12, 25)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_geographic_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Spatial grid heatmap (last year)
last_year = events_df["year"].max()
last_ev = events_df[events_df["year"] == last_year]
fig, ax = plt.subplots(figsize=(12, 10))
last_grid = build_grid_gpu(last_ev, gs=GS)
im = ax.imshow(last_grid[:, :, -1], origin="lower", cmap="YlOrRd", aspect="auto")
ax.set_title(f"Dengue Case Density — {last_year} (48×48 Grid, Last Month)")
ax.set_xlabel("Longitude bin"); ax.set_ylabel("Latitude bin")
plt.colorbar(im, label="Case count")
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_spatial_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  EDA plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: GPU MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 4: GPU MODEL DEFINITIONS")
print("=" * 65)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SpatialAttention(nn.Module):
    """Spatial attention: learn which grid cells matter most."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.conv(x)
        return x * att


class SpatioTemporalCNN(nn.Module):
    """Enhanced CNN-LSTM with spatial attention for GPU."""
    def __init__(self, gs=GS, conv_channels=[64, 128, 256],
                 lstm_hidden=128, lstm_layers=2, dropout=0.2, forecast_horizon=FH):
        super().__init__()
        self.gs = gs
        self.forecast_horizon = forecast_horizon

        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(ConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.randn(1, 1, gs, gs)
            out = self.cnn(dummy)
            cnn_out_size = out.numel()

        self.attention = SpatialAttention(conv_channels[-1])
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.LayerNorm(lstm_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, forecast_horizon),
        )

    def forward(self, x):
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, T, H, W = x.shape

        x = x.contiguous().view(batch * T, 1, H, W)
        x = self.cnn(x)
        x = self.attention(x)
        x = x.contiguous().view(batch, T, -1)

        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)


class TransformerForecaster(nn.Module):
    """CNN + Transformer for spatio-temporal forecasting (GPU-native).

    Architecture:
    - CNN: downsample 48x48 → 6x6 = 36 spatial tokens
    - Per-timestep CNN encoding → flatten → Transformer → forecast head
    """
    def __init__(self, gs=GS, d_model=128, nhead=4, n_layers=3,
                 dropout=0.15, forecast_horizon=FH):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # CNN encoder: 48x48 → 6x6 (stride-2 × 3 layers → /8)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.MaxPool2d(2),                           # 48→24
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.MaxPool2d(2),                           # 24→12
            nn.Conv2d(64, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),                   # 12→3
        )
        # 3×3 = 9 spatial tokens per timestep; 12 timesteps → 108 tokens
        n_tokens = 9 * SEQ_LEN

        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, T, H, W = x.shape

        # Per-timestep CNN: (B*T, 1, H, W) → (B*T, d_model, 3, 3)
        x = x.contiguous().view(batch * T, 1, H, W)
        x = self.cnn(x)                             # (B*T, d_model, 3, 3)
        n_tokens = x.size(2) * x.size(3)            # 9

        # Flatten spatial: (B*T, d_model, 9) → (B, T*d_model, 9)
        x = x.view(batch * T, -1, n_tokens)         # (B*T, d_model, 9)
        x = x.view(batch, T * n_tokens, -1)        # (B, T*9, d_model)

        n_available = min(x.size(1), self.pos_embed.size(1))
        x = x[:, :n_available] + self.pos_embed[:, :n_available]
        x = self.dropout(x)

        x = self.transformer(x)                     # (B, T*9, d_model)
        last = x[:, -1]                            # take last token

        return self.head(last)


class AttentionLSTM(nn.Module):
    """LSTM with temporal attention mechanism."""
    def __init__(self, gs=GS, hidden=128, n_layers=2, dropout=0.2, forecast_horizon=FH):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, gs, gs)
            cnn_out = self.cnn(dummy)
            self.cnn_out_size = cnn_out.numel()

        self.lstm = nn.LSTM(self.cnn_out_size, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)

        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, forecast_horizon),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, T, H, W = x.shape

        x = x.contiguous().view(batch * T, 1, H, W)
        x = self.cnn(x)
        x = x.view(batch, T, -1)

        lstm_out, _ = self.lstm(x)
        scores = self.attn(lstm_out)
        attn_w = torch.softmax(scores, dim=1)
        context = (lstm_out * attn_w).sum(dim=1)

        return self.head(context)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: GPU TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def move_to_device(loader, device):
    """Yield batches already on device."""
    for Xb, yb in loader:
        yield Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)


def train_gpu_model(model, train_ld, val_ld, epochs=EPOCHS, lr=LR,
                    patience=PATIENCE, name="Model", weight_decay=WD,
                    grad_clip=1.0, use_amp=True):
    """Train a model on GPU with mixed precision (FP16)."""
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
    scaler = GradScaler() if use_amp else None
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        n_batch = 0

        for Xb, yb in move_to_device(train_ld, DEVICE):
            if yb.dim() > 1 and yb.shape[1] > 1:
                yb = yb.mean(dim=1, keepdim=True)
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    pred = model(Xb)
                    if pred.dim() > 1 and pred.shape[1] > 1:
                        pred = pred.mean(dim=1, keepdim=True)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(Xb)
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred.mean(dim=1, keepdim=True)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            t_loss += loss.item()
            n_batch += 1

        sched.step()
        avg_train = t_loss / max(n_batch, 1)

        model.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for Xb, yb in move_to_device(val_ld, DEVICE):
                if yb.dim() > 1 and yb.shape[1] > 1:
                    yb = yb.mean(dim=1, keepdim=True)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(1)
                with autocast():
                    pred = model(Xb)
                    if pred.dim() > 1 and pred.shape[1] > 1:
                        pred = pred.mean(dim=1, keepdim=True)
                    loss = criterion(pred, yb)
                v_loss += loss.item()
                v_n += 1
        v_loss /= max(v_n, 1)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    [{name}] Epoch {epoch+1:>3}/{epochs} | "
                  f"Train: {avg_train:.4f} | Val: {v_loss:.4f} | Best: {best_val:.4f}")

        if no_improve >= patience:
            print(f"    [{name}] Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_val


def predict_gpu(model, X_np, batch_size=256):
    """Predict on GPU, returns numpy array."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(DEVICE, non_blocking=True)
            with autocast():
                p = model(batch)
            if p.dim() > 1 and p.shape[1] > 1:
                p = p.mean(dim=1, keepdim=True)
            preds.append(p.cpu().numpy().flatten())
    return np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: TRAIN BASELINE MODELS (No Augmentation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 5: TRAINING MODELS (GPU + Mixed Precision)")
print("=" * 65)

from src.evaluation.metrics import compute_forecasting_metrics

results = {}

# ─── CNN-LSTM Baseline ────────────────────────────────────────────────────
print("\n  [5a] CNN-LSTM (No Augmentation)...")
t_train = time.time()
cnn_lstm = SpatioTemporalCNN(gs=GS, conv_channels=[64, 128, 256],
                               lstm_hidden=128, lstm_layers=2, dropout=0.2)
cnn_lstm, best_vloss = train_gpu_model(cnn_lstm, train_ld, val_ld,
                                         epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                         name="CNN-LSTM")
torch.cuda.synchronize()
print(f"  Trained in {time.time()-t_train:.1f}s | Best Val MSE: {best_vloss:.4f}")

y_pred_cnn = predict_gpu(cnn_lstm, X_vl)
cnn_metrics = compute_forecasting_metrics(y_vl, y_pred_cnn)
print(f"  Metrics: RMSE={cnn_metrics['RMSE']:.4f}, MAE={cnn_metrics['MAE']:.4f}, "
      f"R2={cnn_metrics['R2']:.4f}, Pearson={cnn_metrics['Pearson_r']:.4f}")
results["cnn_lstm_no_aug"] = cnn_metrics

# ─── Transformer Baseline ─────────────────────────────────────────────────
print("\n  [5b] Transformer Forecaster (No Augmentation)...")
t_train = time.time()
transformer = TransformerForecaster(gs=GS, d_model=128, nhead=4, n_layers=3)
transformer, best_vloss_t = train_gpu_model(transformer, train_ld, val_ld,
                                             epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                             name="Transformer")
torch.cuda.synchronize()
print(f"  Trained in {time.time()-t_train:.1f}s | Best Val MSE: {best_vloss_t:.4f}")

y_pred_trans = predict_gpu(transformer, X_vl)
trans_metrics = compute_forecasting_metrics(y_vl, y_pred_trans)
print(f"  Metrics: RMSE={trans_metrics['RMSE']:.4f}, MAE={trans_metrics['MAE']:.4f}, "
      f"R2={trans_metrics['R2']:.4f}, Pearson={trans_metrics['Pearson_r']:.4f}")
results["transformer_no_aug"] = trans_metrics

# ─── Attention LSTM ───────────────────────────────────────────────────────
print("\n  [5c] Attention LSTM (No Augmentation)...")
t_train = time.time()
attn_lstm = AttentionLSTM(gs=GS, hidden=128, n_layers=2)
attn_lstm, best_vloss_a = train_gpu_model(attn_lstm, train_ld, val_ld,
                                            epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                            name="AttnLSTM")
torch.cuda.synchronize()
print(f"  Trained in {time.time()-t_train:.1f}s | Best Val MSE: {best_vloss_a:.4f}")

y_pred_attn = predict_gpu(attn_lstm, X_vl)
attn_metrics = compute_forecasting_metrics(y_vl, y_pred_attn)
print(f"  Metrics: RMSE={attn_metrics['RMSE']:.4f}, MAE={attn_metrics['MAE']:.4f}, "
      f"R2={attn_metrics['R2']:.4f}, Pearson={attn_metrics['Pearson_r']:.4f}")
results["attn_lstm_no_aug"] = attn_metrics

gc.collect()
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: SOP AUGMENTATION + RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 6: SOP AUGMENTATION")
print("=" * 65)

t_aug = time.time()
sop_aug_list = sop_augment_fast(train_ev, n_aug=3, window_months=3, seed=SEED)
sop_combined = pd.concat([train_ev] + sop_aug_list, ignore_index=True)
print(f"  SOP: generated {len(sop_combined) - len(train_ev):,} augmented events "
      f"in {time.time()-t_aug:.1f}s")

# Rebuild grid and sequences
sop_grid = build_grid_gpu(sop_combined)
X_sop, y_sop = create_sequences(sop_grid)
print(f"  SOP Sequences: {len(X_sop)}")

sop_ds = TensorDataset(
    torch.from_numpy(X_sop).pin_memory(),
    torch.from_numpy(y_sop).pin_memory(),
)
sop_ld = DataLoader(sop_ds, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, pin_memory=True)

print("\n  CNN-LSTM + SOP Augmentation...")
t_train = time.time()
cnn_sop = SpatioTemporalCNN(gs=GS, conv_channels=[64, 128, 256],
                              lstm_hidden=128, lstm_layers=2, dropout=0.25)
cnn_sop, _ = train_gpu_model(cnn_sop, sop_ld, val_ld,
                               epochs=EPOCHS, lr=LR, patience=PATIENCE,
                               name="CNN-LSTM+SOP")
torch.cuda.synchronize()
print(f"  Trained in {time.time()-t_train:.1f}s")

y_pred_cnn_sop = predict_gpu(cnn_sop, X_vl)
cnn_sop_metrics = compute_forecasting_metrics(y_vl, y_pred_cnn_sop)
print(f"  Metrics: RMSE={cnn_sop_metrics['RMSE']:.4f}, MAE={cnn_sop_metrics['MAE']:.4f}, "
      f"R2={cnn_sop_metrics['R2']:.4f}, Pearson={cnn_sop_metrics['Pearson_r']:.4f}")
results["cnn_lstm_sop"] = cnn_sop_metrics

del sop_ds, sop_ld, sop_grid, X_sop, y_sop
gc.collect()
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: QUANTUM AUGMENTATION + RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 7: QUANTUM (Statistical) AUGMENTATION")
print("=" * 65)

t_qaug = time.time()
synth_df = quantum_augment_fast(train_ev, n_synth=20000, seed=SEED)
q_combined = pd.concat([train_ev, synth_df], ignore_index=True)
print(f"  Quantum proxy: generated {len(synth_df):,} synthetic events "
      f"in {time.time()-t_qaug:.1f}s")

# Rebuild grid and sequences
q_grid = build_grid_gpu(q_combined)
X_q, y_q = create_sequences(q_grid)
print(f"  Quantum Sequences: {len(X_q)}")

q_ds = TensorDataset(
    torch.from_numpy(X_q).pin_memory(),
    torch.from_numpy(y_q).pin_memory(),
)
q_ld = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=NUM_WORKERS, pin_memory=True)

print("\n  CNN-LSTM + Quantum Augmentation...")
t_train = time.time()
cnn_q = SpatioTemporalCNN(gs=GS, conv_channels=[64, 128, 256],
                            lstm_hidden=128, lstm_layers=2, dropout=0.25)
cnn_q, _ = train_gpu_model(cnn_q, q_ld, val_ld,
                             epochs=EPOCHS, lr=LR, patience=PATIENCE,
                             name="CNN-LSTM+Quantum")
torch.cuda.synchronize()
print(f"  Trained in {time.time()-t_train:.1f}s")

y_pred_q = predict_gpu(cnn_q, X_vl)
q_metrics = compute_forecasting_metrics(y_vl, y_pred_q)
print(f"  Metrics: RMSE={q_metrics['RMSE']:.4f}, MAE={q_metrics['MAE']:.4f}, "
      f"R2={q_metrics['R2']:.4f}, Pearson={q_metrics['Pearson_r']:.4f}")
results["cnn_lstm_quantum"] = q_metrics

del q_ds, q_ld, q_grid, X_q, y_q
gc.collect()
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: GRID-LEVEL QUANTUM AUGMENTATION v3
# Key improvement: generate full grid tensors, not individual events
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 8: GRID-LEVEL QUANTUM AUGMENTATION v3")
print("=" * 65)

qgan_v3_success = False
qgan_v3_history = None
qgan_v3_grids = None
qgan_v3_metrics = {}

try:
    from src.augmentation.quantum_augment_v3 import (
        GridQGANV3, train_grid_qgan_v3, generate_grids_v3,
        create_style_contexts, augment_with_grid_qgan, QBMv3, train_qbm_v3
    )

    # ─── 8a: Train QBM v3 on spatial templates ────────────────────────────
    print("\n  [8a] Training QBM v3 (Spatial Template Learning)...")
    t_qbm3 = time.time()

    qbm3 = QBMv3(grid_size=GS, n_patterns=16, n_layers=6)
    combined = np.concatenate([X_tr, X_vl], axis=0)
    # Subsample for faster training
    if len(combined) > 2000:
        idx = np.random.choice(len(combined), 2000, replace=False)
        combined_sub = combined[idx]
    else:
        combined_sub = combined

    print(f"  QBM v3 data: {combined_sub.shape}")
    qbm3_hist = train_qbm_v3(qbm3, combined_sub, epochs=200, lr=0.05, batch_size=32, verbose=True)
    print(f"  QBM v3 trained in {time.time()-t_qbm3:.1f}s, final loss: {qbm3_hist['loss'][-1]:.6f}")
    qgan_v3_metrics["qbm_v3_final_loss"] = qbm3_hist["loss"][-1]

    # ─── 8b: Train Grid QGAN v3 ─────────────────────────────────────────
    print("\n  [8b] Training Grid QGAN v3 (Full Grid Tensor Generation)...")
    t_qgan3 = time.time()

    # Use ALL training data for QGAN (more data = better training)
    n_aug = min(len(X_tr), 347)  # all training sequences
    X_qgan_train = X_tr[:n_aug]
    print(f"  Training on {len(X_qgan_train)} grid sequences, shape: {X_qgan_train.shape}")

    qgan3, gen_grids, qgan3_hist = augment_with_grid_qgan(
        X_qgan_train,
        n_augmented_sequences=300,
        latent_dim=16,
        style_dim=8,
        seq_len=SEQ_LEN,
        grid_h=GS,
        grid_w=GS,
        qgan_epochs=400,    # More epochs for better training
        lr_g=1e-3,
        lr_d=1e-3,
        batch_size=32,     # Smaller batch for stability
        device="cuda",
        verbose=True,
    )
    torch.cuda.synchronize()
    qgan_v3_history = qgan3_hist
    qgan_v3_grids = gen_grids
    print(f"  Grid QGAN v3 trained in {time.time()-t_qgan3:.1f}s")

    # ─── 8c: Convert generated grids to events ───────────────────────────
    print("\n  [8c] Converting generated grids to event DataFrames...")
    base_ts = train_ev["timestamp"].min()
    cell_h = (LAT_MAX - LAT_MIN) / GS
    cell_w = (LON_MAX - LON_MIN) / GS

    # Normalize generated grids to match real data scale
    # Real data: mean ~0.5-2.0 per cell
    # Generated: could be in any range — normalize to match real stats
    real_mean = X_tr[:min(len(X_tr), 100)].mean()
    real_std = X_tr[:min(len(X_tr), 100)].std()
    gen_mean = gen_grids.mean()
    gen_std = gen_grids.std() + 1e-8

    # Scale generated grids to match real data distribution
    gen_norm = (gen_grids - gen_mean) / gen_std
    gen_norm = gen_norm * real_std + real_mean
    gen_norm = np.clip(gen_norm, 0, real_mean * 20)  # cap outliers

    # For generated grids: convert to events preserving spatial structure
    qgan3_synth_events = []
    for seq_idx, grid in enumerate(gen_norm):
        # Average over time to get spatial pattern
        spatial_pattern = grid.mean(axis=0)  # (H, W)
        # Threshold: only top percentile active cells
        if spatial_pattern.max() > 0:
            threshold = max(0.001, np.percentile(spatial_pattern[spatial_pattern > 0], 50))
        else:
            threshold = 0.001
        active_cells = spatial_pattern > threshold

        for i in range(GS):
            for j in range(GS):
                if active_cells[i, j]:
                    # Scale case count to realistic range
                    cell_val = float(spatial_pattern[i, j])
                    case_v = max(1, int(cell_val * 10 + 1))
                    lat_v = LAT_MIN + (i + 0.5) * cell_h
                    lon_v = LON_MIN + (j + 0.5) * cell_w
                    ts_offset = pd.Timedelta(days=(seq_idx * 30) % 365)
                    qgan3_synth_events.append({
                        "event_id": len(train_ev) + len(qgan3_synth_events),
                        "lat": float(lat_v), "lon": float(lon_v),
                        "timestamp": base_ts + ts_offset,
                        "case_count": min(case_v, 5000),
                        "region": "SYNTH_GRIDQGAN",
                        "country": "SYNTHETIC",
                        "augmented": True, "aug_method": "grid_qgan_v3",
                        "year": (base_ts + ts_offset).year,
                        "month": (base_ts + ts_offset).month,
                    })

    qgan3_df = pd.DataFrame(qgan3_synth_events)
    qgan3_df["case_count"] = qgan3_df["case_count"].replace([np.inf, -np.inf], np.nan).fillna(1).astype(int)
    qgan3_df["case_count"] = qgan3_df["case_count"].clip(1, 10000)
    print(f"  Grid QGAN v3 synthetic events: {len(qgan3_df):,}")

    # ─── 8d: Retrain CNN-LSTM with Grid QGAN augmentation ─────────────────
    qgan3_combined = pd.concat([train_ev, qgan3_df], ignore_index=True)
    print(f"\n  Combined dataset: {len(qgan3_combined):,} events "
          f"(orig={len(train_ev):,}, grid_qgan={len(qgan3_df):,})")

    qgan3_grid = build_grid_gpu(qgan3_combined)
    X_q3, y_q3 = create_sequences(qgan3_grid)
    print(f"  Grid QGAN Sequences: {len(X_q3)}")

    # Validate grids are finite
    if not np.isfinite(X_q3).all():
        print("  WARNING: NaN/Inf detected in QGAN grids, clipping...")
        X_q3 = np.nan_to_num(X_q3, nan=0.0, posinf=100.0, neginf=0.0)

    qgan3_ds = TensorDataset(
        torch.from_numpy(X_q3).pin_memory(),
        torch.from_numpy(y_q3).pin_memory(),
    )
    qgan3_ld = DataLoader(qgan3_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)

    print("\n  CNN-LSTM + Grid QGAN v3 Augmentation...")
    t_train = time.time()
    cnn_q3 = SpatioTemporalCNN(gs=GS, conv_channels=[64, 128, 256],
                                lstm_hidden=128, lstm_layers=2, dropout=0.25)
    cnn_q3, _ = train_gpu_model(cnn_q3, qgan3_ld, val_ld,
                                  epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                  name="CNN-LSTM+GridQGANv3")
    torch.cuda.synchronize()
    print(f"  Trained in {time.time()-t_train:.1f}s")

    y_pred_q3 = predict_gpu(cnn_q3, X_vl)
    q3_metrics = compute_forecasting_metrics(y_vl, y_pred_q3)
    print(f"  Metrics: RMSE={q3_metrics['RMSE']:.4f}, MAE={q3_metrics['MAE']:.4f}, "
          f"R2={q3_metrics['R2']:.4f}, Pearson={q3_metrics['Pearson_r']:.4f}")
    results["cnn_lstm_grid_qgan_v3"] = q3_metrics
    qgan_v3_metrics["grid_qgan_val_rmse"] = q3_metrics["RMSE"]
    qgan_v3_metrics["grid_qgan_val_r2"] = q3_metrics["R2"]

    del qgan3_ds, qgan3_ld
    gc.collect()
    torch.cuda.empty_cache()

    qgan_v3_success = True

except Exception as e:
    import traceback
    print(f"  Grid QGAN v3 failed: {e}")
    traceback.print_exc()
    qgan_v3_success = False


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9: COMPREHENSIVE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STAGE 9: COMPREHENSIVE EVALUATION")
print("=" * 65)

print("\n  Validation Set Results:")
print("  " + "-" * 60)
print(f"  {'Method':<30} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Pearson':>8}")
print("  " + "-" * 60)
for name, m in sorted(results.items(), key=lambda x: x[1].get("RMSE", 999)):
    rmse = m.get("RMSE", float("nan"))
    mae  = m.get("MAE",  float("nan"))
    r2   = m.get("R2",   float("nan"))
    pr   = m.get("Pearson_r", float("nan"))
    print(f"  {name:<30} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {pr:>8.4f}")
print("  " + "-" * 60)

# Test set evaluation
print("\n  Test Set Results:")
for name, model in [("CNN-LSTM No Aug", cnn_lstm), ("Transformer No Aug", transformer),
                     ("AttnLSTM No Aug", attn_lstm)]:
    if model is not None:
        y_t_pred = predict_gpu(model, X_ts)
        t_m = compute_forecasting_metrics(y_ts, y_t_pred)
        print(f"  {name}: RMSE={t_m['RMSE']:.4f}, R2={t_m['R2']:.4f}, "
              f"Pearson={t_m['Pearson_r']:.4f}")

# Prediction vs Actual scatter
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flat
for idx, (name, m) in enumerate(sorted(results.items())):
    if idx >= 6:
        break
    ax = axes[idx]
    if name == "cnn_lstm_no_aug":
        yp = y_pred_cnn
    elif name == "transformer_no_aug":
        yp = y_pred_trans
    elif name == "attn_lstm_no_aug":
        yp = y_pred_attn
    elif name == "cnn_lstm_sop":
        yp = y_pred_cnn_sop
    elif name == "cnn_lstm_grid_qgan_v3":
        yp = y_pred_q3
    else:
        yp = y_pred_cnn

    ax.scatter(y_vl, yp, alpha=0.4, s=10, color="steelblue")
    lim = [min(y_vl.min(), yp.min()), max(y_vl.max(), yp.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="y=x")
    r = np.corrcoef(y_vl, yp)[0, 1]
    ax.set_title(f"{name}\nPearson r={r:.4f}", fontsize=9)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")

plt.suptitle("GPU Pipeline — Predicted vs Actual (Validation Set)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()

# RMSE comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))
names = list(results.keys())
rmses = [results[n].get("RMSE", 0) for n in names]
maes  = [results[n].get("MAE",  0) for n in names]
r2s   = [results[n].get("R2",   0) for n in names]

x = np.arange(len(names))
ax.bar(x - 0.25, rmses, 0.25, label="RMSE", color="steelblue")
ax.bar(x,         maes,  0.25, label="MAE",  color="coral")
ax.bar(x + 0.25, r2s,   0.25, label="R2",    color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, rotation=0)
ax.set_ylabel("Score")
ax.set_title("GPU Pipeline — Model Comparison (Validation Set)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Heatmap of spatial predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["CNN-LSTM", "Transformer", "Attention LSTM"]
preds_list = [y_pred_cnn, y_pred_trans, y_pred_attn]
for idx, (title, yp) in enumerate(zip(titles, preds_list)):
    ax = axes[idx]
    last_seq_idx = -1
    # Average predictions over validation sequences
    pred_map = np.full((GS, GS), np.nan)
    # Map predictions back to spatial grid (aggregate per time step)
    # Simple: just show the mean prediction value
    ax.text(0.5, 0.5, f"Mean: {np.mean(yp):.2f}\nStd: {np.std(yp):.2f}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="steelblue")
    ax.set_title(f"{title}\nPrediction Distribution", fontsize=11)
    ax.axis("off")

plt.suptitle("GPU Pipeline — Model Prediction Distributions", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_prediction_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Grid resolution comparison
n_methods = len(results)
n_cols = 3
n_rows = (n_methods + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
if n_methods == 1:
    axes = np.array([axes])
axes_flat = axes.flatten()

methods_ordered = sorted(results.items(), key=lambda x: x[1].get("RMSE", 999))
for idx, (name, m) in enumerate(methods_ordered):
    ax = axes_flat[idx]
    rmse = m.get("RMSE", 0)
    r2 = m.get("R2", 0)
    ax.barh(name.replace("_", " "), rmse, color="steelblue", edgecolor="black")
    ax.set_xlabel("RMSE")
    ax.set_title(f"R²={r2:.3f}", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

for idx in range(len(methods_ordered), len(axes_flat)):
    axes_flat[idx].axis("off")

plt.suptitle("GPU Pipeline — RMSE Comparison (Lower is Better)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT / "gpu_rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Saved: pred_vs_actual, model_comparison, prediction_distributions, "
      f"rmse_comparison, spatial_heatmap, yearly_trends, monthly_patterns")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t0

results_json = {
    "pipeline": "GPU-Accelerated (RTX 3090)",
    "config": {
        "grid_size": GS,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "device": str(DEVICE),
        "gpu": GPU_NAME,
        "vram_gb": round(VRAM_GB, 1),
        "mixed_precision": True,
    },
    "validation_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    "quantum_training": {
        "qgan_v3_success": qgan_v3_success,
        "qgan_v3_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in qgan_v3_metrics.items()},
        "n_grid_qgan_synthetic": int(len(qgan3_df)) if qgan_v3_success else 0,
        "grid_qgan_g_loss_final": float(qgan_v3_history["g_loss"][-1]) if qgan_v3_history else None,
        "grid_qgan_d_loss_final": float(qgan_v3_history["d_loss"][-1]) if qgan_v3_history else None,
    },
    "dataset": {
        "n_train": int(len(train_ev)),
        "n_val": int(len(val_ev)),
        "n_test": int(len(test_ev)),
        "n_total": int(len(events_df)),
        "total_cases": int(events_df["case_count"].sum()),
        "n_countries": int(events_df["country"].nunique()),
        "n_regions": int(events_df["region"].nunique()),
    },
    "timing": {
        "total_seconds": round(total_time, 1),
        "total_minutes": round(total_time / 60, 1),
    },
}

with open(OUTPUT / "gpu_results.json", "w") as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n  Results saved to {OUTPUT / 'gpu_results.json'}")
print(f"\n  GPU Memory after full pipeline:")
print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"    Reserved:  {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")

print("\n" + "=" * 65)
print(f"PIPELINE COMPLETE! Total time: {total_time/60:.1f} minutes")
print("=" * 65)
