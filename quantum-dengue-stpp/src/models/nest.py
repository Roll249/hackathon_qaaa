"""
Neural Spatio-Temporal Point Process (NEST) model.

NEST models the intensity function λ(s, t) as a neural network:
    λ(s, t) = exp(w^T · h(s, t) + b)

where h(s, t) = encoder(spatial_context, temporal_hidden_state).

This implementation uses a grid-based approach where the intensity
at each grid cell is predicted using a shared encoder + per-cell head.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class NESTIntensity(nn.Module):
    """
    NEST: Neural Spatio-Temporal intensity model.

    Predicts per-cell intensity rates from a sequence of spatial grids.
    Uses: spatial CNN encoder → LSTM temporal → per-cell intensity head.
    """

    def __init__(self, gs: int = 20, hidden: int = 64, temporal_hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gs = gs

        # Spatial encoder — CNN that outputs a fixed-size spatial embedding
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),   # → (B, hidden, 4, 4)
            nn.Flatten(2),             # → (B, hidden*16)
        )
        spatial_dim = hidden * 16

        # Temporal module — LSTM over spatial embeddings
        self.temporal = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=temporal_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Intensity head — maps hidden state to per-cell log-intensity
        self.intensity_head = nn.Sequential(
            nn.Linear(temporal_hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, gs * gs),        # log-intensity per cell
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H, W) — sequence of spatial grids
        Returns: (B, H, W) — predicted intensity for last time step
        """
        B, T, H, W = x.shape

        # Encode each time step spatially
        spatial_embeds = []
        for t in range(T):
            emb = self.spatial_encoder(x[:, t:t+1])   # (B, spatial_dim)
            spatial_embeds.append(emb)
        emb_tensor = torch.stack(spatial_embeds, dim=1)  # (B, T, spatial_dim)

        # Temporal dynamics
        _, (h_T, _) = self.temporal(emb_tensor)        # (n_layers, B, temporal_hidden)
        h_last = h_T[-1]                               # (B, temporal_hidden)

        # Per-cell intensity
        log_intensity = self.intensity_head(h_last)   # (B, gs*gs)
        return log_intensity.view(B, self.gs, self.gs)  # (B, H, W)


class NESTForecaster:
    """
    Trainable wrapper around NESTIntensity for dengue forecasting.

    Minimizes negative log-likelihood of observed counts:
        -log p(counts | λ) = Σ cells [λ[cell] - counts[cell] * log(λ[cell])]
    (Poisson NLL — appropriate for count data)
    """

    def __init__(self, gs: int = 20, hidden: int = 64, temporal_hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        self.gs = gs
        self.model = NESTIntensity(gs, hidden, temporal_hidden, n_layers, dropout)
        self.hidden = hidden
        self.temporal_hidden = temporal_hidden
        self.n_layers = n_layers
        self.dropout = dropout

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 25, lr: float = 1e-3, patience: int = 5,
            batch_size: int = 16, verbose: bool = True, device: str = "cpu") -> dict:

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

        if X_val is not None:
            val_ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(y_val)
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=32, shuffle=False, num_workers=0
            )
        else:
            val_loader = None

        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        # Poisson NLL loss: λ - count * log(λ)
        criterion = lambda pred_log, target: (pred_log.exp() - target * pred_log).mean()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.contiguous().to(device), yb.to(device)
                optimizer.zero_grad()
                log_rates = self.model(Xb)                # (B, gs, gs)
                target = yb.view(-1, self.gs, self.gs)    # (B, gs, gs)
                loss = criterion(log_rates, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(yb)
            train_loss /= len(train_loader.dataset)

            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        Xb, yb = Xb.contiguous().to(device), yb.to(device)
                        log_rates = self.model(Xb)
                        target = yb.view(-1, self.gs, self.gs)
                        val_loss += criterion(log_rates, target).item() * len(yb)
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch + 1) % 5 == 0:
                tag = f"val={val_loss:.2f}" if val_loss is not None else "val=N/A"
                print(f"    [NEST] Epoch {epoch+1:>2}/{epochs}  train={train_loss:.2f}  {tag}")

            if no_improve >= patience:
                if verbose:
                    print(f"    [NEST] Early stop at epoch {epoch+1}")
                break

        if best_state:
            self.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        self.model.eval()
        return {"best_val_loss": float(best_val_loss)}

    def predict(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).contiguous().to(device)
            log_rates = self.model(X_t)   # (N, gs, gs)
            rates = log_rates.exp().cpu().numpy()
        return rates.mean(axis=(1, 2))    # scalar per sequence (mean cell rate)
