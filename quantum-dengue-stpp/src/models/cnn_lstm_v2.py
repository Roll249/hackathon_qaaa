"""
Improved CNN-LSTM model v2 with attention and spatial mean feature.

Key improvements over v1:
1. Spatial mean feature (critical for CNN-LSTM to beat NEST)
2. 3-layer CNN with residual connections
3. Bidirectional LSTM
4. Temporal attention mechanism
5. Support for Negative Binomial loss
6. Multiple loss functions (MSE, Poisson, NB, Tweedie)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from .losses import get_loss_fn


class ConvBlockWithResidual(nn.Module):
    """
    ConvBlock with optional residual connection.

    Architecture: Conv -> BN -> Activation -> Conv -> BN -> Activation -> Pool -> Dropout
    Residual path: 1x1 Conv to match dimensions (if needed)
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.3, use_residual: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        if use_residual and in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.MaxPool2d(2),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for LSTM outputs.

    Computes attention weights over time steps and produces a weighted
    context vector that captures which time steps are most important.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor):
        """
        Args:
            lstm_output: (B, T, hidden_size)

        Returns:
            context: (B, hidden_size) - weighted sum over time
            weights: (B, T) - attention weights
        """
        weights = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(weights, dim=1)

        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class SpatioTemporalCNNv2(nn.Module):
    """
    Improved CNN-LSTM v2 with attention and spatial mean feature.

    Architecture:
    1. 3-layer CNN with residual connections (captures spatial patterns)
    2. Spatial mean pooling (KEY FEATURE: what made CNN-LSTM beat NEST)
    3. Bidirectional LSTM (captures forward/backward temporal dependencies)
    4. Temporal attention (focuses on important time steps)
    5. Prediction head with LayerNorm and GELU

    Reference: SYNTHESIS.md analysis shows spatial mean is critical
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        grid_size: int = 32,
        forecast_horizon: int = 1,
        bidirectional: bool = True,
        use_attention: bool = True,
        loss: str = 'mse',
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.forecast_horizon = forecast_horizon
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.loss_name = loss

        layers = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(ConvBlockWithResidual(in_ch, out_ch, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.randn(1, 1, grid_size, grid_size)
            cnn_out = self.cnn(dummy)
            self.cnn_out_size = cnn_out.numel()
            self.cnn_spatial_h = cnn_out.shape[2]
            self.cnn_spatial_w = cnn_out.shape[3]

        lstm_input_size = self.cnn_out_size + 1

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        lstm_output_size = lstm_hidden * (2 if bidirectional else 1)

        if use_attention:
            self.attention = TemporalAttention(lstm_output_size)
            head_input_size = lstm_output_size * 2
        else:
            self.attention = None
            head_input_size = lstm_output_size

        self.head = nn.Sequential(
            nn.Linear(head_input_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_hidden // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, H, W) or (batch, H, W)

        Returns:
            predictions: (batch, forecast_horizon)
        """
        x = x.float()

        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 5:
            x = x.squeeze(2)

        batch, T, H, W = x.shape

        x = x.contiguous().view(batch * T, 1, H, W)
        x = self.cnn(x)
        x = x.contiguous().view(batch, T, -1)

        spatial_mean = x.mean(dim=-1, keepdim=True)
        x = torch.cat([x, spatial_mean], dim=-1)

        lstm_out, _ = self.lstm(x)

        if self.use_attention and self.attention is not None:
            context, _ = self.attention(lstm_out)
            last = lstm_out[:, -1, :]
            combined = torch.cat([last, context], dim=1)
        else:
            combined = lstm_out[:, -1, :]

        return self.head(combined)


def create_sequences_v2(
    grid: np.ndarray,
    seq_len: int = 12,
    forecast_horizon: int = 1,
    include_spatial_mean: bool = True
):
    """
    Create sequences from spatial grid data.

    Args:
        grid: (H, W, T) spatial grid
        seq_len: input sequence length
        forecast_horizon: how many steps ahead to predict
        include_spatial_mean: include spatial mean in sequence

    Returns:
        X: (n_samples, seq_len, H, W) or (n_samples, seq_len, H, W, 2) if include_spatial_mean
        y: (n_samples, forecast_horizon)
    """
    H, W, T = grid.shape
    X, y = [], []

    for t in range(seq_len, T - forecast_horizon + 1):
        seq = grid[:, :, t - seq_len:t].transpose(2, 0, 1)
        future = grid[:, :, t:t + forecast_horizon].sum()
        X.append(seq)
        y.append(future)

    if not X:
        return np.zeros((0, seq_len, H, W)), np.zeros((0, forecast_horizon))

    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32)


def train_cnn_lstm_v2(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 15,
    seed: int = 42,
    verbose: bool = True,
    loss_name: str = 'mse',
    use_scheduler: bool = True,
):
    """
    Train CNN-LSTM v2 with configurable loss function.

    Args:
        model: SpatioTemporalCNNv2 instance
        train_loader: training data loader
        val_loader: validation data loader
        epochs: max training epochs
        lr: learning rate
        device: 'cpu' or 'cuda'
        patience: early stopping patience
        seed: random seed
        verbose: print progress
        loss_name: one of 'mse', 'poisson', 'nb', 'tweedie'
        use_scheduler: use cosine annealing scheduler
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)

    criterion = get_loss_fn(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )
    else:
        scheduler = None

    best_loss = float("inf")
    counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        n_batch = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device).float()
            yb = yb.to(device).float()

            if yb.dim() > 1 and yb.shape[1] > 1:
                yb = yb.mean(dim=1, keepdim=True)
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)

            optimizer.zero_grad()
            pred = model(Xb)

            if pred.dim() > 1 and pred.shape[1] > 1:
                pred = pred.mean(dim=1, keepdim=True)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)

            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item()
            n_batch += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = t_loss / max(n_batch, 1)

        v_loss = None
        if val_loader:
            model.eval()
            v_tot = 0.0
            v_n = 0

            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device).float()
                    yb = yb.to(device).float()

                    if yb.dim() > 1 and yb.shape[1] > 1:
                        yb = yb.mean(dim=1, keepdim=True)
                    if yb.dim() == 1:
                        yb = yb.unsqueeze(1)

                    pred = model(Xb)

                    if pred.dim() > 1 and pred.shape[1] > 1:
                        pred = pred.mean(dim=1, keepdim=True)
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(1)

                    v_tot += criterion(pred, yb).item()
                    v_n += 1

            v_loss = v_tot / max(v_n, 1)

            if v_loss < best_loss:
                best_loss = v_loss
                counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
        else:
            counter += 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            vs = f" | Val: {v_loss:.4f}" if v_loss is not None else ""
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}{vs}")

        if counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def predict_v2(model: nn.Module, X: np.ndarray, device: str = "cpu"):
    """Predict using trained model."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device))
    return pred.cpu().numpy()
