"""CNN-LSTM model for spatio-temporal dengue forecasting."""
import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SpatioTemporalCNN(nn.Module):
    def __init__(self, input_channels=1, conv_channels=None, lstm_hidden=64,
                 lstm_layers=2, dropout=0.3, grid_size=16, forecast_horizon=1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        self.forecast_horizon = forecast_horizon

        layers = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(ConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.randn(1, 1, grid_size, grid_size)
            cnn_out = self.cnn(dummy)
            self.cnn_out_size = cnn_out.numel()

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, forecast_horizon),
        )

    def forward(self, x):
        """x: (batch, seq_len, H, W) or (batch, H, W)"""
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, H, W)
        if x.dim() == 5:
            x = x.squeeze(2)
        batch, T, H, W = x.shape

        x = x.contiguous().view(batch * T, 1, H, W)
        x = self.cnn(x)
        x = x.contiguous().view(batch, T, -1)

        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]

        return self.head(last)


def create_sequences(grid, seq_len=12, forecast_horizon=1):
    """grid: (H, W, T) -> X: (n, seq_len, H, W), y: (n, 1)"""
    H, W, T = grid.shape
    X, y = [], []
    for t in range(seq_len, T - forecast_horizon + 1):
        seq = grid[:, :, t - seq_len:t].transpose(2, 0, 1)
        future = grid[:, :, t:t + forecast_horizon].sum()
        X.append(seq)
        y.append(future)
    if not X:
        return np.zeros((0, seq_len, H, W)), np.zeros((0, 1))
    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32)


def train_cnn_lstm(model, train_loader, val_loader=None, epochs=50, lr=1e-3,
                   device="cpu", patience=10, seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

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

            opt.zero_grad()
            pred = model(Xb)
            if pred.dim() > 1 and pred.shape[1] > 1:
                pred = pred.mean(dim=1, keepdim=True)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
            n_batch += 1

        sched.step()
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


def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device))
    return pred.cpu().numpy()
