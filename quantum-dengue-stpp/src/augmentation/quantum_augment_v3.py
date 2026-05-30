"""
Quantum Augmentation v3 — Grid-Level Generation.

Key insight: The previous versions failed because they generated individual events
and then converted to grid — losing ALL spatial correlations.

v3 approach:
1. QBM learns the DISTRIBUTION of grid-level activity patterns (spatial templates)
2. QGAN generates FULL GRID TENSORS directly (same shape as X_tr)
3. Generated grids are used directly for training — no event conversion

This preserves spatial structure entirely and allows quantum models to learn
the true spatio-temporal distribution.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from scipy.stats import wasserstein_distance
import time
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# QUANTUM BORN MACHINE v3 — Learning Spatial Templates
# =============================================================================

class QBMv3(nn.Module):
    """
    QBM v3: Learns spatial activity patterns.

    Instead of generating individual events, it learns which grid cells are
    typically active together. The output is a spatial probability mask
    over the grid, which can be sampled to create activity patterns.

    Architecture:
    - 2D convolutional structure in parameter space
    - Each "qubit" corresponds to a spatial region
    - Entanglement patterns reflect spatial autocorrelation
    """

    def __init__(self, grid_size=48, n_patterns=16, n_layers=4):
        super().__init__()
        self.grid_size = grid_size
        self.n_patterns = n_patterns
        self.n_layers = n_layers
        self.n_patterns = n_patterns
        n_qubits = min(n_patterns, 12)

        # Parameters: spatial pattern weights
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        # Spatial projection: maps pattern bits to full grid
        self.spatial_proj = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, grid_size * grid_size),
        )

        # Quantum device for true quantum simulation
        self.qdev = qml.device("default.qubit", wires=n_qubits)

        # Build QBM circuit with reduced parameters
        n_q = min(n_patterns, 12)

        @qml.qnode(self.qdev, diff_method="backprop")
        def circuit(params_flat):
            params = params_flat.view(n_layers, n_q)
            for layer in range(n_layers):
                for i in range(n_q):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_q > 2:
                    qml.CNOT(wires=[n_q - 1, 0])
            return qml.probs(wires=range(n_q))

        self.circuit = circuit
        self.n_qubits = n_q

    def forward(self, params=None):
        if params is None:
            params = self.theta
        probs = self.circuit(params.flatten())
        if isinstance(probs, (list, tuple)):
            probs = torch.stack(probs)
        return probs

    def generate_spatial_mask(self, n_samples=100):
        """Generate spatial activity masks for the grid."""
        probs = self.forward().detach().cpu().numpy()
        probs = np.clip(probs, 0, None)
        probs = probs / (probs.sum() + 1e-15)

        masks = []
        for _ in range(n_samples):
            # Sample a state from the QBM distribution
            state_idx = np.random.choice(len(probs), p=probs)
            state_bits = np.array([int(b) for b in format(state_idx, f"0{self.n_qubits}b")])

            # Map pattern bits to spatial grid
            mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            # Each bit controls a region of the grid
            region_h = self.grid_size // 4
            region_w = self.grid_size // 4
            for bit_idx, bit_val in enumerate(state_bits):
                if bit_idx < 16:
                    r = bit_idx // 4
                    c = bit_idx % 4
                    mask[r*region_h:(r+1)*region_h, c*region_w:(c+1)*region_w] = bit_val
            masks.append(mask)

        return np.array(masks)  # (n_samples, grid_h, grid_w)


def train_qbm_v3(model, X_grid, epochs=300, lr=0.05, batch_size=32, verbose=True):
    """
    Train QBM v3 to learn spatial activity patterns from grid tensors.

    X_grid: (n_samples, seq_len, grid_h, grid_w)
    """
    model = model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    n_samples = len(X_grid)
    n_bits = model.n_qubits

    # Create binary spatial patterns: sum over seq_len, threshold
    spatial_activity = X_grid.sum(axis=1)  # (n, h, w)
    spatial_binary = (spatial_activity > 0).astype(np.float32)

    # Reduce to n_patterns regions
    n_r = 4
    n_c = 4
    block_h = spatial_binary.shape[1] // n_r
    block_w = spatial_binary.shape[2] // n_c
    patterns = np.zeros((n_samples, n_r * n_c), dtype=np.float32)
    for i in range(n_samples):
        idx = 0
        for r in range(n_r):
            for c in range(n_c):
                block = spatial_binary[i, r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                patterns[i, idx] = block.mean()
                idx += 1

    patterns = np.clip(patterns, 0, 1)

    # Target distribution over 2^n_bits states
    n_states = min(2 ** n_bits, 4096)
    target_dist = torch.zeros(n_states)
    for row in patterns[:, :n_bits]:
        idx = int("".join(str(int(b > 0.5)) for b in row), 2)
        if idx < n_states:
            target_dist[idx] += 1
    target_dist = target_dist / (target_dist.sum() + 1e-10)

    history = {"loss": []}
    t0 = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        # MMD loss
        gen_probs = model.forward()
        diff = gen_probs - target_dist
        loss = torch.sum(diff ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  QBM v3 Epoch {epoch+1:>4}/{epochs} | Loss: {loss.item():.6f} | "
                  f"Time: {time.time()-t0:.1f}s")

    return history


# =============================================================================
# QGAN v3 — Direct Grid Generation with Quantum Latent Space
# =============================================================================

class QGeneratorGrid(nn.Module):
    """
    Quantum-Inspired Generator for Full Grid Tensors.

    Key improvement over v2:
    - Generates COMPLETE grid tensors directly (seq_len × H × W)
    - Uses latent space manipulation to create diverse spatial patterns
    - Output is directly usable for training (no conversion)

    Architecture mirrors a Variational Quantum Circuit:
    - Latent encoding via RY rotations (data embedding)
    - Style encoding via RZ rotations (temporal context)
    - Entangling layers simulate quantum correlations
    - Convolutional decoder to full grid resolution
    """

    def __init__(self, latent_dim=16, style_dim=8, seq_len=12, grid_h=48, grid_w=48):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Quantum-inspired variational parameters
        # These mirror the RY rotation angles in a real VQC
        self.vqc_ry = nn.Parameter(torch.randn(latent_dim) * 0.1)  # RY angles
        self.vqc_rz = nn.Parameter(torch.randn(style_dim) * 0.1)   # RZ angles
        self.entangle_weights = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

        # Spatial encoder: learns to map latent patterns to grid structure
        self.spatial_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.GELU(),
        )

        # Style processor
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
        )

        # Grid decoder: generates full (seq_len, H, W) tensor
        self.grid_decoder = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, seq_len * grid_h * grid_w),
        )

    def forward(self, z, style):
        """
        Args:
            z: (batch, latent_dim) latent codes from real data
            style: (batch, style_dim) temporal context
        Returns:
            grid: (batch, seq_len, grid_h, grid_w) — full grid tensors
        """
        batch_size = z.size(0)

        # Quantum-inspired latent transformation
        # RY encoding: rotate latent vector
        z_enc = z * torch.pi + self.vqc_ry
        z_ang = torch.sin(z_enc)  # non-linear encoding

        # RZ style modulation
        s_enc = style * torch.pi + self.vqc_rz
        s_ang = torch.cos(s_enc)

        # Entanglement: learned mixing (simulates CNOT gates)
        z_entangled = z_ang @ torch.tanh(self.entangle_weights)

        # Process latent and style
        z_feat = self.spatial_encoder(z_entangled)  # (batch, 256)
        s_feat = self.style_encoder(s_ang)           # (batch, 64)

        # Combine and decode to grid
        combined = torch.cat([z_feat, s_feat], dim=1)  # (batch, 320)
        grid_flat = self.grid_decoder(combined)          # (batch, seq_len*h*w)

        grid = grid_flat.view(batch_size, self.seq_len, self.grid_h, self.grid_w)
        # Ensure non-negative (Poisson-like)
        grid = F.softplus(grid)

        return grid


class QDiscriminatorGrid(nn.Module):
    """Discriminator for grid tensors."""

    def __init__(self, seq_len=12, grid_h=48, grid_w=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, grid):
        if grid.dim() == 5:
            # (batch, seq_len, H, W) — sum over time dim
            grid = grid.sum(dim=1, keepdim=True)
        elif grid.dim() == 4:
            grid = grid.mean(dim=1, keepdim=True)
        return self.net(grid)


class GridQGANV3(nn.Module):
    """
    Grid-Level QGAN v3.

    Generates FULL GRID TENSORS (seq_len × H × W) directly.
    This is the key improvement over v2:
    - v2: generated individual features → random lat/lon → lost structure
    - v3: generates full grids → perfect spatial structure preservation
    """

    def __init__(self, latent_dim=16, style_dim=8, seq_len=12, grid_h=48, grid_w=48,
                 lr_g=1e-3, lr_d=1e-3, device="cuda"):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.device_str = device

        # Latent encoder: compress grid to latent
        self.encoder = nn.Sequential(
            nn.Conv2d(seq_len, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
        )

        # Style encoder
        self.style_net = nn.Sequential(
            nn.Linear(style_dim, 32),
            nn.GELU(),
            nn.Linear(32, style_dim),
        )

        # Quantum-inspired generator
        self.generator = QGeneratorGrid(latent_dim, style_dim, seq_len, grid_h, grid_w)

        # Discriminator
        self.discriminator = QDiscriminatorGrid(seq_len, grid_h, grid_w)

        # Optimizers
        self.opt_g = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.style_net.parameters()) +
            list(self.generator.parameters()),
            lr=lr_g, weight_decay=1e-4
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr_d, weight_decay=1e-4
        )

        self.history = {"g_loss": [], "d_loss": [], "g_adv": [], "g_rec": []}

    def encode(self, grid):
        return self.encoder(grid)

    def encode_style(self, style):
        return self.style_net(style)

    def generate(self, grid, style):
        """Generate synthetic grid from real grid + style."""
        z = self.encode(grid)
        return self.generator(z, style)

    def gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        score = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=score, inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True
        )[0]
        return ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    def train_step(self, real_grids, styles):
        """Single training step."""
        batch_size = real_grids.size(0)
        real_labels = torch.ones(batch_size, 1, device=real_grids.device)
        fake_labels = torch.zeros(batch_size, 1, device=real_grids.device)

        # ── Train Discriminator ──────────────────────────────────
        self.opt_d.zero_grad(set_to_none=True)
        fake_grids = self.generate(real_grids, styles).detach()

        d_real = self.discriminator(real_grids)
        d_fake = self.discriminator(fake_grids)

        gp = self.gradient_penalty(real_grids, fake_grids)
        d_loss = (
            F.binary_cross_entropy_with_logits(d_real, real_labels) +
            F.binary_cross_entropy_with_logits(d_fake, fake_labels) +
            10.0 * gp
        )
        d_loss.backward()
        self.opt_d.step()

        # ── Train Generator ───────────────────────────────────────
        self.opt_g.zero_grad(set_to_none=True)
        fake_grids = self.generate(real_grids, styles)
        d_fake = self.discriminator(fake_grids)

        # Adversarial loss
        g_adv = F.binary_cross_entropy_with_logits(d_fake, real_labels)

        # Reconstruction: generator should produce similar grids
        # But with different spatial patterns (diversity)
        g_rec = F.mse_loss(fake_grids, real_grids)

        # Diversity: encourage different patterns
        g_div = -F.mse_loss(fake_grids.mean(dim=1), real_grids.mean(dim=1)) * 0.1

        g_loss = g_adv + 0.3 * g_rec + g_div
        g_loss.backward()
        self.opt_g.step()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item(),
                "g_adv": g_adv.item(), "g_rec": g_rec.item()}


def train_grid_qgan_v3(model, X_grids, style_contexts, epochs=300, batch_size=32, verbose=True):
    """
    Train Grid QGAN v3.

    X_grids: (n_samples, seq_len, grid_h, grid_w)
    style_contexts: (n_samples, style_dim)
    """
    device = next(model.parameters()).device
    model.train()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_grids),
        torch.FloatTensor(style_contexts),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, drop_last=True)

    t0 = time.time()
    for epoch in range(epochs):
        epoch_g = 0.0
        epoch_d = 0.0
        n_b = 0

        for grids, styles in loader:
            grids = grids.to(device)
            styles = styles.to(device)
            metrics = model.train_step(grids, styles)
            epoch_g += metrics["g_loss"]
            epoch_d += metrics["d_loss"]
            n_b += 1

        avg_g = epoch_g / max(n_b, 1)
        avg_d = epoch_d / max(n_b, 1)
        model.history["g_loss"].append(avg_g)
        model.history["d_loss"].append(avg_d)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  QGAN v3 Epoch {epoch+1:>4}/{epochs} | G_loss: {avg_g:.4f} | "
                  f"D_loss: {avg_d:.4f} | Time: {time.time()-t0:.1f}s")

    return model.history


def generate_grids_v3(model, X_ref, style_contexts, n_samples=None, batch_size=256):
    """Generate full grid tensors using trained QGAN v3."""
    model.eval()
    device = next(model.parameters()).device

    if n_samples is None:
        n_samples = len(X_ref)

    all_grids = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            grids = torch.FloatTensor(X_ref[i:end]).to(device)
            styles = torch.FloatTensor(style_contexts[i:end]).to(device)

            gen_grids = model.generate(grids, styles)
            all_grids.append(gen_grids.cpu().numpy())

    return np.concatenate(all_grids, axis=0)


# =============================================================================
# QUANTUM AUGMENTATION PIPELINE v3 — Grid-Level
# =============================================================================

def create_style_contexts(X_grids, n_styles=8):
    """
    Create style/temporal context for grid tensors.

    X_grids: (n, seq_len, H, W)
    Returns: (n, style_dim) style vectors
    """
    n = len(X_grids)

    # Temporal features from the grid data itself
    temporal_agg = X_grids.sum(axis=(2, 3))  # (n, seq_len)

    features = np.stack([
        temporal_agg.mean(axis=1),                    # 0: mean activity
        temporal_agg.std(axis=1),                     # 1: temporal variability
        (temporal_agg > 0).mean(axis=1),              # 2: occupancy
        np.percentile(temporal_agg, 25, axis=1),     # 3: Q25
        np.percentile(temporal_agg, 75, axis=1),     # 4: Q75
        (temporal_agg[:, -1] - temporal_agg[:, 0]) / (temporal_agg.sum(axis=1) + 1e-9),  # 5: trend
        X_grids.mean(axis=(1, 2, 3)),                  # 6: overall mean
        (X_grids > 0).mean(axis=(1, 2, 3)),          # 7: sparsity
    ], axis=1)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features.astype(np.float32)


def grid_to_events(grid, seq_idx, base_lat, base_lon, cell_size):
    """Convert a single grid (H, W) to event DataFrame."""
    h, w = grid.shape
    events = []
    for i in range(h):
        for j in range(w):
            if grid[i, j] > 0.01:
                lat = base_lat + (i + 0.5) * cell_size
                lon = base_lon + (j + 0.5) * cell_size
                events.append({
                    "lat": float(lat), "lon": float(lon),
                    "case_count": int(np.clip(grid[i, j], 1, 10000)),
                    "seq_idx": seq_idx,
                })
    return events


def augment_with_grid_qgan(
    X_train,
    n_augmented_sequences=500,
    latent_dim=16,
    style_dim=8,
    seq_len=12,
    grid_h=48,
    grid_w=48,
    qgan_epochs=300,
    lr_g=1e-3,
    lr_d=1e-3,
    batch_size=64,
    device="cuda",
    verbose=True,
):
    """
    Main augmentation function using Grid QGAN v3.

    Pipeline:
    1. Train Grid QGAN v3 on training grid tensors
    2. Generate augmented grids
    3. Convert grids to events (preserving spatial structure)
    4. Return augmented events DataFrame

    This preserves ALL spatial correlations because we generate at the grid level.
    """
    print(f"\n  [GridQGAN v3] Training on {len(X_train)} sequences...")
    t0 = time.time()

    # Create style contexts
    style_contexts = create_style_contexts(X_train, n_styles=style_dim)

    # Initialize model
    model = GridQGANV3(
        latent_dim=latent_dim,
        style_dim=style_dim,
        seq_len=seq_len,
        grid_h=grid_h,
        grid_w=grid_w,
        lr_g=lr_g, lr_d=lr_d,
        device=device,
    ).to(device)

    # Train
    history = train_grid_qgan_v3(
        model, X_train, style_contexts,
        epochs=qgan_epochs, batch_size=batch_size, verbose=verbose
    )

    print(f"  Grid QGAN v3 trained in {time.time()-t0:.1f}s")

    # Generate augmented grids
    print(f"  Generating {n_augmented_sequences} augmented sequences...")
    aug_grids = generate_grids_v3(
        model, X_train[:min(len(X_train), n_augmented_sequences)],
        style_contexts[:min(len(X_train), n_augmented_sequences)],
        n_samples=n_augmented_sequences, batch_size=256
    )

    # Validate quality: compare distribution of generated vs real
    real_agg = X_train[:len(aug_grids)].sum(axis=(1, 2, 3))
    gen_agg = aug_grids.sum(axis=(1, 2, 3))
    mmd = np.sqrt(np.mean((real_agg - gen_agg) ** 2))
    corr = np.corrcoef(real_agg, gen_agg)[0, 1]

    print(f"  Generated grids: {aug_grids.shape}")
    print(f"  Quality — MMD: {mmd:.4f}, Correlation: {corr:.4f}")

    return model, aug_grids, history
