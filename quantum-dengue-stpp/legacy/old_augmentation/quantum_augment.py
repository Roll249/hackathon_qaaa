"""
Quantum Generative Models for Spatio-Temporal Event Sequence Augmentation.

Implements:
1. Quantum Born Machine (QBM) - Liu & Wang 2018
2. Variational Quantum Circuit Generator - for discrete event sequence generation
3. Hybrid Latent Style-Based QGAN - Baglio/Liepelt 2024-2026

The quantum generator learns the probability distribution of dengue event sequences
and generates synthetic events that preserve spatio-temporal structure.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.templates.layers import BasicEntanglerLayers
from scipy.stats import wasserstein_distance
from typing import Tuple, Optional
import time


# =============================================================================
# QUANTUM CIRCUIT DEFINITIONS
# =============================================================================

def create_qbm_circuit(n_qubits, n_layers=4):
    """
    Create a Quantum Born Machine circuit for probability distribution learning.

    The circuit uses parameterized RY rotations and entangling CNOT gates.
    After measurement, the probability distribution over basis states
    represents the learned distribution.

    Based on: Liu & Wang, "Differentiable Learning of QCBM" (PR A 2018)
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(params, wires):
        for layer in range(n_layers):
            for i in range(len(wires)):
                qml.RY(params[layer, i], wires=i)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[i, i + 1])
            if len(wires) > 2:
                qml.CNOT(wires=[len(wires) - 1, 0])

        return qml.probs(wires=wires)

    return circuit


def create_qgan_generator_circuit(n_qubits, n_layers=4, latent_dim=None):
    """
    Create a VQC-based generator circuit for QGAN.

    Uses amplitude encoding of latent vector + variational layers.
    Generates a quantum state whose measurement yields synthetic events.

    Based on: Baglio "Style-Based QGAN" (arXiv:2405.04401)
    """
    if latent_dim is None:
        latent_dim = n_qubits

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(latent_params, weights, wires):
        AngleEmbedding(latent_params, wires=range(min(len(latent_params), n_qubits)), rotation="Y")

        StronglyEntanglingLayers(weights, wires=range(n_qubits))

        return qml.probs(wires=wires)

    return circuit


def create_qgan_discriminator_circuit(n_qubits=8, n_layers=3):
    """
    Classical discriminator implemented as quantum circuit.
    Uses Hadamard test + measurement to compare real vs generated.
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(data_params, wires):
        AngleEmbedding(data_params[:n_qubits], wires=range(n_qubits), rotation="Y")
        BasicEntanglerLayers(qml.math.ones((n_layers, n_qubits)) * 0.5, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(wires=0))

    return circuit


# =============================================================================
# QUANTUM BORN MACHINE
# =============================================================================

class QuantumBornMachine:
    """
    Quantum Born Machine for learning probability distributions over event sequences.

    The QBM encodes a probability distribution P(x) over n-qubit basis states.
    Each basis state |x⟩ corresponds to a discrete event configuration.
    The MMD loss measures distance between generated and target distributions.

    Key advantages:
    - Native discrete distribution modeling (no discretization needed)
    - Exponential expressibility in qubit count
    - Can represent complex correlated distributions efficiently

    Reference: Liu & Wang, PR A 2018; Gili et al., npj Quantum Inf 2020
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 4,
        device: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.fitted = False

        self.n_params = n_layers * n_qubits

        dev = qml.device(device, wires=n_qubits)
        self.dev = dev

        @qml.qnode(dev, diff_method="backprop")
        def circuit(params):
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit
        self.params = None

    def _mmd_loss(self, probs, target_probs):
        """
        Maximum Mean Discrepancy loss using quantum kernel.

        MMD(P, Q) = E[K(x,y)] - 2E[K(x,y')] + E[K(y,y')]
        where K(x,y) = |<x|y>|^2 is the quantum kernel (swaps test).
        """
        p = probs + 1e-10
        q = target_probs + 1e-10
        p = p / p.sum()
        q = q / q.sum()

        kernel_xx = np.sum(np.outer(p, p) * (1 - np.eye(len(p))))
        kernel_yy = np.sum(np.outer(q, q) * (1 - np.eye(len(q))))
        kernel_xy = np.sum(np.outer(p, q))

        n_p = len(p)
        n_q = len(q)
        kernel_xx = np.sum(p[:, None] * p[None, :] * (1 - np.eye(n_p))) / (n_p * n_p - n_p + 1e-9)
        kernel_yy = np.sum(q[:, None] * q[None, :] * (1 - np.eye(n_q))) / (n_q * n_q - n_q + 1e-9)
        kernel_xy = np.sum(p[:, None] * q[None, :]) / (n_p * n_q)

        return kernel_xx - 2 * kernel_xy + kernel_yy

    def _mmd_loss_quantum(self, probs, target_probs):
        """
        Simplified MMD using Hilbert-Schmidt norm.
        More efficient for large state spaces.
        """
        p = probs / (probs.sum() + 1e-10)
        q = target_probs / (target_probs.sum() + 1e-10)

        diff = p - q
        hs_norm = np.sqrt(np.sum(diff ** 2))

        return hs_norm

    def fit(
        self,
        training_data: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """
        Train the QBM on discrete event data.

        Args:
            training_data: array of shape (n_samples, n_bits) where each row
                          is a binary event vector (e.g., presence/absence in spatial cells)
            epochs: number of training epochs
            lr: learning rate
            batch_size: batch size for gradient estimation
            verbose: print progress

        Returns:
            training history dict
        """
        n_samples = len(training_data)
        n_bits = min(self.n_qubits, training_data.shape[1])

        indices = np.arange(n_samples)

        self.params = np.random.uniform(0, np.pi, size=(self.n_layers, self.n_qubits))

        history = {"loss": [], "mmd": []}
        optimizer = "Adam"

        for epoch in range(epochs):
            np.random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = max(n_samples // batch_size, 1)

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_samples)
                batch_indices = indices[batch_start:batch_end]

                batch_data = training_data[batch_indices, :n_bits]

                target_probs = np.zeros(2 ** n_bits)
                for row in batch_data:
                    idx = int("".join(str(int(b)) for b in row), 2)
                    target_probs[idx] += 1
                target_probs /= target_probs.sum() + 1e-10

                grad = np.zeros_like(self.params)
                eps = 0.01

                for i in range(self.n_layers):
                    for j in range(self.n_qubits):
                        params_plus = self.params.copy()
                        params_plus[i, j] += eps
                        probs_plus = self.circuit(params_plus)
                        loss_plus = self._mmd_loss_quantum(probs_plus, target_probs)

                        params_minus = self.params.copy()
                        params_minus[i, j] -= eps
                        probs_minus = self.circuit(params_minus)
                        loss_minus = self._mmd_loss_quantum(probs_minus, target_probs)

                        grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

                self.params -= lr * grad

                current_probs = self.circuit(self.params)
                current_loss = self._mmd_loss_quantum(current_probs, target_probs)
                epoch_loss += current_loss

            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)

            current_probs = self.circuit(self.params)
            mmd = self._mmd_loss_quantum(current_probs, target_probs)
            history["mmd"].append(mmd)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  QBM Epoch {epoch+1}/{epochs} | MMD: {mmd:.6f} | Loss: {avg_loss:.6f}")

        self.fitted = True
        return history

    def generate(self, n_samples: int = 100, shots: int = 1000) -> np.ndarray:
        """
        Generate synthetic event samples from the trained QBM.

        Args:
            n_samples: number of samples to generate
            shots: number of circuit evaluations per sample

        Returns:
            Array of shape (n_samples, n_qubits) with binary samples
        """
        if not self.fitted:
            raise ValueError("QBM not fitted. Call fit() first.")

        probs = self.circuit(self.params)

        samples = []
        for _ in range(n_samples):
            outcome = np.random.choice(len(probs), p=probs)
            binary = np.array([int(b) for b in format(outcome, f"0{self.n_qubits}b")])
            samples.append(binary)

        return np.array(samples)

    def generate_with_shots(self, n_samples: int = 1000, shots: int = 100) -> np.ndarray:
        """Generate using shot-based measurement (more realistic for NISQ)."""
        dev_shots = qml.device(self.device, wires=self.n_qubits, shots=shots)

        @qml.qnode(dev_shots, diff_method=None)
        def circuit_shots(params):
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.sample(qml.PauliZ(i)) for i in range(self.n_qubits)]

        samples = []
        for _ in range(n_samples):
            result = circuit_shots(self.params)
            sample = np.array([(1 - r) / 2 for r in result])
            samples.append(sample)

        return np.array(samples)


# =============================================================================
# HYBRID LATENT STYLE-BASED QGAN
# =============================================================================

class HybridStyleQGAN:
    """
    Hybrid Latent Style-Based Quantum Generative Adversarial Network.

    Architecture:
    1. Classical Autoencoder: compress event sequences → latent space (dim=latent_dim)
    2. Quantum Generator: VQC operating on latent vector → synthetic quantum state
    3. Classical Discriminator: distinguishes real from synthetic latent codes
    4. Style Vector: encodes temporal context (season, year, region type)

    Key advantage: exponential parameter scaling (Liepelt & Baglio 2026)
    Fewer quantum parameters achieve same quality as large classical generators.

    Reference: Baglio style-based QGAN (arXiv:2405.04401, 2406.02668, 2601.05036)
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 4,
        latent_dim: int = 16,
        event_dim: int = 32,
        style_dim: int = 8,
        device: str = "default.qubit",
        torch_device: str = "cpu",
        lr_g: float = 1e-3,
        lr_d: float = 1e-3,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.event_dim = event_dim
        self.style_dim = style_dim
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Classical components can run on CUDA; quantum circuit stays on CPU
        self.torch_device = torch.device(torch_device)

        self.autoencoder = Autoencoder(event_dim, latent_dim, style_dim).to(self.torch_device)
        self.discriminator = Discriminator(latent_dim + style_dim).to(self.torch_device)

        # QGeneratorVQC kept on CPU — PennyLane simulation is CPU-based
        self.q_generator = QGeneratorVQC(latent_dim, n_qubits, n_layers, style_dim)

        self.opt_g = torch.optim.AdamW(
            list(self.autoencoder.parameters()) + list(self.q_generator.parameters()),
            lr=lr_g, weight_decay=1e-4
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr_d, weight_decay=1e-4
        )

        self.history = {"g_loss": [], "d_loss": [], "mmd": []}
        self.fitted = False

    def _create_quantum_circuit(self):
        """Create the VQC generator circuit using PennyLane."""
        dev = qml.device(self.device, wires=self.n_qubits)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(latent_input, style_input, weights):
            AngleEmbedding(latent_input[:self.n_qubits], wires=range(self.n_qubits), rotation="Y")

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i] * latent_input[i % len(latent_input)], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def _quantum_kernel(self, probs1, probs2):
        """Compute quantum kernel MMD between two distributions."""
        p = probs1 / (probs1.sum() + 1e-10)
        q = probs2 / (probs2.sum() + 1e-10)
        return np.sqrt(np.sum((p - q) ** 2))

    def _encode_to_fake(self, real_events, real_styles):
        """Encode real events, run through quantum generator, return latent and style."""
        z_real = self.autoencoder.encode(real_events)
        real_style = self.autoencoder.encode_style(real_styles)
        # quantum generator runs on CPU; move inputs/outputs across device boundary
        z_fake_cpu = self.q_generator(z_real.cpu(), real_style.cpu())
        z_fake = z_fake_cpu.to(self.torch_device)
        return z_real, real_style, z_fake

    def train_step(self, real_events, real_styles, lambda_gp=10.0):
        """Single generator training step (G only, no D update)."""
        batch_size = real_events.shape[0]
        real_labels = torch.ones(batch_size, 1).to(self.torch_device)

        z_real, real_style, z_fake = self._encode_to_fake(real_events, real_styles)

        combined_fake = torch.cat([z_fake, real_style], dim=1)
        d_fake_for_g = self.discriminator(combined_fake)
        g_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            d_fake_for_g, real_labels
        )

        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return g_loss.item()

    def _gradient_penalty(self, disc, real, fake):
        alpha = torch.rand(real.size(0), 1).to(self.torch_device)
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)
        disc_interp = disc(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interp, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interp),
            create_graph=True, retain_graph=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def fit(
        self,
        event_sequences: np.ndarray,
        temporal_contexts: np.ndarray,
        epochs: int = 300,
        batch_size: int = 32,
        n_critic: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train the Hybrid Style-Based QGAN.

        Args:
            event_sequences: array of shape (n_samples, event_dim)
                             e.g., binned case counts per spatial cell
            temporal_contexts: array of shape (n_samples, style_dim)
                              e.g., [month_sin, month_cos, year, ...]
            epochs: training epochs
            batch_size: batch size
            n_critic: discriminator updates per generator update
            verbose: print progress
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(event_sequences),
            torch.FloatTensor(temporal_contexts)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            for events, styles in loader:
                events = events.to(self.torch_device)
                styles = styles.to(self.torch_device)

                # Train D n_critic times (standard WGAN-GP pattern)
                for _ in range(n_critic):
                    d_loss = self._train_discriminator(events, styles)

                # Train G exactly once
                g_loss = self.train_step(events, styles)
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                n_batches += 1

            self.history["g_loss"].append(epoch_g_loss / max(n_batches, 1))
            self.history["d_loss"].append(epoch_d_loss / max(n_batches, 1))

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  QGAN Epoch {epoch+1}/{epochs} | G_loss: {epoch_g_loss/n_batches:.4f} | D_loss: {epoch_d_loss/n_batches:.4f}")

        self.fitted = True
        return self.history

    def _train_discriminator(self, events, styles):
        batch_size = events.shape[0]
        real_labels = torch.ones(batch_size, 1).to(self.torch_device)
        fake_labels = torch.zeros(batch_size, 1).to(self.torch_device)

        with torch.no_grad():
            z_real = self.autoencoder.encode(events)
            style = self.autoencoder.encode_style(styles)
        combined_real = torch.cat([z_real, style], dim=1)
        d_real = self.discriminator(combined_real)

        # quantum generator on CPU
        z_fake_cpu = self.q_generator(z_real.cpu(), style.cpu())
        z_fake = z_fake_cpu.to(self.torch_device)
        combined_fake = torch.cat([z_fake, style], dim=1)
        d_fake = self.discriminator(combined_fake.detach())

        # both combined_real and combined_fake have shape (batch, latent_dim+style_dim)
        gp = self._gradient_penalty(self.discriminator, combined_real, combined_fake.detach())

        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(d_real, real_labels) +
            torch.nn.functional.binary_cross_entropy_with_logits(d_fake, fake_labels) +
            10.0 * gp
        )

        self.opt_d.zero_grad()
        loss.backward()
        self.opt_d.step()

        return loss.item()

    def generate(
        self,
        event_sequences: np.ndarray,
        temporal_contexts: np.ndarray,
    ) -> np.ndarray:
        """
        Generate synthetic event sequences conditioned on real data.

        Args:
            event_sequences: conditioning events (n_samples, event_dim)
            temporal_contexts: temporal context vectors (n_samples, style_dim)

        Returns:
            Generated synthetic event sequences (n_samples, event_dim)
        """
        if not self.fitted:
            raise ValueError("QGAN not fitted. Call fit() first.")

        self.autoencoder.eval()
        self.q_generator.eval()

        with torch.no_grad():
            events_t = torch.FloatTensor(event_sequences).to(self.torch_device)
            styles_t = torch.FloatTensor(temporal_contexts).to(self.torch_device)

            z = self.autoencoder.encode(events_t)
            style = self.autoencoder.encode_style(styles_t)

            # quantum generator on CPU, then move output back for decoder
            z_fake = self.q_generator(z.cpu(), style.cpu()).to(self.torch_device)
            synthetic = self.autoencoder.decode(z_fake)

        return synthetic.cpu().numpy()

    def compute_distribution_quality(self, real, generated):
        """Compute MMD and Wasserstein distance between real and generated."""
        real_flat = real.reshape(len(real), -1)
        gen_flat = generated.reshape(len(generated), -1)

        n_r = len(real_flat)
        n_g = len(gen_flat)

        mmd = 0.0
        for _ in range(min(n_r, 100)):
            i = np.random.randint(n_r)
            j = np.random.randint(n_g)
            diff = real_flat[i] - gen_flat[j]
            mmd += np.dot(diff, diff)
        mmd /= (n_r * n_g)

        w_dist = np.mean([
            wasserstein_distance(real_flat[i], gen_flat[i])
            for i in range(min(n_r, n_g))
        ])

        return {"mmd": np.sqrt(mmd), "wasserstein": w_dist}


# =============================================================================
# HYBRID MODULE DEFINITIONS
# =============================================================================

class Autoencoder(nn.Module):
    """Classical autoencoder for event sequence compression."""

    def __init__(self, event_dim, latent_dim, style_dim):
        super().__init__()
        self.style_dim = style_dim

        self.encoder = nn.Sequential(
            nn.Linear(event_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        )

        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 16),
            nn.GELU(),
            nn.Linear(16, style_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, event_dim),
            nn.Softplus(),
        )

    def encode(self, x):
        return self.encoder(x)

    def encode_style(self, s):
        return self.style_encoder(s)

    def decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Classical discriminator for latent space."""

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class QGeneratorVQC(nn.Module):
    """
    Variational Quantum Circuit Generator as a PyTorch nn.Module.

    The QGeneratorVQC uses PennyLane's qnode inside a PyTorch module
    for seamless integration with autograd.
    """

    def __init__(self, latent_dim, n_qubits, n_layers, style_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.style_dim = style_dim

        self.dev = qml.device("default.qubit", wires=n_qubits)

        self.weight_shapes = {
            "weights": (n_layers, n_qubits, 3)
        }

        # backprop is much faster than parameter-shift for simulation
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(latent, style, weights):
            n_encode = min(len(latent), n_qubits)
            for i in range(n_encode):
                qml.RY(latent[i % len(latent)] * np.pi, wires=i)
            for i in range(n_encode):
                qml.RZ(style[i % len(style)] * np.pi, wires=i)

            StronglyEntanglingLayers(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit
        # StronglyEntanglingLayers expects shape (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, latent, style):
        """latent/style arrive on CPU (quantum circuits are CPU-only)."""
        target_device = latent.device
        batch_size = latent.shape[0]

        results = []
        for i in range(batch_size):
            z_vals = self.quantum_circuit(latent[i], style[i], self.weights)
            # z_vals is a list of scalar tensors from expval
            z_t = torch.stack([
                v if isinstance(v, torch.Tensor) else torch.tensor(float(v), dtype=torch.float32)
                for v in z_vals
            ])
            results.append(z_t)

        # PennyLane returns float64; cast to float32 before projection
        z_batch = torch.stack(results).float().to(target_device)  # (batch, n_qubits)
        return self.output_proj(z_batch)


# =============================================================================
# AUGMENTATION PIPELINE
# =============================================================================

class QuantumAugmentationPipeline:
    """
    Full pipeline for quantum-augmented spatio-temporal event generation.

    Pipeline:
    1. Encode event sequences to quantum-compatible representation
    2. Train quantum generative model (QBM or QGAN)
    3. Generate synthetic events
    4. Validate with K/L function comparison
    5. Combine with original data
    """

    def __init__(
        self,
        model_type: str = "qgan",
        n_qubits: int = 8,
        n_layers: int = 4,
        latent_dim: int = 16,
        augmentation_ratio: int = 3,
        seed: int = 42,
        torch_device: str = "cpu",
    ):
        self.model_type = model_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.augmentation_ratio = augmentation_ratio
        self.seed = seed
        self.torch_device = torch_device
        self.fitted = False

    def _encode_events(self, events_df, n_bins=8):
        """
        Encode event sequences into quantum-compatible representations.

        Returns:
            event_vectors: discretized event count vectors
            context_vectors: temporal/spatial context
        """
        np.random.seed(self.seed)

        if len(events_df) == 0:
            return np.array([]), np.array([])

        events_df = events_df.copy()
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
        events_df = events_df.sort_values("timestamp")

        max_cases = events_df["case_count"].quantile(0.99) + 1
        events_df["case_bin"] = np.clip(
            (events_df["case_count"] / max_cases * (n_bins - 1)).astype(int), 0, n_bins - 1
        )

        events_df["month_sin"] = np.sin(2 * np.pi * events_df["timestamp"].dt.month / 12)
        events_df["month_cos"] = np.cos(2 * np.pi * events_df["timestamp"].dt.month / 12)
        events_df["year_norm"] = (events_df["year"] - events_df["year"].min()) / (
            events_df["year"].max() - events_df["year"].min() + 1e-9
        )
        events_df["lat_norm"] = (events_df["lat"] - events_df["lat"].min()) / (
            events_df["lat"].max() - events_df["lat"].min() + 1e-9
        )
        events_df["lon_norm"] = (events_df["lon"] - events_df["lon"].min()) / (
            events_df["lon"].max() - events_df["lon"].min() + 1e-9
        )

        event_vectors = events_df[["case_bin", "lat_norm", "lon_norm"]].values.astype(np.float32)

        context_vectors = events_df[["month_sin", "month_cos", "year_norm", "lat_norm", "lon_norm"]].values.astype(np.float32)

        if event_vectors.shape[1] < self.latent_dim:
            pad = np.zeros((len(event_vectors), self.latent_dim - event_vectors.shape[1]))
            event_vectors = np.concatenate([event_vectors, pad], axis=1)
        elif event_vectors.shape[1] > self.latent_dim:
            event_vectors = event_vectors[:, :self.latent_dim]

        return event_vectors, context_vectors

    def fit(self, events_df, epochs=300, verbose=True):
        """Train the quantum augmentation model."""
        event_vectors, context_vectors = self._encode_events(events_df)

        if len(event_vectors) < 10:
            print("  Warning: Too few events for quantum augmentation")
            return self

        n_total = len(event_vectors)
        n_train = int(0.8 * n_total)

        perm = np.random.permutation(n_total)
        train_events = event_vectors[perm[:n_train]]
        train_contexts = context_vectors[perm[:n_train]]

        if self.model_type == "qbm":
            self.model = QuantumBornMachine(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
            )

            discretized = np.zeros((len(train_events), self.n_qubits))
            for i, ev in enumerate(train_events):
                for j in range(min(len(ev), self.n_qubits)):
                    discretized[i, j] = 1 if ev[j] > 0.5 else 0

            self.model.fit(discretized, epochs=epochs, lr=0.01, verbose=verbose)

        elif self.model_type == "qgan":
            self.model = HybridStyleQGAN(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                latent_dim=self.latent_dim,
                event_dim=event_vectors.shape[1],
                style_dim=context_vectors.shape[1],
                lr_g=1e-3,
                lr_d=1e-3,
                seed=self.seed,
                torch_device=self.torch_device,
            )
            self.model.fit(train_events, train_contexts, epochs=epochs, verbose=verbose)

        self.event_vectors = event_vectors
        self.context_vectors = context_vectors
        self.fitted = True
        return self

    def generate(self, n_samples=None, events_df=None):
        """Generate synthetic event sequences."""
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if n_samples is None:
            n_samples = len(self.event_vectors) * self.augmentation_ratio

        if events_df is not None:
            ev, ctx = self._encode_events(events_df)
            if len(ev) > 0:
                generated = self.model.generate(ev, ctx)
                return generated

        if self.model_type == "qbm":
            samples = self.model.generate(n_samples=n_samples)
            return samples

        elif self.model_type == "qgan":
            n_ref = min(len(self.event_vectors), n_samples)
            idx = np.random.choice(len(self.event_vectors), n_ref, replace=True)
            ref_events = self.event_vectors[idx]
            ref_contexts = self.context_vectors[idx]

            generated = self.model.generate(ref_events, ref_contexts)

            extra = n_samples - n_ref
            if extra > 0:
                extra_idx = np.random.choice(len(self.event_vectors), extra, replace=True)
                extra_events = self.event_vectors[extra_idx]
                extra_contexts = self.context_vectors[extra_idx]
                extra_gen = self.model.generate(extra_events, extra_contexts)
                generated = np.concatenate([generated, extra_gen], axis=0)

            return generated

    def validate_distribution(self, original, generated):
        """Validate generated distribution matches original."""
        if len(original) == 0 or len(generated) == 0:
            return {"error": "Empty data"}

        return {
            "mmd": np.sqrt(np.mean((original.mean(axis=0) - generated.mean(axis=0)) ** 2)),
            "wasserstein_mean": wasserstein_distance(
                original[:, 0], generated[:, 0]
            ) if original.shape[1] > 0 else 0.0,
            "mean_cases_diff": abs(original[:, 0].mean() - generated[:, 0].mean()),
        }
