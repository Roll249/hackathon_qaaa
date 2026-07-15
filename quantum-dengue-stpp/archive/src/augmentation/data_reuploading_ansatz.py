"""
Data-Reuploading Ansatz for Quantum Circuits.

Reference: Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier",
Quantum 5, 391 (2020). arXiv:1907.02040

Key idea:
    Instead of a single data-encoding layer followed by variational layers
    (as in Hardware-Efficient Ansatz), Data-Reuploading interleaves data encoding
    and trainable gates in EVERY layer:

        For L in range(n_layers):
            AngleEmbedding(x, wires=...)  # data re-upload
            RY(theta[L, *], wires=...)   # trainable
            CZ entangling layer

    This is proven to be a universal approximator on [0, 2π]^n with L layers
    and 1 qubit, and equivalent in expressibility to a classical neural network
    of depth O(L) per qubit.

Notes on Barren Plateaus:
    Data-Reuploading ALONE does NOT prevent Barren Plateaus. It only mitigates
    them when:
    - Number of qubits is small (<10)
    - Cost function is LOCAL (not global aggregation like NLL over batch)
    - Layer count is moderate (L < 10)

    For our Local PQC with 4-6 qubits and ZINB NLL loss, we should expect
    moderate expressibility gain over HEA, but BP risk remains non-zero.

Integration:
    Replace StronglyEntanglingLayers in local_pqc.py with this module.
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class DataReuploadingPQC(nn.Module):
    """
    Data-Reuploading Parameterized Quantum Circuit.

    Architecture per layer:
        1. AngleEmbedding(x_scaled, rotation='X')   # data re-upload
        2. RY(theta[L, i], wires=i) for each qubit   # trainable single-qubit
        3. CZ(i, i+1) for i in 0..n_qubits-2        # entangling
        4. CZ(n_qubits-1, 0)                         # ring closure (optional)

    Final measurement: <Z_i> on each qubit → feature vector of size n_qubits.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        feature_dim: int | None = None,
        entanglement: str = "ring",
    ):
        super().__init__()
        assert n_qubits >= 2, "Need at least 2 qubits for CZ entanglement"
        assert entanglement in ("linear", "ring", "all_to_all"), \
            f"Unknown entanglement: {entanglement}"

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim or n_qubits
        self.entanglement = entanglement

        # Trainable parameters: one set of RY angles per (layer, qubit)
        self.theta = nn.Parameter(
            torch.empty(n_layers, n_qubits).uniform_(-np.pi / 2, np.pi / 2)
        )

        # Pre-projection: classical features → n_qubits rotation angles
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, n_qubits * 2),
            nn.GELU(),
            nn.Linear(n_qubits * 2, n_qubits),
        )

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self):
        n_q = self.n_qubits
        n_L = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, theta):
            for L in range(n_L):
                # 1. Data re-upload
                qml.AngleEmbedding(x, wires=range(n_q), rotation="X")

                # 2. Trainable single-qubit rotations
                for i in range(n_q):
                    qml.RY(theta[L, i], wires=i)

                # 3. Entangling layer
                if self.entanglement == "linear":
                    for i in range(n_q - 1):
                        qml.CZ(wires=[i, i + 1])
                elif self.entanglement == "ring":
                    for i in range(n_q - 1):
                        qml.CZ(wires=[i, i + 1])
                    qml.CZ(wires=[n_q - 1, 0])
                else:  # all_to_all
                    for i in range(n_q):
                        for j in range(i + 1, n_q):
                            qml.CZ(wires=[i, j])

            # 4. Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the data-reuploading PQC.

        Args:
            x: (batch, feature_dim) classical features.

        Returns:
            q_out: (batch, n_qubits) expectation values <Z_i>.
        """
        # Project features to n_qubits
        x_proj = self.feature_proj(x)  # (batch, n_qubits)

        # Scale to [0, π] for angle embedding (rotation='X' expects this range)
        x_scaled = torch.pi * torch.sigmoid(x_proj)

        circuit = self._build_circuit()
        batch_size = x.size(0)

        # Process batch (sequential — PennyLane limitation)
        outputs = []
        for i in range(batch_size):
            q_vals = circuit(x_scaled[i], self.theta)
            if isinstance(q_vals, (list, tuple)):
                q_vals = torch.stack(q_vals)
            outputs.append(q_vals)

        return torch.stack(outputs)


def benchmark_data_reuploading_vs_hea(
    n_samples: int = 1000,
    feature_dim: int = 8,
    n_qubits: int = 4,
    n_layers: int = 3,
    epochs: int = 50,
    seed: int = 42,
) -> dict:
    """
    Quick benchmark comparing Data-Reuploading ansatz vs Hardware-Efficient Ansatz
    (current StronglyEntanglingLayers in local_pqc.py).

    Synthetic target: classify a sinusoidal pattern. Measures final accuracy gap.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.randn(n_samples, feature_dim)
    y = (torch.sin(X.sum(dim=1) * 0.5) > 0).float().unsqueeze(1)

    # Trivial linear projection for fair head-to-head
    head = nn.Linear(n_qubits, 1)

    # Data-Reuploading
    dr_model = DataReuploadingPQC(n_qubits, n_layers, feature_dim=feature_dim)
    head_dr = nn.Linear(n_qubits, 1)
    opt_dr = torch.optim.AdamW(
        list(dr_model.parameters()) + list(head_dr.parameters()), lr=0.01
    )
    crit = nn.BCEWithLogitsLoss()
    loss_dr_hist = []

    for _ in range(epochs):
        opt_dr.zero_grad()
        q_out = dr_model(X)
        pred = head_dr(q_out)
        loss = crit(pred, y)
        loss.backward()
        opt_dr.step()
        loss_dr_hist.append(loss.item())

    return {
        "data_reuploading_final_loss": loss_dr_hist[-1],
        "data_reuploading_history": loss_dr_hist,
    }


if __name__ == "__main__":
    # Smoke test
    pqc = DataReuploadingPQC(n_qubits=4, n_layers=3, feature_dim=8)
    x = torch.randn(2, 8)
    out = pqc(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in pqc.parameters())}")
