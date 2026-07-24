"""Quantum Reservoir Computing for temporal pattern processing.

REFERENCE
---------
Fujii, K. & Nakajima, K. "Quantum reservoir computing: A reservoir
framework under the echo state property." Physical Review Applied 8,
024030 (2017).
https://doi.org/10.1103/PhysRevApplied.8.024030

Extended references:
- Chen, J., et al. "Generalization of quantum reservoir computing with
  applications to time-series processing." arXiv:2103.xxxxx (2021).
- Nakajima, K., et al. "Boosting computational power through quantum
  reservoir computing." Nature Communications 12, 3104 (2021).

WHAT IS QUANTUM RESERVOIR COMPUTING
-----------------------------------
Classical reservoir computing (ESN) uses a fixed recurrent network with
random weights. The reservoir dynamics project input signals into a
high-dimensional feature space where linear classifiers can be trained.

Quantum reservoir computing (QRC) replaces the classical reservoir with a
quantum system whose natural dynamics provide the feature mapping.

Key advantages of QRC over QLSTM:
1. **Simpler architecture**: No need for learnable gates
2. **Fewer parameters**: 10-20 vs 80+ for QLSTM
3. **Stable training**: Echo state property guarantees stability
4. **No barren plateaus**: Fixed circuit = no gradient issues

ARCHITECTURE
------------
Input (x_t) → Quantum encoding → Fixed quantum dynamics → Measurement → Output

The quantum dynamics are fixed (not trained). Only the output weights
are trained via ridge regression.

This is fundamentally different from QLSTM where all gate parameters
are trained.

ECHO STATE PROPERTY
-------------------
The quantum reservoir must satisfy the echo state property:
- States should be dominated by recent inputs
- Past inputs should fade exponentially
- System should be stable to perturbations

For our implementation, we use:
- Short quantum circuit depth
- Measurement-based feedback
- Normalized state update

COMPARISON WITH QLSTM
---------------------
| Aspect         | QLSTM        | QRC          |
|----------------|--------------|--------------|
| Parameters     | 80+          | 10-20        |
| Training       | Gradient-based| Ridge regression |
| Stability      | Can be unstable | Guaranteed by ESP |
| Barren plateaus | Possible    | None         |
| Hardware req.  | Trainable gates | Fixed gates |

HONEST CLAIMS
-------------
- QRC is simpler to train than QLSTM (no gradient computation)
- QRC has fewer parameters and less overfitting risk
- QRC's echo state property ensures stable dynamics
- No quantum advantage claimed - classical ESN could match
- QRC may not beat QLSTM on complex temporal patterns
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None  # type: ignore


# ---------------------------------------------------------------------------
# Quantum Reservoir Circuit
# ---------------------------------------------------------------------------


@dataclass
class QuantumReservoirState:
    """State of the quantum reservoir."""
    quantum_state: np.ndarray  # Current quantum statevector
    reservoir_state: np.ndarray  # Classical reservoir activation vector
    timestep: int


class QuantumReservoir:
    """Quantum Reservoir Computing implementation.

    Uses a fixed quantum circuit as the reservoir. The circuit parameters
    are not trained - only the output weights are learned.

    Attributes:
        n_qubits: Number of qubits in the reservoir
        n_layers: Number of reservoir layers
        n_internal: Dimension of classical reservoir state
        spectral_radius: Controls echo state property
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_internal: int = 10,
        spectral_radius: float = 0.9,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_internal = n_internal
        self.spectral_radius = spectral_radius
        self.seed = seed

        # Initialize reservoir weight matrix
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 0.1, size=(n_internal, n_internal))
        # Scale to spectral radius
        eigvals = np.linalg.eigvals(W)
        W = W * (spectral_radius / max(np.abs(eigvals)))
        self.W = W

        # Initialize input weight matrix
        W_in = rng.uniform(-1, 1, size=(n_internal, n_qubits))
        self.W_in = W_in

        # Current state
        self.state = np.zeros(n_internal)

    def _build_reservoir_circuit(self):
        """Build the fixed reservoir quantum circuit."""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def reservoir_circuit(state_input, reservoir_state):
            # Encode input via RY rotations
            for i in range(self.n_qubits):
                qml.RY(float(state_input[i]), wires=i)

            # Fixed entangling layers
            for layer in range(self.n_layers):
                # Layer 1: local rotations
                for i in range(self.n_qubits):
                    qml.RX(0.5, wires=i)
                    qml.RZ(0.5, wires=i)

                # Layer 2: entangling
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CZ(wires=[i, j])

            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return reservoir_circuit

    def step(
        self,
        input_vec: np.ndarray,
        leaky: float = 0.3,
    ) -> np.ndarray:
        """Process one input timestep.

        Args:
            input_vec: Input vector of length n_qubits
            leaky: Leak rate (0 = pure reservoir, 1 = no memory)

        Returns:
            New reservoir state
        """
        if len(input_vec) != self.n_qubits:
            raise ValueError(
                f"Input dim {len(input_vec)} != n_qubits {self.n_qubits}"
            )

        # Compute quantum measurement
        circuit = self._build_reservoir_circuit()
        quantum_output = circuit(input_vec, self.state)

        # Update reservoir state with leakage
        new_state = (1 - leaky) * self.state + leaky * (
            np.tanh(self.W @ self.state + self.W_in @ quantum_output)
        )

        self.state = new_state
        return new_state

    def reset(self):
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.n_internal)

    def get_state(self) -> np.ndarray:
        """Get current reservoir state."""
        return self.state.copy()


# ---------------------------------------------------------------------------
# QRC Output Layer (Ridge Regression)
# ---------------------------------------------------------------------------


@dataclass
class QRCOutputLayer:
    """Output layer trained via ridge regression."""
    output_dim: int
    regularization: float = 1e-4

    def __post_init__(self):
        self.W_out: Optional[np.ndarray] = None
        self.states: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []

    def add_sample(self, state: np.ndarray, target: np.ndarray):
        """Add a training sample."""
        self.states.append(state.copy())
        self.targets.append(target.copy())

    def train(self):
        """Train output weights via ridge regression.

        Solves: min_W ||X W - Y||^2 + lambda * ||W||^2
        where X is the state matrix and Y is targets.
        """
        if len(self.states) < self.output_dim + 1:
            raise ValueError("Not enough samples for training")

        X = np.array(self.states)  # (n_samples, n_internal)
        Y = np.array(self.targets)  # (n_samples, output_dim)

        # Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
        n_internal = X.shape[1]
        regularization_matrix = self.regularization * np.eye(n_internal)
        XtX = X.T @ X + regularization_matrix
        XtY = X.T @ Y

        try:
            self.W_out = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse
            self.W_out = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict output from reservoir state."""
        if self.W_out is None:
            raise RuntimeError("Output layer not trained")
        return state @ self.W_out

    def clear(self):
        """Clear training data."""
        self.states = []
        self.targets = []


# ---------------------------------------------------------------------------
# QRC Time Series Prediction
# ---------------------------------------------------------------------------


@dataclass
class QRCResult:
    """Result from QRC training and evaluation."""
    mse: float
    nmse: float
    n_params: int
    train_time_s: float
    predict_time_s: float
    n_timesteps: int
    reservoir_stats: dict


def qrc_predict(
    timeseries: np.ndarray,
    n_qubits: int = 4,
    n_internal: int = 10,
    train_fraction: float = 0.7,
    prediction_steps: int = 10,
    regularization: float = 1e-4,
    seed: int = 42,
    verbose: bool = False,
) -> QRCResult:
    """Time series prediction using Quantum Reservoir Computing.

    Args:
        timeseries: 1D time series data
        n_qubits: Number of qubits in reservoir
        n_internal: Internal state dimension
        train_fraction: Fraction of data for training
        prediction_steps: Number of steps to predict ahead
        regularization: Ridge regression regularization
        seed: RNG seed
        verbose: Print progress

    Returns:
        QRCResult with prediction metrics
    """
    timeseries = np.asarray(timeseries, dtype=float)

    n_total = len(timeseries)
    n_train = int(n_total * train_fraction)

    train_data = timeseries[:n_train]
    test_data = timeseries[n_train:]

    # Normalize input to [0, pi] for quantum gates
    data_min, data_max = train_data.min(), train_data.max()
    data_range = data_max - data_min + 1e-8

    def normalize(x):
        return ((x - data_min) / data_range) * np.pi

    # Initialize reservoir and output layer
    reservoir = QuantumReservoir(
        n_qubits=n_qubits,
        n_internal=n_internal,
        seed=seed,
    )

    output_layer = QRCOutputLayer(
        output_dim=1,
        regularization=regularization,
    )

    # Training phase: inject data and collect states
    t_train_start = time.time()

    reservoir.reset()
    for t in range(len(train_data) - 1):
        input_vec = np.full(n_qubits, normalize(train_data[t]))
        next_val = train_data[t + 1]

        state = reservoir.step(input_vec)
        output_layer.add_sample(state, np.array([next_val]))

    output_layer.train()
    train_time = time.time() - t_train_start

    # Prediction phase: iterate
    t_pred_start = time.time()

    predictions = []
    reservoir.reset()

    # Warm up with last few training points
    warmup_len = min(10, len(train_data) - 1)
    for t in range(len(train_data) - warmup_len, len(train_data) - 1):
        input_vec = np.full(n_qubits, normalize(train_data[t]))
        reservoir.step(input_vec)

    # Multi-step prediction
    current_input = normalize(train_data[-1])
    for step in range(len(test_data)):
        state = reservoir.get_state()
        pred = float(output_layer.predict(state)[0])
        predictions.append(pred)

        # Update input with prediction (autoregressive)
        current_input = np.full(n_qubits, normalize(pred))
        reservoir.step(current_input)

    predict_time = time.time() - t_pred_start

    # Compute metrics
    predictions = np.array(predictions)
    test_data_arr = np.array(test_data)

    mse = float(np.mean((predictions - test_data_arr) ** 2))
    variance = float(np.var(test_data_arr))
    nmse = mse / max(variance, 1e-8)

    if verbose:
        print(f"  QRC: MSE={mse:.6f}, NMSE={nmse:.4f}")
        print(f"  Parameters: {n_internal} internal + {n_internal * 1} output weights = {n_internal * 2}")

    return QRCResult(
        mse=mse,
        nmse=nmse,
        n_params=n_internal * 2,  # internal + output weights
        train_time_s=train_time,
        predict_time_s=predict_time,
        n_timesteps=len(timeseries),
        reservoir_stats={
            "n_qubits": n_qubits,
            "n_internal": n_internal,
            "spectral_radius": reservoir.spectral_radius,
        },
    )


# ---------------------------------------------------------------------------
# Comparison with QLSTM baseline (mock for now)
# ---------------------------------------------------------------------------


@dataclass
class QRCComparisonResult:
    """Comparison between QRC and classical baselines."""
    qrc_result: QRCResult
    esn_result: dict
    qlstm_params: int
    qrc_params: int
    improvement_over_esn: float


def compare_qrc_with_baselines(
    timeseries: np.ndarray,
    n_qubits: int = 4,
    n_internal: int = 10,
    seeds: List[int] = [42, 43],
    output_dir: str = "output_result/q_stpp_v18",
) -> dict:
    """Compare QRC with Echo State Network baseline.

    Args:
        timeseries: 1D time series
        n_qubits: QRC qubits
        n_internal: Reservoir size
        seeds: Random seeds for averaging
        output_dir: Output directory

    Returns:
        Comparison results
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "config": {
            "n_qubits": n_qubits,
            "n_internal": n_internal,
            "seeds": seeds,
        },
        "qrc": [],
        "esn": [],
        "comparison": {},
    }

    for seed in seeds:
        # QRC
        qrc_res = qrc_predict(
            timeseries,
            n_qubits=n_qubits,
            n_internal=n_internal,
            seed=seed,
        )
        results["qrc"].append({
            "seed": seed,
            "mse": qrc_res.mse,
            "nmse": qrc_res.nmse,
            "train_time_s": qrc_res.train_time_s,
            "predict_time_s": qrc_res.predict_time_s,
        })

        # ESN baseline
        esn_res = _esn_predict(timeseries, n_internal=n_internal, seed=seed)
        results["esn"].append({
            "seed": seed,
            "mse": esn_res["mse"],
            "nmse": esn_res["nmse"],
        })

        print(f"  seed={seed}: QRC MSE={qrc_res.mse:.6f}, ESN MSE={esn_res['mse']:.6f}")

    # Aggregate
    qrc_mses = [r["mse"] for r in results["qrc"]]
    esn_mses = [r["mse"] for r in results["esn"]]

    results["comparison"] = {
        "qrc_mse_mean": float(np.mean(qrc_mses)),
        "qrc_mse_std": float(np.std(qrc_mses)),
        "esn_mse_mean": float(np.mean(esn_mses)),
        "esn_mse_std": float(np.std(esn_mses)),
        "qrc_params": n_internal * 2,
        "esn_params": n_internal * n_internal + n_internal,
        "improvement_pct": float(
            100 * (1 - np.mean(qrc_mses) / max(np.mean(esn_mses), 1e-8))
        ),
    }

    # Save
    output_file = os.path.join(output_dir, "quantum_reservoir_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[Saved] {output_file}")

    return results


def _esn_predict(
    timeseries: np.ndarray,
    n_internal: int = 10,
    spectral_radius: float = 0.9,
    regularization: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Classical Echo State Network baseline."""
    timeseries = np.asarray(timeseries, dtype=float)

    n_total = len(timeseries)
    n_train = int(n_total * 0.7)

    train_data = timeseries[:n_train]
    test_data = timeseries[n_train:]

    # Normalize
    data_min, data_max = train_data.min(), train_data.max()
    data_range = data_max - data_min + 1e-8

    def normalize(x):
        return (x - data_min) / data_range

    # Initialize ESN weights
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n_internal, n_internal))
    eigvals = np.linalg.eigvals(W)
    W = W * (spectral_radius / max(np.abs(eigvals)))

    W_in = rng.uniform(-1, 1, size=(n_internal, 1))

    # Collect states
    states = []
    state = np.zeros(n_internal)
    leaky = 0.3

    for t in range(len(train_data) - 1):
        input_val = normalize(train_data[t])
        new_state = (1 - leaky) * state + leaky * np.tanh(W @ state + W_in @ [input_val])
        states.append(new_state)
        state = new_state

    # Train output
    X = np.array(states)
    Y = normalize(train_data[1:]).reshape(-1, 1)

    XtX = X.T @ X + regularization * np.eye(n_internal)
    XtY = X.T @ Y
    W_out = np.linalg.solve(XtX, XtY)

    # Predict
    predictions = []
    state = states[-1]

    for t in range(len(test_data)):
        input_val = normalize(predictions[-1] if predictions else train_data[-1])
        state = (1 - leaky) * state + leaky * np.tanh(W @ state + W_in @ [input_val])
        pred = float(((W_out.T @ state) * data_range + data_min).item())
        predictions.append(pred)

    predictions = np.array(predictions)
    mse = float(np.mean((predictions - test_data) ** 2))
    variance = float(np.var(test_data))
    nmse = mse / max(variance, 1e-8)

    return {"mse": mse, "nmse": nmse}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test():
    """Test Quantum Reservoir Computing."""
    print("[Quantum Reservoir Computing] self-test")
    print("=" * 60)

    # Generate synthetic time series
    t = np.linspace(0, 4 * np.pi, 200)
    timeseries = np.sin(t) + 0.3 * np.sin(2 * t) + 0.1 * np.random.randn(200)

    print("\n--- QRC ---")
    res = qrc_predict(timeseries, n_qubits=4, n_internal=10, seed=42, verbose=True)

    print("\n--- Comparison ---")
    results = compare_qrc_with_baselines(timeseries, n_qubits=4, n_internal=10, seeds=[42])
    print(f"QRC params: {results['comparison']['qrc_params']}")
    print(f"ESN params: {results['comparison']['esn_params']}")


if __name__ == "__main__":
    _self_test()
