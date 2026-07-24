"""Quantum Probability Image Encoding (QPIE) for risk intensity map.

Embeds classical risk scores into quantum state amplitudes, so that
high-risk communes have HIGH amplitude (will be amplified by Grover),
and low-risk communes have LOW amplitude (will be TRIỆT TIÊU).

Key insight: triệt tiêu amplitude ở low-risk valleys là cơ chế tự nhiên
của amplitude amplification - không cần explicit prediction.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def encode_risk_qpie(risk_scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode risk scores into a normalized probability amplitude vector.

    Args:
        risk_scores: shape (N,) non-negative floats

    Returns:
        statevector: shape (N,) complex, normalized so sum(|a|^2) = 1
        n_qubits: int, ceil(log2(N))

    Mathematical: |ψ⟩ = Σ_i sqrt(r_i) |i⟩ / ||√r||
    """
    r = np.asarray(risk_scores, dtype=float)
    r = np.clip(r, 0, None)
    if r.sum() == 0:
        raise ValueError("All risk scores are zero")

    amplitudes = np.sqrt(r)
    statevector = amplitudes / np.linalg.norm(amplitudes)

    n = len(risk_scores)
    n_qubits = int(np.ceil(np.log2(n)))
    target_dim = 2 ** n_qubits
    if len(statevector) < target_dim:
        # Pad with zeros
        padded = np.zeros(target_dim, dtype=complex)
        padded[:len(statevector)] = statevector
        statevector = padded

    return statevector, n_qubits


def qpie_qnode(n_qubits: int, risk_scores: np.ndarray):
    """Build a PennyLane QNode that prepares QPIE state from classical risks.

    State preparation via Mottonen decomposition: any statevector can be
    prepared with O(2^n) gates using uniformly controlled rotations.
    For small n_qubits (8-12), this is exact.

    Returns: callable that returns the prepared statevector.
    """
    statevector, _ = encode_risk_qpie(risk_scores)
    statevector = statevector.astype(complex)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(statevector, wires=range(n_qubits))
        return qml.state()

    return circuit


def get_top_k_from_probs(probs: np.ndarray, k: int) -> list[int]:
    """Get indices of top-k probability amplitudes (most likely measurement outcomes)."""
    return list(np.argsort(probs)[-k:][::-1])


if __name__ == "__main__":
    # Quick test
    rng = np.random.default_rng(42)
    risk = rng.uniform(0, 1, 16)
    risk[[3, 7, 12]] = [0.9, 0.95, 0.85]  # 3 hotspots

    statevector, n_qubits = encode_risk_qpie(risk)
    print(f"Risk scores: {risk}")
    print(f"Statevector shape: {statevector.shape}, n_qubits={n_qubits}")
    print(f"  Probability mass at hotspots (3, 7, 12): "
          f"{statevector[3]**2 + statevector[7]**2 + statevector[12]**2:.4f}")
    print(f"  Total probability: {np.sum(np.abs(statevector)**2):.4f}")

    # Run through PennyLane circuit
    circuit = qpie_qnode(n_qubits, risk)
    result = circuit()
    print(f"  PennyLane state matches: {np.allclose(result, statevector)}")
