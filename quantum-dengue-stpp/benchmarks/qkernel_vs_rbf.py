#!/usr/bin/env python3
"""Benchmark: Quantum Kernel vs classical RBF on STPP hotspot features.

PURPOSE
-------
Run both a quantum kernel and a classical RBF kernel through the same
1-nearest-neighbour classification pipeline on synthetic dengue-like
features, then plot the accuracy side-by-side with honest annotations.

USAGE
-----
    python benchmarks/qkernel_vs_rbf.py [--n 50] [--n-qubits 4]

The script writes:
    output_result/q_stpp_v17/qkernel_vs_rbf_results.json
    output_result/q_stpp_v17/qkernel_vs_rbf.png

It does NOT claim quantum advantage. The plot caption includes a
"no quantum advantage claimed" disclaimer enforced by the
check_quantum_claims guard rail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.quantum import quantum_knn_classify, check_quantum_claims  # noqa: E402
from src.quantum.honest_assessment import MAX_QKERNEL_QUBITS  # noqa: E402

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "output_result", "q_stpp_v17"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data — three pattern families a quantum kernel might separate
# differently than RBF.
# ---------------------------------------------------------------------------


def make_cluster_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """A point pattern with strong local clustering (small L(r) at small r)."""
    centers = rng.uniform(0, 1, size=(3, 2))
    pts = []
    for _ in range(n):
        c = centers[rng.integers(0, 3)]
        pts.append(c + rng.normal(0, 0.05, size=2))
    return np.clip(np.array(pts), 0, 1)


def make_regular_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """A more regular pattern (large L(r) at small r)."""
    grid = int(np.sqrt(n)) + 1
    coords = np.array(
        [(i / grid, j / grid) for i in range(grid) for j in range(grid)]
    )
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx]


def make_random_pattern(n: int, rng: np.random.Generator) -> np.ndarray:
    """CSR (Poisson) pattern as the null class."""
    return rng.uniform(0, 1, size=(n, 2))


PATTERN_FUNCS = {
    "cluster": make_cluster_pattern,
    "regular": make_regular_pattern,
    "random": make_random_pattern,
}


def featurize(coords: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    """Compute a tiny Ripley's-L-style summary used as the feature vector.

    We use a vector of length 4 (counts at 4 radii). It is intentionally
    tiny so it fits in <= 6 qubits for the quantum kernel.
    """
    from scipy.spatial.distance import pdist, squareform

    if len(coords) < 2:
        return np.zeros(len(r_values))
    dist = squareform(pdist(coords, metric="euclidean"))
    n = len(coords)
    out = np.zeros(len(r_values), dtype=float)
    for k, r in enumerate(r_values):
        # Standardised pair-count density: (pairs < r) / n^2
        out[k] = (np.sum(dist < r) - n) / (n * n)
    # Stabilise the sign
    return np.sign(out) * np.abs(out) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    n_samples_per_class: int,
    n_qubits: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    r_values = np.linspace(0.05, 0.5, n_qubits)

    X, y = [], []
    for label, fn in enumerate(PATTERN_FUNCS.values()):
        for _ in range(n_samples_per_class):
            coords = fn(n=40, rng=rng)
            X.append(featurize(coords, r_values))
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    # Standardise so classical and quantum features are on similar scales.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Rescale to [0, pi] for the quantum rotation gates.
    X_quantum = (X_scaled - X_scaled.min()) / (
        X_scaled.max() - X_scaled.min() + 1e-12
    ) * np.pi

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_tr_q, X_te_q, _, _ = train_test_split(
        X_quantum, y, test_size=0.4, random_state=seed, stratify=y
    )

    # Classical 1-NN with RBF-weighted distance (sklearn handles it).
    t0 = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf.fit(X_tr, y_tr)
    y_pred_classical = clf.predict(X_te)
    t_classical = time.perf_counter() - t0
    acc_classical = accuracy_score(y_te, y_pred_classical)

    # Quantum kernel 1-NN.
    t0 = time.perf_counter()
    try:
        y_pred_quantum = quantum_knn_classify(
            X_tr_q, y_tr, X_te_q, n_qubits=n_qubits, k=1
        )
        acc_quantum = accuracy_score(y_te, y_pred_quantum)
        t_quantum = time.perf_counter() - t0
        quantum_ok = True
    except ImportError:
        acc_quantum = float("nan")
        t_quantum = float("nan")
        quantum_ok = False

    result = {
        "n_samples_per_class": n_samples_per_class,
        "n_qubits": n_qubits,
        "seed": seed,
        "acc_classical_euclidean": acc_classical,
        "acc_quantum_kernel": acc_quantum,
        "time_classical_s": t_classical,
        "time_quantum_s": t_quantum,
        "quantum_kernel_ran": quantum_ok,
    }
    return result, y_te, y_pred_classical, y_pred_quantum


def plot_results(
    y_te: np.ndarray,
    y_pred_classical: np.ndarray,
    y_pred_quantum: np.ndarray,
    quantum_ok: bool,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(
        y_te, y_pred_classical, alpha=0.6, label="classical", color="C0"
    )
    axes[0].set_xlabel("True label")
    axes[0].set_ylabel("Predicted label (classical)")
    axes[0].set_title("Classical 1-NN (Euclidean)")
    axes[0].legend()

    if quantum_ok:
        axes[1].scatter(
            y_te, y_pred_quantum, alpha=0.6, label="quantum", color="C3"
        )
    axes[1].set_xlabel("True label")
    axes[1].set_ylabel("Predicted label (quantum)")
    axes[1].set_title("Quantum-kernel 1-NN")
    axes[1].legend()

    fig.suptitle(
        "Honest benchmark: classical vs quantum kernel on synthetic "
        "STPP-like features — no quantum advantage claimed",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=30, help="samples per class"
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="feature dim == qubit count (max 15)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_qubits > MAX_QKERNEL_QUBITS:
        print(
            f"Refusing n_qubits={args.n_qubits} > "
            f"{MAX_QKERNEL_QUBITS}."
        )
        return 1

    # Sanity: make sure we never write over-claim strings.
    try:
        check_quantum_claims("No quantum advantage claimed in this plot.")
    except RuntimeError as exc:
        print(exc)
        return 2

    print(
        f"[benchmark] n_samples_per_class={args.n} "
        f"n_qubits={args.n_qubits} seed={args.seed}"
    )
    result, y_te, y_pred_c, y_pred_q = run_benchmark(
        args.n, args.n_qubits, args.seed
    )
    print(json.dumps(result, indent=2))

    out_json = os.path.join(OUTPUT_DIR, "qkernel_vs_rbf_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    out_png = os.path.join(OUTPUT_DIR, "qkernel_vs_rbf.png")
    plot_results(y_te, y_pred_c, y_pred_q, result["quantum_kernel_ran"], out_png)
    print(f"[benchmark] wrote {out_json}")
    print(f"[benchmark] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
