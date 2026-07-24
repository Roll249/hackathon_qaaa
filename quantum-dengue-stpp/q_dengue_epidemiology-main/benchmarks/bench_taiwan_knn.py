"""Experiment A: does regularizing the Kaohsiung graph (k-NN, near-uniform
degree/weight) restore quantum walk resonance that the radius-threshold
graph (bench_taiwan_kaohsiung.py) does not show?

Hypothesis (docs/QUANTUM_ADVANTAGE_REPORT.md): resonance needs (1) connected,
(2) near-regular degree, (3) low weight variance among each vertex's edges.
Radius-threshold graphs on real geography violate all three (many small
components, heterogeneous local density). A k-NN graph forces every vertex
to have the same out-degree by construction, which should fix (2) directly
and improve (1); testing both binary (max weight-uniformity) and
distance-kernel-weighted k-NN isolates whether uniformity or connectivity
is the bigger factor.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_weighted_walk import (
    quantum_search_run, classical_hitting_weighted, connectivity_info,
)

VECTOR_MEAN_DISPERSAL_M = 200.0


def load_positions():
    df = pd.read_csv(ROOT / "data" / "taiwan_kaohsiung_villages.csv")
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    diff_lat_m = (lats[:, None] - lats[None, :]) * 111_000
    diff_lon_m = (lons[:, None] - lons[None, :]) * 111_000 * np.cos(np.deg2rad(lats.mean()))
    dist_m = np.sqrt(diff_lat_m ** 2 + diff_lon_m ** 2)
    return df, dist_m


def build_knn_graph(dist_m: np.ndarray, k: int, weighted: bool):
    """Symmetric k-NN graph. weighted=False -> binary (max uniformity);
    weighted=True -> exp(-d/200m) kernel on the kept k-NN edges."""
    n = dist_m.shape[0]
    A = np.zeros((n, n))
    d = dist_m.copy()
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argpartition(d, k, axis=1)[:, :k]
    for v in range(n):
        for u in nn_idx[v]:
            w = np.exp(-dist_m[v, u] / VECTOR_MEAN_DISPERSAL_M) if weighted else 1.0
            A[v, u] = w
            A[u, v] = w  # symmetrize (union of mutual/one-directional NN)
    return A


def restrict_to_largest_component(A, risk, df):
    n_comp, labels = connectivity_info(A)
    sizes = np.bincount(labels)
    largest_label = int(np.argmax(sizes))
    keep = np.where(labels == largest_label)[0]
    return A[np.ix_(keep, keep)], risk[keep], df.iloc[keep].reset_index(drop=True), n_comp


def run_one(A_full, risk_full, df_full, label):
    A, risk, df, n_comp = restrict_to_largest_component(A_full, risk_full, df_full)
    n = len(df)
    mean_deg = float((A > 0).sum(axis=1).mean())
    arc_dim = int((A > 0).sum())

    start_v = 0
    marked = int(np.argmax(risk))
    if A[start_v].sum() == 0 or A[marked].sum() == 0:
        print(f"  [{label}] SKIP: start or marked isolated after restriction")
        return None

    if arc_dim > 6000:
        print(f"  [{label:<28}] N={n:4d} mean_deg={mean_deg:6.2f} arc_dim={arc_dim:6d}  "
              f"SKIP quantum walk: dense-matrix mem ~{3*arc_dim**2*8/1e9:.1f}GB too large")
        return {
            "label": label, "N": n, "n_components_full": n_comp, "mean_degree": mean_deg,
            "arc_dim": arc_dim, "skipped_too_large": True,
        }

    t_class = classical_hitting_weighted(A, marked, n_trials=200)
    crossing_t, peak_p, peak_t, final_p = quantum_search_run(
        A, start_v, marked, max_t=2000, threshold=0.05)

    print(f"  [{label:<28}] N={n:4d} n_comp_full={n_comp:3d} mean_deg={mean_deg:6.2f} "
          f"arc_dim={arc_dim:6d}  t_class={t_class:9.2f}  peak_p={peak_p:.4f} at t={peak_t}  "
          f"resonance={'YES' if crossing_t else 'NO'}")

    return {
        "label": label, "N": n, "n_components_full": n_comp, "mean_degree": mean_deg,
        "arc_dim": arc_dim, "classical_hitting": t_class, "quantum_peak_p": float(peak_p),
        "quantum_peak_t": int(peak_t), "quantum_crossing_t": crossing_t,
        "resonance": crossing_t is not None,
    }


def main():
    print("=" * 90)
    print("EXPERIMENT A: k-NN regularization vs radius-threshold on Kaohsiung real graph")
    print("=" * 90)

    df, dist_m = load_positions()
    n_cases = df["n_cases"].to_numpy(dtype=float)
    risk = np.log1p(n_cases)
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)

    results = []

    # Baseline: radius-threshold (same as bench_taiwan_kaohsiung.py, radius=500m)
    kernel = np.exp(-dist_m / VECTOR_MEAN_DISPERSAL_M)
    A_radius = np.where(dist_m < 500.0, kernel, 0.0)
    np.fill_diagonal(A_radius, 0.0)
    results.append(run_one(A_radius, risk, df, "radius=500m (baseline)"))

    print()
    for k in [2, 3, 4, 6, 8, 12]:
        A_bin = build_knn_graph(dist_m, k, weighted=False)
        results.append(run_one(A_bin, risk, df, f"k-NN binary, k={k}"))

    print()
    for k in [2, 3, 4, 6, 8, 12]:
        A_w = build_knn_graph(dist_m, k, weighted=True)
        results.append(run_one(A_w, risk, df, f"k-NN weighted, k={k}"))

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    import json
    with open(out_dir / "taiwan_knn_experiment.json", "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2)
    print(f"\n[SAVED] {out_dir / 'taiwan_knn_experiment.json'}")


if __name__ == "__main__":
    main()
