"""Experiment B: does restricting to a small, DENSE sub-region (single
district instead of the whole city) restore resonance?

Toy graphs that resonated well (bench_weighted_walk.py) were small AND dense
relative to N: Ring degree=2 (but N small, fully cyclic-symmetric) -> 1.0,
sparse_binary N=48 with degree ~N/2 -> 0.23. The whole-city graphs (radius
and k-NN, experiment A) are large (N~200-850) but sparse relative to N
(mean_deg << N/2). This restricts to Sanmin Dist. (86 villages, contains
4/5 top real hotspots including the global max) and sweeps radius from
sparse up to fully-connected, to see if a much higher degree/N ratio at
small N recovers resonance.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_weighted_walk import (
    quantum_search_run, classical_hitting_weighted, connectivity_info,
)

VECTOR_MEAN_DISPERSAL_M = 200.0


def load_district(name: str):
    df = pd.read_csv(ROOT / "data" / "taiwan_kaohsiung_villages.csv")
    df = df[df["township"] == name].reset_index(drop=True)
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    diff_lat_m = (lats[:, None] - lats[None, :]) * 111_000
    diff_lon_m = (lons[:, None] - lons[None, :]) * 111_000 * np.cos(np.deg2rad(lats.mean()))
    dist_m = np.sqrt(diff_lat_m ** 2 + diff_lon_m ** 2)
    n_cases = df["n_cases"].to_numpy(dtype=float)
    risk = np.log1p(n_cases)
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)
    return df, dist_m, risk


def restrict_to_largest_component(A, risk, df):
    n_comp, labels = connectivity_info(A)
    sizes = np.bincount(labels)
    keep = np.where(labels == int(np.argmax(sizes)))[0]
    return A[np.ix_(keep, keep)], risk[keep], df.iloc[keep].reset_index(drop=True), n_comp


def run_one(A_full, risk_full, df_full, label):
    A, risk, df, n_comp = restrict_to_largest_component(A_full, risk_full, df_full)
    n = len(df)
    mean_deg = float((A > 0).sum(axis=1).mean())
    arc_dim = int((A > 0).sum())

    start_v = 0
    marked = int(np.argmax(risk))
    if n < 3 or A[start_v].sum() == 0 or A[marked].sum() == 0:
        print(f"  [{label}] SKIP: too small or isolated")
        return None

    t_class = classical_hitting_weighted(A, marked, n_trials=200)
    crossing_t, peak_p, peak_t, final_p = quantum_search_run(
        A, start_v, marked, max_t=2000, threshold=0.05)
    t_quant = crossing_t if crossing_t else float("inf")
    speedup = t_class / t_quant if t_quant not in (0, float("inf")) else None

    print(f"  [{label:<26}] N={n:4d} deg/N={mean_deg/n:5.2f} mean_deg={mean_deg:6.2f} "
          f"arc_dim={arc_dim:6d}  t_class={t_class:8.2f}  crossing_t={str(crossing_t):>6}  "
          f"peak_p={peak_p:.4f} at t={peak_t}  speedup={speedup if speedup else 'N/A':>6}")
    return {
        "label": label, "N": n, "mean_degree": mean_deg, "deg_over_N": mean_deg / n,
        "arc_dim": arc_dim, "classical_hitting": t_class,
        "quantum_crossing_t": crossing_t, "quantum_peak_p": float(peak_p),
        "quantum_peak_t": int(peak_t), "resonance": crossing_t is not None,
        "empirical_speedup": speedup,
    }


def main():
    print("=" * 95)
    print("EXPERIMENT B: dense small sub-region (Sanmin Dist., 86 villages, contains global hotspot)")
    print("=" * 95)

    df, dist_m, risk = load_district("Sanmin Dist.")
    print(f"N villages in Sanmin Dist.: {len(df)}")
    kernel = np.exp(-dist_m / VECTOR_MEAN_DISPERSAL_M)

    results = []
    for radius in [500, 1000, 1500, 2000, 3000, 5000, 10000]:
        A = np.where(dist_m < radius, kernel, 0.0)
        np.fill_diagonal(A, 0.0)
        results.append(run_one(A, risk, df, f"radius={radius}m"))

    # Fully-connected weighted (radius=inf) — closest analog to toy "fully" regime
    A_full = kernel.copy()
    np.fill_diagonal(A_full, 0.0)
    results.append(run_one(A_full, risk, df, "fully-connected (weighted)"))

    # Fully-connected BINARY (exact toy "fully" regime: A = ones - eye)
    n = len(df)
    A_binary_full = np.ones((n, n)) - np.eye(n)
    results.append(run_one(A_binary_full, risk, df, "fully-connected (binary)"))

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    import json
    with open(out_dir / "taiwan_dense_experiment.json", "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2)
    print(f"\n[SAVED] {out_dir / 'taiwan_dense_experiment.json'}")


if __name__ == "__main__":
    main()
