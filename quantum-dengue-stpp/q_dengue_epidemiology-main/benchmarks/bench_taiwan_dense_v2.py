"""Experiment B (v2) — fair test: v1 always used start_v=0 for the quantum
walk, but on Sanmin Dist. vertex 0 turned out to be the 2nd-closest village
to the marked hotspot (412m out of 86 candidates) while the classical
baseline (classical_hitting_weighted) averages hitting time over a RANDOM
start each trial. That mismatch alone could explain the huge "speedup"
numbers in v1 — not genuine quantum interference.

This reruns the best v1 config (radius=1000m) with several different start
vertices (closest, farthest, and a few random ones) to see whether
resonance/speedup is a structural property of the graph or an artifact of
picking a lucky start.
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
RADIUS_M = 1000.0


def main():
    df = pd.read_csv(ROOT / "data" / "taiwan_kaohsiung_villages.csv")
    df = df[df["township"] == "Sanmin Dist."].reset_index(drop=True)
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    diff_lat_m = (lats[:, None] - lats[None, :]) * 111_000
    diff_lon_m = (lons[:, None] - lons[None, :]) * 111_000 * np.cos(np.deg2rad(lats.mean()))
    dist_m = np.sqrt(diff_lat_m ** 2 + diff_lon_m ** 2)
    n_cases = df["n_cases"].to_numpy(dtype=float)
    risk = np.log1p(n_cases)
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)

    kernel = np.exp(-dist_m / VECTOR_MEAN_DISPERSAL_M)
    A = np.where(dist_m < RADIUS_M, kernel, 0.0)
    np.fill_diagonal(A, 0.0)

    n_comp, labels = connectivity_info(A)
    keep = np.where(labels == int(np.argmax(np.bincount(labels))))[0]
    A = A[np.ix_(keep, keep)]
    risk = risk[keep]
    df = df.iloc[keep].reset_index(drop=True)
    n = len(df)
    marked = int(np.argmax(risk))

    print("=" * 100)
    print(f"Sanmin Dist., radius={RADIUS_M:.0f}m, N={n}, marked={marked} "
          f"({df.iloc[marked]['village_name']}, {int(df.iloc[marked]['n_cases'])} cases)")
    print(f"mean_degree={(A>0).sum(axis=1).mean():.2f}")
    print("=" * 100)

    d_to_marked = dist_m[keep][:, keep][:, marked]
    order = np.argsort(d_to_marked)
    closest = int(order[1]) if order[0] == marked else int(order[0])  # nearest OTHER village
    farthest = int(order[-1])
    rng = np.random.default_rng(7)
    random_starts = [int(x) for x in rng.choice([v for v in range(n) if v != marked], size=5, replace=False)]

    candidates = {
        "closest-to-marked": closest,
        "farthest-from-marked": farthest,
    }
    for i, v in enumerate(random_starts):
        candidates[f"random-{i}"] = v

    print(f"\n{'start':<22} {'dist_to_marked(m)':>18} {'t_class':>10} {'crossing_t':>11} "
          f"{'peak_p':>8} {'peak_t':>7} {'speedup':>10}")
    print("-" * 100)

    results = []
    for label, start_v in candidates.items():
        if A[start_v].sum() == 0:
            print(f"{label:<22} SKIP: isolated")
            continue
        t_class = classical_hitting_weighted(A, marked, n_trials=200)
        crossing_t, peak_p, peak_t, final_p = quantum_search_run(
            A, start_v, marked, max_t=2000, threshold=0.05)
        t_quant = crossing_t if crossing_t else float("inf")
        speedup = t_class / t_quant if t_quant not in (0, float("inf")) else None
        d = d_to_marked[start_v]
        print(f"{label:<22} {d:18.1f} {t_class:10.2f} {str(crossing_t):>11} "
              f"{peak_p:8.4f} {peak_t:7d} {str(speedup) if speedup else 'N/A':>10}")
        results.append({
            "label": label, "start_v": start_v, "dist_to_marked_m": float(d),
            "classical_hitting": t_class, "quantum_crossing_t": crossing_t,
            "quantum_peak_p": float(peak_p), "quantum_peak_t": int(peak_t),
            "empirical_speedup": speedup,
        })

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    import json
    with open(out_dir / "taiwan_dense_v2_fair_starts.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_dir / 'taiwan_dense_v2_fair_starts.json'}")


if __name__ == "__main__":
    main()
