"""Quantum walk search on REAL data: Kaohsiung City dengue outbreak (Taiwan CDC, 1998-2024).

Real village-level graph (847 villages, real GPS centroids from case geocoding)
vs the synthetic Dien Bien graph. Same kernel/threshold methodology as
graph_dien_bien.py, same quantum walk search as bench_weighted_walk.py —
only the INPUT DATA changes (real vector positions + real historical case
counts instead of synthetic random placement).

Data: data/taiwan_kaohsiung_villages.csv (prepared from
data/taiwan_dengue_daily.csv, Taiwan CDC Open Data, 107k line-list dengue
cases 1998-2024).
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
from src.durr_hoyer_max import dur_hoyer_max_finding

VECTOR_MEAN_DISPERSAL_M = 200.0
NEIGHBORHOOD_RADIUS_M = 500.0  # arc-dim ~2.5k on largest component (vs 1500m -> ~22k, OOM)


def build_kaohsiung_graph(csv_path: Path, radius_m: float = NEIGHBORHOOD_RADIUS_M):
    df = pd.read_csv(csv_path)
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    diff_lat_m = (lats[:, None] - lats[None, :]) * 111_000
    diff_lon_m = (lons[:, None] - lons[None, :]) * 111_000 * np.cos(np.deg2rad(lats.mean()))
    dist_m = np.sqrt(diff_lat_m ** 2 + diff_lon_m ** 2)

    kernel = np.exp(-dist_m / VECTOR_MEAN_DISPERSAL_M)
    A = np.where(dist_m < radius_m, kernel, 0.0)
    np.fill_diagonal(A, 0.0)

    n_cases = df["n_cases"].to_numpy(dtype=float)
    risk = np.log1p(n_cases)
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)

    return A, risk, df


def restrict_to_largest_component(A, risk, df):
    n_comp, labels = connectivity_info(A)
    sizes = np.bincount(labels)
    largest_label = int(np.argmax(sizes))
    keep = np.where(labels == largest_label)[0]
    return A[np.ix_(keep, keep)], risk[keep], df.iloc[keep].reset_index(drop=True), n_comp, sizes


def main():
    csv_path = ROOT / "data" / "taiwan_kaohsiung_villages.csv"
    print("=" * 80)
    print("REAL DATA: Kaohsiung City dengue outbreak (Taiwan CDC, 1998-2024)")
    print(f"Graph construction: same kernel as graph_dien_bien.py "
          f"(exp(-d/{VECTOR_MEAN_DISPERSAL_M:.0f}m)), radius={NEIGHBORHOOD_RADIUS_M:.0f}m")
    print("=" * 80)

    A_full, risk_full, df_full = build_kaohsiung_graph(csv_path)
    n_comp, labels = connectivity_info(A_full)
    sizes = sorted(np.bincount(labels))[::-1]
    print(f"\nFull graph: N={len(df_full)}, components={n_comp}, "
          f"largest 5 component sizes={sizes[:5]}")
    print(f"Mean degree (full): {(A_full > 0).sum(axis=1).mean():.2f}")

    A, risk, df, n_comp0, sizes0 = restrict_to_largest_component(A_full, risk_full, df_full)
    n = len(df)
    mean_deg = float((A > 0).sum(axis=1).mean())
    print(f"\nLargest component: N={n} ({100*n/len(df_full):.1f}% of villages), "
          f"mean_degree={mean_deg:.2f}")
    print(f"(for comparison: Dien Bien synthetic — N=130, 6 components, mean_degree=2.86)")

    # Real top hotspot by actual historical case count
    top5 = np.argsort(risk)[::-1][:5]
    print(f"\nTop-5 real hotspot villages (by historical dengue cases):")
    for idx in top5:
        row = df.iloc[idx]
        print(f"  {row['village_name']} ({row['township']}): "
              f"cases={int(row['n_cases'])}, risk={risk[idx]:.4f}")

    # ===== Quantum walk search: start at vertex 0, target = real max hotspot =====
    start_v = 0
    marked = int(np.argmax(risk))
    reachable = True  # by construction (largest component)

    print(f"\n[Quantum walk search] start_v={start_v}, marked={marked} "
          f"({df.iloc[marked]['village_name']}, {int(df.iloc[marked]['n_cases'])} cases)")

    t_class = classical_hitting_weighted(A, marked, n_trials=300)
    crossing_t, peak_p, peak_t, final_p = quantum_search_run(
        A, start_v, marked, max_t=2000, threshold=0.05)
    t_quant = crossing_t if crossing_t else float("inf")
    speedup = t_class / t_quant if t_quant not in (0, float("inf")) else None

    print(f"  Classical hitting time (empirical, 300 trials): {t_class:.2f} steps")
    print(f"  Quantum peak P(marked): {peak_p:.4f} at t={peak_t}")
    print(f"  Quantum crossing t (P>0.05): {crossing_t}")
    print(f"  Empirical speedup: {speedup if speedup else 'N/A (no resonance)'}")

    # Try a few more marked targets among real hotspots for robustness
    print(f"\n[Additional targets: other real top-5 hotspots]")
    for idx in top5:
        if idx == marked:
            continue
        t_class_i = classical_hitting_weighted(A, int(idx), n_trials=150)
        c_t, p_p, p_t, f_p = quantum_search_run(A, start_v, int(idx), max_t=2000, threshold=0.05)
        print(f"  target={df.iloc[idx]['village_name']:<15} t_class={t_class_i:8.2f}  "
              f"peak_p={p_p:.4f} at t={p_t}  crossing_t={c_t}")

    # ===== Durr-Hoyer max finding on real risk data =====
    print(f"\n[Durr-Hoyer max finding on REAL risk scores]")
    true_idx = int(np.argmax(risk))
    q_idx, q_score = dur_hoyer_max_finding(risk, seed=42, verbose=True)
    print(f"  True argmax: idx={true_idx} ({df.iloc[true_idx]['village_name']})")
    print(f"  Durr-Hoyer result: idx={q_idx} ({df.iloc[q_idx]['village_name']})")
    print(f"  Match: {q_idx == true_idx}")

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    import json
    result = {
        "source": "Taiwan CDC Open Data, Kaohsiung City, 1998-2024",
        "n_villages_total": int(len(df_full)),
        "n_villages_largest_component": int(n),
        "n_components_full_graph": int(n_comp),
        "mean_degree_largest_component": mean_deg,
        "quantum_peak_p_marked": float(peak_p),
        "quantum_crossing_t": float(t_quant) if t_quant != float("inf") else None,
        "classical_hitting_time": float(t_class),
        "empirical_speedup": float(speedup) if speedup else None,
        "durr_hoyer_match": bool(q_idx == true_idx),
    }
    with open(out_dir / "taiwan_kaohsiung_benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {out_dir / 'taiwan_kaohsiung_benchmark.json'}")


if __name__ == "__main__":
    main()
