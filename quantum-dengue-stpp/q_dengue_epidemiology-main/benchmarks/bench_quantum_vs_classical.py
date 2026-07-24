"""Benchmark: Dürr-Høyer max-finding with REAL oracle + query counter.

This benchmark measures oracle queries empirically, NOT from a formula.
Every Grover iteration = 1 oracle query. Counter increments on each call.

IMPORTANT: Dürr-Høyer O(√N) is the ITERATION count, not query count.
Each iteration = O(√(N/M)) queries. For M>1, total = O(N).
The "quantum speedup" applies to MINIMUM finding with M=1, not maximum.

Results:
- DH/Class < 1 → Dürr-Høyer uses fewer oracle queries than classical scan
- DH/Class ≈ 1 → Dürr-Høyer matches classical scan
- Match rate < 100% → Grover search is probabilistic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.graph_dien_bien import build_synthetic_dien_bien
from src.durr_hoyer_max import dur_hoyer_benchmark


def main():
    print("=" * 75)
    print("BENCHMARK: Dürr-Høyer with REAL oracle counter")
    print("(Every Grover iteration = 1 oracle query. Counter increments empirically.)")
    print("=" * 75)

    grid_sizes = [16, 32, 64, 128]
    seeds = [42, 43, 44]

    rows = []

    print(f"\n{'N':>4} | {'Class':>7} | {'DH':>7} | {'Theory':>7} | "
          f"{'DH/Cl':>7} | {'iters':>6} | {'Match':>6}")
    print("-" * 75)

    for n_target in grid_sizes:
        for seed in seeds:
            result = dur_hoyer_benchmark(n_target, seed)
            print(f"{result['n']:>4} | {result['classical_queries']:>7} | "
                  f"{result['durr_hoyer_queries']:>7} | {result['theory_sqrt_n']:>7} | "
                  f"{result['ratio_vs_classical']:>7.2f} | "
                  f"{result['total_grover_iters']:>6} | "
                  f"{'✓' if result['match'] else '✗':>6}")
            rows.append(result)

    # Aggregate statistics
    print("\n" + "=" * 75)
    print("AGGREGATE STATISTICS")
    print("=" * 75)

    by_n = {}
    for r in rows:
        n = r["n"]
        if n not in by_n:
            by_n[n] = {"ratios": [], "matches": [], "dh_queries": [], "iters": []}
        by_n[n]["ratios"].append(r["ratio_vs_classical"])
        by_n[n]["matches"].append(r["match"])
        by_n[n]["dh_queries"].append(r["durr_hoyer_queries"])
        by_n[n]["iters"].append(r["total_grover_iters"])

    print(f"\n{'N':>4} | {'mean DH/Cl':>11} | {'min':>7} | {'max':>7} | "
          f"{'match%':>8} | {'mean iters':>10}")
    print("-" * 75)

    for n in sorted(by_n):
        data = by_n[n]
        ratios = data["ratios"]
        matches = data["matches"]
        iters = data["iters"]
        dh_q = data["dh_queries"]

        mean_ratio = np.mean(ratios)
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        match_pct = 100 * np.mean(matches)
        mean_iters = np.mean(iters)
        mean_dh_q = np.mean(dh_q)

        print(f"{n:>4} | {mean_ratio:>11.3f} | {min_ratio:>7.3f} | {max_ratio:>7.3f} | "
              f"{match_pct:>7.1f}% | {mean_iters:>10.1f}")

    print("\n" + "=" * 75)
    print("INTERPRETATION:")
    print("  DH/Class < 1.0  → Dürr-Høyer uses FEWER oracle queries than classical scan")
    print("  DH/Class ≈ 1.0  → Dürr-Høyer matches classical scan")
    print("  Match rate < 100% → Grover search is probabilistic (expected)")
    print("")
    print("The O(√N) bound in Dürr-Høyer applies to ITERATION count.")
    print("Each iteration = 1 oracle query. Total queries scale as measured.")
    print("")
    print("For MINIMUM finding (M=1): total = O(N) → O(√N) after √N iterations.")
    print("For MAXIMUM finding (M>1): more queries per iteration → O(N) similar to classical.")
    print("=" * 75)

    # Save JSON
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "durr_hoyer_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[SAVED] {out_file}")

    return rows


if __name__ == "__main__":
    main()
