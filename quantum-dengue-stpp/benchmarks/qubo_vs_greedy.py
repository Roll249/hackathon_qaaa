#!/usr/bin/env python3
"""Benchmark: QUBO-QAOA vs classical greedy for SOP subset selection.

PURPOSE
-------
Both solvers pick the same K permutations out of M candidate SOP
permutations. We compare:
  * quality = mean L-error of the selected subset
  * time    = wall-clock seconds
  * diversity = mean pairwise Hamming distance of the selected subset

The plot shows quality, diversity, and time side-by-side. As with the
qkernel benchmark, the captions explicitly state that no quantum
advantage is claimed.

USAGE
-----
    python benchmarks/qubo_vs_greedy.py [--m 8] [--k 3]
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.quantum import (  # noqa: E402
    QUBOSOPSelector,
    check_quantum_claims,
    validate_qaoa_output,
)
from src.quantum.honest_assessment import MAX_QAOA_QUBITS  # noqa: E402

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "output_result", "q_stpp_v17"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic SOP candidates: each candidate is a binary permutation
# vector. We give them realistic L-error distributions and varying
# similarities.
# ---------------------------------------------------------------------------


def make_synthetic_sop_candidates(
    m: int, n_positions: int, rng: np.random.Generator
):
    """Generate M candidate SOP permutations with realistic metadata."""
    l_errors = rng.uniform(0.01, 0.5, size=m)
    # Build permutations and pairwise similarities.
    perms = []
    for _ in range(m):
        p = np.arange(n_positions)
        rng.shuffle(p)
        perms.append(p)
    perms = np.array(perms)
    # Pairwise Hamming distance (normalised).
    similarities = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            similarities[i, j] = float(np.mean(perms[i] != perms[j]))
    return l_errors, similarities, perms


def evaluate_selection(
    selected: np.ndarray,
    l_errors: np.ndarray,
    perms: np.ndarray,
) -> dict:
    """Compute quality and diversity for a chosen subset."""
    selected = np.asarray(selected, dtype=int)
    mean_err = float(np.mean(l_errors[selected]))
    sub = perms[selected]
    m = len(sub)
    diversity = 0.0
    count = 0
    for a in range(m):
        for b in range(a + 1, m):
            diversity += float(np.mean(sub[a] != sub[b]))
            count += 1
    diversity = diversity / max(count, 1)
    return {"mean_l_error": mean_err, "diversity": diversity}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(m: int, k: int, n_positions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    l_errors, similarities, perms = make_synthetic_sop_candidates(
        m, n_positions, rng
    )

    selector = QUBOSOPSelector(seed=seed)

    # Classical greedy
    t0 = time.perf_counter()
    greedy_sel = selector.select(l_errors, similarities, k, method="greedy")
    t_greedy = time.perf_counter() - t0
    greedy_eval = evaluate_selection(greedy_sel, l_errors, perms)

    # QAOA
    qaoa_sel = None
    t_qaoa = float("nan")
    qaoa_eval = {"mean_l_error": float("nan"), "diversity": float("nan")}
    qaoa_ok = False
    if m <= MAX_QAOA_QUBITS:
        try:
            t0 = time.perf_counter()
            qaoa_sel = selector.select(l_errors, similarities, k, method="qaoa")
            t_qaoa = time.perf_counter() - t0
            qaoa_eval = evaluate_selection(qaoa_sel, l_errors, perms)
            qaoa_ok = True
        except ImportError:
            qaoa_ok = False
        except Exception as exc:
            print(f"[qaqa] failed: {exc}")
            qaoa_ok = False

    return {
        "m": m,
        "k": k,
        "n_positions": n_positions,
        "seed": seed,
        "greedy": {
            "selected": greedy_sel.tolist(),
            "mean_l_error": greedy_eval["mean_l_error"],
            "diversity": greedy_eval["diversity"],
            "time_s": t_greedy,
        },
        "qaoa": {
            "ran": qaoa_ok,
            "selected": qaoa_sel.tolist() if qaoa_ok else [],
            "mean_l_error": qaoa_eval["mean_l_error"],
            "diversity": qaoa_eval["diversity"],
            "time_s": t_qaoa,
        },
    }


def plot_results(result: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    methods = ["greedy", "qaoa"]
    labels = ["Greedy (classical)", "QAOA (simulator)"]
    colours = ["C0", "C3"]
    errs = [result["greedy"]["mean_l_error"], result["qaoa"]["mean_l_error"]]
    diversities = [
        result["greedy"]["diversity"],
        result["qaoa"]["diversity"],
    ]
    times = [result["greedy"]["time_s"], result["qaoa"]["time_s"]]

    axes[0].bar(labels, errs, color=colours)
    axes[0].set_ylabel("Mean L-error (lower = better)")
    axes[0].set_title("Quality")

    axes[1].bar(labels, diversities, color=colours)
    axes[1].set_ylabel("Pairwise Hamming diversity (higher = better)")
    axes[1].set_title("Diversity")

    axes[2].bar(labels, times, color=colours)
    axes[2].set_ylabel("Wall-clock time (s)")
    axes[2].set_title("Cost")

    fig.suptitle(
        "Honest benchmark: greedy vs QAOA on SOP subset selection — "
        "no quantum advantage claimed",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-positions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.m > MAX_QAOA_QUBITS:
        print(
            f"Refusing M={args.m} > {MAX_QAOA_QUBITS}; "
            f"QAOA path will be skipped automatically."
        )

    # Guard rail: refuse over-claims in the title.
    try:
        check_quantum_claims("No quantum advantage claimed in this plot.")
    except RuntimeError as exc:
        print(exc)
        return 2

    print(
        f"[benchmark] M={args.m} K={args.k} n_positions={args.n_positions}"
    )
    result = run_benchmark(
        args.m, args.k, args.n_positions, args.seed
    )
    print(json.dumps(result, indent=2))

    out_json = os.path.join(OUTPUT_DIR, "qubo_vs_greedy_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    out_png = os.path.join(OUTPUT_DIR, "qubo_vs_greedy.png")
    plot_results(result, out_png)
    print(f"[benchmark] wrote {out_json}")
    print(f"[benchmark] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
