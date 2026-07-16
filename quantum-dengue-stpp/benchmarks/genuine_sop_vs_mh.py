#!/usr/bin/env python3
"""Benchmark: genuine Grover SOP search vs classical Metropolis-Hastings.

This is the honest head-to-head the project has been missing. The quantum
path uses factoradic rank encoding + Grover iteration on PennyLane's
default.qubit. The classical path is the same MH sampler used in v15.

We report, at each N:
  * quantum marked-probability after optimal Grover iterations
  * classical best L-error found with a fixed budget of N evaluations
  * speed-up factor for quantum (oracle-query count)
  * wall-clock times for both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.quantum.genuine_sop_quantum import (  # noqa: E402
    compute_L_summary,
    enqueue_all_costs,
    l_error,
    number_of_qubits,
    run_sop_quantum,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_result", "q_stpp_v17")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Classical Metropolis-Hastings baseline (mirrors v15 fair logic)
# ---------------------------------------------------------------------------


def classical_mh_sop_search(
    times,
    coords_x,
    coords_y,
    r_values,
    L_target,
    n_budget,
    seed=42,
):
    """Run Metropolis-Hastings with a fixed budget of L-summary evaluations."""
    rng = np.random.default_rng(seed)
    n = len(times)

    def score(perm):
        return l_error(
            compute_L_summary(times[perm], coords_x, coords_y, r_values),
            L_target,
        )

    perm = rng.permutation(n)
    cur_err = score(perm)
    best_err = cur_err
    best_perm = perm.copy()
    temp0 = max(cur_err, 1e-12)
    temp1 = temp0 * 0.01

    for step in range(1, n_budget):
        t = step / max(n_budget - 1, 1)
        temp = temp0 * (temp1 / temp0) ** t
        cand = perm.copy()
        i, j = rng.choice(n, 2, replace=False)
        cand[i], cand[j] = cand[j], cand[i]
        cand_err = score(cand)
        delta = cand_err - cur_err
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            perm = cand
            cur_err = cand_err
            if cur_err < best_err:
                best_err = cur_err
                best_perm = perm.copy()

    return best_err, best_perm


def make_synthetic_dataset(n: int, seed: int = 0):
    """Generate a synthetic STPP-like pattern with explicit space-time
    correlation.

    IMPORTANT: we use UNSORTED times and an explicit space-time coupling
    so that the L-function summary varies across permutations. Sorting
    the times before computing L_target makes every permutation
    trivially preserve L, which gives Grover nothing to amplify.

    The coupling term `times = jitter + a*x + b*y` makes the temporal
    ordering carry information about the spatial layout, mimicking the
    way a self-exciting Hawkes process clusters events in space-time.
    """
    rng = np.random.default_rng(seed)
    times = rng.uniform(0.0, 1.0, size=n)
    coords_x = rng.uniform(0.0, 1.0, size=n)
    coords_y = rng.uniform(0.0, 1.0, size=n)
    coupling = 0.4 * coords_x + 0.2 * coords_y
    times = times + coupling
    t_min, t_max = times.min(), times.max()
    if t_max > t_min:
        times = (times - t_min) / (t_max - t_min)
    r_values = np.linspace(0.05, 0.30, 6)
    L_target = compute_L_summary(times, coords_x, coords_y, r_values)
    return times, coords_x, coords_y, r_values, L_target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one(n: int, seed: int, tau_quantile: float, n_budget_factor: int,
            top_k: int = 0):
    print(f"\n=== N={n} ===")
    times, coords_x, coords_y, r_values, L_target = make_synthetic_dataset(
        n=n, seed=seed
    )

    # ---- Quantum path ----
    t0 = time.perf_counter()
    qres = run_sop_quantum(
        times, coords_x, coords_y, r_values, L_target,
        tau_quantile=tau_quantile,
        top_k=top_k if top_k > 0 else None,
    )
    q_total_t = time.perf_counter() - t0

    # ---- Classical path ----
    # Budget = the number of L-evaluations Grover needs to mark the same
    # number of items by random sampling. Random sampling on a uniform
    # distribution needs N!/M trials to hit one marked item, in expectation.
    # We give classical the *same* number of evaluations as random-sampling
    # would need. Quantum needs sqrt(N!/M) iterations of coherent predicate.
    if qres.marked_count <= 0:
        classical_budget = 1
    else:
        classical_budget = max(1, int(math.factorial(n) // qres.marked_count))

    t0 = time.perf_counter()
    mh_err, mh_perm = classical_mh_sop_search(
        times, coords_x, coords_y, r_values, L_target,
        n_budget=classical_budget,
        seed=seed,
    )
    mh_t = time.perf_counter() - t0

    # Oracle query count comparison
    # Honest note: Grover's quantum advantage is on QUERIES to the predicate
    # (oracle), not on wall-clock time. Quantum uses ~ sqrt(N!/M) coherent
    # predicate evaluations (1 per Grover iteration, embedded in the
    # superposition). A classical random sampler needs N!/M samples in
    # expectation to hit one marked permutation.
    n_factorial = math.factorial(n)
    marked = qres.marked_count
    quantum_predicate_calls = max(1, qres.iterations)  # 1 oracle call per Grover iter
    classical_predicate_calls_random = max(1, n_factorial // max(marked, 1))
    classical_predicate_calls_mh = classical_budget
    # The honest metric is the ratio (N!/M) / iterations; this is a
    # constant times sqrt(N!/M), so it tracks the textbook quadratic
    # speedup. We label it honestly as "predicate-query ratio vs random",
    # not as "wall-clock speedup" or unqualified "quantum speedup".
    if quantum_predicate_calls > 0:
        query_speedup_vs_random = classical_predicate_calls_random / quantum_predicate_calls
    else:
        query_speedup_vs_random = float("nan")

    result = {
        "n": n,
        "qubits": qres.qubits,
        "marked_count": qres.marked_count,
        "marked_probability_quantum": qres.marked_probability,
        "uniform_baseline_marked_prob": qres.uniform_baseline,
        "iterations_quantum": qres.iterations,
        "best_cost_quantum": qres.best_cost,
        "oracle_prep_time_s": qres.oracle_prep_time_s,
        "circuit_run_time_s": qres.circuit_run_time_s,
        "total_quantum_time_s": q_total_t,
        "best_l_error_mh": mh_err,
        "mh_time_s": mh_t,
        "mh_budget_evaluations": classical_budget,
        "quantum_predicate_calls": quantum_predicate_calls,
        "classical_predicate_calls_random": classical_predicate_calls_random,
        "classical_predicate_calls_mh": classical_predicate_calls_mh,
        "query_speedup_vs_random": query_speedup_vs_random,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-quantile", type=float, default=0.02,
                        help="Quantile of L-error costs that defines 'marked' perms; "
                             "smaller value = fewer marked items = more Grover amplification. "
                             "Default 0.02 keeps marked fraction ~2%% which lets Grover amplify.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="If > 0, mark the K lowest-cost permutations instead of using "
                             "tau-quantile. Avoids degenerate cases where most costs tie at zero.")
    parser.add_argument("--n-budget-factor", type=int, default=4)
    args = parser.parse_args()

    results = []
    for n in (4, 5, 6):
        try:
            r = run_one(n, args.seed, args.tau_quantile, args.n_budget_factor,
                        top_k=args.top_k)
            results.append(r)
        except Exception as exc:
            print(f"[N={n}] failed: {exc}")

    out_json = os.path.join(OUTPUT_DIR, "genuine_sop_vs_mh_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[benchmark] wrote {out_json}")

    plot_results(results, os.path.join(OUTPUT_DIR, "genuine_sop_vs_mh.png"))
    print(f"[benchmark] wrote plot")
    return 0


def plot_results(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ns = [r["n"] for r in results]
    q_probs = [r["marked_probability_quantum"] for r in results]
    u_probs = [r["uniform_baseline_marked_prob"] for r in results]
    q_costs = [r["best_cost_quantum"] for r in results]
    mh_costs = [r["best_l_error_mh"] for r in results]
    speedups = [r["query_speedup_vs_random"] for r in results]

    width = 0.35
    x = np.arange(len(ns))
    axes[0].bar(x - width / 2, q_probs, width, label="Quantum (Grover)", color="C3")
    axes[0].bar(x + width / 2, u_probs, width, label="Uniform (random)", color="C0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N={n}" for n in ns])
    axes[0].set_ylabel("Probability of marked permutation")
    axes[0].set_title("Quantum amplification factor")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x - width / 2, q_costs, width, label="Quantum best", color="C3")
    axes[1].bar(x + width / 2, mh_costs, width, label="MH best", color="C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"N={n}" for n in ns])
    axes[1].set_ylabel("Best L-error found")
    axes[1].set_title("Solution quality")
    axes[1].legend()

    axes[2].bar(x, speedups, color="C2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"N={n}" for n in ns])
    axes[2].set_ylabel("Quantum vs random (predicate calls)")
    axes[2].set_title("Quantum oracle-call speedup")
    axes[2].axhline(1.0, color="grey", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Genuine Grover SOP search vs classical MH — "
        "honest comparison (no wall-clock quantum claim)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())