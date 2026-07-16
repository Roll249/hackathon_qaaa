#!/usr/bin/env python3
"""Sweep SOP instances to expose genuine Grover amplification.

Run genuine Grover SOP search on multiple synthetic datasets, each
with a different tau threshold that yields a different marked-fraction
ratio M / N!. As M/N! shrinks, the Grover amplification factor
(qprob / baseline) should grow as sqrt(N!/M) — the textbook quadratic
speedup.

This is the empirical evidence that the quantum-superposition
structure genuinely helps SOP search, not just a heuristic.
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
    build_iterated_circuit,
    compute_L_summary,
    enqueue_all_costs,
    l_error,
    number_of_qubits,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_result", "q_stpp_v17")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_unsorted_dataset(n: int, seed: int):
    """Generate an STPP-style synthetic dataset.

    For the SOP cost to depend on the permutation order, the dataset
    MUST have non-trivial space-time correlation. With uniformly-random
    times + uniformly-random spatial coordinates the L-function still
    varies across permutations (because the distance matrix
        ||z_i - z_j||^2 = (x_i - x_j)^2 + (y_i - y_j)^2 + (t_i - t_j)^2
    depends on which time stamp lands at which location), but the
    variation is smaller than for genuinely correlated data.

    We construct a dataset with explicit space-time correlation by
    pairing each (x, y) location with a time that depends linearly on
    the position, then adding small jitter. This is a stand-in for a
    Hawkes process when we want reproducible N-event data.
    """
    rng = np.random.default_rng(seed)
    times = rng.uniform(0.0, 1.0, size=n)
    coords_x = rng.uniform(0.0, 1.0, size=n)
    coords_y = rng.uniform(0.0, 1.0, size=n)
    # Add a space-time coupling term so that the cost function has a
    # genuine, permutation-dependent structure.
    coupling = 0.4 * coords_x + 0.2 * coords_y
    times = times + coupling
    # Re-scale times to [0, 1] so the L summary's time_scale = 1/T still
    # works.
    t_min, t_max = times.min(), times.max()
    if t_max > t_min:
        times = (times - t_min) / (t_max - t_min)
    r_values = np.linspace(0.05, 0.30, 6)
    L_target = compute_L_summary(times, coords_x, coords_y, r_values)
    return times, coords_x, coords_y, r_values, L_target


def grover_iteration_count(marked: int, n_total: int, max_iter: int = 32):
    """Standard Grover iteration count: floor(pi/4 * sqrt(N/M))."""
    if marked <= 0:
        return 1
    ratio = n_total / marked
    return min(max_iter, max(1, int(round(math.pi / 4.0 * math.sqrt(ratio)))))


def run_one(n: int, tau: float, seed: int, top_k: int = 0):
    times, coords_x, coords_y, r_values, L_target = make_unsorted_dataset(n, seed)
    costs = enqueue_all_costs(times, coords_x, coords_y, r_values, L_target)
    n_factorial = math.factorial(n)

    if top_k > 0:
        order = np.argsort(costs)
        marked_idx = order[:top_k]
        marked = int(top_k)
    else:
        marked = int(np.sum(costs <= tau))
        marked_idx = None

    if marked == 0:
        return None

    iters = grover_iteration_count(marked, n_factorial)
    if marked_idx is not None:
        circuit, q, marked_check = build_iterated_circuit(
            n, costs, iters, marked_indices=marked_idx
        )
    else:
        circuit, q, marked_check = build_iterated_circuit(
            n, costs, iters, tau=tau
        )
    assert marked_check == marked

    # Run a single QNode that performs all `iters` Grover iterations internally.
    probs = circuit()
    probs = np.asarray(probs, dtype=np.float64).real
    valid = np.zeros_like(probs, dtype=bool)
    valid[:n_factorial] = True
    if marked_idx is not None:
        valid[:] = False
        valid[marked_idx] = True
    else:
        valid[valid] = costs <= tau
    qprob_amp = float(probs[valid].sum())
    baseline = marked / n_factorial
    best_idx = int(np.argmax(probs))
    best_cost = (
        float(costs[best_idx]) if best_idx < n_factorial else float("nan")
    )

    return {
        "n": n,
        "q": q,
        "tau": tau,
        "marked": marked,
        "n_total": n_factorial,
        "marked_fraction": marked / n_factorial,
        "iters": iters,
        "baseline_prob": baseline,
        "qprob_after_amp": qprob_amp,
        "amplification_factor": qprob_amp / max(baseline, 1e-12),
        "best_cost": best_cost,
        "expected_optimal_iter": grover_iteration_count(marked, n_factorial, max_iter=10**6),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,
                        help="If > 0, mark the K lowest-cost perms per N "
                             "(avoids degenerate cost distributions).")
    args = parser.parse_args()

    # We sweep n and tau so we hit a range of marked-fractions.
    # For each n we choose several taus to make marked_fraction small
    # enough that Grover has something to amplify.
    rows = []
    if args.top_k > 0:
        # Mode: top-k. For each n, sweep over several K values.
        for n in (5, 6):
            data = make_unsorted_dataset(n, args.seed)
            costs = enqueue_all_costs(*data)
            n_factorial = math.factorial(n)
            for k in (3, 6, 12, 24):
                if k > n_factorial // 4:
                    continue
                row = run_one(n, tau=float("nan"), seed=args.seed, top_k=k)
                if row is None:
                    continue
                rows.append(row)
                print(
                    f"N={n}, top_k={k}, marked={row['marked']}/{row['n_total']} "
                    f"({100 * row['marked_fraction']:.1f}%), iters={row['iters']}, "
                    f"baseline={row['baseline_prob']:.3f}, qprob={row['qprob_after_amp']:.3f}, "
                    f"amp={row['amplification_factor']:.2f}x, "
                    f"optimal_iters={row['expected_optimal_iter']}"
                )
    else:
        # Original mode: pick taus to give a range of marked fractions.
        for n in (5, 6):
            times, coords_x, coords_y, r_values, L_target = make_unsorted_dataset(n, args.seed)
            costs = enqueue_all_costs(times, coords_x, coords_y, r_values, L_target)
            for frac_target in (1.0 / 16, 1.0 / 8, 1.0 / 4):
                n_factorial = math.factorial(n)
                target_marked = int(round(n_factorial * frac_target))
                if target_marked < 1:
                    continue
                sorted_costs = np.sort(costs)
                tau = float(sorted_costs[target_marked])
                row = run_one(n, tau=tau, seed=args.seed)
                if row is None:
                    continue
                rows.append(row)
                print(
                    f"N={n}, tau={tau:.5f}, marked={row['marked']}/{row['n_total']} "
                    f"({100 * row['marked_fraction']:.1f}%), iters={row['iters']}, "
                    f"baseline={row['baseline_prob']:.3f}, qprob={row['qprob_after_amp']:.3f}, "
                    f"amp={row['amplification_factor']:.2f}x, "
                    f"optimal_iters={row['expected_optimal_iter']}"
                )

    out_json = os.path.join(OUTPUT_DIR, "grover_amp_sweep_results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    plot_results(rows, os.path.join(OUTPUT_DIR, "grover_amp_sweep.png"))

    print(f"\n[done] wrote {out_json}")


def plot_results(rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = [r["marked_fraction"] for r in rows]
    amps = [r["amplification_factor"] for r in rows]
    ns = [r["n"] for r in rows]
    scatter = ax.scatter(fracs, amps, c=ns, cmap="viridis", s=80)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Marked fraction M / N!")
    ax.set_ylabel("Grover amplification factor (qprob / baseline)")
    ax.set_title(
        "Genuine Grover SOP amplification vs marked fraction\n"
        "(data points colored by N; theoretical sqrt(N!/M) shown dashed)"
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("N (number of events)")

    fracs_arr = np.array(fracs)
    # Theoretical line: amp = sqrt(N!/M) = sqrt(1/fraction)
    theoretical = 1.0 / np.sqrt(fracs_arr)
    theoretical = np.clip(theoretical, 1.0, None)
    ax.plot(
        sorted(fracs),
        [theoretical[i] for i in np.argsort(fracs)],
        "r--",
        alpha=0.6,
        label="theoretical sqrt(N!/M)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()