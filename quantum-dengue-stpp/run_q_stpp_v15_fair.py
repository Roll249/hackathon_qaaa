#!/usr/bin/env python3
"""
Q-STPP v15 (corrected): FAIR comparison of SOP permutation-generation strategies.

WHAT THIS DOES
--------------
Second-Order Preserving (SOP) permutations are used to AUGMENT spatio-temporal
point-process data (Mohler & Mateu 2024): we permute event time-stamps while
trying to preserve Ripley's L-function summary. A useful augmenter must do BOTH:
  (1) preserve L(r)                 -> LOW  L(r) error, and
  (2) produce a DIVERSE set of perms -> HIGH set diversity.
Reporting only (1) rewards a method that collapses to a single low-error
permutation, which is useless for augmentation. We therefore report both.

We compare three CLASSICAL local-search strategies under a strictly fair
protocol -- identical RNG seed, identical evaluation budget (same number of
L-function evaluations, the dominant O(N^2) cost), and an error scale-adaptive
acceptance rule for the sampler:

  * mh    - Metropolis-Hastings sampler (annealed, scale-adaptive temperature)
  * grover- "Grover-inspired" focused hill-climbing (greedy, single swap)
  * qaoa  - "QAOA-inspired" multi-swap perturbation (greedy, mixer-like)

HONEST NOTE ON "QUANTUM"
------------------------
There is NO quantum hardware and NO quantum simulator in this script. All three
methods are classical NumPy. "Grover-/QAOA-inspired" means the heuristic borrows
an idea (focused amplification / mixer perturbations); it does not run a quantum
circuit. Any quality/diversity difference reported here is a property of the
classical heuristic, NOT a quantum advantage.
"""

import os
import time
import json
import argparse
import warnings
import tempfile

# Point Matplotlib at a writable config dir before importing it, so machines
# with a read-only/absent HOME (e.g. a fresh VPS) don't print the
# "Matplotlib config dir is not writable" warning.
os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(tempfile.gettempdir(), f'mplconfig-{os.getuid() if hasattr(os, "getuid") else "user"}'),
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_result', 'q_stpp_v15_fair')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = ('mh', 'grover', 'qaoa')
METHOD_LABELS = {
    'mh': 'Metropolis-Hastings',
    'grover': 'Grover-inspired (greedy)',
    'qaoa': 'QAOA-inspired (multi-swap)',
}


# ============================================================================
# Data + L-function summary
# ============================================================================

def simulate_hawkes(n_events_target=50, mu=2.0, theta=0.9, omega=5.0,
                    T=2.0, space_size=0.5, seed=42):
    """Simulate a self-exciting (Hawkes-like) space-time point pattern.

    Uses a single seeded Generator so the pattern is fully reproducible.
    """
    rng = np.random.default_rng(seed)

    events = []
    t = rng.exponential(1.0 / mu) if mu > 0 else 0.01
    max_time = 0.0

    while max_time < T * 0.8 or len(events) < n_events_target // 2:
        lam = mu
        for (t_i, x_i, y_i) in events:
            dt = t - t_i
            if 0 < dt < 2.0:
                lam += theta * omega * np.exp(-omega * dt)

        lam = max(lam, 0.1)
        lam_max = mu + theta * omega * max(len(events), 1)

        if rng.random() < lam / lam_max:
            events.append((t, rng.uniform(0, space_size), rng.uniform(0, space_size)))

        t += rng.exponential(1.0 / max(lam_max, 0.1))
        max_time = t

        if len(events) > 500 or t > T * 2:
            break

    if not events:
        return np.array([]), np.array([]), np.array([])

    events = np.array(events)
    return events[:, 0], events[:, 1], events[:, 2]


def compute_L_summary(times, coords_x, coords_y, r_values, T=1.0, space_size=1.0):
    """Space-time second-order summary derived from Ripley's K.

    We build a distance matrix over (x, y, time*scale), count neighbour pairs
    within each radius r to estimate K(r) = pairs(<r) / N^2, then return a
    stabilised transform L(r) = sign(K) * |K|^(1/3). The exact transform is not
    important; it is applied identically to every method, so relative
    comparisons are valid.
    """
    if len(times) < 2:
        return np.zeros_like(r_values)

    time_scale = space_size / T
    z = np.column_stack([coords_x, coords_y, times * time_scale])
    n = len(times)
    dist = squareform(pdist(z, metric='euclidean'))

    K = np.zeros_like(r_values)
    for k, r in enumerate(r_values):
        K[k] = (np.sum(dist < r) - n) / (n * n)
    return np.sign(K) * np.abs(K) ** (1.0 / 3.0)


def l_error(L_perm, L_target):
    """Mean squared deviation of a permutation's L(r) from the target's."""
    return float(np.mean((L_perm - L_target) ** 2))


def set_diversity(perms):
    """Mean pairwise normalised Hamming distance across a set of permutations.

    0.0 => all permutations identical (mode collapse; useless for augmentation).
    ~1.0 => every position differs between every pair (maximally diverse).
    """
    m = len(perms)
    if m < 2:
        return 0.0
    dists = [np.mean(perms[a] != perms[b])
             for a in range(m) for b in range(a + 1, m)]
    return float(np.mean(dists))


# ============================================================================
# Fair local search: identical evaluation budget for every method
# ============================================================================

def _generate_perms(strategy, times, coords_x, coords_y, r_values, L_target,
                    n_perms, evals_per_perm, rng, anneal_ratio=0.01):
    """Generate `n_perms` permutations with `strategy`.

    Every strategy spends EXACTLY `evals_per_perm` L-summary evaluations per
    permutation (1 for the initial state + evals_per_perm-1 candidate probes),
    so all methods share an identical computational budget. `rng` is the single
    per-run generator shared across methods for a controlled comparison.
    """
    n = len(times)

    def score(perm):
        return l_error(
            compute_L_summary(times[perm], coords_x, coords_y, r_values), L_target
        )

    perms = []
    n_steps = max(1, evals_per_perm - 1)

    for _ in range(n_perms):
        perm = rng.permutation(n)
        cur_err = score(perm)                       # counted eval #1

        # Scale-adaptive annealing temperature for the MH sampler: start near
        # the initial error magnitude, decay geometrically. No magic constant.
        temp0 = max(cur_err, 1e-12)
        temp1 = temp0 * anneal_ratio

        for step in range(n_steps):
            if strategy == 'qaoa':
                # mixer-like multi-swap proposal
                cand = perm.copy()
                n_swaps = int(rng.integers(1, max(2, n // 4)))
                for _ in range(n_swaps):
                    i, j = rng.choice(n, 2, replace=False)
                    cand[i], cand[j] = cand[j], cand[i]
            else:
                # single-swap proposal (mh and grover)
                i, j = rng.choice(n, 2, replace=False)
                cand = perm.copy()
                cand[i], cand[j] = cand[j], cand[i]

            cand_err = score(cand)                  # the one counted probe/step

            if strategy == 'mh':
                frac = step / n_steps
                temp = temp0 * (temp1 / temp0) ** frac
                accept = (cand_err < cur_err or
                          rng.random() < np.exp(-(cand_err - cur_err) / max(temp, 1e-12)))
            else:  # grover, qaoa -> greedy hill-climbing
                accept = cand_err < cur_err

            if accept:
                perm, cur_err = cand, cand_err

        perms.append(perm)

    return perms


def evaluate_method(strategy, times, coords_x, coords_y, r_values, L_target,
                    n_perms, evals_per_perm, seed):
    """Run one method and return its quality/diversity/time metrics."""
    rng = np.random.default_rng(seed)   # identical seed across methods
    t0 = time.time()
    perms = _generate_perms(strategy, times, coords_x, coords_y, r_values,
                            L_target, n_perms, evals_per_perm, rng)
    elapsed = time.time() - t0

    errors = [l_error(compute_L_summary(times[p], coords_x, coords_y, r_values), L_target)
              for p in perms]
    return {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'diversity': set_diversity(perms),
        'time': float(elapsed),
    }


# ============================================================================
# Runner
# ============================================================================

def run_single(seed, n_events, n_perms, evals_per_perm):
    """One (seed, N) cell: run every method under the same budget."""
    times, cx, cy = simulate_hawkes(n_events_target=n_events, seed=seed)
    if len(times) < 10:
        return None

    r_values = np.linspace(0.05, 0.3, 8)
    L_target = compute_L_summary(times, cx, cy, r_values)

    out = {'seed': seed, 'n_events': int(len(times)), 'n_events_target': n_events,
           'evals_per_perm': evals_per_perm, 'n_perms': n_perms}
    for m in METHODS:
        out[m] = evaluate_method(m, times, cx, cy, r_values, L_target,
                                 n_perms, evals_per_perm, seed)
    return out


def aggregate(rows):
    """Average per-method metrics across seeds, grouped by target N."""
    by_n = {}
    for r in rows:
        by_n.setdefault(r['n_events_target'], []).append(r)

    agg = {}
    for n, group in sorted(by_n.items()):
        entry = {'n_events_target': n, 'n_seeds': len(group)}
        for m in METHODS:
            errs = np.array([g[m]['mean_error'] for g in group])
            divs = np.array([g[m]['diversity'] for g in group])
            tim = np.array([g[m]['time'] for g in group])
            entry[m] = {
                'mean_error': float(np.mean(errs)),
                'std_error': float(np.std(errs)),
                'mean_diversity': float(np.mean(divs)),
                'mean_time': float(np.mean(tim)),
            }
        agg[n] = entry
    return agg


def ratio(numer, denom, eps=1e-9):
    """Bounded ratio: guards against divide-by-near-zero blow-ups."""
    return numer / max(denom, eps)


def print_summary(agg):
    print("\n" + "=" * 78)
    print("  FAIR SOP COMPARISON — quality (L(r) error) AND diversity")
    print("=" * 78)
    header = f"  {'N':>4} │ {'method':<28} │ {'L(r) err':>12} │ {'diversity':>10} │ {'t(s)':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for n in sorted(agg):
        e = agg[n]
        for m in METHODS:
            md = e[m]
            print(f"  {n:>4} │ {METHOD_LABELS[m]:<28} │ {md['mean_error']:>12.6f} │ "
                  f"{md['mean_diversity']:>10.3f} │ {md['mean_time']:>7.2f}")
        print("  " + "·" * (len(header) - 2))

    print("\n  Reading the table:")
    print("   • Lower L(r) error  = better preservation of second-order structure.")
    print("   • Higher diversity  = more useful set for data augmentation.")
    print("   • Greedy methods usually reach lower error but lower diversity;")
    print("     the MH sampler trades a little error for much more diversity.")
    print("   • These are classical heuristics under an identical budget —")
    print("     no quantum hardware is involved and no quantum advantage is claimed.")


def plot_summary(agg, path):
    ns = sorted(agg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'mh': 'gray', 'grover': 'green', 'qaoa': 'steelblue'}

    ax = axes[0]
    for m in METHODS:
        errs = [agg[n][m]['mean_error'] for n in ns]
        ax.plot(ns, errs, 'o-', label=METHOD_LABELS[m], color=colors[m])
    ax.set_xlabel('Target number of events (N)')
    ax.set_ylabel('Mean L(r) error (lower = better)')
    ax.set_title('Quality: L(r) preservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for m in METHODS:
        divs = [agg[n][m]['mean_diversity'] for n in ns]
        ax.plot(ns, divs, 'o-', label=METHOD_LABELS[m], color=colors[m])
    ax.set_xlabel('Target number of events (N)')
    ax.set_ylabel('Set diversity (higher = better)')
    ax.set_title('Diversity: usefulness for augmentation')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Q-STPP v15 (corrected): fair SOP comparison — identical seed & budget\n'
                 '(all methods classical; "quantum-inspired" heuristics only)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Q-STPP v15 (corrected): fair SOP comparison')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_events', type=int, nargs='+', default=[20, 30, 50])
    parser.add_argument('--n_perms', type=int, default=10)
    parser.add_argument('--evals_per_perm', type=int, default=200,
                        help='L-summary evaluations per permutation (identical for every method)')
    args = parser.parse_args()

    print("=" * 78)
    print("  Q-STPP v15 (corrected): FAIR SOP COMPARISON")
    print("  Same seed · same evaluation budget · quality + diversity reported")
    print("  All methods classical — no quantum hardware, no quantum advantage claimed")
    print("=" * 78)

    rows = []
    for n in args.n_events:
        for seed in args.seeds:
            r = run_single(seed, n, args.n_perms, args.evals_per_perm)
            if r is not None:
                rows.append(r)
                line = "  ".join(
                    f"{m}: err={r[m]['mean_error']:.5f} div={r[m]['diversity']:.2f}"
                    for m in METHODS)
                print(f"  N={n:>3} seed={seed}: {line}")

    agg = aggregate(rows)
    print_summary(agg)

    with open(os.path.join(OUTPUT_DIR, 'fair_comparison_results.json'), 'w') as f:
        json.dump({'runs': rows, 'aggregate': agg, 'config': vars(args)},
                  f, indent=2, default=float)
    plot_path = os.path.join(OUTPUT_DIR, 'fair_comparison_plot.png')
    plot_summary(agg, plot_path)

    print(f"\n  Results: {os.path.join(OUTPUT_DIR, 'fair_comparison_results.json')}")
    print(f"  Plot:    {plot_path}")


if __name__ == '__main__':
    main()
