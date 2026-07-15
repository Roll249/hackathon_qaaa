"""
COMPREHENSIVE QUANTUM vs CLASSICAL SOP BENCHMARK v2
============================================

KEY INSIGHT: We need a MEANINGFUL optimization target.
- "Match L-function of original" → cost=0 for original, >0 for any perm
- "Match between two realizations" → meaningful comparison

APPROACH: CROSS-CORRELATION MATCHING
- Realization A: generate from Hawkes(μ=1.0, θ=0.8, ω=10)
- Realization B: generate from Hawkes(μ=0.5, θ=0.5, ω=5)  
- TASK: Find permutation π such that π(A_times) best matches B's temporal structure
- Cost = ||ACF(A_times_permuted) - ACF(B_times)||_2

This is meaningful because:
- Original A has ACF(A)
- B has different ACF(B) from different process params
- We want the permutation of A that makes A look most like B
- Cost=0 only if ACF(A_perm) ≈ ACF(B)

This tests whether quantum can find the "best matching" permutation
more efficiently than classical search.
"""

import numpy as np
import torch
import time
import json
import warnings
import sys
import os
warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.augmentation.xy_mixer_qaoa import XYMixerQAOA, apply_swap_chain

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'output_result/quantum_advantage_study')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_acf(times, max_lag=10):
    """Autocorrelation function of event times."""
    n = len(times)
    t = times
    # Normalize (NO SORT - permutation invariance is the BUG we want to fix)
    t = (t - t.mean()) / (t.std() + 1e-6)
    acf = np.correlate(t, t, mode='full')
    acf = acf[n - 1:n - 1 + max_lag]
    acf = acf / acf[0] if acf[0] != 0 else acf
    return acf


def cost_acf_match(times_A, times_B, perm=None):
    """
    Cost = ||ACF(A_permuted) - ACF(B)||_2

    Measures how well a permutation of A's times matches B's temporal structure.
    """
    acf_B = compute_acf(times_B)
    acf_A = compute_acf(times_A)

    if perm is not None:
        permuted_A = times_A[perm]
        acf_A_perm = compute_acf(permuted_A)
    else:
        acf_A_perm = acf_A

    cost = float(np.sum((acf_A_perm - acf_B) ** 2))
    return cost


# ============================================================================
# DATASET: Paired Hawkes realizations
# ============================================================================

def generate_paired_hawkes(n, seed_A, seed_B, mu_A=1.0, theta_A=0.8, omega_A=10.0,
                           mu_B=0.5, theta_B=0.5, omega_B=5.0):
    """Generate two realizations from different Hawkes processes."""
    def hawkes(n, mu, theta, omega, seed):
        rng = np.random.default_rng(seed)
        times = [rng.exponential(1.0 / mu)]
        attempts = 0
        while len(times) < n and attempts < n * 20:
            t = times[-1]
            lam = mu
            for t_i in times[:-1]:
                if t - t_i > 0:
                    lam += theta * omega * np.exp(-omega * (t - t_i))
            lam = max(lam, 1e-6)
            dt = rng.exponential(1.0 / lam)
            t_new = t + dt
            lam_new = mu
            for t_i in times:
                if t_new - t_i > 0:
                    lam_new += theta * omega * np.exp(-omega * (t_new - t_i))
            if rng.random() < lam_new / lam:
                times.append(t_new)
            attempts += 1

        times = np.array(times[:n])
        times = (times - times.min()) / (times.max() - times.min() + 1e-6)
        return times

    times_A = hawkes(n, mu_A, theta_A, omega_A, seed_A)
    times_B = hawkes(n, mu_B, theta_B, omega_B, seed_B)

    # Same spatial coords for fair comparison
    rng = np.random.default_rng(seed_A)
    coords = rng.uniform(0.05, 0.95, (n, 2))

    return coords, times_A, times_B


# ============================================================================
# METHOD 1: XY-Mixer QAOA
# ============================================================================

def run_xy_qaoa(coords, times_A, times_B, n_layers, n_iter, n_samples, seed):
    """Run XY-QAOA on cross-correlation matching."""
    n = len(times_A)
    n_qubits = min(n, 10)
    n_swap = max(1, n - 1)

    # Reference cost (no permutation)
    ref_cost = cost_acf_match(times_A, times_B)

    model = XYMixerQAOA(n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    best_cost = float('inf')
    best_perm = np.arange(n)
    all_costs = []
    wall_start = time.time()

    for it in range(n_iter):
        swap_bits = model.sample_swap_bits(n_samples=n_samples)[0]
        perm = apply_swap_chain(np.arange(n), swap_bits[:n_swap])
        c = cost_acf_match(times_A, times_B, perm)
        all_costs.append(c)

        if c < best_cost:
            best_cost = c
            best_perm = perm.copy()

        optimizer.zero_grad()
        loss = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    wall_time = time.time() - wall_start

    return {
        'method': 'XY-QAOA',
        'best_cost': best_cost,
        'final_cost': all_costs[-1],
        'ref_cost': ref_cost,
        'all_costs': all_costs,
        'wall_time': wall_time,
        'iterations': n_iter,
        'func_evals': n_iter * n_samples,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# METHOD 2: Classical SOP (Mohler-Mateu 2024)
# ============================================================================

def run_classical_sop(coords, times_A, times_B, n_perms, n_swap_iters, seed):
    """Run classical SOP."""
    rng = np.random.default_rng(seed)
    n = len(times_A)

    best_perm = np.arange(n)
    best_cost = float('inf')
    all_costs = []
    wall_start = time.time()

    for k in range(n_perms):
        perm = rng.permutation(n)
        for _ in range(n_swap_iters):
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            c_old = cost_acf_match(times_A, times_B, perm)
            c_new = cost_acf_match(times_A, times_B, new_perm)
            if c_new < c_old:
                perm = new_perm

        c_final = cost_acf_match(times_A, times_B, perm)
        all_costs.append(c_final)
        if c_final < best_cost:
            best_cost = c_final
            best_perm = perm.copy()

    wall_time = time.time() - wall_start

    return {
        'method': 'Classical SOP',
        'best_cost': best_cost,
        'final_cost': all_costs[-1] if all_costs else best_cost,
        'all_costs': all_costs,
        'wall_time': wall_time,
        'func_evals': n_perms * n_swap_iters,
        'n_params': 0,
    }


# ============================================================================
# METHOD 3: Random Search (Lower bound)
# ============================================================================

def run_random_search(coords, times_A, times_B, n_candidates, seed):
    """Run random permutation search."""
    rng = np.random.default_rng(seed)
    n = len(times_A)

    best_perm = np.arange(n)
    best_cost = float('inf')
    all_costs = []
    wall_start = time.time()

    for _ in range(n_candidates):
        perm = rng.permutation(n)
        c = cost_acf_match(times_A, times_B, perm)
        all_costs.append(c)
        if c < best_cost:
            best_cost = c
            best_perm = perm.copy()

    wall_time = time.time() - wall_start

    return {
        'method': 'Random Search',
        'best_cost': best_cost,
        'final_cost': all_costs[-1] if all_costs else best_cost,
        'all_costs': all_costs,
        'wall_time': wall_time,
        'func_evals': n_candidates,
        'n_params': 0,
    }


# ============================================================================
# METHOD 4: Simulated Annealing
# ============================================================================

def run_annealing(coords, times_A, times_B, n_iters, seed):
    """Run simulated annealing."""
    rng = np.random.default_rng(seed)
    n = len(times_A)

    perm = rng.permutation(n)
    cost = cost_acf_match(times_A, times_B, perm)
    best_perm = perm.copy()
    best_cost = cost
    all_costs = [cost]
    wall_start = time.time()

    T = 1.0
    for it in range(n_iters):
        T = T * 0.99
        i, j = rng.integers(0, n, 2)
        new_perm = perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        c_new = cost_acf_match(times_A, times_B, new_perm)

        delta = c_new - cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
            perm = new_perm
            cost = c_new
            if cost < best_cost:
                best_cost = cost
                best_perm = perm.copy()
        all_costs.append(cost)

    wall_time = time.time() - wall_start

    return {
        'method': 'Simulated Annealing',
        'best_cost': best_cost,
        'final_cost': all_costs[-1],
        'all_costs': all_costs,
        'wall_time': wall_time,
        'func_evals': n_iters,
        'n_params': 0,
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUANTUM vs CLASSICAL SOP — COMPREHENSIVE BENCHMARK                   ║
║  Task: Match ACF of A_permuted to ACF of B                          ║
║  Dataset: Paired Hawkes realizations (different process params)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    N_values = [8, 12, 16, 20]
    seeds = [42, 123, 456, 789, 1000]

    configs = {
        'XY-QAOA':           {'n_layers': 3, 'n_iter': 100, 'n_samples': 5},
        'Classical SOP':     {'n_perms': 100, 'n_swap_iters': 30},
        'Simulated Annealing': {'n_iters': 200},
        'Random Search':     {'n_candidates': 500},
    }

    print(f"  N values: {N_values}")
    print(f"  Seeds per config: {len(seeds)}")
    print(f"  Methods: {list(configs.keys())}")

    results = []

    for N in N_values:
        print(f"\n{'='*70}")
        print(f"  N={N}")
        print(f"{'='*70}\n")

        row = {'N': N, 'methods': {}}

        for method_name, cfg in configs.items():
            all_best = []
            all_final = []
            all_times = []
            all_funcs = []
            all_refs = []

            for seed in seeds:
                coords, times_A, times_B = generate_paired_hawkes(
                    N, seed_A=seed, seed_B=seed+10000,
                    mu_A=1.0, theta_A=0.8, omega_A=10.0,
                    mu_B=0.5, theta_B=0.5, omega_B=5.0
                )

                ref = cost_acf_match(times_A, times_B, None)

                if method_name == 'XY-QAOA':
                    r = run_xy_qaoa(coords, times_A, times_B, cfg['n_layers'],
                                   cfg['n_iter'], cfg['n_samples'], seed)
                elif method_name == 'Classical SOP':
                    r = run_classical_sop(coords, times_A, times_B,
                                          cfg['n_perms'], cfg['n_swap_iters'], seed)
                elif method_name == 'Simulated Annealing':
                    r = run_annealing(coords, times_A, times_B, cfg['n_iters'], seed)
                else:
                    r = run_random_search(coords, times_A, times_B,
                                         cfg['n_candidates'], seed)

                all_best.append(r['best_cost'])
                all_final.append(r['final_cost'])
                all_times.append(r['wall_time'])
                all_funcs.append(r['func_evals'])
                all_refs.append(ref)

            row['methods'][method_name] = {
                'best_mean': float(np.mean(all_best)),
                'best_std': float(np.std(all_best)),
                'final_mean': float(np.mean(all_final)),
                'time_mean': float(np.mean(all_times)),
                'funcs_mean': float(np.mean(all_funcs)),
                'ref_mean': float(np.mean(all_refs)),
            }

            b = np.mean(all_best)
            s = np.std(all_best)
            f = np.mean(all_final)
            t = np.mean(all_times)
            print(f"  {method_name:<22} best={b:.6f}±{s:.4f}  "
                  f"final={f:.6f}  t={t:.2f}s")

        results.append(row)

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Best Cost by Method (lower is better)")
    print(f"{'='*70}\n")

    print(f"  {'N':>4}  {'Random':>10}  {'Annealing':>10}  {'Classical':>10}  {'XY-QAOA':>10}  {'Winner':>12}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    for row in results:
        N = row['N']
        r = row['methods']
        rnd = r.get('Random Search', {}).get('best_mean', 0)
        ann = r.get('Simulated Annealing', {}).get('best_mean', 0)
        cls = r.get('Classical SOP', {}).get('best_mean', 0)
        xy = r.get('XY-QAOA', {}).get('best_mean', 0)
        winners = [('Random', rnd), ('Annealing', ann), ('Classical', cls), ('XY-QAOA', xy)]
        winner = min(winners, key=lambda x: x[1])[0]
        print(f"  {N:>4}  {rnd:>10.6f}  {ann:>10.6f}  {cls:>10.6f}  {xy:>10.6f}  {winner:>12}")

    # === TIME SCALING ===
    print(f"\n{'='*70}")
    print(f"  WALL CLOCK TIME (seconds)")
    print(f"{'='*70}\n")
    print(f"  {'N':>4}  {'Random':>10}  {'Annealing':>10}  {'Classical':>10}  {'XY-QAOA':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for row in results:
        N = row['N']
        r = row['methods']
        rnd = r.get('Random Search', {}).get('time_mean', 0)
        ann = r.get('Simulated Annealing', {}).get('time_mean', 0)
        cls = r.get('Classical SOP', {}).get('time_mean', 0)
        xy = r.get('XY-QAOA', {}).get('time_mean', 0)
        print(f"  {N:>4}  {rnd:>10.2f}s  {ann:>10.2f}s  {cls:>10.2f}s  {xy:>10.2f}s")

    # === PARAMETER EFFICIENCY ===
    print(f"\n{'='*70}")
    print(f"  QUALITY/TIME RATIO (cost_improvement_per_second)")
    print(f"{'='*70}\n")
    print(f"  {'N':>4}  {'Random':>10}  {'Annealing':>10}  {'Classical':>10}  {'XY-QAOA':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for row in results:
        N = row['N']
        r = row['methods']
        rnd = r.get('Random Search', {}).get('best_mean', 1)
        ann = r.get('Simulated Annealing', {}).get('best_mean', 1)
        cls = r.get('Classical SOP', {}).get('best_mean', 1)
        xy = r.get('XY-QAOA', {}).get('best_mean', 1)
        rnd_t = r.get('Random Search', {}).get('time_mean', 1)
        ann_t = r.get('Simulated Annealing', {}).get('time_mean', 1)
        cls_t = r.get('Classical SOP', {}).get('time_mean', 1)
        xy_t = r.get('XY-QAOA', {}).get('time_mean', 1)
        print(f"  {N:>4}  {(1/rnd_t if rnd_t>0 else 0):>10.4f}  {(1/ann_t if ann_t>0 else 0):>10.4f}  {(1/cls_t if cls_t>0 else 0):>10.4f}  {(1/xy_t if xy_t>0 else 0):>10.4f}")

    # === THEORETICAL ===
    print(f"\n{'='*70}")
    print(f"  THEORETICAL QUANTUM ADVANTAGE")
    print(f"{'='*70}\n")
    from math import factorial, sqrt
    print(f"  {'N':>4}  {'N!':>20}  {'√N!':>15}  {'Ratio':>10}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*15}  {'-'*10}")
    for N in [8, 12, 16, 20]:
        nf = factorial(N)
        snf = int(sqrt(nf))
        print(f"  {N:>4}  {nf:>20,}  {snf:>15,}  {nf//snf:>10,}x")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'comprehensive_benchmark.json'), 'w') as f:
        json.dump({'results': results, 'configs': configs, 'N_values': N_values,
                  'seeds': seeds}, f, indent=2, default=str)

    print(f"\n  Results: {os.path.join(OUTPUT_DIR, 'comprehensive_benchmark.json')}")
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    run_benchmark()