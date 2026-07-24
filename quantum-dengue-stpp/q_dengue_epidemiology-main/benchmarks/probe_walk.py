"""Probe: what max P(marked) does quantum walk actually achieve?

Rewritten to use bench_weighted_walk.py which has the correct quantum walk.
This file probes whether quantum walk can achieve resonance (P > threshold)
on toy graphs vs the real Điện Biên graph.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from benchmarks.bench_weighted_walk import (
    build_graph, quantum_search_run, classical_hitting_weighted,
    connectivity_info
)

print(f"{'Regime':<20} N={'N':>4} marked={'m':>4} reachable={'reach':>6} "
      f"peak_P={'peak_p':>7} t_cross={'t_cross':>7} resonance={'res':>5}")
print("-" * 75)

# Toy graphs
for N in [8, 16, 32, 48]:
    for regime in ["sparse_binary", "sparse_weighted", "ring", "grid"]:
        A = build_graph(regime, N)
        n_comp, labels = connectivity_info(A)
        largest = np.argmax(np.bincount(labels))
        candidates = [v for v in range(N) if labels[v] == largest and v != 0]
        marked = candidates[N // 4] if len(candidates) > N // 4 else candidates[-1]
        reachable = labels[0] == labels[marked]

        crossing_t, peak_p, peak_t, final_p = quantum_search_run(
            A, start_v=0, marked=marked, max_t=500, threshold=0.05)

        t_class = classical_hitting_weighted(A, marked, n_trials=100)
        speedup = t_class / crossing_t if crossing_t else float('inf')
        resonance = "YES" if crossing_t else "NO"

        print(f"{regime:<20} {N:>4} {marked:>4} {str(reachable):>6} "
              f"{peak_p:>7.4f} {str(crossing_t):>7} {resonance:>5}")

print()
print("=" * 75)
print("REAL ĐIỆN BIÊN GRAPH")
print("=" * 75)

from src.graph_dien_bien import build_synthetic_dien_bien
g = build_synthetic_dien_bien()
A = g.adjacency
n_comp, labels = connectivity_info(A)
start_comp = labels[0]

for m in [10, 65, 120]:
    reachable = labels[m] == start_comp
    crossing_t, peak_p, peak_t, final_p = quantum_search_run(
        A, start_v=0, marked=m, max_t=2000, threshold=0.05)
    t_class = classical_hitting_weighted(A, m, n_trials=100)
    resonance = "YES" if crossing_t else "NO"

    print(f"marked={m:>3} reachable={str(reachable):>5} "
          f"peak_p={peak_p:.4f} at_t={peak_t:>5} "
          f"t_cross={str(crossing_t):>7} resonance={resonance}")
