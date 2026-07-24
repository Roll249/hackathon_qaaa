"""Benchmark: QPIA vs Coined Quantum Walk for Epidemiology.

This benchmark compares the QPIA (Quantum Path Integral Approach) against
the existing coined quantum walk implementation to see if path-based
interference can achieve resonance where node-based interference failed.

Key question: Does QPIA's path-based approach align better with the problem
structure (finding transmission chains leading to hotspots)?

Test methodology:
1. Use the same synthetic Dien Bien graph
2. Mark the top-5 highest-risk communes as "hotspots"
3. Run both algorithms and measure P(marked) = sum of probabilities at hotspots
4. Resonance threshold: P(marked) < 0.05 (current coined QW failed this)

Reference: Gautam & Ahn 2024 "QPIA for VRP" (IEEE TITS)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
import json
import time

_parent = _Path(__file__).parent.parent
if str(_parent) not in _sys.path:
    _sys.path.insert(0, str(_parent))

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def create_ring_graph(n: int, n_marked: int = 1) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a ring graph with marked nodes.
    
    Returns:
        adjacency, risk, marked_indices
    """
    adjacency = np.zeros((n, n))
    for i in range(n):
        adjacency[i, (i + 1) % n] = 1.0
        adjacency[i, (i - 1) % n] = 1.0
    
    # Risk: uniform + one high-risk marked node
    risk = np.ones(n) * 0.3
    marked = [(n // 4) % n]  # Default marked at position 1/4 around the ring
    risk[marked] = 0.95
    
    return adjacency, risk, marked


def create_star_graph(n: int, n_marked: int = 1) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a star graph centered at node 0.
    
    Returns:
        adjacency, risk, marked_indices
    """
    adjacency = np.zeros((n, n))
    for i in range(1, n):
        adjacency[0, i] = 1.0
        adjacency[i, 0] = 1.0
    
    # Risk: center is hotspot
    risk = np.ones(n) * 0.3
    marked = [0]
    risk[marked] = 0.95
    
    return adjacency, risk, marked


def create_grid_graph(size: int, n_marked: int = 1) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a grid graph with marked nodes at corners/center.
    
    Returns:
        adjacency, risk, marked_indices
    """
    n = size * size
    adjacency = np.zeros((n, n))
    
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            if j + 1 < size:
                adjacency[idx, idx + 1] = 1.0
                adjacency[idx + 1, idx] = 1.0
            if i + 1 < size:
                adjacency[idx, idx + size] = 1.0
                adjacency[idx + size, idx] = 1.0
    
    # Risk: center is hotspot
    risk = np.ones(n) * 0.3
    center = (size // 2) * size + (size // 2)
    marked = [center]
    risk[marked] = 0.95
    
    return adjacency, risk, marked


def create_path_graph(n: int, n_marked: int = 1) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a path graph (line) with marked node at end.
    
    Returns:
        adjacency, risk, marked_indices
    """
    adjacency = np.zeros((n, n))
    for i in range(n - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    
    # Risk: end node is hotspot
    risk = np.ones(n) * 0.3
    marked = [n - 1]
    risk[marked] = 0.95
    
    return adjacency, risk, marked


# ═══════════════════════════════════════════════════════════════════════════════
# QPIA IMPLEMENTATION (from qpia_epidemiology.py)
# ═══════════════════════════════════════════════════════════════════════════════

from src.qpia_epidemiology import (
    qpia_search, QPIAResult, PathState, enumerate_paths,
    compute_path_amplitudes, aggregate_to_nodes,
    compute_interference_analysis, EpidemicAction, RiskWeightedAction,
    qpia_grover_hybrid, qpia_index_case_finding
)


def run_qpia(
    adjacency: np.ndarray,
    risk: np.ndarray,
    marked: list[int],
    max_path_length: int = 4,
    action_scale: float = 1.0,
    risk_weight: float = 2.0,
    start_nodes: Optional[list[int]] = None,
    use_target_aware: bool = False,
    use_backward: bool = False,
) -> dict:
    """Run QPIA and return metrics."""
    n = adjacency.shape[0]
    
    # Run QPIA search
    result = qpia_search(
        adjacency=adjacency,
        risk=risk,
        start_nodes=start_nodes if start_nodes else list(range(n)),
        max_path_length=max_path_length,
        action_type="target_aware" if use_target_aware else "risk_weighted",
        action_scale=action_scale,
        risk_weight=risk_weight,
        marked_nodes=marked,
        target_action=use_target_aware,
        backward_paths=use_backward,
    )
    
    # Compute P(marked) - sum of probabilities at marked nodes
    p_marked = float(np.sum(result.node_probabilities[marked]))
    
    # Compute interference analysis
    analysis = compute_interference_analysis(result, adjacency, risk)
    
    return {
        "algorithm": "QPIA",
        "p_marked": p_marked,
        "n_paths": result.n_paths,
        "top_5": result.top_k_nodes(5),
        "phase_coherence": analysis["phase_coherence"],
        "convergence": analysis["convergence_metric"],
        "marked_in_top_5": sum(1 for m in marked if m in [idx for idx, _ in result.top_k_nodes(5)]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COINED QUANTUM WALK (simplified from existing implementation)
# ═══════════════════════════════════════════════════════════════════════════════

def grover_mark(psi: np.ndarray, marked: list[int], n_qubits: int) -> np.ndarray:
    """Apply Grover oracle marking phase flip."""
    psi_marked = psi.copy()
    for m in marked:
        if m < len(psi):
            psi_marked[m] = -psi_marked[m]
    return psi_marked


def diffusion(psi: np.ndarray) -> np.ndarray:
    """Grover diffusion operator (inversion about mean)."""
    n = len(psi)
    mean = np.mean(psi)
    return 2 * mean - psi


def run_coined_walk(
    adjacency: np.ndarray,
    risk: np.ndarray,
    marked: list[int],
    n_steps: int = 20,
    seed: int = 42,
) -> dict:
    """Run coined quantum walk (Grover-based) and return metrics.
    
    This is a simplified version of the arc-space coined walk.
    Uses Grover amplification for marked states.
    """
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    n_qubits = int(np.ceil(np.log2(n)))
    dim = 2 ** n_qubits
    
    # Initialize uniform superposition
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Markov chain mixing (simplified)
    degree = np.sum(adjacency > 0, axis=1)
    
    for step in range(n_steps):
        # Apply walk operator (approximated by uniform mixing)
        # In arc-space QW, this would be: U_walk = H ⊗ (2|0⟩⟨0| - I)
        # Here we approximate with diffusion
        
        # Mark phase flip (oracle)
        psi = grover_mark(psi, marked, n_qubits)
        
        # Diffusion
        psi = diffusion(psi)
        
        # Renormalize
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))
    
    # Final probabilities
    probs = np.abs(psi[:n]) ** 2
    probs = probs / np.sum(probs)  # Renormalize
    
    # P(marked)
    p_marked = float(np.sum(probs[marked]))
    
    # Top 5
    top5 = list(np.argsort(probs)[::-1][:5])
    
    return {
        "algorithm": "CoinedQW",
        "p_marked": p_marked,
        "n_steps": n_steps,
        "top_5": top5,
        "marked_in_top_5": sum(1 for m in marked if m in top5),
    }


def run_spatial_search(
    adjacency: np.ndarray,
    risk: np.ndarray,
    marked: list[int],
    seed: int = 42,
) -> dict:
    """Run spatial search (amplification on spatial structure)."""
    n = adjacency.shape[0]
    
    # Use risk as initial amplitudes
    amplitudes = np.sqrt(risk + 1e-10)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    # Grover-like amplification (iterative)
    n_iterations = int(np.ceil(np.sqrt(n)))
    
    for _ in range(n_iterations):
        # Phase flip on marked
        for m in marked:
            amplitudes[m] = -amplitudes[m]
        
        # Diffusion (inversion about mean on full space)
        mean = np.mean(amplitudes)
        amplitudes = 2 * mean - amplitudes
        
        # Renormalize
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    probs = np.abs(amplitudes) ** 2
    p_marked = float(np.sum(probs[marked]))
    top5 = list(np.argsort(probs)[::-1][:5])
    
    return {
        "algorithm": "SpatialSearch",
        "p_marked": p_marked,
        "n_iterations": n_iterations,
        "top_5": top5,
        "marked_in_top_5": sum(1 for m in marked if m in top5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def run_classical_scan(risk: np.ndarray, marked: list[int]) -> dict:
    """Classical scan (always finds maximum risk)."""
    n = len(risk)
    top_k = list(np.argsort(risk)[::-1][:5])
    
    return {
        "algorithm": "ClassicalScan",
        "p_marked": float(np.sum(risk[marked]) / np.sum(risk)),  # Weighted by risk
        "top_5": top_k,
        "marked_in_top_5": sum(1 for m in marked if m in top_k),
    }


def run_random_baseline(risk: np.ndarray, marked: list[int], n_trials: int = 100) -> dict:
    """Random sampling baseline."""
    n = len(risk)
    rng = np.random.default_rng(42)
    
    # Sample from uniform distribution
    samples = rng.integers(0, n, size=(n_trials, 5))
    
    # Count how often marked appears in top-5
    hits = 0
    for sample in samples:
        if any(m in sample for m in marked):
            hits += 1
    
    return {
        "algorithm": "RandomBaseline",
        "hit_rate": hits / n_trials,
        "expected_p_marked": len(marked) / n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result from one benchmark trial."""
    graph_type: str
    n_nodes: int
    algorithm: str
    p_marked: float
    marked_in_top_5: int
    resonance_achieved: bool  # P(marked) < 0.05
    execution_time_ms: float
    details: dict


def benchmark_on_graph(
    graph_type: str,
    adjacency: np.ndarray,
    risk: np.ndarray,
    marked: list[int],
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run all algorithms on one graph."""
    n = adjacency.shape[0]
    results = []
    
    # QPIA - test multiple configurations
    # Standard configurations (path forward, risk-weighted action)
    for max_len in [2, 3, 4]:
        for action_scale in [0.5, 1.0, 2.0]:
            start_time = time.time()
            try:
                qpia_result = run_qpia(
                    adjacency=adjacency,
                    risk=risk,
                    marked=marked,
                    max_path_length=max_len,
                    action_scale=action_scale,
                    risk_weight=2.0,
                    use_target_aware=False,
                    use_backward=False,
                )
                elapsed = (time.time() - start_time) * 1000
                results.append(BenchmarkResult(
                    graph_type=graph_type,
                    n_nodes=n,
                    algorithm=f"QPIA(len={max_len},scale={action_scale})",
                    p_marked=qpia_result["p_marked"],
                    marked_in_top_5=qpia_result["marked_in_top_5"],
                    resonance_achieved=qpia_result["p_marked"] < 0.05,
                    execution_time_ms=elapsed,
                    details=qpia_result,
                ))
            except Exception as e:
                print(f"  QPIA failed: {e}")
    
    # QPIA with target-aware action (rewards reaching marked nodes)
    for max_len in [2, 3, 4]:
        for target_reward in [2.0, 4.0]:
            start_time = time.time()
            try:
                qpia_result = run_qpia(
                    adjacency=adjacency,
                    risk=risk,
                    marked=marked,
                    max_path_length=max_len,
                    action_scale=1.0,
                    risk_weight=2.0,
                    use_target_aware=True,
                    use_backward=False,
                )
                elapsed = (time.time() - start_time) * 1000
                results.append(BenchmarkResult(
                    graph_type=graph_type,
                    n_nodes=n,
                    algorithm=f"QPIA-Target(len={max_len},reward={target_reward})",
                    p_marked=qpia_result["p_marked"],
                    marked_in_top_5=qpia_result["marked_in_top_5"],
                    resonance_achieved=qpia_result["p_marked"] < 0.05,
                    execution_time_ms=elapsed,
                    details=qpia_result,
                ))
            except Exception as e:
                print(f"  QPIA-Target failed: {e}")
    
    # QPIA with backward path enumeration (for index case finding)
    # This finds paths that lead TO the hotspot, helping identify transmission sources
    for max_len in [2, 3, 4]:
        start_time = time.time()
        try:
            qpia_result = run_qpia(
                adjacency=adjacency,
                risk=risk,
                marked=marked,
                max_path_length=max_len,
                action_scale=1.0,
                risk_weight=2.0,
                use_target_aware=False,
                use_backward=True,
            )
            elapsed = (time.time() - start_time) * 1000
            results.append(BenchmarkResult(
                graph_type=graph_type,
                n_nodes=n,
                algorithm=f"QPIA-Backward(len={max_len})",
                p_marked=qpia_result["p_marked"],
                marked_in_top_5=qpia_result["marked_in_top_5"],
                resonance_achieved=qpia_result["p_marked"] < 0.05,
                execution_time_ms=elapsed,
                details=qpia_result,
            ))
        except Exception as e:
            print(f"  QPIA-Backward failed: {e}")
    
    # QPIA-Grover Hybrid (honest quantum algorithm)
    for max_len in [2, 3, 4]:
        start_time = time.time()
        try:
            hybrid_result = qpia_grover_hybrid(
                adjacency=adjacency,
                risk=risk,
                marked=marked,
                max_path_length=max_len,
            )
            elapsed = (time.time() - start_time) * 1000
            results.append(BenchmarkResult(
                graph_type=graph_type,
                n_nodes=n,
                algorithm=f"QPIA-Grover(len={max_len})",
                p_marked=hybrid_result["p_marked"],
                marked_in_top_5=hybrid_result["marked_in_top_5"],
                resonance_achieved=hybrid_result["p_marked"] < 0.05,
                execution_time_ms=elapsed,
                details=hybrid_result,
            ))
        except Exception as e:
            print(f"  QPIA-Grover failed: {e}")
    
    # Coined QW
    start_time = time.time()
    try:
        qw_result = run_coined_walk(
            adjacency=adjacency,
            risk=risk,
            marked=marked,
            n_steps=20,
            seed=seed,
        )
        elapsed = (time.time() - start_time) * 1000
        results.append(BenchmarkResult(
            graph_type=graph_type,
            n_nodes=n,
            algorithm="CoinedQW",
            p_marked=qw_result["p_marked"],
            marked_in_top_5=qw_result["marked_in_top_5"],
            resonance_achieved=qw_result["p_marked"] < 0.05,
            execution_time_ms=elapsed,
            details=qw_result,
        ))
    except Exception as e:
        print(f"  CoinedQW failed: {e}")
    
    # Spatial Search
    start_time = time.time()
    try:
        ss_result = run_spatial_search(
            adjacency=adjacency,
            risk=risk,
            marked=marked,
            seed=seed,
        )
        elapsed = (time.time() - start_time) * 1000
        results.append(BenchmarkResult(
            graph_type=graph_type,
            n_nodes=n,
            algorithm="SpatialSearch",
            p_marked=ss_result["p_marked"],
            marked_in_top_5=ss_result["marked_in_top_5"],
            resonance_achieved=ss_result["p_marked"] < 0.05,
            execution_time_ms=elapsed,
            details=ss_result,
        ))
    except Exception as e:
        print(f"  SpatialSearch failed: {e}")
    
    # Classical baselines
    classical = run_classical_scan(risk, marked)
    results.append(BenchmarkResult(
        graph_type=graph_type,
        n_nodes=n,
        algorithm="ClassicalScan",
        p_marked=classical["p_marked"],
        marked_in_top_5=classical["marked_in_top_5"],
        resonance_achieved=False,  # Classical doesn't "achieve resonance"
        execution_time_ms=0.0,
        details=classical,
    ))
    
    random = run_random_baseline(risk, marked)
    results.append(BenchmarkResult(
        graph_type=graph_type,
        n_nodes=n,
        algorithm="Random",
        p_marked=random["expected_p_marked"],
        marked_in_top_5=0,
        resonance_achieved=False,
        execution_time_ms=0.0,
        details=random,
    ))
    
    return results


def run_comprehensive_benchmark(seeds: list[int] = [42, 43, 44]) -> list[BenchmarkResult]:
    """Run comprehensive benchmark across multiple graph types and seeds."""
    all_results = []
    
    print("=" * 80)
    print("QPIA vs Coined Quantum Walk Benchmark")
    print("Testing if path-based interference achieves resonance where node-based failed")
    print("=" * 80)
    
    # Graph configurations
    graphs = [
        ("ring_20", lambda: create_ring_graph(20)),
        ("star_15", lambda: create_star_graph(15)),
        ("grid_4x4", lambda: create_grid_graph(4)),
        ("path_20", lambda: create_path_graph(20)),
    ]
    
    for graph_name, graph_fn in graphs:
        print(f"\n{'='*60}")
        print(f"Testing on {graph_name}")
        print(f"{'='*60}")
        
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            adjacency, risk, marked = graph_fn()
            n = adjacency.shape[0]
            print(f"    Nodes: {n}, Marked: {marked}")
            
            results = benchmark_on_graph(
                graph_type=graph_name,
                adjacency=adjacency,
                risk=risk,
                marked=marked,
                seed=seed,
            )
            
            all_results.extend(results)
            
            # Print summary for this trial
            for r in results:
                resonance_str = "✓ RESONANCE" if r.resonance_achieved else ""
                print(f"    {r.algorithm:30s} P(marked)={r.p_marked:.4f}  top5={r.marked_in_top_5} {resonance_str}")
    
    return all_results


def analyze_results(results: list[BenchmarkResult]) -> dict:
    """Analyze benchmark results."""
    by_algo = {}
    for r in results:
        if r.algorithm not in by_algo:
            by_algo[r.algorithm] = []
        by_algo[r.algorithm].append(r)
    
    analysis = {}
    for algo, res_list in by_algo.items():
        p_marked_mean = np.mean([r.p_marked for r in res_list])
        p_marked_std = np.std([r.p_marked for r in res_list])
        resonance_rate = np.mean([r.resonance_achieved for r in res_list])
        top5_hit_rate = np.mean([r.marked_in_top_5 for r in res_list])
        
        analysis[algo] = {
            "p_marked_mean": float(p_marked_mean),
            "p_marked_std": float(p_marked_std),
            "resonance_rate": float(resonance_rate),
            "top5_hit_rate": float(top5_hit_rate),
            "n_trials": len(res_list),
        }
    
    return analysis


def print_analysis(analysis: dict):
    """Print analysis summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Sort by P(marked) ascending (lower = better for finding rare events)
    sorted_algos = sorted(analysis.items(), key=lambda x: x[1]["p_marked_mean"])
    
    print(f"\n{'Algorithm':<35} {'P(marked)':>12} {'Resonance':>12} {'Top5 Hits':>12}")
    print("-" * 75)
    
    for algo, stats in sorted_algos:
        resonance_str = f"{stats['resonance_rate']*100:.0f}%" if stats['resonance_rate'] > 0 else "-"
        print(f"{algo:<35} {stats['p_marked_mean']:>10.4f} ± {stats['p_marked_std']:.4f} {resonance_str:>12} {stats['top5_hit_rate']:>10.1f}/5")
    
    print("\n" + "-" * 75)
    print("INTERPRETATION:")
    print("  - P(marked) < 0.05 = resonance achieved (quantum speedup for rare event)")
    print("  - Resonance rate = % of trials where P(marked) < 0.05")
    print("  - Top5 Hits = avg number of marked nodes in predicted top-5")
    print("  - QPIA goal: Match or beat CoinedQW on these metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QPIA vs Coined QW Benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                       help="Random seeds for trials")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_benchmark(seeds=args.seeds)
    
    # Analyze
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Save results
    if args.output:
        output_data = {
            "results": [
                {
                    "graph_type": r.graph_type,
                    "n_nodes": r.n_nodes,
                    "algorithm": r.algorithm,
                    "p_marked": r.p_marked,
                    "marked_in_top_5": r.marked_in_top_5,
                    "resonance_achieved": r.resonance_achieved,
                    "execution_time_ms": r.execution_time_ms,
                    "details": r.details,
                }
                for r in results
            ],
            "analysis": analysis,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
