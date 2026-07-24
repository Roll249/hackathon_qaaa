# QPIA Analysis Report: Why Path Integral Fails for Index Finding

## Executive Summary

After implementing and testing the Quantum Path Integral Approach (QPIA) from Gautam & Ahn 2024 for epidemiology index finding, we found that **QPIA does NOT achieve resonance for hotspot detection** despite achieving mathematically correct path interference.

**Key Finding**: The fundamental issue is a **structural mismatch** between the problem (find marked NODE) and the algorithm (compute amplitude per PATH).

## Theoretical Analysis

### How QPIA Works (Gautam & Ahn 2024)

The QPIA algorithm computes:

$$|\psi\rangle = \sum_{\text{paths}} e^{iS[\text{path}]} |\text{path}\rangle$$

Where $S[\text{path}]$ is the action along the path. For VRP (Vehicle Routing Problem):

- Problem: Find optimal **PATH** (sequence of cities)
- Algorithm: Compute amplitude per **PATH**
- Result: Paths with lower action constructively interfere → optimal route emerges

### Why It Fails for Epidemiology

For epidemiology index finding:

- Problem: Find **NODE** (transmission source/hotspot)  
- Algorithm: Compute amplitude per **PATH**, then aggregate to nodes
- Result: **Structural mismatch** causes interference pattern to NOT align with hotspots

The aggregation step:
$$P(\text{node } j) = \left|\sum_{\text{paths ending at } j} e^{iS[\text{path}]}\right|^2$$

This loses the path-level interference information. The interference pattern highlights nodes that are "central" in the path graph sense, not nodes with high risk.

## Benchmark Results

### Synthetic Graphs

| Algorithm | P(marked) | Resonance | Top-5 Hits |
|-----------|-----------|-----------|-------------|
| QPIA(len=3,scale=1.0) | 0.028 | 75% | 0.2/5 |
| QPIA-Backward(len=3) | 0.028 | 75% | 0.2/5 |
| QPIA-Grover(len=4) | 0.604 | 0% | 1.0/5 |
| Classical Scan | 0.161 | N/A | 1.0/5 |
| Random | 0.057 | N/A | 0.0/5 |

**Observation**: QPIA achieves low P(marked) (resonance) but marked nodes are NOT in top-5.

### Realistic Dien Bien Graph (130 communes)

| Configuration | Hits | Best Hotspot Rank |
|---------------|------|-------------------|
| Classical (risk sort) | **5/5** | #1, #2, #3, #4, #5 |
| QPIA len=2 | 1/5 | #5 (node 97) |
| QPIA len=3 | **0/5** | #32 (node 97) |
| QPIA len=4 | **0/5** | #19 (node 97) |

**True hotspot 30 (highest risk 0.958)**: Ranked #62-67 by QPIA!

## Root Cause Analysis

### Mathematical Explanation

The path integral computes:
$$\text{Amplitude}(j) = \sum_{\text{paths ending at } j} e^{iS[\text{path}]}$$

For constructive interference to boost node $j$, we need many paths ending at $j$ to have similar phases. This happens when:

1. Many paths reach $j$ (high connectivity/centrality)
2. Paths have similar lengths/costs

This biases toward **central nodes**, not **high-risk nodes**.

### Epidemiological Interpretation

In the Dien Bien graph:
- High-risk nodes are at specific geographic locations with favorable conditions
- Central nodes (high connectivity) are NOT necessarily high-risk
- The path integral favors nodes reachable by many short paths = network hubs

## Comparison: Coined Quantum Walk vs QPIA

| Aspect | Coined QW (arc-space) | QPIA (path-space) |
|--------|----------------------|-------------------|
| State space | Individual nodes | Paths/sequences |
| Interference | Between node states | Between path phases |
| Amplitude source | Grover oracle marking | Action functional |
| Works for | Unstructured search | Structured path problems |
| Fails for | Dense graphs (resonance) | Node-target problems |

**Both fail for the same reason**: The problem structure (find marked NODE) doesn't align with the algorithm mechanism.

## What WOULD Work

### Option 1: Grover with Risk Oracle (Current Working Approach)

The Dürr-Høyer algorithm with proper oracle access achieves correct results because:

1. It directly operates on node states
2. Grover amplification is designed for "find marked element" problems
3. No aggregation step that loses information

### Option 2: Hybrid QPIA-Grover

Combining QPIA's problem encoding with Grover's oracle:

```python
# QPIA provides initial amplitude structure (encodes transmission dynamics)
psi = qpia_initial_state(adjacency, risk)

# Grover amplifies marked states on top
for _ in range(√N):
    psi = grover_oracle(psi, marked)
    psi = diffusion(psi)
```

This uses QPIA for what it's good at (encoding problem structure) and Grover for what it's good at (search amplification).

### Option 3: QPIA for Index Case Finding (Different Problem)

QPIA might work for a different question:

> "Given a known hotspot, what are likely index cases (transmission sources)?"

This is a **path-finding** problem where QPIA's mechanism aligns with the problem:
- Find paths leading TO the hotspot
- Paths from index cases to hotspots have specific properties

## Conclusion

### Main Finding

**QPIA does NOT achieve resonance for epidemiology index finding** because of a fundamental structural mismatch between the problem (find marked NODE) and the algorithm (compute amplitude per PATH).

### Why VRP Works

Gautam & Ahn 2024's QPIA works for VRP because:
- VRP explicitly asks for an optimal **PATH**
- QPIA computes amplitude per **PATH**
- Problem structure ALIGNS with algorithm mechanism

### Why Epidemiology Fails

Epidemiology index finding asks for:
- An optimal **NODE** (hotspot)
- But QPIA computes amplitude per **PATH**
- Problem structure DOES NOT align with algorithm mechanism

### Recommendation

For epidemiology hotspot detection, continue using:
1. **Dürr-Høyer (Grover-based)** for maximum finding
2. **Classical sorting** for top-K detection (simpler, works correctly)

QPIA could be explored for:
- Transmission chain reconstruction
- Index case identification given known hotspots
- Other path-centric epidemiological questions

## References

1. Gautam & Ahn, "Quantum Path Integral Approach for Vehicle Routing Optimization With Limited Qubit", IEEE TITS 2024
2. Feynman, "Space-time approach to non-relativistic quantum mechanics", Rev. Mod. Phys. 1948
3. Childs & Goldstone, "Spatial search by quantum walk", PRA 2004
