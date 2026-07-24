# Proposed Quantum Architecture for Dengue STPP Pipeline

**Version:** 1.0  
**Date:** July 22, 2026  
**Status:** Proposal for v18

---

## Current Architecture vs Proposed Architecture

### Current Architecture (v17)

```
Real Dengue CSV → Hawkes Sim → L(r) Features → [CLASSICAL] SOP Candidates
                                                          ↓
                                                    QAOA/Grover Selection
                                                          ↓
                                              [CLASSICAL] 1-NN RBF/Q-Kernel
                                                          ↓
                                                    Hotspot Prediction
```

**Problem:** Quantum handles the easy part (selection from near-optimal candidates).

### Proposed Architecture (v18)

```
Real Dengue CSV → Hawkes Sim → L(r) Features
                                      ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            [CLASSICAL]                   [QUANTUM]
            SOP Candidates ←── Preprocessing ── Spatial Regions
                    ↓                               ↓
            Greedy Selection                 Quantum Walk Search
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                          Ensemble Voting
                                    ↓
                            Hotspot Prediction
```

**Key insight:** Quantum handles the exponential sub-problem (spatial search over M regions), classical handles everything else.

---

## Component Specifications

### 1. Classical Preprocessing (Stage 0-1)

```python
def classical_preprocessing(times, coords_x, coords_y):
    """Classical feature extraction and spatial discretization."""
    # L(r) computation
    L_features = compute_L_summary(times, coords_x, coords_y, r_values)
    
    # Spatial discretization: divide region into M grid cells
    M = 100  # 10×10 grid
    regions = discretize_spatial_grid(coords_x, coords_y, M)
    
    # Region adjacency graph
    G = build_region_graph(regions)
    
    return L_features, regions, G
```

### 2. Quantum Spatial Search (Stage 2)

```python
def quantum_spatial_search(G, regions, target_hotspot_profile):
    """
    Quantum walk search over M spatial regions.
    
    Uses Childs-Goldstone quantum walk for O(√M) search
    vs O(M) classical scan.
    
    Args:
        G: Region adjacency graph (M nodes)
        regions: Region metadata
        target_hotspot_profile: L(r) profile of known hotspots
    
    Returns:
        top_k_regions: Regions most likely to be hotspots
    """
    # Build quantum walk oracle marking high-probability regions
    oracle = build_spatial_oracle(G, target_hotspot_profile)
    
    # Quantum walk search
    walk = quantum_walk_search(oracle, n_steps=sqrt(M))
    
    # Measure and return top-k regions
    measured = walk.measure()
    return decode_region_indices(measured, regions)
```

### 3. Ensemble Integration (Stage 3)

```python
def ensemble_prediction(classical_pred, quantum_pred, weights=None):
    """Ensemble classical and quantum predictions."""
    if weights is None:
        weights = [0.5, 0.5]  # Equal weight initially
    
    # Weighted voting
    combined = weights[0] * classical_pred + weights[1] * quantum_pred
    return combined > 0.5
```

---

## Implementation Plan

### Phase A: Classical Baseline Enhancement

1. Implement spatial discretization module
2. Build region adjacency graph
3. Add classical spatial search baseline

### Phase B: Quantum Spatial Search

1. Implement quantum walk oracle
2. Build quantum walk search circuit
3. Integrate with PennyLane

### Phase C: Ensemble Integration

1. Design ensemble voting mechanism
2. Implement adaptive weighting
3. Benchmark ensemble vs individual methods

---

## Expected Performance

| Metric | Current (v17) | Proposed (v18) | Improvement |
|--------|---------------|-----------------|-------------|
| Spatial search | O(M) classical | O(√M) quantum | 10× at M=100 |
| Hotspot accuracy | ~89% | ~90-95% | +1-6% |
| Wall-clock | 1.09s | 0.5-0.8s | 1.5-2× faster |
| Scalability | N ≤ 30 | N ≤ 100 | 3× better |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Quantum walk implementation complexity | Use PennyLane's built-in operations |
| Hardware noise | Start with simulator, then hardware |
| Ensemble bias | Validate on held-out test set |

---

*Proposal prepared by Research Agent — July 22, 2026*
