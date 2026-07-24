# Optimization Proposal: Coupling Grover + Doi-Peliti

**Author:** Research Team  
**Version:** v18  
**Date:** July 2026

---

## Mục lục

1. [Tổng quan Pipeline hiện tại](#1-tổng-quan-pipeline-hiện-tại)
2. [Phân tích Coupling Points](#2-phân-tích-coupling-points)
3. [Đề xuất Improvements](#3-đề-xuất-improvements)
4. [Implementation Plan](#4-implementation-plan)
5. [Honest Assessment](#5-honest-assessment)

---

## 1. Tổng quan Pipeline hiện tại

### Current Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CURRENT PIPELINE v18                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ Raw Dengue  │───▶│ Doi-Peliti      │───▶│ Filtered Signal  │   │
│  │ Incidence   │    │ Decomposition   │    │ (endo/exo)       │   │
│  └─────────────┘    └─────────────────┘    └────────┬─────────┘   │
│                                                      │              │
│                                                      ▼              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ Risk Map    │───▶│ Grover Spatial  │───▶│ Hotspot          │   │
│  │ (from DP)   │    │ Search          │    │ Rankings         │   │
│  └─────────────┘    └─────────────────┘    └──────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Modules hiện tại

| Module | Input | Output | Status |
|--------|-------|--------|--------|
| Doi-Peliti | Incidence time series | Endogenous/exogenous signals | ✅ Done |
| Grover Search | Risk map | Top-K hotspots | ✅ Done |

### Weaknesses

1. **No feedback loop** - Doi-Peliti output không influence Grover oracle
2. **Independent optimization** - Mỗi module tối ưu riêng, không joint
3. **Risk map generation** - Ad-hoc, không dùng DP decomposition
4. **No uncertainty propagation** - Confidence từ DP không được propagate

---

## 2. Phân tích Coupling Points

### 2.1 Risk Map Generation

**Current approach (ad-hoc):**

```python
# Trong compare_spatial_search()
risk = np.zeros((nx, ny), dtype=float)
risk += rng.uniform(0.1, 0.3, size=(nx, ny))  # Background
# Add hotspots...
risk_map = RiskMap(grid=grid, values=risk)
```

**Problem:** Không dùng actual incidence data hay DP decomposition!

### 2.2 Oracle Construction

**Current approach (threshold-based):**

```python
# Trong run_grover_search()
if target_indices is not None:
    for idx in target_indices:
        if idx < n:
            diag[idx] = -1.0  # Mark
```

**Problem:** Oracle marks cells, nhưng KHÔNG incorporate endogenous/exogenous signals!

### 2.3 Information Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MISSING INFORMATION FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Doi-Peliti                    Grover Search                         │
│  ┌───────────┐                ┌───────────────┐                     │
│  │ Endogenous│ ──────────────▶│ Oracle should │                     │
│  │ (clustered)│   SHOULD       │ mark hotspots │                     │
│  └───────────┘                 └───────────────┘                     │
│                                                                      │
│  ┌───────────┐                ┌───────────────┐                     │
│  │ Exogenous │ ──────────────▶│ Background    │                     │
│  │ (noise)   │   SHOULD       │ should be     │                     │
│  └───────────┘                 │ deprioritized │                     │
│                                 └───────────────┘                     │
│                                                                      │
│  ┌───────────┐                ┌───────────────┐                     │
│  │ Branching │ ──────────────▶│ Search depth  │                     │
│  │ Ratio     │   POSSIBLE     │ should vary   │                     │
│  └───────────┘                 └───────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Đề xuất Improvements

### 3.1 Joint Optimization Framework

**Concept:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 JOINT OPTIMIZATION FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐      │
│  │ Incidence  │───▶│ Doi-Peliti     │───▶│ Endogenous Map  │      │
│  │ Data       │    │ Decomposition  │    │ (clusters)      │      │
│  └─────────────┘    └─────────────────┘    └────────┬─────────┘      │
│                                                      │               │
│                                                      ▼               │
│                              ┌───────────────────────────────────┐   │
│                              │  Weighted Risk Map Generation     │   │
│                              │  λ(s) = α·endo(s) + (1-α)·exo(s)│   │
│                              └───────────────┬───────────────────┘   │
│                                              │                       │
│                                              ▼                       │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐      │
│  │ Branching  │───▶│ Adaptive Oracle │◀───│ Weighted Risk    │      │
│  │ Ratio      │    │ Construction    │    │ Map              │      │
│  └─────────────┘    └─────────────────┘    └──────────────────┘      │
│                              │                                       │
│                              ▼                                       │
│                       ┌─────────────┐                               │
│                       │ Grover      │                               │
│                       │ Search      │                               │
│                       └─────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Specific Proposals

#### Proposal A: Endogenous-Weighted Risk Map

**Idea:** Dùng endogenous signal để weight risk scores.

```python
def generate_weighted_risk_map(
    spatial_grid,
    incidence_matrix,  # shape: (n_districts, n_weeks)
    dp_result,  # DoiPeliti result
    alpha=0.8  # weight for endogenous
):
    """Generate risk map weighted by endogenous activity."""
    
    # Compute spatial risk from incidence
    spatial_risk = np.mean(incidence_matrix, axis=1)  # Average over time
    spatial_risk = spatial_risk.reshape(spatial_grid.nx, spatial_grid.ny)
    
    # Get endogenous intensity
    endogenous = dp_result.endogenous_signal
    
    # Weight by endogenous activity
    # High endogenous = high risk
    endogenous_weight = endogenous[-1]  # Most recent time point
    
    # Combine: risk weighted by endogenous activity
    weighted_risk = spatial_risk * (alpha * endogenous_weight + (1 - alpha))
    
    return RiskMap(grid=spatial_grid, values=weighted_risk)
```

**Rationale:** 
- Cells với high endogenous activity (clustered transmission) → higher risk
- Cells với high exogenous activity (imported cases) → lower risk for hotspot search

#### Proposal B: Adaptive Oracle Construction

**Idea:** Oracle weights phản ánh branching structure.

```python
def build_adaptive_oracle(
    risk_map,
    dp_result,
    top_k=5,
    base_threshold=None
):
    """Build oracle with adaptive threshold based on branching ratio."""
    
    branching_ratio = dp_result.branching_ratio
    
    # Adjust threshold based on branching ratio
    # High n (near critical) → more spread → lower threshold
    # Low n (subcritical) → less spread → higher threshold
    if branching_ratio < 0.5:
        # Subcritical: sharp clusters
        threshold_multiplier = 1.2
    elif branching_ratio < 0.9:
        # Near critical: spread clusters
        threshold_multiplier = 0.8
    else:
        # Near/at critical: power-law spread
        threshold_multiplier = 0.5
    
    # Get top-K cells
    top_indices = risk_map.get_top_k_indices(top_k)
    
    # Create weighted oracle
    oracle_diag = np.ones(2**risk_map.grid.n_qubits, dtype=complex)
    
    for idx in top_indices:
        state_idx = grid_idx_to_state_idx(idx, risk_map.grid.n_qubits)
        # Weight: higher for cells with high endogenous
        weight = risk_map.get_by_index(idx) * threshold_multiplier
        oracle_diag[state_idx] = -np.exp(weight)  # Phase based on weight
    
    return np.diag(oracle_diag)
```

**Rationale:**
- Subcritical process (n<0.5): transmission localized, sharp hotspots
- Critical process (n≈1): power-law spread, diffuse risk

#### Proposal C: Iterative Refinement

**Idea:** Grover iterations adapt dựa trên DP output.

```python
def adaptive_grover_iterations(n_targets, branching_ratio):
    """Determine iterations based on branching structure."""
    
    # Branching affects search difficulty
    # High n → more spread → harder to isolate single hotspot
    
    base_iters = int(np.pi / 4 * np.sqrt(1.0 / (n_targets + 1e-10)))
    
    # Adjust for branching
    if branching_ratio < 0.5:
        # Subcritical: easier search
        adjustment = 0.8
    elif branching_ratio < 0.9:
        # Near critical: standard
        adjustment = 1.0
    else:
        # Critical: harder search
        adjustment = 1.2
    
    return max(1, int(base_iters * adjustment))
```

### 3.3 Joint Loss Function

**Concept:** Optimize cả DP decomposition và Grover search jointly.

```python
def joint_loss(dp_params, grover_params, ground_truth):
    """
    Joint loss for DP + Grover optimization.
    
    Minimize: L = L_dp + λ * L_grover
    
    Where:
    - L_dp: Decomposition error (RMSE endogenous/exogenous)
    - L_grover: Hotspot ranking error
    """
    
    # DP loss: fit Hawkes parameters
    dp_loss = decomposition_rmse(predicted, ground_truth)
    
    # Grover loss: top-K accuracy
    grover_loss = ranking_error(grover_predictions, ground_truth_hotspots)
    
    # Combined
    return dp_loss + lambda_param * grover_loss
```

---

## 4. Implementation Plan

### Phase 1: Data Integration (Days 1-2)

**Milestone:** Risk map từ actual data + DP output.

```python
# Step 1: Load real dengue data
incidence_data = load_dengue_data("data/dengue_vietnam_weekly.csv")

# Step 2: Run DP decomposition
dp = DoiPelitiDecomposer(kernel_type='exponential')
dp_result = dp.decompose(incidence_data.counts, incidence_data.weeks)

# Step 3: Generate weighted risk map
risk_map = generate_weighted_risk_map(
    grid=spatial_grid,
    incidence_matrix=incidence_data.matrix,
    dp_result=dp_result,
    alpha=0.7
)

# Step 4: Run Grover with adaptive oracle
grover_result = run_grover_search(
    risk_map,
    use_adaptive_oracle=True,
    dp_result=dp_result
)
```

### Phase 2: Adaptive Oracle (Days 3-4)

**Milestone:** Oracle construction incorporates DP output.

```python
# Replace fixed oracle with adaptive
oracle = build_adaptive_oracle(
    risk_map=risk_map,
    dp_result=dp_result,
    top_k=5
)

# Use adaptive iterations
n_iters = adaptive_grover_iterations(
    n_targets=5,
    branching_ratio=dp_result.branching_ratio
)
```

### Phase 3: Evaluation (Days 5-6)

**Milestone:** Compare joint vs independent optimization.

```python
# Independent baseline
baseline_result = run_grover_search(risk_map, top_k=5)

# Joint optimization
joint_result = run_grover_search(
    risk_map,
    use_adaptive_oracle=True,
    dp_result=dp_result
)

# Compare
print(f"Baseline Accuracy@5: {baseline_result.accuracy_top5}")
print(f"Joint Accuracy@5: {joint_result.accuracy_top5}")
print(f"Improvement: {joint_result.accuracy_top5 - baseline_result.accuracy_top5}")
```

---

## 5. Honest Assessment

### Feasibility

| Proposal | Feasibility | Time | Expected Impact |
|----------|-------------|------|-----------------|
| A: Weighted Risk Map | HIGH | 2 days | MEDIUM |
| B: Adaptive Oracle | MEDIUM | 3 days | HIGH |
| C: Joint Optimization | LOW | 1 week | HIGH |

### Realistic Timeline (Hackathon)

**With 1 week remaining:**

```
Day 1-2: Proposal A (Weighted Risk Map)
          - Connect DP output to risk map
          - Test on real data

Day 3-4: Proposal B (Adaptive Oracle)  
          - Implement adaptive oracle
          - Adjust iterations based on n

Day 5-6: Evaluation
          - Compare baselines
          - Document results

Day 7:   Buffer / Polish
```

### Expected Outcomes

| Metric | Current | After Proposals A+B |
|--------|---------|---------------------|
| Hotspot Accuracy | 80-90% | 85-95% (estimated) |
| Risk Map Quality | Ad-hoc | Data-driven |
| Adaptivity | Fixed | Adaptive to branching |

### Limitations

1. **Data dependency:** Cần real incidence data, không chỉ synthetic
2. **Validation:** Khó validate without ground truth hotspots
3. **Complexity:** Joint optimization có thể overfit

### Recommendation

**Implement Proposals A + B (not C):**

- A: Weighted Risk Map - straightforward, data-driven
- B: Adaptive Oracle - moderate complexity, high impact
- C: Joint Optimization - deferred (too complex for hackathon)

**Why not C?**
- Joint optimization requires careful tuning
- Risk of overfitting
- Hard to validate in short timeframe

---

## Appendix: Code Skeleton

```python
"""
Joint Optimization: Doi-Peliti + Grover Spatial Search
=====================================================

This module implements the coupling between DP decomposition
and Grover search for improved hotspot detection.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class JointConfig:
    """Configuration for joint DP-Grover optimization."""
    endogenous_weight: float = 0.7  # Alpha for risk map
    branching_adjustment: bool = True
    adaptive_iterations: bool = True


def joint_pipeline(
    incidence_data: np.ndarray,
    spatial_grid,
    dp_result,
    config: JointConfig = None
):
    """
    Run joint DP-Grover pipeline.
    
    Args:
        incidence_data: Incidence counts by location and time
        spatial_grid: SpatialGrid object
        dp_result: DecompositionResult from DoiPeliti
        config: JointConfig with hyperparameters
    
    Returns:
        JointResult with hotspots and metadata
    """
    if config is None:
        config = JointConfig()
    
    # Step 1: Generate weighted risk map
    risk_map = generate_weighted_risk_map(
        grid=spatial_grid,
        incidence=incidence_data,
        dp_result=dp_result,
        alpha=config.endogenous_weight
    )
    
    # Step 2: Determine iterations adaptively
    if config.adaptive_iterations:
        n_iters = adaptive_grover_iterations(
            n_targets=5,
            branching_ratio=dp_result.branching_ratio
        )
    else:
        n_iters = None
    
    # Step 3: Build adaptive oracle
    oracle = build_adaptive_oracle(
        risk_map=risk_map,
        dp_result=dp_result,
        top_k=5
    )
    
    # Step 4: Run Grover
    grover_result = run_grover_search(
        risk_map,
        n_iterations=n_iters,
        oracle_matrix=oracle,
        top_k=5
    )
    
    return JointResult(
        risk_map=risk_map,
        grover_result=grover_result,
        dp_result=dp_result,
        config=config
    )
```

---

*Document này là phần của RAPID-DENGUE v18 pipeline optimization proposal.*
