# TASK 2.1: K-function Computation - QUANTUM OPPORTUNITY

## Thông tin chung

| Field | Value |
|-------|-------|
| **Task ID** | MODULE-2.1 |
| **Module** | Layer 1: Feature Extraction |
| **Priority** | P1 - HIGH |
| **Assigned to** | [ASSIGN] |
| **Due Date** | Week 4 |

---

## 1. Mục tiêu

Tối ưu hóa K-function computation - **THE BOTTLENECK** của toàn bộ pipeline.

```
K(r) = (1/λ) × (1/N²) × Σ 𝟙(dij < r)
```

**Complexity hiện tại**: O(N²) pairwise distance computation
**Được gọi**: ~1000 lần trong SOP search
**Impact**: Nếu giảm được O(N²), toàn bộ pipeline speedup đáng kể

---

## 2. Input/Output

```
Input:  Events (t, x, y) - N points
Output: K(r) values for r in [r_min, r_max]
```

---

## 3. Pipeline Context

```
┌────────────────────────────────────────┐
│  MODULE 1: Data Pipeline              │
│  (Output: Clean events)               │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  MODULE 2.1: K-function ⭐ HERE       │ ← BOTTLENECK!
│  (Output: K(r) values)                │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  MODULE 2.2: L-function               │
│  (Output: L(r) = sign(K)|K|^(1/3))    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  MODULE 3: Prediction (DỰ ĐOÁN)       │
└────────────────────────────────────────┘
```

---

## 4. Tài liệu cần đọc

### 4.1 Bắt buộc
- [ ] `ARCHITECTURE.md` - Section 2.2 (Feature Extraction Layer)
- [ ] `THEORY.md` - Section 2.1-2.2 (K-function và L-function)
- [ ] `Q_STPP_V16_REPORT.md` - Section 2.1 (Architecture)

### 4.2 Research Papers cần tìm
1. **Quantum Linear Algebra**
   - "Quantum algorithm for linear systems" - Harrow, Hassidim, Lloyd (2009)
   - "Quantum gradient descent" - Recent work

2. **Quantum Distance Calculations**
   - "Quantum distance oracle" papers
   - "Quantum nearest neighbor algorithms"

3. **Classical Accelerations**
   - GPU-accelerated K-function
   - Approximate K-function methods
   - Tree-based methods (Ball Trees, KD-Trees)

---

## 5. Research Questions

### 5.1 Can quantum help K-function?

```
K(r) computation involves:
1. Pairwise distances: O(N²) → can quantum do O(log N)?
2. Counting: O(N²) → can amplitude amplification help?
3. Normalization: O(N) → trivial
```

### 5.2 Specific algorithms to research

| Algorithm | Classical | Quantum | Speedup? |
|-----------|-----------|---------|----------|
| Pairwise distance | O(N²) | ? | ? |
| Counting within r | O(N²) | ? | ? |
| K(r) for all r | O(N² × R) | ? | ? |

### 5.3 Requirements analysis

```
For quantum advantage:
- Number of qubits needed: ?
- Circuit depth: ?
- Error rate requirements: ?
- QRAM needed: Yes/No?
```

---

## 6. Implementation Checklist

### 6.1 Literature Survey (Week 1)
- [ ] Tìm 5-10 papers về quantum K-function/similar
- [ ] Đọc và summarize methods
- [ ] Xác định best candidate approach
- [ ] Write literature report

### 6.2 Classical Baseline (Week 2)
- [ ] Implement current K-function in Python
- [ ] Profile to find exact bottleneck
- [ ] Benchmark for N = 10, 50, 100, 500, 1000
- [ ] Document timing results

### 6.3 Classical Optimizations (Week 2-3)
- [ ] Try NumPy vectorization
- [ ] Try numba JIT compilation
- [ ] Try GPU (CuPy) if available
- [ ] Compare all approaches

### 6.4 Quantum Analysis (Week 3-4)
- [ ] Design quantum circuit for pairwise distance
- [ ] Estimate qubit requirements
- [ ] Compare theoretical vs practical
- [ ] Write recommendation report

---

## 7. Expected Deliverables

### Week 1: Literature Report
```
📄 K-function Quantum Research Report

1. Summary of 5-10 relevant papers
2. Methods that might apply
3. Initial assessment: feasible or not?
4. Recommended approach
```

### Week 2: Classical Baseline + Optimizations
```
📊 K-function Benchmark Results

1. Current implementation profile
2. Bottleneck identification
3. Classical speedups achieved
4. Remaining gap
```

### Week 3-4: Quantum Analysis + Final Report
```
📄 K-function Optimization Final Report

1. Classical optimizations (final)
2. Quantum approach (if feasible)
3. Recommendations for pipeline
4. Implementation plan for v17
```

---

## 8. Benchmark Design

### 8.1 Test Data
```python
# Synthetic Hawkes data
N_values = [10, 20, 50, 100, 200, 500, 1000]
seeds = [42, 123, 456, 789, 1000]
```

### 8.2 Metrics to Track
```python
metrics = {
    'time': 'seconds',
    'memory': 'MB',
    'flops': 'floating point operations',
    'calls': 'number of K-function evaluations'
}
```

### 8.3 Comparison Points
- [ ] NumPy (baseline)
- [ ] NumPy + numba
- [ ] GPU (CuPy)
- [ ] Proposed quantum approach

---

## 9. Red Flags (What to Watch)

⚠️ **QRAM requirement**: If quantum approach needs QRAM, it's not practical now
⚠️ **Qubit count**: If N > 100 requires > 100 qubits, not feasible
⚠️ **Circuit depth**: If depth > 1000, NISQ noise kills advantage
⚠️ **Classical simulation**: If quantum is slower on classical sim, why bother?

---

## 10. Questions for Team Lead

1. What N values are realistic for our dengue data?
2. Is GPU available for testing?
3. Should we prioritize speed or accuracy?
4. What's the timeline for v17?

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Assigned | | |
| Team Lead | | |
