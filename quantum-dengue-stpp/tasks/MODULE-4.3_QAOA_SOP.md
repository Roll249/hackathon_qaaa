# TASK 4.3: QAOA for SOP - MAIN QUANTUM RESEARCH

## Thông tin chung

| Field | Value |
|-------|-------|
| **Task ID** | MODULE-4.3 |
| **Module** | Layer 3: SOP Augmentation |
| **Priority** | P1 - HIGH |
| **Assigned to** | [ASSIGN] |
| **Due Date** | Week 4 |
| **OUTPUT** | Benchmark: QAOA vs Classical |

---

## 1. Mục tiêu

**Research core question**: Can genuine QAOA outperform classical heuristics (MH, Greedy, QAOA-inspired) for SOP permutation search?

```
SOP Problem:
- Input: Events (x, y, t) 
- Goal: Find permutation π that minimizes ||L(π) - L_target||²
- Constraint: Preserve second-order statistics

QAOA Question:
- Can QAOA find better solutions than greedy?
- At what N does quantum advantage emerge?
- What circuit depth (p) is needed?
```

---

## 2. Input/Output

```
Input:  
  - Events: (t, x, y) for N events
  - Target L-function: L_target(r)
  - Budget: Max L-function evaluations
  
Output: 
  - Best permutation found
  - L(r) error
  - Diversity score
  - Comparison with classical methods
```

---

## 3. Pipeline Context

```
┌─────────────────────────────────────────────────────────────┐
│  MODULE 3: Prediction (DỰ ĐOÁN)                         │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 4.3: QAOA SOP ⭐ MAIN RESEARCH                 │
│                                                             │
│  Research Question:                                        │
│  Can genuine QAOA beat classical heuristics?               │
│                                                             │
│  Methods to compare:                                       │
│  1. Classical Greedy                                      │
│  2. Classical MH                                          │
│  3. Classical QAOA-inspired (current)                     │
│  4. GENUINE QAOA (to implement)                         │
│  5. Real QAOA on hardware (if available)                │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 5: Output                                        │
│  (Fair comparison results)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Tài liệu cần đọc

### 4.1 Bắt buộc
- [ ] `THEORY.md` - Section 4 (Optimization Methods)
- [ ] `ARCHITECTURE.md` - Section 2.4 (SOP Augmentation)
- [ ] `run_q_stpp_v15_fair.py` - Lines 156-213 (current SOP implementation)
- [ ] `DEVELOPMENT_HISTORY.md` - Section 3 (What failed)

### 4.2 QAOA Papers
1. **Foundational**
   - Farhi et al. (2014) - "A Quantum Approximate Optimization Algorithm"
   - Zhou et al. (2020) - "QAOA Performance Scaleup"

2. **QAOA for Permutations**
   - "QAOA for scheduling problems"
   - "Permutation optimization with QAOA"
   - "XY-mixer for combinatorial optimization"

3. **Benchmarks**
   - "QAOA vs Classical - when does quantum win?"
   - "Quantum advantage in optimization"
   - Recent NISQ results

---

## 5. QAOA Implementation Plan

### 5.1 Problem Formulation

```python
# SOP as QUBO/MAP
# Decision variables: x_i = position of event i

# Cost function:
# C(π) = ||L(π) - L_target||²

# QAOA formulation:
# H_C = Σ h_i x_i + Σ J_ij x_i x_j
# Need to map L(r) error to QUBO
```

### 5.2 Cost Hamiltonian Design

```python
def cost_hamiltonian(perm, L_target, events):
    """
    H_C = ||L(perm) - L_target||²
    
    How to encode this as Hamiltonian?
    - Permutation constraints
    - L-function approximation
    """
    pass

# Need to discretize L(r) values
# Binary encoding for permutation
```

### 5.3 Mixer Hamiltonian

```python
# XY-Mixer (natural for permutations)
H_M = Σ (X_i X_j + Y_i Y_j)

# This preserves Hamming distance
# Natural for permutation problems
```

### 5.4 Circuit Design

```python
def qaoa_circuit(p, beta, gamma, events):
    """
    QAOA circuit with p layers
    
    |ψ⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^⊗n
    
    Where:
    - U_C(γ) = exp(-iγ H_C)
    - U_B(β) = exp(-iβ H_M)
    """
    pass
```

---

## 6. Benchmark Design

### 6.1 Test Cases

```python
test_cases = [
    # Small N - QAOA should work well
    {'n': 5, 'p': 1, 'shots': 1000},
    {'n': 6, 'p': 1, 'shots': 1000},
    {'n': 7, 'p': 2, 'shots': 1000},
    
    # Medium N - classical might win
    {'n': 10, 'p': 2, 'shots': 1000},
    {'n': 12, 'p': 3, 'shots': 500},
    {'n': 15, 'p': 3, 'shots': 500},
    
    # Large N - classical definitely wins (current)
    {'n': 20, 'p': 4, 'shots': 100},
    {'n': 30, 'p': 5, 'shots': 100},
]
```

### 6.2 Methods to Compare

| Method | Type | Implementation |
|--------|------|----------------|
| Greedy | Classical | From v15 code |
| MH | Classical | From v15 code |
| QAOA-inspired | Classical | From v15 code |
| QAOA (sim) | Quantum sim | To implement |
| QAOA (hardware) | Real QC | IBM/Rigetti |

### 6.3 Metrics

```python
metrics = {
    'best_error': 'Lowest L(r) error found',
    'mean_error': 'Average error',
    'variance': 'Stability',
    'diversity': 'Permutation diversity',
    'time': 'Wall-clock time',
    'circuit_depth': 'QAOA depth p',
    'shots': 'Measurement samples'
}
```

---

## 7. Implementation Checklist

### 7.1 Week 1: Literature + Theory
- [ ] Read QAOA papers (Farhi 2014, Zhou 2020)
- [ ] Study current SOP implementation
- [ ] Design QUBO formulation for SOP
- [ ] Write literature report

### 7.2 Week 2: Classical Baseline
- [ ] Run existing methods (Greedy, MH, QAOA-inspired)
- [ ] Collect benchmark data
- [ ] Establish classical baseline
- [ ] Document results

### 7.3 Week 3: QAOA Implementation
- [ ] Implement QAOA in PennyLane or Qiskit
- [ ] Test on small N (N ≤ 10)
- [ ] Optimize parameters (β, γ via COBYLA)
- [ ] Compare with classical

### 7.4 Week 4: Hardware (if available)
- [ ] Run on IBM Quantum
- [ ] Compare with simulation
- [ ] Analyze noise effects
- [ ] Write final report

---

## 8. Expected Deliverables

### Week 1: Literature Report
```
📄 QAOA for SOP Research

1. QUBO formulation for SOP
2. Mixer Hamiltonian design
3. Initial feasibility assessment
```

### Week 2: Classical Baseline
```
📊 Classical SOP Benchmark

1. Results for Greedy, MH, QAOA-inspired
2. Best method by N
3. Trade-off analysis
```

### Week 3: QAOA Results
```
📊 QAOA Implementation + Results

1. QAOA code
2. Small N results (N ≤ 10)
3. Comparison with classical
4. Optimal p finding
```

### Week 4: Final Report
```
📄 QAOA for SOP - Final Report

1. Complete benchmark
2. Quantum vs Classical analysis
3. When does quantum help?
4. Recommendations for pipeline
```

---

## 9. Red Flags

⚠️ **NISQ limitations**: Current QAOA on NISQ devices has noise
⚠️ **Barren plateaus**: Gradient descent may fail for large p
⚠️ **Classical simulation**: O(2^n) - can't do large N
⚠️ **Parameter optimization**: COBYLA may not find optimal β, γ
⚠️ **Fair comparison**: Same budget as classical required

---

## 10. Questions for Team Lead

1. Is there access to real quantum hardware? (IBM/Rigetti)
2. What PennyLane/Qiskit version to use?
3. Should we prioritize N=5-10 or N=15-20?
4. What's the success criteria for "quantum wins"?

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Assigned | | |
| Team Lead | | |
