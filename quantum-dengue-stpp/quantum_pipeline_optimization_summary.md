# Quantum Pipeline Optimization — Executive Summary

**Project:** Quantum-dengue-stpp v17  
**Date:** July 22, 2026  
**Version:** 1.0

---

## Current State Assessment

### Phase 1 Benchmark Results

| Component | Result | Assessment |
|-----------|--------|------------|
| **QAOA-XY** | Accuracy: 88.9% = classical (Δ = 0.000) | No advantage |
| **QAOA wall-clock** | 1.09s vs 0.08s classical | 10× slower |
| **Grover amplification** | 228× query speedup theoretically | 15× slower wall-clock |
| **Quantum kernel** | Comparable accuracy | 100-1000× slower |
| **Set diversity** | QAOA lower than classical (0.89 vs 0.95) | XY mixer concentrates |

### Root Causes Identified

1. **Candidates too good** — 60-80 classical swap iterations pre-optimize to near-optimal (L-error ~10⁻⁴). No quantum advantage niche.
2. **Problem too small** — M=10, k=4 is trivial for greedy; QUBO landscape has no local minima for QAOA to exploit.
3. **Simulator overhead** — O(2^M) statevector cost negates all theoretical quantum speedup.
4. **Data encoding mismatch** — 8-dim L(r) features mapped to 8 qubits via RY; Hilbert space underutilized.
5. **Hybrid boundary wrong** — Quantum does the easy part (subset selection from near-optimal candidates).

---

## Research Coverage

| Research Area | Status | Key Papers |
|--------------|--------|------------|
| QAOA optimization | ✅ Covered | Farhi 2014, Wang 2020, Zhou 2020, Egger 2021 |
| Quantum-classical hybrid | ✅ Covered | Peruzzo 2014, McClean 2016 |
| Data encoding strategies | ✅ Covered | Havlíček 2019, Schuld 2019 |
| Quantum kernel methods | ✅ Covered | Rebentrost 2014, Bishwas 2018 |
| VQA and barren plateaus | ✅ Covered | McClean 2018, Cerezo 2021 |
| Quantum walks | ✅ Covered | Childs 2004 |
| Amplitude amplification | ✅ Covered | Brassard 2002 |
| NISQ-era considerations | ✅ Covered | Preskill 2018, Kandala 2017 |

**Total References:** 18 papers (15 academic + 3 domain)

---

## 5 Optimization Strategies Identified

### Priority 1: Warm-Start QAOA 🔴

| Aspect | Detail |
|---------|--------|
| **What** | Initialize QAOA (γ, β) from classical greedy solution |
| **Reference** | Egger et al. (2021) "Warm-starting quantum optimization" |
| **Expected gain** | 50% time reduction (1.09s → 0.55s), better solutions |
| **Effort** | SMALL (2 days implementation) |
| **Risk** | LOW |
| **Validation** | Compare iterations-to-convergence vs random init |

### Priority 2: Trainable Quantum Kernels 🔴

| Aspect | Detail |
|---------|--------|
| **What** | End-to-end QNG optimization of feature map parameters |
| **Reference** | Liu 2020, hubless 2021 |
| **Expected gain** | +0-5% accuracy on hard instances |
| **Effort** | MEDIUM (1 week) |
| **Risk** | LOW |
| **Validation** | Compare trainable vs static kernels on SOP data |

### Priority 3: Dataset Scaling to Hard Instances 🔴

| Aspect | Detail |
|---------|--------|
| **What** | Benchmark at N=50, N=100 to find crossover point |
| **Reference** | Zhou 2020 (QAOA needs large, hard instances) |
| **Expected gain** | Find where quantum starts outperforming classical |
| **Effort** | SMALL (1 week profiling) |
| **Risk** | LOW |
| **Validation** | Accuracy vs N curve; identify quantum advantage regime |

### Priority 4: Amplitude Encoding for L(r) 🟡

| Aspect | Detail |
|---------|--------|
| **What** | Encode 8-dim L(r) via amplitude encoding (log₂(8) = 3 qubits) |
| **Reference** | Schuld & Killoran 2019 |
| **Expected gain** | Better Hilbert space utilization |
| **Effort** | MEDIUM (2 weeks) |
| **Risk** | HIGH (requires QRAM for exponential advantage) |
| **Validation** | Compare kernel matrices: amplitude vs angle encoding |

### Priority 5: Hybrid Boundary Refactoring 🟡

| Aspect | Detail |
|---------|--------|
| **What** | Move quantum to spatial search (O(√M) vs O(M)) |
| **Reference** | Childs & Goldstone 2004 |
| **Expected gain** | Quadratic speedup on spatial sub-problem |
| **Effort** | LARGE (4 weeks) |
| **Risk** | MEDIUM |
| **Validation** | Compare quantum walk vs classical nearest-neighbor |

---

## Recommended Roadmap

### Phase 1: Quick Wins (Weeks 1-2)
- [ ] Implement warm-start QAOA initialization
- [ ] Skip Grover for M > 12 (eliminate 15× slow-down)
- [ ] Benchmark warm-start vs random on existing configs

### Phase 2: Medium-Term (Weeks 3-6)
- [ ] Integrate trainable quantum kernels
- [ ] Sweep QUBO hyperparameters (α, β, λ)
- [ ] Run benchmarks at N=50, N=100

### Phase 3: Long-Term (Weeks 7-14)
- [ ] Design quantum spatial search for geographic hotspots
- [ ] Implement amplitude encoding
- [ ] Prepare for hardware deployment

---

## Honest Assessment

### Can Quantum Beat Classical in THIS Pipeline?

**Short answer: NOT YET, but POSSIBLE with changes.**

| Condition | Classical Wins | Quantum Could Win |
|-----------|---------------|-------------------|
| N ≤ 30, M ≤ 15 | ✅ Always | ❌ No |
| N ≥ 50, hard instances | ❌ Maybe not | ✅ Maybe |
| Spatial search (M regions) | O(M) | O(√M) |
| Quantum kernel (high-dim) | O(N²·d) | O(N²·2^n) on hardware |
| Trainable kernels | Limited expressibility | Better with QNG |

**Honest conclusion:** For the current pipeline with N ≤ 30, M = 10, quantum provides no advantage over classical. The 10× wall-clock slowdown and identical accuracy mean quantum is a net negative. Changes needed:

1. Scale to harder instances (N ≥ 50)
2. Warm-start QAOA for 2× time improvement
3. Move quantum to the right sub-problem (spatial search)

---

## Deliverables

| File | Description | Location |
|------|-------------|----------|
| Full Research Report | 5000+ words, 18 references, detailed strategies | `quantum_pipeline_optimization_research.md` |
| Executive Summary | 1-page overview, prioritized strategies | `quantum_pipeline_optimization_summary.md` |

---

*Generated by Deep Research Agent — July 22, 2026*
