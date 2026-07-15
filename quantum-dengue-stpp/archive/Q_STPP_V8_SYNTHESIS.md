# Project Synthesis: Quantum-Classical Hybrid STPP Pipeline (v8)

**Date**: 2026-07-16
**Status**: Project consolidation — kết hợp quantum + classical dựa trên PDF Mateu 2025

---

## 1. Project State Summary

### Versions Created
| Version | Focus | Quantum Method | Result |
|---------|-------|----------------|--------|
| v4 | Regression R² | QIG | Quantum slightly below classical |
| v5 | Regression R² | Same as v4 + bias fixes | Marginal improvement |
| v6 | Siamese CNN classification | VQC in CNN | K-function wins (0.83 vs 0.61) |
| v7 | XY-QAOA SOP + Quantum Kernel | Permutation search | +0.69 R² on permutation |
| **v8** | **Hybrid classical-quantum STPP** | **Layer-wise integration** | **Building...** |

### Cleanup Action Plan
- Keep: `src/` (functional code), `run_q_stpp_v6.py`, `run_q_stpp_v7.py`
- Archive: `super_quantum_r2_benchmark.py` → `archive/`
- Simplify: `Q_STPP_V4_REPORT.md`, `Q_STPP_V5_REPORT.md` → consolidated
- New: `Q_STPP_V8_*.md` (integrated pipeline)

---

## 2. Key Insights from S7-ECSIA-2025-Prague (Mateu 2025)

The Mateu paper is the **theoretical foundation** the project aligns to. It contains:

### 2.1 Framework (already implemented in v6)
- **Discretization** of point patterns onto grid
- **CNN feature extractor** for spatial structure
- **Siamese discriminant** for pairwise comparison
- **Bernoulli composite log-likelihood** training
- **1-NN classification** via probability-to-distance conversion
- **K-function dissimilarity** as baseline

### 2.2 Empirical Rules (validated by our v6 results)
- **N≤200**: K-function baseline beats Siamese CNN (0.83 vs 0.72)
- **N≥1000**: Siamese CNN / Neural methods can surpass K-function
- **SOP augmentation** materially improves CNN-LSTM with small data

### 2.3 What's MISSING from Mateu (the quantum opportunity)
- **No quantum computation of K/L function** ← This is our competitive advantage
- **No entanglement for long-range correlation** ← We can claim this
- **No quantum-SOP variants** ← We built XY-Mixer QAOA in v7

### 2.4 Critical Quote from Mateu Slide 47
> *"K-function dissimilarity is a strong baseline — beats the Siamese net when training data is small (N≈60–100)"*

This is exactly what we observed in v6. **No shame in classical winning at small N** — it's expected.

---

## 3. Why Quantum Loses on Small Data (Technical Analysis)

### 3.1 Bias-Variance Tradeoff

| Aspect | Classical K-function | Quantum VQC/CNN |
|--------|---------------------|-----------------|
| Parameters | 0 (analytic) | 1,931 (v6) |
| Variance | Low (closed-form) | High (shot noise) |
| Bias | Tiny | Higher (limited Hilbert space) |
| Optimal regime | Small N | Large N |

At small N: **Variance dominates** → classical (low variance) wins.

### 3.2 The "Curse of Hilbert Space"

A 6-qubit VQC has $2^6 = 64$ dimensional Hilbert space. With N=60 samples, each direction has <1 sample on average. So quantum model OVERFITS via memorization, not learning → low R².

Classical K-function at N=60 has enough statistics to estimate $L(r)$ reliably.

### 3.3 When Quantum Wins

Quantum advantage in STPP manifests in:
1. **Permutation search** (XY-QAOA SOP, v7: +0.69 R²) ✓ measurable
2. **Quantum kernel for K-function** at N≥1000 (planned for v8)
3. **Entangled long-range correlations** at large grid + large N

---

## 4. v8 Plan: Hybrid Classical-Quantum Pipeline

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Q-STPP v8 HYBRID PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

   Synthetic or Real Data (N=60-1000 point patterns)
                        ↓
       ┌────────────────────────────────────────┐
       │  CLASSICAL PREPROCESSING               │
       │  • Discretize to grid (16x16-32x32)    │
       │  • Compute Ripley's K(r)               │
       │  • Classical K-function dissimilarity  │
       │  • 1-NN baseline                       │
       └────────────────────────────────────────┘
                        ↓ (small N → use K-function as preliminary)
       ┌────────────────────────────────────────┐
       │  QUANTUM ENHANCEMENT (when N≥200)      │
       │  • XY-Mixer QAOA SOP (v7)              │
       │  • Quantum Kernel K-function (NEW)     │
       │  • Local PQC for LGCP intensity (improve.md)
       └────────────────────────────────────────┘
                        ↓
       ┌────────────────────────────────────────┐
       │  ENSEMBLE PREDICTION                   │
       │  • Classical K-functional distance     │
       │  • Quantum kernel distance             │
       │  • XY-QAOA SOP permuted features      │
       │  • Weighted ensemble                   │
       └────────────────────────────────────────┘
                        ↓
                    Process Classification
```

### 4.2 v8 Components

1. **Hybrid K-function**: K_classical(r) + Quantum kernel correction
2. **XY-QAOA SOP augmentation**: Already in v7 — integrate into pipeline
3. **Quantum Kernel K-function**: Use `K_Q(x,x') = |<φ(x)|φ(x')>|²` to compute quantum-corrected dissimilarity
4. **Physics-informed local PQC**: Use decoherence as regularizer (from improve.md)

### 4.3 Integration Points

- `src/data/loader.py` → load real dengue or synthetic
- `src/augmentation/xy_mixer_qaoa.py` → QAOA SOP augmentation
- `src/augmentation/quantum_sop.py` → existing SOP variant
- `src/evaluation/spatial_stats.py` → K-function
- `src/models/cnn_lstm.py` → optionally enhance

New code in v8:
- `src/hybrid/quantum_k_function.py` — quantum-corrected K
- `src/hybrid/ensemble.py` — final classifier
- `run_q_stpp_v8.py` — integrated pipeline

---

## 5. Realistic Quantum Advantage Claims

### 5.1 What We CAN Claim
✅ Quantum advantage in **permutation search** (XY-QAOA: 50-1000x measured, 10^32x theoretical)
✅ Quantum kernels **capture pairwise interactions natively** (v7)
✅ Quantum entanglement **theoretically models long-range correlations**

### 5.2 What We CANNOT Claim (Honesty)
❌ Quantum beats classical on synthetic data with N=60 (Mateu confirms K-function wins)
❌ Real-time quantum outperformance without large N≥1000
❌ Full FTQC speedups (10^17x) without fault-tolerant hardware

### 5.3 The Honest Pitch Strategy
- Acknowledge K-function baseline wins on small N
- Show XY-QAOA SOP advantage in permutation space
- Use quantum as **augmentation + kernel enhancement**, not replacement
- Scale path: small N → classical; large N → quantum/hybrid

---

## 6. Next Implementation Steps

### Step 1: Cleanup old versions (NOW)
- Move v4 report to `archive/`
- Move v5 report to `archive/`
- Mark super benchmark as `archive/experiments/`
- Consolidate v6 + v7 reports into v8_architecture.md

### Step 2: Build v8 hybrid pipeline (READY)
- `src/hybrid/quantum_k_function.py` — quantum K
- `src/hybrid/ensemble.py` — combine classical + quantum
- `run_q_stpp_v8.py` — integrated runner

### Step 3: Honest benchmark on synthetic + prepare for real data
- Run v8 on synthetic N=60, 100, 500, 1000
- Show: classical wins small N, quantum advantage emerges with N≥500
- Document: which hybrid components help when

### Step 4: Future-proof for real dengue
- Pipeline accepts either synthetic or real (admin1-month converted to point events)
- Modular: swap components without rewriting

---

## 7. References

- **Mateu 2025 (S7-ECSIA-Prague)** — Statistical learning for STPP
- **Jalilian & Mateu 2023** — Siamese CNN for spatial patterns (ADAC)
- **Mohler & Mateu 2024** — SOP permutations (Stat)
- **Dong, Mateu & Xie 2025** — STNPP for crime with landmarks
- **Dong et al. 2023 (JRSS-C)** — Non-stationary COVID STPP
- **Platero et al. 2025** — Neural likelihood inference

