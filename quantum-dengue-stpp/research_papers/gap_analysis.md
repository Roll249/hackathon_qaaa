# Gap Analysis: QAAA Dengue STPP vs Existing Literature

**Generated:** July 22, 2026  
**Purpose:** Identify gaps between QAAA proposal and existing quantum ML literature

---

## Executive Summary

The QAAA architecture proposes combining 8+ quantum computing techniques for spatio-temporal dengue prediction. This analysis identifies **12 critical gaps** between the proposal and the existing literature, categorized by severity and feasibility.

---

## Gap Categories

### 🔴 Critical Gaps (Must Address)

#### Gap 1: No Empirical Validation on Epidemiological Data

**Description:** All referenced quantum ML techniques (QWGAN, SuDaI, QCNN, QLSTM) have been validated on:
- Synthetic datasets (MNIST, Gaussian)
- Network security data
- Weather/precipitation data
- Simple time-series (NARMA, damped oscillators)

**Missing:** Validation on dengue case data, disease outbreak sequences, epidemiological spatiotemporal patterns

**Impact:** High - Cannot assess real-world effectiveness

**Recommendation:** Run quantum pipeline on real dengue data from Vietnam Dengue Watch or similar sources

---

#### Gap 2: Scale Mismatch - Limited Qubits vs Complex Task

**Description:** QAAA proposes 8-10 qubits for spatio-temporal forecasting

**Evidence from Literature:**
- Wang et al. (2025): "Limited qubits + deep encoding → predictive performance degenerates to near random guessing"
- QWGAN tested on 3-8 qubits maximum
- QLSTM tested on toy problems (NARMA-5)

**Impact:** Critical - May fundamentally limit quantum advantage

**Recommendation:** 
- Use shallow circuits (p ≤ 3)
- Consider hybrid quantum-classical approaches
- Do not claim advantage for high-dimensional forecasting with limited qubits

---

#### Gap 3: Overclaimed "Quantum" Techniques

**Description:** QAAA uses "quantum" modifier on techniques that are purely classical:

| QAAA Claim | Reality | Gap |
|-----------|---------|-----|
| "Quantum Fractional Hawkes" | No quantum implementation exists | Critical |
| "Quantum-informed priors" | Classical Fractional Hawkes is classical | Critical |
| "Quantum STPP model" | No quantum Hawkes paper found | Critical |

**Impact:** High - Misleading claims damage credibility

**Recommendation:** Use "Classical Fractional Hawkes + Quantum Components" or remove "quantum" modifier

---

#### Gap 4: Hardware Requirements Mismatch

**Description:** QAAA proposes MP-QLSTM requiring distributed quantum computing

**Literature (Chen et al. 2025):**
- Requires M QPUs for M sub-VQCs
- Each VQC needs q qubits
- Total: M × q qubits across distributed system

**Reality:** No access to multiple QPUs; distributed quantum computing is emerging technology

**Impact:** Critical - Proposal not implementable on current hardware

**Recommendation:** List as "Future Work" or remove from current architecture

---

### 🟡 Moderate Gaps (Should Address)

#### Gap 5: SuDaI Novelty Overstated

**Description:** QAAA presents SuDaI as a novel contribution

**Reality:** SuDaI is essentially a variant of data re-uploading:
- Both: Repeated data encoding with trainable parameters
- SuDaI: Specifically for time series with alternating input/variational layers
- Data Re-Uploading: General framework

**Overlap:** 80% similar to data re-uploading (Pérez-Salinas 2020)

**Impact:** Medium - Overstates contribution novelty

**Recommendation:** Frame as "SuDaI (data re-uploading variant for time series)" not as separate contribution

---

#### Gap 6: "30.5% Loss Reduction" Claim Unverified

**Description:** QAAA claims "QCNN achieves 30.5% loss reduction"

**Search Results:** No matching paper found in:
- QCNN papers (Cong, Choi, Lukin)
- Quanvolutional papers (Henderson et al.)
- Environmental/emissions literature

**Possible Sources:** 
- Classical deep learning literature misattributed
- Unpublished internal results
- Different benchmark setup

**Impact:** Medium - Potentially misleading claim

**Recommendation:** Remove claim or provide citation

---

#### Gap 7: QWGAN Mode Collapse Claim Unsubstantiated

**Description:** QAAA claims "QWGAN-GP prevents mode collapse"

**Evidence:**
- Classical WGAN-GP: Proven mode collapse mitigation (Gulrajani et al. 2017)
- Quantum WGAN: No explicit mode collapse mitigation demonstrated
- Chakrabarti et al. (2019): Does not claim mode collapse prevention

**Impact:** Medium - Overclaims quantum advantage

**Recommendation:** Use "Theoretically may prevent mode collapse" not "prevents mode collapse"

---

#### Gap 8: No Comparison with Classical Baselines

**Description:** QAAA does not provide systematic comparison with classical state-of-the-art

**Classical Methods for STPP:**
- Classical Hawkes processes
- LSTM/GRU for time-series
- Transformer models
- XGBoost/LightGBM for tabular

**Missing:** 
- Direct comparison of quantum vs classical on same data
- Ablation studies
- Runtime comparison on equal hardware

**Impact:** Medium - Cannot assess quantum advantage

**Recommendation:** Add systematic classical baselines (LSTM, Hawkes, Transformer)

---

### 🟢 Minor Gaps (Nice to Have)

#### Gap 9: No Discussion of Barren Plateaus

**Description:** Deep quantum circuits can suffer from barren plateaus (vanishing gradients)

**Evidence:**
- McClean et al. (2018): Expressivity → barren plateaus
- Ragone et al. (2024): Lie algebraic theory of barren plateaus

**Missing:** How QAAA architecture mitigates barren plateaus

**Impact:** Low - For shallow circuits (p ≤ 3), may not be critical

---

#### Gap 10: No Error Mitigation Discussion

**Description:** QAAA does not address quantum error mitigation

**Techniques Available:**
- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Dynamic decoupling

**Missing:** How NISQ noise affects results

**Impact:** Low for simulations; High for real hardware

---

#### Gap 11: Parameter Efficiency Claims Need Clarification

**Description:** "80 params = 99% accuracy" claim

**Source:** Hammami et al. (2025) - Network anomaly detection

**Unclear:**
- Is this for the same task?
- Same data distribution?
- Same evaluation metric?

**Impact:** Low - May not transfer to dengue

---

#### Gap 12: No Uncertainty Quantification

**Description:** QAAA does not discuss prediction intervals or confidence

**Importance for Epidemiology:**
- Outbreak predictions need uncertainty bounds
- Classical approaches provide this (e.g., Bayesian LSTM)

**Missing:** How quantum approach handles uncertainty

**Impact:** Medium for practical deployment

---

## Summary Table

| Gap | Severity | Feasibility | Priority |
|-----|----------|-------------|----------|
| No epidemiological validation | 🔴 Critical | Medium | 1 |
| Scale mismatch | 🔴 Critical | Low | 2 |
| Overclaimed "quantum" | 🔴 Critical | High | 3 |
| Hardware requirements | 🔴 Critical | Low | 4 |
| SuDaI novelty overstated | 🟡 Moderate | High | 5 |
| "30.5%" claim unverified | 🟡 Moderate | High | 6 |
| Mode collapse claim | 🟡 Moderate | High | 7 |
| No classical baselines | 🟡 Moderate | High | 8 |
| No barren plateaus discussion | 🟢 Minor | Medium | 9 |
| No error mitigation | 🟢 Minor | Medium | 10 |
| Parameter efficiency claims | 🟢 Minor | Medium | 11 |
| No uncertainty quantification | 🟢 Minor | Medium | 12 |

---

## Recommended Actions

### Immediate (Before Next Submission)

1. **Remove** "Quantum Fractional Hawkes" claim
2. **Remove or cite** "30.5% loss reduction" claim
3. **Relabel** MP-QLSTM as "Future Work"
4. **Add caveat** to Sublinear n-Toffoli (theoretical only)
5. **Revise** QWGAN claim to "theoretically may prevent mode collapse"

### Short-term (Next Iteration)

1. Add systematic classical baselines (LSTM, Hawkes)
2. Run quantum pipeline on real dengue data
3. Compare with classical Hawkes process
4. Add shallow circuit analysis (p ≤ 3)
5. Discuss barren plateaus mitigation

### Long-term (Future Research)

1. Implement on real quantum hardware
2. Validate SuDaI on epidemiological data
3. Test QLSTM on outbreak sequences
4. Develop quantum error mitigation pipeline

---

## What QAAA Gets Right

Despite the gaps, QAAA makes several valid contributions:

1. ✅ **Genuine QAOA-XY implementation** - Well-documented, benchmarked
2. ✅ **Honest assessment in v17 report** - Acknowledges no wall-clock advantage
3. ✅ **Correct framing** of XY mixer as structural advantage
4. ✅ **Proper use** of parameter-shift rule and QNG
5. ✅ **Fractional Hawkes** as classical component - Valid choice

---

## Conclusion

The QAAA architecture has **4 critical gaps** that must be addressed before claiming quantum advantage for dengue STPP:

1. No empirical validation on epidemiological data
2. Scale mismatch (limited qubits vs complex task)
3. Overclaimed "quantum" techniques
4. Hardware requirements not met

The most credible path forward is to:
1. Focus on proven components (XY QAOA, QNG)
2. Frame speculative components as "future work"
3. Add classical baselines for honest comparison
4. Validate on real dengue data
