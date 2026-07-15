# Q-STPP v12 ROI Analysis — Real Advantage or Synthetic Illusion?

**Date**: 2026-07-16
**Status**: Statistical and methodological verification of quantum advantage claims
**Depends on**: results from v12_proper_kernel.py and v12_significance.py

---

## 1. ROI Question

User asked: "sự cải tiến này nó sẽ có ROI thế nào so với cổ điển, ta đang cố gắng giải quyết vấn đề để cho nó có lợi thế thật sự nhé"

Translation: "What's the ROI of this improvement over classical? We're trying to solve the problem so it has REAL advantage."

This is the *right* question — the most important one for QC4SG. Synthetic data is easy to game. Real-world cost-benefit is what matters.

---

## 2. ROI Definitions

### 2.1 Public Health Value (Positive ROI)

Catching an additional dengue case earlier provides:
- **$100-1,000 per case** in avoided healthcare costs
- **2-5 cases prevented** per caught index case (R₀ ≈ 2-3 for dengue)
- **Avoided outbreak**: $1M+ per regional outbreak (CDC estimates)

### 2.2 Quantum Cost
- **Simulator cost**: ~free (CPU runtime)
- **IBM Quantum**: $0.10-1.00 per query (varies by qubit count)
- **Braket (Rigetti/IonQ)**: $0.30-3.00 per query

### 2.3 Net ROI Formula

```
ROI = (accuracy_gain × N_cases × value_per_caught_case) - (N_queries × cost_per_query)
```

Plugging in N_cases = 10,000, value = $100, hybrid = 0.88, classical = 0.69:

```
ROI = (0.88 - 0.69) × 10000 × $100 - 10000 × $0.50
    = 0.19 × 10000 × $100 - $5000
    = $190,000 - $5,000
    = $185,000 NET POSITIVE ROI per 10,000 cases
```

**But this assumes accuracy_gain = +0.19 holds on real data.**

---

## 3. ROI Verification — Three Critical Questions

### Q1: Is the +0.19 advantage real, or a lucky seed?
- **Test**: 10 random seeds + paired t-test
- **Result**: Pending v12_significance.py
- **If NOT significant**: claim is statistical noise → ROI becomes 0

### Q2: Is Hilbert projection the right quantum kernel?
- **Current**: Hilbert projection achieves only 0.33-0.55
- **Test**: IQP-style / data re-uploading / amplitude encoding
- **Result**: Pending v12_proper_kernel.py
- **If still ≤0.55**: quantum kernel alone has no advantage — hybrid only wins because classical is weak

### Q3: Does synthetic STPP match real dengue?
- **Current**: 3 clean process types (Poisson, LGCP, Cluster)
- **Real**: Many overlapping patterns, missing data, measurement noise
- **Risk**: Real-data accuracy will be lower than synthetic
- **Test required**: Real dengue admin-1 data (TYCHO dataset, not yet integrated)

---

## 4. ROI Scenarios

### Scenario A: Best Case (+0.19 reproducible, proper quantum kernel ≥0.71)
- **Status**: ROI = $185,000 per 10,000 cases
- **Pitch**: "Quantum-classical hybrid reduces dengue missed cases by 61%"
- **Likelihood**: TO BE VERIFIED

### Scenario B: Realistic Case (+0.05 to +0.10 on real data, hybrid still > classical)
- **ROI**: $25,000-$95,000 per 10,000 cases
- **Status**: Still positive but moderate
- **Likely outcome** if v9 advantage is real

### Scenario C: Conservative Case (advantage disappears on real data)
- **ROI**: $0 to negative
- **Status**: Need backup approach (e.g., quantum Monte Carlo for K-function estimation is well-known advantage)

---

## 5. What's Actually Needed for Real ROI

### 5.1 Statistical Significance (v12_significance.py output)
- 10 random seeds
- Paired t-test
- p < 0.05 required to claim "advantage"

### 5.2 Proper Quantum Kernel (v12_proper_kernel.py output)
- IQP / data re-uploading feature maps
- Should achieve ≥0.71 alone at N=150+
- If not: honest acknowledgment in report

### 5.3 Real Data Validation (Future)
- TYCHO dengue dataset
- Cross-country validation (Vietnam vs Thailand vs Cambodia)
- Climate covariate integration

---

## 6. Honest Conclusion (To Be Updated)

> The v9 hybrid pipeline claims +0.19 quantum advantage at N=150. This advantage was demonstrated on **synthetic STPP data** with a **lucky single seed** using **a Hilbert projection that is not a standard quantum kernel**.
>
> For REAL ROI, three things must be shown:
> 1. The advantage is statistically significant (10 seeds, paired t-test)
> 2. A proper quantum kernel can match or exceed classical alone
> 3. The advantage generalizes to real dengue data
>
> Without these, the +0.19 number is interesting but NOT a decision-ready result.

---

## 7. References (Pending Verification)

This section will be filled with v12_significance and v12_proper_kernel results when available.