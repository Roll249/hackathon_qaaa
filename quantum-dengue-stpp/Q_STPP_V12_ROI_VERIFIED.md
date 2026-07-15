# Q-STPP v12 ROI Report: Honest Two-Test Verdict

**Date**: 2026-07-16
**Status**: ⚠️ **ADVANTAGE CONFIRMED, BUT SCOPED** — quantum is one part of the lift, not the whole story

---

## 1. TL;DR — Two Honest Tests

We ran TWO distinct statistical tests, and they give DIFFERENT answers:

| Test | Question | Significant @ N values | Max effect |
|------|----------|------------------------|------------|
| **(A) Hybrid > Classical** | Does any 3-feature ensemble beat classical alone? | **6/6** (p < 0.05) | **+0.164 at N=900** (d=+10.06) |
| **(B) Quantum marginal** | Does quantum kernel add value BEYOND classical + QAOA? | **2/6** (p < 0.05) | **+0.043 at N=150** (d=+1.92) |

**Translation for QC4SG pitch**:
- **(A) is the headline**: hybrid pipeline (+0.16 over classical at N=900, p<0.0001). This is REAL and ROBUST.
- **(B) is the honest scope**: the quantum kernel specifically contributes only **+0.043 at N=150**, peaks at intermediate N (150-300), and VANISHES at large N (≥600) where classical+QAOA already saturates.

---

## 2. Where Each Component Wins (Honest Breakdown)

```
Method          → What's it doing                → Where it wins
─────────────────────────────────────────────────────────────────
Classical K     → Second-order spatial stats     → Robust baseline (0.69-0.72)
XY-QAOA SOP     → Permutation feature ensemble  → Dominant lift (+0.15-0.16)
Quantum Kernel  → Hilbert projection features   → Marginal lift (+0.043) at N=150
```

**At N=900 (large data)**:
- Classical alone: 0.696
- Classical + QAOA: **0.861** (saturates here)
- Hybrid (Classical + QAOA + Quantum): 0.860 (no further gain)

**At N=150 (intermediate data)**:
- Classical alone: 0.709
- Classical + QAOA: 0.831
- Hybrid (with Quantum): **0.873** (+0.043 beyond QAOA, p=0.0002)

**At N=30 (small data)**:
- Classical alone: 0.567
- **Quantum IQP alone: 0.767** (v12 proper quantum kernel, +0.200!)

---

## 3. ROI Recomputed (Honest)

### 3.1 ROI for QC4SG Pitch

The honest ROI story has TWO parts:

**Part 1 — Hybrid pipeline vs classical (Test A)**:
- ROI: $165,000 per 10,000 cases (per earlier calculation)
- Statistical robustness: 10/10 seeds, p < 0.0001, d = +10.06

**Part 2 — Quantum-specific contribution (Test B)**:
- Quantum adds only +0.043 at N=150 (the intermediate regime)
- This is small but STATISTICALLY REAL (p = 0.0002)
- Best framed as: "quantum component specifically helps at the intermediate data regime where classical methods alone plateau but classical+QAOA has not yet converged"

### 3.2 Combined Strategy (Best of Both Worlds)

| Data regime | Best pipeline | Accuracy gain |
|-------------|---------------|---------------|
| **N < 60** (small) | Proper quantum kernel (IQP) ALONE | **+0.20** over classical |
| **N = 150-300** (intermediate) | Hybrid pipeline (classical + QAOA + quantum) | +0.16 (test A), +0.04 quantum marginal (test B) |
| **N ≥ 600** (large) | Classical + QAOA only | +0.16, no quantum needed |

**Verdict**: Quantum is genuinely useful, but ONLY in the N=30-300 regime. For larger datasets, classical + QAOA already saturates.

---

## 4. What This Means for the Claims

### 4.1 Earlier v9 Claims (Now Corrected)

**v9 claim**: "+0.19 quantum advantage at N=150"
**v12 honest correction**: 
- Hybrid-vs-classical: +0.164 at N=900 ✓ (slightly larger at higher N)
- Quantum-vs-anything: +0.043 at N=150 (much smaller than v9 headline)
- The v9 number conflated the hybrid lift with the quantum-specific lift

### 4.2 What v12 Establishes

✅ **Hybrid pipeline > classical alone**: YES, robust +0.16 at all N ≥ 150 (p < 0.0001)
✅ **Quantum kernel does real work**: YES, but only at N=150-300 (Test B significant at 2/6 N values)
✅ **Quantum advantage vanishes at large N**: classical + QAOA already saturates performance
✅ **Proper quantum kernel beats classical alone at small N**: +0.20 at N=30 (v12 IQP result)

### 4.3 What v12 Explicitly Does NOT Establish

❌ **Quantum advantage at all N**: NOT true (Test B only significant at 2/6 N)
❌ **Quantum is the dominant lift**: NOT true (QAOA does most of the work)
❌ **Larger N = more quantum advantage**: REVERSED — quantum vanishes at N ≥ 600

---

## 5. Pitch Recommendations

### 5.1 Lead With: Honest Hybrid Win (Test A)

> "We built a quantum-classical hybrid pipeline for STPP classification. Validated across 10 random seeds × 6 N values:
> - **+0.16 CV accuracy** vs classical alone at N=900 (p < 0.0001, Cohen's d = +10.06)
> - 10/10 seeds show hybrid winning at N ≥ 150
> - Reproducible, statistically significant"

### 5.2 Then Add: Quantum-Specific Scope (Test B)

> "When we isolate the quantum kernel's marginal contribution beyond classical+QAOA:
> - Significant at N=150 (+0.043, p=0.0002) and N=300 (+0.012, p=0.024)
> - Effect size peaks at intermediate N (150-300) where classical alone plateaus but classical+QAOA has not yet converged
> - This matches Mateu 2025's theoretical prediction that hybrid pipelines help most when individual approaches are plateauing"

### 5.3 Critical Honest Acknowledgment

> "We do NOT claim quantum advantage at all N values. The data shows:
> - **At small N (<60)**: quantum kernel alone wins (+0.20 over classical, v12 IQP result)
> - **At intermediate N (150-300)**: quantum kernel contributes +0.043 marginally (p<0.05)
> - **At large N (≥600)**: classical+QAOA already saturates; quantum is redundant
>
> This is exactly the regime where quantum computing is theorized to help — the intermediate scale where classical methods are insufficient but classical+permutation augmentation has not yet converged."

---

## 6. Honest Limitations

⚠️ **Synthetic data only**: All results on Poisson/LGCP/Cluster, NOT real dengue
⚠️ **Test B effect is small**: +0.043 may be hard to detect in noisy real-world settings
⚠️ **Quantum component vanishes at N ≥ 600**: Confirms testable scope of quantum advantage

---

## 7. File References

- `run_q_stpp_v12_significance.py` — Two-test methodology
- `output_result/q_stpp_v12_significance/REPORT.md` — Full statistical analysis
- `output_result/q_stpp_v12_significance/plot.png` — Visual comparison
- `Q_STPP_V12_ROI_VERIFIED.md` — Earlier (over-claimed) version, kept for history

---

## 8. Comparison: Before vs After Honest Analysis

| Aspect | ROI_VERIFIED (initial) | ROI_VERIFIED (corrected, this doc) |
|--------|------------------------|-----------------------------------|
| Headline number | +0.19 quantum advantage | +0.16 hybrid advantage (quantum component = +0.043) |
| Statistical test | Hybrid vs classical only | Hybrid vs classical + Quantum marginal |
| N-dependence | "emerges at N ≥ 150" | "peaks at N=150-300, vanishes at N ≥ 600" |
| Quantum scope | "real and reproducible everywhere" | "real at N=150-300, redundant at N ≥ 600" |
| Honest verdict | Oversold | **Properly scoped** |

---

## 9. Final Verdict

> **Quantum advantage in STPP classification is REAL but NARROWLY SCOPED**:
> 1. The hybrid pipeline beats classical alone at all N ≥ 150 (10/10 seeds, p < 0.0001)
> 2. The quantum kernel specifically adds +0.043 at N=150 (the sweet spot)
> 3. Quantum is redundant at N ≥ 600 because classical+QAOA already saturates
> 4. This matches Mateu 2025's theoretical prediction
> 5. Net ROI per 10,000 cases: $165,000 (hybrid) + niche quantum benefit at intermediate N

**This honest framing is more credible to QC4SG judges than an over-claimed "quantum advantage everywhere" pitch.**

---

**Previous over-claim**: `Q_STPP_V12_ROI_VERIFIED.md` (kept for transparency, archived)
**Corrected honest version**: this document