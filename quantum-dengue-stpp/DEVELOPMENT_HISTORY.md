# Q-STPP Project: Complete Development History

**Generated**: 2026-07-16
**Source**: Git history from initial commit to current HEAD
**Total commits**: 30
**Authors**: Roll249 (Khang Le)
**Branch**: master (default)

This document records the COMPLETE development history of the Quantum-Dengue-STPP
project. It captures every major decision, what was kept, and what was archived —
**including the claims that were later withdrawn**.

> ⚠️ **Withdrawal banner (read first).** Several results reported in the history
> below were later found to be over-stated or not backed by the code, and are
> **withdrawn**:
> - the v9–v12 "quantum kernel" classification advantage (+0.16 accuracy) — the
>   "quantum kernel" was a classical random-Fourier projection, not quantum;
> - the v15-initial "~33× better L(r) error" — an unfair comparison;
> - all monetary ROI figures ("$165,000 / 10,000 cases") and the "R² ~0.97".
>
> The current, honest state is **v15 corrected**: a fair, fully classical
> comparison of SOP permutation-search heuristics (see §11, §13, and
> `Q_STPP_V15_REPORT.md`). Where the history text still reads as if an old claim
> were current, treat this banner as authoritative.

---

## 1. Timeline Overview

```
2026-05-30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2026-07-16
   │                                                              │
   ▼                                                              ▼
 d3da3d1 (initial)                                       bf40d8f (v12 current)
       "update"                                                "v12: ROI verified"

   v0 ─→ v1 ─→ v2 ─→ v3 ─→ v4 ─→ v5 ─→ v6 ─→ v7 ─→ v8 ─→ v9 ─→ v10 ─→ v11 ─→ v12
  pre  inf. R²   fix+API quapp  R²   R²   Mateu  XY-QAOA Linear Hybrid Quantum ARCH ROI
                                                        SOP    hybrid        Zoo      VERIFIED
```

---

## 2. Complete Commit History (30 commits, oldest first)

### 2.1 Phase 1: Foundational Exploration (May 30 – June 22)

| # | Hash | Date | Author | Message | Status |
|---|------|------|--------|---------|--------|
| 1 | `dabf61a` | 2026-05-30 19:23 | Roll249 | update | archived |
| 2 | `5751224` | 2026-06-11 22:13 | Roll249 | update | archived |
| 3 | `9d5a835` | 2026-06-21 22:13 | Roll249 | update ZINB | archived |
| 4 | `9fdb6c2` | 2026-06-21 22:24 | Roll249 | fix: Data leakage prevention + Adaptive gridding + Softplus output | archived |
| 5 | `295ed3a` | 2026-06-21 22:30 | Roll249 | feat: Testing, logging, serialization, and FastAPI API | archived |
| 6 | `7e582bb` | 2026-06-21 22:31 | Roll249 | docs: Update README with v2 improvements | archived |
| 7 | `499fd56` | 2026-06-21 22:33 | Roll249 | feat: Add FastAPI dependencies to requirements.txt | archived |
| 8 | `0a2040f` | 2026-06-22 00:38 | Roll249 | feat: Production-ready infrastructure for v3 | archived |
| 9 | `39e6569` | 2026-06-22 14:35 | Roll249 | feat: QuApp quantum computing platform integration | archived |
| 10 | `fff27ca` | 2026-06-22 14:47 | Roll249 | chore: Add setup.sh script for environment setup | archived |
| 11 | `f41ab19` | 2026-06-22 14:47 | Roll249 | fix: Remove accidental file | archived |
| 12 | `75985f2` | 2026-06-22 15:39 | Roll249 | add quapp | archived |

**Phase 1 summary**:
- Initial exploratory development (R² regression for dengue intensity prediction)
- ZINB (Zero-Inflated Negative Binomial) loss implementation
- Data leakage prevention fixes
- Adaptive gridding and Softplus output
- FastAPI service layer for production deployment
- QuApp quantum computing platform integration
- **Status**: All archived — R² regression approach was eventually superseded by STPP classification

---

### 2.2 Phase 2: Quantum Advantage Exploration (July 14)

| # | Hash | Date | Author | Message | Status |
|---|------|------|--------|---------|--------|
| 13 | `93c031d` | 2026-07-14 11:45 | Roll249 | update quantum | archived |
| 14 | `b0f3108` | 2026-07-14 11:59 | Roll249 | update quantum | archived |
| 15 | `7654a7f` | 2026-07-14 15:17 | Roll249 | update quantum | archived |

**Phase 2 summary**:
- First quantum experiments (before version numbering)
- Initial probes of QIG (Quantum Intensity Generator)
- Multiple iterations of quantum module updates
- **Status**: All archived — exploratory phase before Mateu 2025 alignment

---

### 2.3 Phase 3: Mateu 2025 Alignment + STPP Framework (July 15)

| # | Hash | Date | Author | Message | Status |
|---|------|------|--------|---------|--------|
| 16 | `af41a05` | 2026-07-15 01:51 | Roll249 | chore: cleanup legacy files + add v5 (post code-review fixes) | archived |
| 17 | `b818f51` | 2026-07-15 01:51 | Roll249 | docs: add reference paper Mateu ECSIA 2025 (Prague) | archived (info in THEORY.md) |
| 18 | `d13f90f` | 2026-07-15 01:57 | Roll249 | **feat: v6 aligned with Mateu ECSIA 2025** (Siamese CNN + 1-NN classification) | archived |
| 19 | `c302f32` | 2026-07-15 11:30 | Roll249 | **feat: XY-Mixer QAOA for SOP permutation search** | archived (integrated into v9) |
| 20 | `c8198c3` | 2026-07-15 12:11 | Roll249 | feat: Quantum advantage benchmark - comprehensive study | archived |
| 21 | `b3da3d2` | 2026-07-15 15:19 | Roll249 | feat: Super R² benchmark for GPU (3090 Ti, 128GB RAM) | archived |
| 22 | `b0243fb` | 2026-07-15 23:51 | Roll249 | docs: Viết file logic chi tiết tiếng Việt | archived (consolidated) |
| 23 | `10a939d` | 2026-07-15 23:56 | Roll249 | docs: Add English version of logic details | archived (consolidated) |

**Phase 3 summary** (KEY MILESTONE):
- **v6**: First Mateu 2025-aligned version. Implemented:
  - Discretization (12×12 grid from spatial point pattern)
  - Siamese CNN feature extractor (10,049 params)
  - Composite Bernoulli loss (per Mateu slide 36)
  - 1-NN classification (per Mateu slide 32)
  - K-function baseline (per Mateu slide 13)
  - **Result**: K-function = 0.8333 (best), Classical Siamese = 0.7222, Quantum Siamese = 0.6111
- **XY-Mixer QAOA**: First quantum-augmented SOP permutation search using quantum mixer Hamiltonian
- **Super R² benchmark**: GPU-targeted benchmark (later not used for STPP)
- **Status**: All archived except XY-QAOA concept (integrated into v9)

---

### 2.4 Phase 4: Quantum Algorithm Zoo (July 16, first half)

| # | Hash | Date | Author | Message | Status |
|---|------|------|--------|---------|--------|
| 24 | `f930c16` | 2026-07-16 00:06 | Roll249 | feat: Quantum Advantage x100 honest proof script | archived |
| 25 | `30048e1` | 2026-07-16 00:11 | Roll249 | **feat: v7 — XY-Mixer QAOA SOP + Quantum Kernel K-function** | archived (in v9) |
| 26 | `785e37c` | 2026-07-16 00:18 | Roll249 | **v8: Hybrid classical-quantum STPP pipeline** + project synthesis | archived |
| 27 | `602b3fe` | 2026-07-16 00:24 | Roll249 | v9: Hybrid quantum-classical "wins at N≥150" | **WITHDRAWN** (see banner) |
| 28 | `a3aab9a` | 2026-07-16 00:30 | Roll249 | v10: "Quantum Algorithm Zoo" — 5 algorithms | **WITHDRAWN** |
| 29 | `eea55fb` | 2026-07-16 00:33 | Roll249 | v11: Consolidate to v9+v10 + ARCHITECTURE/THEORY | superseded |
| 30 | `bf40d8f` | 2026-07-16 00:43 | Roll249 | v12: "ROI verified — quantum advantage is REAL" | **WITHDRAWN** |

Later commits (not in the table above): v13–v14 (superseded) and
**`f824dd3` v15** — consolidation to the corrected fair comparison, the current
state (see §11).

**Phase 4 summary** — all WITHDRAWN or superseded (see the banner at the top):
- **v7**: XY-Mixer QAOA SOP + a "Quantum Kernel K-function". Archived.
- **v8**: First linear hybrid. Archived.
- **v9**: claimed "+0.19 over classical at N=150". **WITHDRAWN** — the "quantum kernel" was a classical random-Fourier projection.
- **v10**: five "quantum algorithms" (QBOOT/GAS/QAE/QFT/TSQS), several with unverifiable citations. **WITHDRAWN**.
- **v11**: codebase consolidation.
- **v12**: claimed a "proper IQP quantum kernel" and "statistically significant" advantage. **WITHDRAWN** — the significance script still used the classical RFF, not the IQP kernel. See §9 and `Q_STPP_V15_REPORT.md`.

---

## 3. Version Milestones

### 3.1 v0-v3: Pre-quantitative phase (R² regression)
**Focus**: Predicting dengue case counts via ZINB loss
**Key files**:
- `src/models/cnn_lstm.py` (now archived)
- `src/models/zinb.py` (now archived)
**Lesson learned**: R² regression doesn't match the Mateu 2025 framework for STPP classification. Switched to classification.

### 3.2 v4-v5: R² regression refined
**Focus**: Post code-review fixes for R²
**Files**: `run_q_stpp_v4.py`, `run_q_stpp_v5.py` (archived)
**Lesson learned**: Warm-start bias corrections applied. Still not the right approach for STPP.

### 3.3 v6: Mateu 2025 Alignment (CRITICAL MILESTONE)
**Date**: 2026-07-15
**File**: `run_q_stpp_v6.py` (archived)
**Focus**: Mateu S7-ECSIA-Prague 2025 paper alignment

Implementation:
| Component | Mateu Reference | v6 Implementation |
|-----------|-----------------|-------------------|
| Discretization | Slide 14 | `discretize_to_grid` |
| CNN feature extractor | Slides 17-19, 43 | `CNNFeatureExtractor` |
| Siamese discriminant | Slide 30 | `SiameseDiscriminant` |
| Composite Bernoulli loss | Slide 36 | `composite_bernoulli_loss` |
| SOP permutations | Mohler-Mateu 2024 | `sop_permute_grid` |
| 1-NN classification | Slide 32 | `one_nn_accuracy` |
| K-function baseline | Slide 13 | `ripley_k` |

**Results** (synthetic Poisson/LGCP/Cluster, 42 train + 18 test):
| Method | 1-NN Accuracy | Params |
|--------|---------------|--------|
| K-function (baseline) | **0.8333** | - |
| Classical Siamese CNN | 0.7222 | 10,049 |
| Quantum Siamese CNN (hybrid) | 0.6111 | 1,931 |

**Honest finding**: K-function baseline wins on small synthetic data. This is consistent with Mateu 2025 slide 47.

### 3.4 v7: XY-Mixer QAOA SOP
**Date**: 2026-07-16
**File**: `run_q_stpp_v7.py` (archived, integrated into v9)
**Focus**: Quantum-augmented SOP permutations via XY-Mixer QAOA
**Key innovation**: 
- Mixer Hamiltonian $H_M = \sum_{\langle i,j \rangle} (X_i X_j + Y_i Y_j)$
- Naturally produces SOP-preserving permutations (symmetric under exchange)
- Cost: $O(N^2)$ per swap, vs $O(N^2)$ classical

### 3.5 v8: Linear Hybrid (FAILED)
**Date**: 2026-07-16
**File**: `run_q_stpp_v8.py` (archived)
**Focus**: First hybrid pipeline combining classical + quantum
**Lesson learned**: Linear combination failed because:
$$D_{\text{hybrid}} = \alpha D_{\text{classical}} + \beta D_{\text{quantum}} + \gamma D_{\text{QAOA}}$$
did not decorrelate the errors of each component. Result: equal to best individual.

### 3.6–3.9 v9–v12 (all WITHDRAWN)

> These four versions built the "quantum advantage" narrative that was later
> found to be unsupported by the code. They are **withdrawn**; the numbers below
> are recorded only as history, not as current results.

- **v9 "Smart Hybrid"** (`run_q_stpp_v9.py`, removed): claimed "+0.19 over
  classical at N=150" from decision-level voting. **WITHDRAWN** — the "quantum
  kernel" feature it voted on was a classical random-Fourier projection.
- **v10 "Quantum Algorithm Zoo"** (`run_q_stpp_v10.py`, removed): five
  "quantum algorithms" (GAS, QBOOT, QAE, QFT-Symmetric, TSQS), several citing
  arXiv IDs that could not be verified (e.g. "arXiv 2604.00951", "2603.22401").
  **WITHDRAWN**.
- **v11**: architecture/theory consolidation. Superseded by the corrected docs.
- **v12 "ROI Verification"** (`run_q_stpp_v12_significance.py`,
  `run_q_stpp_v12_proper_kernel.py`, removed): claimed a "proper IQP quantum
  kernel" and a "statistically significant" advantage (p<0.0001, Cohen's d≈10).
  **WITHDRAWN** — a genuine PennyLane IQP kernel existed in
  `run_q_stpp_v12_proper_kernel.py`, but the *significance* script that produced
  the headline p-values still used the classical RFF, so the reported
  significance was not about a quantum kernel at all.

---

## 4. ROI "Verified" (v12) — WITHDRAWN

> This ROI block is **withdrawn** (see the banner at the top). It was built on the
> v12 "quantum kernel" classification advantage, which turned out to be a
> classical random-Fourier projection, not a quantum kernel. The figures below
> are kept only as a record of what was claimed at the time — **not** as a
> current result.

```
[WITHDRAWN] Per 10,000 dengue cases:
[WITHDRAWN] - Error reduction: 1,700 cases caught extra by quantum
[WITHDRAWN] - Value at $100/caught: $170,000
[WITHDRAWN] - Quantum hardware cost: -$5,000
[WITHDRAWN] - NET ROI: $165,000 per 10,000 cases
```

---

## 5. Historical snapshot (v12 — superseded)

> This section described the repo *at commit `bf40d8f`*. It is **out of date**:
> those v9/v10/v12 scripts and reports were removed when the project consolidated
> to v15. For the current file list see §11. Kept only as a record.

The v12 snapshot listed `run_q_stpp_v9/v10/v12_*.py`, `Q_STPP_V9/V10/V12_*.md`,
and `output_result/q_stpp_v9/v10/v12*` — **all removed**. The `output_result/data/`
CSVs (OpenDengue-derived, see `dengue_dataset/`) remain.

---

## 6. Key Lessons Learned

### 6.1 What Worked
✅ **Alignment with published work** (Mateu 2025) provided structure
✅ **Fair-comparison discipline** (shared seed, equal budget) once it was applied
✅ **Reporting two metrics** (quality AND diversity) prevents mode-collapse from looking like a win
✅ **Withdrawing over-claims** — the honest, scoped result is more credible than the inflated one

### 6.2 What Failed
❌ **R² regression** for STPP (mismatch with Mateu 2025 framework)
❌ **Linear ensemble** doesn't decorrelate errors
❌ **Hilbert projection** is not a quantum kernel
❌ **v9 quantum kernel alone** is weak without proper feature maps

### 6.3 Honest Limitations
⚠️ All results on **synthetic STPP** (Poisson/LGCP/Cluster)
⚠️ **Real dengue data validation** pending (TYCHO dataset not yet integrated)
⚠️ Effect size on real data may be smaller than synthetic

---

## 7. Files by Commit

This section can be reconstructed with:
```bash
git log --all --reverse --name-status --pretty=format:"%h | %ai | %an | %s"
```

For brevity, see `archive/` for all archived files.

---

## 8. Author

**Khang Le (Roll249)**
- Project: Quantum-Dengue-STPP (QC4SG Hackathon)
- Branch: master (default)
- Time span: 2026-05-30 to 2026-07-16 (47 days)
- Commits: 30
- Final state: **v15 corrected** — a fair, fully classical comparison (see §11).
  Earlier "quantum advantage" claims are withdrawn.

---

## 9. v13-v15: Quantum-Inspired SOP (July 16, afternoon)

### 9.1 v13: "Quantum Native SOP" (removed)
**Date**: 2026-07-16
**File**: `run_q_stpp_v13_quantum_native.py` (removed)
**Claim**: a PennyLane QAOA circuit sampling SOP permutations. Superseded and
removed when the project consolidated to the single classical v15 script.

### 9.2 v14: Quick QAOA Comparison (removed)
**Date**: 2026-07-16
**File**: `run_q_stpp_v14_quick_qaoa.py` (removed)

### 9.3 v15: fair SOP comparison (current)
**Date**: 2026-07-16
**Files**:
- `run_q_stpp_v15_qaoa_sop.py` (initial — unfair, removed)
- `run_q_stpp_v15_fair.py` (**current**)
**Focus**: fair comparison of three classical permutation-search heuristics

#### v15 Discovery: CHEATING in Initial Implementation

**Problem Found**:
The initial v15 implementation had a CRITICAL FLAW:
```python
n_quick_samples = min(500, math.factorial(n))
for _ in range(n_quick_samples):
    perm = list(rng.permutation(n))  # BRUTE FORCE = CHEATING!
```

This was brute-force enumeration disguised as "quantum-inspired"!

**FIX Applied**:
- All methods now have **same computational budget** (3333 operations)
- No brute-force enumeration
- No unlimited sampling
- Fair comparison: Classical MH vs Quantum-Inspired vs QAOA

#### v15-initial Results — WITHDRAWN

> The v15-initial "~33× better L(r) error" table (values like 35×/62×/42×) is
> **withdrawn**. It came from an unfair comparison: the Metropolis-Hastings
> baseline used the unseeded global RNG and a broken fixed acceptance
> temperature that made it accept ~90% of worsening moves (so it barely
> optimised), the three methods did not share an equal evaluation budget, and
> the "×" figure divided by a near-zero denominator. It also reported only mean
> error, which rewards mode collapse. **No such multiplier is claimed.**

#### v15 (corrected) — fair comparison

`run_q_stpp_v15_fair.py` was rewritten to fix all of the above:

1. Every method is seeded identically — each re-instantiates `default_rng(seed)`
   with the same `seed`, so they start from the same random state (differences in
   the stream come only from the differing proposal/acceptance, not the seed).
2. Each method spends exactly `evals_per_perm` L-summary evaluations (equal budget).
3. Scale-adaptive annealed temperature for MH (no magic constant).
4. Both **L(r) error** (quality) and **set diversity** are reported.
5. Ratios are clamped; the headline is the two metrics, not a multiplier.

All three methods are **classical**; "Grover-/QAOA-inspired" are heuristic
analogies, not quantum circuits. Result numbers are produced by running the
script (see `Q_STPP_V15_REPORT.md`); none are hard-coded here.

---

## 10. Complete Version Timeline

```
2026-05-30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2026-07-16
   │                                                                        │
   ▼                                                                        ▼
d3da3d1 (initial)                                                  bf40d8f (v12)
       "update"                                                             "v12: ROI"
                                                                             
v0 ─→ v1 ─→ v2 ─→ v3 ─→ v4 ─→ v5 ─→ v6 ─→ v7 ─→ v8 ─→ v9 ─→ v10 ─→ v11 ─→ v12
pre  inf. R²   fix+API quapp  R²   R²   Mateu  XY-QAOA Linear Hybrid Quantum ARCH ROI
                                                    SOP    hybrid        Zoo      VERIFIED
                                                                             
                                                                             
Then: v13 ─→ v14 ─→ v15 (FAIR)
   Quantum   Quick   QI-SOP
   Native    QAOA   BREAKTHROUGH
             Comparison
```

---

## 11. Current State (v15 corrected)

### 11.1 Active files
```
├── ARCHITECTURE.md            # System design (matches the code)
├── THEORY.md                  # L-function + local-search background
├── DEVELOPMENT_HISTORY.md     # This file
├── README.md                  # Quick start
├── Q_STPP_V15_REPORT.md       # Methodology + results template
│
├── run_q_stpp_v15_fair.py     # ★ the entire pipeline (classical)
├── run.sh                     # convenience wrapper
├── requirements.txt           # numpy, scipy, matplotlib
│
├── output_result/
│   ├── data/                  # cached synthetic event CSVs
│   └── q_stpp_v15_fair/       # results.json + plot.png (created by a run)
│
└── archive/                   # withdrawn / superseded versions (v4–v12)
```

Note: the earlier scripts (`run_q_stpp_v9.py`, `run_q_stpp_v10.py`,
`run_q_stpp_v12_*.py`, `run_q_stpp_v13_*.py`) were removed when the project
consolidated to the single fair-comparison script.

### 11.2 Key Findings Summary

| Version | Focus | Status |
|---------|-------|--------|
| v9–v12 | "Quantum kernel" classification (+0.16 accuracy) | **WITHDRAWN** — the "quantum kernel" was a classical random-Fourier projection, not quantum |
| v15-initial | QI-SOP "~33×" | **WITHDRAWN** — unfair comparison (see §9) |
| v15 corrected | Fair SOP comparison | classical MH vs Grover-/QAOA-inspired; reports L(r) error **and** diversity; no quantum advantage claimed |

---

## 12. Key Lessons (Updated)

### 12.1 What Worked
✅ **Alignment with published work** (Mateu 2025) provided structure
✅ **Fair-comparison discipline** — shared seed + equal evaluation budget
✅ **Reporting two metrics** (L(r) error AND diversity) prevents mode-collapse from looking like a win
✅ **Withdrawing over-claims** — the honest scope is more credible than the inflated numbers

### 12.2 What Failed
❌ **R² regression** for STPP (mismatch with Mateu 2025 framework)
❌ **Linear ensemble** doesn't decorrelate errors
❌ **Hilbert projection** is not a quantum kernel
❌ **Cheating in v15 initial** - brute force enumeration not allowed
❌ **Unlimited sampling** - must match computational budgets

### 12.3 Honest Limitations
⚠️ All results on **synthetic STPP** (Poisson/LGCP/Cluster)
⚠️ **Real dengue data validation** pending (TYCHO dataset not yet integrated)
⚠️ Effect size on real data may be smaller than synthetic

---

## 13. ROI Summary — WITHDRAWN

> The previous ROI figures ("$165,000 per 10,000 cases", "1,700 extra cases
> caught by quantum", "~33× better L(r) error", "R² ~0.97") are **withdrawn**.
> They were built on the withdrawn quantum-kernel classification result and the
> unfair v15-initial comparison, and the "R²" was never computed by any code.
> No monetary ROI or speedup multiplier is claimed. The corrected project
> reports only L(r) preservation error and set diversity, produced by running
> `run_q_stpp_v15_fair.py`.

---

## 14. Files by Commit

This section can be reconstructed with:
```bash
git log --all --reverse --name-status --pretty=format:"%h | %ai | %an | %s"
```

---

## 15. Author

**Khang Le (Roll249)**
- Project: Quantum-Dengue-STPP (QC4SG Hackathon)
- Branch: master (default)
- Time span: 2026-05-30 to 2026-07-16 (47 days)
- Commits: 30+
- Final state: v15 corrected — a fair, honest, **classical** comparison of SOP
  permutation-search heuristics. Earlier "quantum advantage" claims are withdrawn.

---

**This document is the canonical history of the Quantum-Dengue-STPP project. It
records every version, including the claims that were later withdrawn for being
over-stated or not backed by the code.**