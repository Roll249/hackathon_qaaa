# Q-STPP Project: Complete Development History

**Generated**: 2026-07-16
**Current Version**: v16
**Status**: Honest hybrid architecture with classical-first design

---

## 1. Timeline Overview

```
2026-05-30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2026-07-16
   │                                                                      │
   ▼                                                                      ▼
 d3da3d1 (initial)                                               v16 (current)
       "update"                                                    "Honest Hybrid"
                                                                              
 v0 ─→ v1 ─→ v2 ─→ v3 ─→ v4 ─→ v5 ─→ v6 ─→ v7 ─→ v8 ─→ v9 ─→ v10 ─→ v11 ─→ v12
pre  inf. R²   fix+API quapp  R²   R²   Mateu  XY-QAOA Linear Hybrid Quantum ARCH ROI
                                                    SOP    hybrid        Zoo      VERIFIED
                                                                              
                                                                              
Then: v13 ─→ v14 ─→ v15 ──────────────────────────────────────────────→ v16
   Quantum   Quick   FAIR     Honest Hybrid Architecture
   Native    QAOA   SOP      Classical-First
```

---

## 2. Version Summary

| Version | Focus | Status | Key Finding |
|---------|-------|--------|-------------|
| v0-v5 | R² regression | Archived | Wrong approach for STPP |
| v6 | Mateu 2025 alignment | Archived | K-function baseline wins on small data |
| v7 | XY-Mixer QAOA SOP | Archived | Concept valid, but NISQ too noisy |
| v8 | Linear Hybrid | Failed | Linear ensemble doesn't decorrelate errors |
| v9-v12 | Quantum kernel claims | **WITHDRAWN** | "Quantum kernel" was classical RFF |
| v13-v14 | Quantum-native SOP | Superseded | Fair comparison needed |
| **v15** | Fair SOP comparison | Corrected | Classical heuristics win |
| **v16** | Honest Hybrid Architecture | **Current** | Classical-first, quantum-where-useful |

---

## 3. Honest Assessment (v16)

### What Works

✅ **Classical-first approach**
- K/L-function computation: O(N²) deterministic
- MH sampler: High diversity, proven convergence
- Greedy search: Lowest error, fast
- QAOA-inspired: Balances quality and diversity

✅ **Honest methodology**
- Same seed, same budget for fair comparison
- Report BOTH quality AND diversity
- No over-claims about quantum advantage

### What Doesn't Work (Yet)

❌ **Quantum advantage for current problem sizes**
- N < 100: Classical wins decisively
- N ≈ 100-200: No proven quantum advantage
- N > 200: Potential crossover point, but unproven

❌ **Previous over-claims**
- "Quantum kernel" was classical RFF projection
- "33x improvement" was unfair comparison
- "ROI $165,000" was built on withdrawn claims

---

## 4. v16 Architecture Principles

### 4.1 Layer Design

```
Layer 0: Data Pipeline      → 100% Classical
Layer 1: Feature Extract    → 100% Classical (K/L-function, CNN, GNN)
Layer 2: Prediction        → 100% Classical (1-NN, risk scoring)
Layer 3: SOP Augmentation → Classical+ (MH, greedy, QAOA-inspired)
Layer 4: Quantum Layer     → Future research only
Layer 5: Output            → 100% Classical
```

### 4.2 Quantum Assessment (per quantum-computing-expert)

| Use Case | Potential | Practical | Timeline |
|----------|-----------|-----------|----------|
| QAOA for SOP (N>200) | High | Unproven | 2-5 years |
| Quantum kernels | Medium | Unvalidated | Research |
| VQE optimization | Low | Speculative | Unknown |

### 4.3 Key Messages

1. **Classical v16 is production-ready** for current dengue data
2. **Quantum is future research**, not current solution
3. **No quantum advantage claimed** for any benchmark
4. **Honest comparison** between classical methods only

---

## 5. Files by Version

### v15 (Corrected Fair Comparison)
```
├── run_q_stpp_v15_fair.py          # Classical SOP comparison
├── output_result/q_stpp_v15_qaoa_sop_fixed/
│   ├── fair_comparison_results.json
│   └── quantum_advantage_regimes.png
├── Q_STPP_V15_REPORT.md
├── ARCHITECTURE.md (old)
├── THEORY.md (old)
└── DEVELOPMENT_HISTORY.md
```

### v16 (Honest Hybrid Architecture)
```
├── run_q_stpp_v16.py               # [NEW] Main v16 pipeline
├── src/
│   ├── data/                        # Layer 0: Data pipeline
│   ├── features/                    # Layer 1: Feature extraction
│   ├── prediction/                 # Layer 2: Prediction
│   ├── augmentation/               # Layer 3: SOP augmentation
│   ├── quantum/                    # Layer 4: Quantum benchmarks (future)
│   └── output/                     # Layer 5: Metrics & visualization
├── docs/
│   ├── ARCHITECTURE.md             # [UPDATED] Layer design
│   ├── THEORY.md                  # [UPDATED] Mathematical foundations
│   ├── QUANTUM_ASSESSMENT.md      # [NEW] Honest quantum analysis
│   ├── Q_STPP_V16_REPORT.md       # [NEW] Technical report
│   └── DEVELOPMENT_HISTORY.md     # [UPDATED] This file
└── tests/                          # Unit tests
```

---

## 6. Key Lessons Learned

### 6.1 What Worked
✅ **Alignment with Mateu 2025** provided rigorous methodology
✅ **Fair comparison discipline** (shared seed + equal budget)
✅ **Reporting two metrics** (quality AND diversity)
✅ **Withdrawing over-claims** improved credibility

### 6.2 What Failed
❌ **R² regression** for STPP - wrong framework
❌ **Linear ensemble** - doesn't decorrelate errors
❌ **Classical RFF** as "quantum kernel" - misleading
❌ **Brute force enumeration** disguised as "quantum-inspired"

### 6.3 Honest Limitations
⚠️ All results on **synthetic STPP** (Hawkes/Poisson/LGCP)
⚠️ **Real dengue data validation** pending (TYCHO dataset)
⚠️ Effect size on real data may differ from synthetic
⚠️ Quantum advantage unproven for any current benchmark

---

## 7. Path Forward

### 7.1 Immediate Priorities
1. **Integrate real dengue data** (TYCHO/OpenDengue)
2. **Validate v16 on real outbreak scenarios**
3. **Add proper unit tests**
4. **Generate reproducible benchmarks**

### 7.2 Future Research Directions
1. **QAOA for large N** (N > 200) with proper benchmarking
2. **Quantum kernel validation** on specific pattern families
3. **VQE for kernel parameter optimization**
4. **Non-stationary kernel learning** (per Mateu 2025)

### 7.3 What to Avoid
- Don't claim quantum advantage without proof
- Don't use classical simulation as "quantum experiment"
- Don't over-state ROI without real data validation
- Don't rush the theory before implementation

---

## 8. Author

**Khang Le (Roll249)**
- Project: Quantum-Dengue-STPP (QC4SG Hackathon)
- Time span: 2026-05-30 to 2026-07-16 (47+ days)
- Current state: **v16 Honest Hybrid** - Classical-first, quantum-where-useful

---

## 9. Citation

If you use this work, please cite:

```bibtex
@misc{quantum-dengue-stpp,
  title={Q-STPP v16: Honest Hybrid Architecture for Dengue Prediction},
  author={Khang Le},
  year={2026},
  note={Based on Mateu 2025 (ECSIA Prague)}
}
```
