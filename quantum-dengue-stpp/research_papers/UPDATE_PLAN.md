# Update Plan: Quantum Dengue STPP Architecture v18

**Date:** 22/07/2026
**Source:** `UNIFIED_CONCLUSION.md`
**Goal:** Tái cấu trúc pipeline dựa trên findings từ deep research (29+ papers verified)

---

## Strategic Direction

Dựa trên honest assessment, chuyển từ "khoe tất cả techniques" sang **"honest về proven vs future"**:

- **Phần A — Implement ngay (đã có evidence):** XY QAOA, QNG, QLSTM, QCNN, Classical Fractional Hawkes
- **Phần B — Frame là future work:** QWGAN-GP, SuDaI, MP-QLSTM, Sublinear n-Toffoli

---

## Phase 1: Immediate Cleanup (1-2 days)

### 1.1. Sửa báo cáo chính (QUANTUM_AUDIT_REPORT.md family)

**Tasks:**
- [ ] Tạo file mới: `RESEARCH_VERIFIED_REPORT.md` thay thế claims không có nguồn
- [ ] Loại bỏ "30,5% loss reduction" claim
- [ ] Loại bỏ "18,6% accuracy improvement" claim
- [ ] Đổi "Quantum Fractional Hawkes" → "Classical Fractional Hawkes + quantum components"
- [ ] Đổi "prevents mode collapse" → "theoretically may prevent mode collapse"
- [ ] Frame MP-QLSTM là "Future Work" (không phải current implementation)
- [ ] Frame Sublinear n-Toffoli là "theoretical direction" (không phải implementable)

### 1.2. Cập nhật code comments và docs

**Files to update:**
- [ ] `src/quantum/__init__.py` — đánh dấu status cho mỗi module
- [ ] `src/quantum/pipeline_v17.py` — thêm caveats
- [ ] `src/quantum/qng_optimizer.py` — OK, không cần sửa (proven)
- [ ] `src/quantum/xy_qaoa_sop.py` — OK, không cần sửa (proven)

### 1.3. Add disclaimers vào slides

**Tasks:**
- [ ] Thêm slide "Honest Limitations" vào presentation
- [ ] Phân biệt rõ: ✅ PROVEN vs ⚡ SPECULATIVE vs 🔴 RISKY

---

## Phase 2: Empirical Validation (3-5 days)

### 2.1. Run on real dengue data

**Tasks:**
- [ ] Source data từ Vietnam Dengue Watch, WHO, hoặc GitHub dengue datasets
- [ ] Format data: time series + spatial coordinates
- [ ] Run pipeline trên real data với proven components
- [ ] Document results

### 2.2. Add classical baselines

**Tasks:**
- [ ] Implement Classical Hawkes baseline (Rizoiu 2018)
- [ ] Implement LSTM baseline cho time-series
- [ ] Implement Transformer baseline
- [ ] Compare quantum vs classical trên same data

### 2.3. Honest benchmark

**Tasks:**
- [ ] Document quantum runtime vs classical runtime
- [ ] Report Δ accuracy giữa quantum và classical
- [ ] Acknowledge nếu quantum KHÔNG thắng (theo reality check)

---

## Phase 3: Implement Optimizations (5-7 days)

### 3.1. Warm-Start QAOA (Egger 2021)

**Tasks:**
- [ ] Implement warm-start initialization từ greedy classical solution
- [ ] Benchmark: time reduction, solution quality
- [ ] Document expected 50% time reduction

### 3.2. Trainable Quantum Kernels với QNG

**Tasks:**
- [ ] Integrate QNG optimizer vào kernel training
- [ ] Test trên hard instances
- [ ] Document +0-5% accuracy improvement

### 3.3. Scale to N ≥ 50

**Tasks:**
- [ ] Test pipeline ở N=50, N=100
- [ ] Find crossover point quantum thắng classical
- [ ] Document scale-dependent advantage

### 3.4. Amplitude Encoding

**Tasks:**
- [ ] Implement amplitude encoding cho L(r) features
- [ ] Compare với current angle encoding
- [ ] Document Hilbert space utilization improvement

---

## Phase 4: Add Recommended Papers (2-3 days)

### 4.1. Add to literature review

**Critical papers:**
- [ ] Yang et al. (2023) "Provably superior accuracy in quantum stochastic modeling"
- [ ] McClean et al. (2018) "Barren plateaus in quantum neural network training landscapes"
- [ ] Kübler et al. (2023) "Supervised quantum machine learning kernels"
- [ ] Rizoiu et al. (2018) "Hawkes processes for epidemiological dynamics"
- [ ] Basso et al. (2022) "Obstacles on the path to quantum advantage"

### 4.2. Add new components

**Quantum Reservoir Computing (Fujii 2017):**
- [ ] Implement QRC baseline
- [ ] Compare với QLSTM (parameters, training stability)
- [ ] Consider as alternative to QWGAN

**Error Mitigation:**
- [ ] Implement ZNE (Zero-Noise Extrapolation) — Temme 2017
- [ ] Implement PEC (Probabilistic Error Cancellation) — Zhang 2020
- [ ] Document NISQ performance

---

## Phase 5: Documentation & Presentation (2-3 days)

### 5.1. Update README và ARCHITECTURE

**Tasks:**
- [ ] Thêm section "Verified vs Speculative" vào README
- [ ] Update ARCHITECTURE.md với honest assessment
- [ ] Add reference list với DOI

### 5.2. Create slides deck

**Structure:**
1. Introduction (problem statement)
2. ✅ **Proven contributions** (XY QAOA, QNG, classical Hawkes)
3. ⚡ **Empirical results on real dengue data**
4. ⚠️ **Honest limitations** (barren plateaus, scale, NISQ noise)
5. 🔬 **Future research directions** (QWGAN-GP, MP-QLSTM, etc.)
6. References (29+ papers với DOI)

### 5.3. Create "honest claims" document

**Tasks:**
- [ ] List những gì CHƯA claim (no quantum advantage at N ≤ 30)
- [ ] List những gì CHỦ ĐỘNG claim (XY QAOA optimality, QNG works)
- [ ] Add disclaimers về simulator vs real hardware

---

## Priority Matrix

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| Cleanup báo cáo (remove unverified claims) | HIGH | LOW | 🔴 P0 |
| Add classical baselines | HIGH | MEDIUM | 🔴 P0 |
| Run on real dengue data | HIGH | HIGH | 🔴 P0 |
| Warm-start QAOA | HIGH | MEDIUM | 🟡 P1 |
| Add McClean 2018 (barren plateaus) | MEDIUM | LOW | 🟡 P1 |
| Add Yang 2023 (quantum advantage stochastic) | MEDIUM | MEDIUM | 🟡 P1 |
| Trainable quantum kernels | MEDIUM | MEDIUM | 🟡 P1 |
| Quantum Reservoir Computing | MEDIUM | HIGH | 🟢 P2 |
| Error mitigation | MEDIUM | HIGH | 🟢 P2 |
| Scale to N ≥ 50 | HIGH | HIGH | 🟡 P1 |

---

## Timeline

```
Week 1 (immediate):
  Day 1-2: Cleanup báo cáo, code comments
  Day 3-5: Source real dengue data, implement baselines

Week 2 (optimization):
  Day 6-8: Warm-start QAOA + QNG kernel
  Day 9-10: Scale testing (N=50)

Week 3 (documentation):
  Day 11-13: Update README, slides
  Day 14:   Final review với team
```

---

## Success Metrics

### Quantitative
- [ ] Real dengue data tested, results documented
- [ ] Classical baselines implemented, comparison done
- [ ] Warm-start QAOA reduces time ≥30%
- [ ] No false claims trong final report

### Qualitative
- [ ] Honest framing về proven vs speculative
- [ ] 29+ references với DOI properly cited
- [ ] Slides có "Honest Limitations" section
- [ ] Reviewers có thể verify từng claim qua literature

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Không tìm được real dengue data | Dùng synthetic data với clear documentation |
| Quantum KHÔNG outperform classical | Acknowledge honestly, frame as "tool not silver bullet" |
| Team pushback về removing claims | Cite specific papers showing claims unsupported |
| Timeline quá dài | Cut P2 items, focus on P0 |

---

## Deliverables

1. ✅ `research_papers/UNIFIED_CONCLUSION.md` — DONE
2. ⏳ `research_papers/UPDATE_PLAN.md` — DONE (this file)
3. ⏳ Updated `QUANTUM_AUDIT_REPORT.md` với honest claims
4. ⏳ Updated slides deck
5. ⏳ Real dengue data benchmark results
6. ⏳ Classical baseline comparison
7. ⏳ Warm-start QAOA implementation
8. ⏳ Updated literature review (29+ papers)

---

## Notes

- **Không cần overclaim:** Honest reporting tăng credibility dài hạn
- **Phase 1 critical:** Cleanup unverified claims trước khi làm gì khác
- **Empirical > theoretical:** Real data validation là proof của pudding
- **Keep proven, frame speculative:** Đây là cách quantum community chấp nhận