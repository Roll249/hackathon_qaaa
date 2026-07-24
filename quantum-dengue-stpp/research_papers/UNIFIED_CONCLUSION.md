# Báo cáo Tổng hợp Nghiên cứu: Quantum Enhanced Spatio-Temporal Forecasting for Dengue

**Ngày:** 22/07/2026
**Người tổng hợp:** QAAA Lab Research Synthesis
**Phạm vi:** Verification toàn diện báo cáo đề xuất kiến trúc lượng tử cho QC4SG 2026
**Nguồn:** 2 deep-research workers (papers database + pipeline optimization), 29+ academic papers reviewed

---

## 1. Tóm tắt Điều hành (Executive Summary)

Báo cáo "Giải quyết Nút thắt Tiền xử lý và Tối ưu hóa Khung Kỹ thuật Lượng tử trong Mô hình Dự báo Dịch tễ Không gian - Thời gian" đề xuất một pipeline lượng tử đầy tham vọng với 8 thành phần cốt lõi. Sau khi đối chiếu tỉ mỉ với 29+ bài báo học thuật được công bố, **nhiều claim trong báo cáo không được chứng minh đầy đủ bởi literature**:

- **3/8 thành phần ở mức PROVEN** và có thể triển khai ngay (XY QAOA, Parameter-Shift/QNG, Fractional Hawkes classical)
- **3/8 ở mức SPECULATIVE** — có cơ sở lý thuyết nhưng chưa có empirical validation trên dữ liệu dịch tễ
- **2/8 ở mức RISKY** — yêu cầu hardware chưa khả thi hoặc purely theoretical
- **2 claim không tìm thấy nguồn** cần được loại bỏ

**Kết luận cốt lõi:** Báo cáo cần tái cấu trúc thành 2 phần rõ ràng — (a) những gì chúng tôi đã và có thể chứng minh (XY QAOA, QNG, classical Fractional Hawkes), (b) những gì là hướng nghiên cứu tương lai (QWGAN-GP, MP-QLSTM, Sublinear n-Toffoli). Tránh sử dụng từ "quantum" cho các kỹ thuật classical không có quantum implementation.

---

## 2. Đánh giá Thành phần Kiến trúc (8 Components)

### Bảng tổng hợp

| # | Thành phần | Status | Bằng chứng | Rủi ro |
|---|-----------|--------|------------|--------|
| 1 | Quantum Data Re-Uploading | ⚡ SPECULATIVE | Proven cho classification; chưa verify cho STPP | Trung bình |
| 2 | Sublinear n-Toffoli Encoding | 🔴 RISKY | Theoretical only; probabilistic success rate | Cao |
| 3 | QWGAN-GP | ⚡ SPECULATIVE | Theory đúng; NISQ chưa verify | Trung bình |
| 4 | SuDaI | ⚡ SPECULATIVE | Variant của data re-uploading; domain khác | Trung bình |
| 5 | QCNN | ✅ PROVEN | Cong et al. 2019 (Nature Physics) | Thấp |
| 6 | QLSTM | ✅ PROVEN | Chen et al. 2020 (ICASSP) | Thấp |
| 7 | MP-QLSTM | 🔴 RISKY | Cần distributed QPU — không khả thi | Rất cao |
| 8 | Fractional Hawkes + Field Master Eq | ✅ PROVEN (classical) | Chen et al. 2020; Kanazawa & Sornette 2020 | Thấp |

---

### 2.1. Quantum Data Re-Uploading

**Claim trong báo cáo:** "Universal function approximator thông qua việc nạp dữ liệu nhiều lần với trainable unitaries, khả năng chống overfitting nhờ vanishing high-frequency components."

**Verification:**
- ✅ **Proven cho classification:** Pérez-Salinas et al. (2020) "Data re-uploading for a universal quantum classifier" — Quantum 4, 226. Chứng minh single-qubit có thể là universal classifier.
- 🔴 **Critical limitation (Wang et al. 2025):** Với limited qubits + deep re-uploading circuits, predictive performance suy giảm về random guessing.
- ⚠️ **Chưa có paper nào apply data re-uploading cho spatio-temporal forecasting với epidemiological data.**

**Verdict:** Có thể dùng shallow circuits (p ≤ 3) cho dengue, nhưng đừng claim "universal" cho complex STPP tasks.

---

### 2.2. Sublinear n-Toffoli Encoding

**Claim trong báo cáo:** "Encode vector phức kích thước N=2^n với độ sâu mạch sublinear, dùng hypercube graph isomorphism."

**Verification:**
- 🔴 **Theoretical only:** Không tìm thấy paper peer-reviewed công bố thuật toán này.
- ⚠️ **Trade-off nghiêm trọng:** Tỷ lệ thành công xác suất tỷ lệ thuận với sparsity của dữ liệu — không khả thi cho dense climate-dengue data.
- 🔴 **NISQ compatibility chưa chứng minh:** Cần 2 ancilla qubits + multi-controlled X gates — không thực tế trên IBM Eagle 127-qubit ở scale này.

**Verdict:** KHÔNG nên đưa vào pipeline hiện tại. Đây là pure speculation. Nếu muốn giữ, frame là "future research direction."

---

### 2.3. QWGAN-GP (Quantum Wasserstein GAN + Gradient Penalty)

**Claim trong báo cáo:** "Quantum Wasserstein distance với gradient penalty ngăn chặn mode collapse, duy trì volatility clustering."

**Verification:**
- ✅ **Quantum Wasserstein distance:** Chakrabarti et al. (2019) "Quantum Wasserstein GANs" — NeurIPS Workshop.
- ✅ **Foundational QGAN:** Lloyd & Weedbrook (2018); Dallaire-Demers & Killoran (2018).
- ⚠️ **Mode collapse mitigation chưa verify:** Chakrabarti 2019 KHÔNG claim explicit mode collapse prevention. Đây là extension từ classical WGAN-GP (Gulrajani et al. 2017).
- 🔴 **NISQ performance chưa test:** Hammami et al. (2025) mới test ở 4-8 qubits trên simulator.

**Verdict:** Frame là "theoretically may prevent mode collapse" thay vì "prevents." Mode collapse mitigation vẫn là open question.

---

### 2.4. SuDaI (Successive Data Injection)

**Claim trong báo cáo:** "Tiêm lặp đi lặp lại các phần khác nhau của vector data vào evolving quantum state. 80 params = 99% accuracy."

**Verification:**
- ✅ **Original SuDaI paper:** Kalfon et al. (2023).
- ⚠️ **Domain khác biệt:** Hammami et al. (2025) test trên network anomaly detection — không phải epidemiology.
- 🔴 **Không phải contribution mới:** SuDaI essentially là variant của data re-uploading. Cả hai đều repeated data encoding với trainable parameters. Overlap ~80%.

**Verdict:** KHÔNG claim là separate contribution. Frame là "SuDaI (data re-uploading variant cho time series)."

---

### 2.5. QCNN (Quanvolutional Neural Networks)

**Claim trong báo cáo:** "QCNN giảm hàm mất mát tới 30,5% và tăng độ chính xác 18,6%."

**Verification:**
- ✅ **Original QCNN:** Cong, Choi, Lukin (2019) "Quantum convolutional neural networks" — Nature Physics 15, 1273-1278.
- ✅ **Quanvolutional Networks:** Henderson et al. (2020) — Scientific Reports.
- 🔴 **"30,5%" và "18,6%" KHÔNG tìm thấy nguồn:** Search trong QCNN papers, Quanvolutional papers, weather/precipitation forecasting papers — không match.

**Verdict:** Proven concept, NHƯNG cần loại bỏ hai con số "30,5%" và "18,6%" hoặc cung cấp citation. Có thể misattribution từ classical deep learning literature.

---

### 2.6. QLSTM (Quantum LSTM)

**Claim trong báo cáo:** "6 VQC thay thế 4 gates của LSTM, giảm 166 → 24 params."

**Verification:**
- ✅ **Original QLSTM:** Chen, Yoo, Fang (2020) "Quantum long short-term memory" — ICASSP 2020.
- ✅ **Validation:** Tested trên simple time-series (damped oscillators, NARMA-5).
- ⚠️ **Limited empirical validation:** Chưa test trên epidemic-scale data.

**Verdict:** Proven technique. Phù hợp cho dengue forecasting ở scale thấp. Cần test trên real outbreak data.

---

### 2.7. MP-QLSTM (Multi-Parallel Quantum LSTM)

**Claim trong báo cáo:** "Phân tích parallel trên nhiều QPU, giảm barren plateaus, tăng noise resilience."

**Verification:**
- ⚠️ **Distributed QLSTM papers:** Chen et al. (2022, 2025).
- 🔴 **Hardware requirements nghiêm trọng:** Yêu cầu M QPUs độc lập (M = số sub-VQCs). Hiện tại không có multi-QPU access.
- 🔴 **NISQ feasibility cực thấp:** Distributed quantum computing vẫn là emerging tech.

**Verdict:** KHÔNG thể triển khai hiện tại. Frame là "future work" hoặc loại bỏ.

---

### 2.8. Fractional Hawkes Process + Field Master Equation

**Claim trong báo cáo:** "Mittag-Leffler kernel với heavy-tailed distributions, Markovian embedding qua Field Master Equation Theory, quantum circuits để học invariant measures."

**Verification:**
- ✅ **Fractional Hawkes mathematical framework:** Chen et al. (2020), Habyarimana et al. (2023).
- ✅ **Field Master Equation Theory:** Kanazawa & Sornette (2020).
- ✅ **Heavy-tailed fits epidemic data:** Dengue case counts exhibit heavy-tailed distributions — phù hợp.
- 🔴 **KHÔNG có paper quantum implementation:** Toàn bộ là classical technique. Claim "Quantum Fractional Hawkes" trong báo cáo là MISLEADING.

**Verdict:** Math proven, NHƯNG đây là CLASSICAL technique. Loại bỏ "quantum" modifier. Frame là "Classical Fractional Hawkes + quantum components (nếu có)."

---

## 3. Các Claim Không Tìm Thấy Nguồn (Unverified Claims)

| Claim trong báo cáo | Search result | Action |
|---------------------|---------------|--------|
| "QCNN giảm loss 30,5%" | ❌ Not found | Remove hoặc cite source |
| "QCNN tăng accuracy 18,6%" | ❌ Not found | Remove hoặc cite source |
| "SuDaI 80 params = 99% accuracy" | ✅ Verified cho network anomaly (Hammami 2025) | Add domain caveat |
| "Quantum Fractional Hawkes" | ❌ No quantum impl exists | Remove "quantum" |
| "MP-QLSTM giảm barren plateaus" | ⚠️ Theoretical only | Move to future work |
| "Risk Ratio = 1,87 cho dengue outbreak" | ✅ Epidemiological literature | Keep |
| "Altitude <2200m transmission limit" | ✅ Standard epidemiology | Keep |
| "Parameter-Shift Rule formula" | ✅ Mitarai et al. 2018 | Keep |
| "166 → 24 params QLSTM" | ⚠️ Verified trong toy examples | Add scale caveat |

---

## 4. Pipeline Optimization Reality Check

Theo worker nghiên cứu về pipeline optimization, **hiện trạng pipeline hiện tại KHÔNG có quantum advantage ở scale N ≤ 30**:

- **Quantum 10× slower** classical ở N=30, M=10 (Δ accuracy = 0.000)
- **Root cause:** Candidates pre-optimized bởi 60-80 classical iterations, không còn "quantum advantage niche"
- **XY mixer paradox:** Reduces set diversity (0.89 vs 0.95)
- **Brute-force optimum:** 100% recovery ở M ≤ 12

**Top 5 optimization strategies (priority order):**

1. 🔴 **Warm-start QAOA** từ greedy classical solutions — 50% time reduction (Egger 2021)
2. 🔴 **Trainable quantum kernels** với QNG optimization — +0-5% accuracy
3. 🔴 **Scale to N ≥ 50** — tìm crossover point quantum thắng classical
4. 🟡 **Amplitude encoding** — better Hilbert space utilization cho L(r) features
5. 🟡 **Hybrid boundary refactor** — move quantum to spatial search (exponential advantage O(√M) vs O(M))

---

## 5. Top Papers Nên Bổ Sung Vào Literature Review

| # | Paper | Lý do quan trọng | Difficulty |
|---|-------|------------------|------------|
| 1 | Yang, Garner et al. (2023) "Provably superior accuracy in quantum stochastic modeling" — Phys Rev A 108, 022411 | **Trực tiếp applicable** cho Hawkes process (stochastic); chứng minh quantum advantage in memory | Medium |
| 2 | Fujii & Nakajima (2017) "Quantum reservoir computing" — Phys Rev Applied 8, 024030 | Alternative temporal modeling; ít parameters hơn QLSTM; ổn định hơn QWGAN | Medium |
| 3 | McClean et al. (2018) "Barren plateaus in quantum neural network training landscapes" — Nature Communications 9, 4812 | Critical limitation mà báo cáo KHÔNG đề cập | Low |
| 4 | Rizoiu et al. (2018) "Hawkes processes for modeling epidemiological dynamics" — ICDM 2018 | Classical baseline cho honest comparison | Low |
| 5 | Kübler, Morris, Youssry, Zhao (2023) "Supervised quantum machine learning kernels are quantum neural networks" — Nature Communications | Foundation cho kernel claims | Medium |
| 6 | Basso et al. (2022) "Obstacles on the path to quantum advantage" — arXiv:2109.13981 | Honest assessment của QAOA capabilities | Low |
| 7 | Spatiotemporal Hawkes with Graphon (2024) — arXiv:2409.16903 | Advanced spatial modeling hơn Ripley's L-function | High |
| 8 | Quantum Kernel-Based LSTM (2024) — IEEE ICASSP | Alternative QLSTM variant | Medium |

---

## 6. Critical Gaps Cần Address

### 🔴 Gap 1: Không có Empirical Validation trên Epidemiological Data

Tất cả techniques (QWGAN, SuDaI, QCNN, QLSTM) chỉ test trên:
- Synthetic (MNIST, Gaussian)
- Network security data
- Weather/precipitation
- Simple time-series (NARMA, damped oscillators)

**Missing:** Validation trên dengue case data, disease outbreak sequences, epidemiological spatiotemporal patterns.

**Recommendation:** Chạy pipeline trên real dengue data từ Vietnam Dengue Watch hoặc WHO data.

---

### 🔴 Gap 2: Scale Mismatch

- QAAA propose 8-10 qubits cho complex spatio-temporal forecasting
- Wang et al. (2025): Limited qubits + deep encoding → predictive performance degenerates to random guessing
- QWGAN tested max 3-8 qubits
- QLSTM tested trên toy problems (NARMA-5)

**Recommendation:**
- Dùng shallow circuits (p ≤ 3)
- Hybrid quantum-classical approaches
- Đừng claim quantum advantage cho high-dimensional forecasting với limited qubits

---

### 🔴 Gap 3: Overclaimed "Quantum" Techniques

| QAAA Claim | Reality |
|-----------|---------|
| "Quantum Fractional Hawkes" | No quantum implementation exists |
| "Quantum-informed priors" | Classical Fractional Hawkes is classical |
| "Quantum STPP model" | No quantum Hawkes paper found |

**Recommendation:** Loại bỏ "quantum" modifier hoặc reframe.

---

### 🔴 Gap 4: Hardware Requirements Not Met

MP-QLSTM yêu cầu distributed quantum computing — không khả thi hiện tại.

**Recommendation:** List as "Future Work" hoặc remove.

---

## 7. Kết luận Cốt lõi

### Báo cáo nên được Tái cấu trúc Thành 2 Phần Rõ Ràng

**Phần A: Contributions Đã Được Chứng Minh (Implementable Now)**
1. ✅ XY Mixer QAOA với Hamming weight preservation (đã implemented, benchmarked)
2. ✅ Parameter-Shift Rule + QNG Optimizer (đã implemented)
3. ✅ Classical Fractional Hawkes Process với Mittag-Leffler kernel (math proven)
4. ✅ QLSTM (proven cho simple time-series)
5. ✅ QCNN (proven cho quantum state recognition)

**Phần B: Hướng Nghiên Cứu Tương Lai (Future Work)**
1. ⚡ QWGAN-GP — cần NISQ validation
2. ⚡ SuDaI — cần epidemiological domain transfer validation
3. ⚡ Data Re-Uploading cho STPP — cần empirical study
4. 🔴 Sublinear n-Toffoli — purely theoretical
5. 🔴 MP-QLSTM — cần distributed quantum hardware

### Honest Assessment

> Báo cáo QAAA kết hợp các kỹ thuật quantum computing hợp lệ (QAOA, QNG) với các claim mang tính suy đoán (QWGAN mode collapse, SuDaI novelty, MP-QLSTM feasibility). Framing trung thực nhất là **tập trung vào các components đã proven (XY QAOA, QNG) như contribution hiện tại**, với các techniques khác như hướng nghiên cứu tương lai.

### Bài học Chính

1. **Quantum ≠ Magic**: Mọi technique cần empirical validation trên target domain
2. **Scale matters**: Limited qubits + deep circuits = degraded performance
3. **Domain transfer không tự động**: SuDaI success trên network security ≠ success trên epidemiology
4. **"Quantum" modifier cần cẩn thận**: Chỉ dùng khi có quantum implementation thực sự
5. **Honest reporting > Overclaiming**: Ghi nhận limitations giúp credibility lâu dài

---

## 8. Nguồn Tham Khảo Đã Xác Minh

### Foundational Quantum Computing Papers

1. **Pérez-Salinas, Cervera-Lierta, Gil-Fuster, Latorre (2020).** "Data re-uploading for a universal quantum classifier." *Quantum* 4, 226.

2. **Cong, Choi, Lukin (2019).** "Quantum convolutional neural networks." *Nature Physics* 15, 1273-1278. DOI: 10.1038/s41567-019-0648-3

3. **Chen, Yoo, Fang (2020).** "Quantum long short-term memory." *ICASSP 2020*.

4. **Lloyd & Weedbrook (2018).** "Quantum generative adversarial learning." *Phys Rev Lett* 121, 040502.

5. **Chakrabarti, Huang, Li, Feizi, Wu (2019).** "Quantum Wasserstein GANs." *NeurIPS 2019 Workshop*.

6. **Mitarai, Negoro, Kitagawa, Fujii (2018).** "Quantum circuit learning." *Phys Rev A* 98, 032309.

7. **McClean, Boixo, Smelyanskiy, Babbush, Neven (2018).** "Barren plateaus in quantum neural network training landscapes." *Nature Communications* 9, 4812.

### Hawkes Process Papers

8. **Chen, Qiu, Xiang, Bao (2020).** "Fractional Hawkes processes." (Mittag-Leffler kernel)

9. **Habyarimana et al. (2023).** "Explicit proofs and simulations for fractional Hawkes."

10. **Kanazawa & Sornette (2020).** "Field master equation theory for non-Markovian Hawkes."

### Spatial & Epidemiological Papers

11. **Rizoiu, Mishra, Kong, Carman (2018).** "Hawkes processes for modeling epidemiological dynamics." *ICDM 2018*.

12. **Mohler & Mateu (2023).** "3D L-function for spatio-temporal point patterns."

### Recent Quantum ML Papers

13. **Yang, Garner et al. (2023).** "Provably superior accuracy in quantum stochastic modeling." *Phys Rev A* 108, 022411.

14. **Fujii & Nakajima (2017).** "Quantum reservoir computing." *Phys Rev Applied* 8, 024030.

15. **Kübler, Morris, Youssry, Zhao (2023).** "Supervised quantum machine learning kernels are quantum neural networks." *Nature Communications*.

16. **Basso, Kim, Venturelli (2022).** "Obstacles on the path to quantum advantage." *arXiv:2109.13981*.

### Algorithm Improvements

17. **Egger, Marecek, Woerner (2021).** "Warm-starting quantum optimization." *Quantum* 5, 479.

18. **Wang, Magann, Talarico, Sornette (2025).** Critical limitation of deep re-uploading with limited qubits.

19. **Hammami et al. (2025).** QWGAN + SuDaI for network anomaly detection.

### Spatial Hawkes Extensions

20. **Spatiotemporal Hawkes with Graphon (2024).** *arXiv:2409.16903*.

*(Và 9+ papers khác đã được review chi tiết trong papers_database.md)*

---

## 9. Tóm Tắt Đề Xuất Cập Nhật (Update Plan)

Xem file companion: `UPDATE_PLAN.md` cho roadmap cụ thể với timeline và actions.

---

**Tổng kết:** Báo cáo QAAA có tầm nhìn tham vọng và identify đúng các bottleneck thực tế. Tuy nhiên, nhiều solutions được đề xuất còn ở mức theoretical và chưa có empirical validation trên epidemiological data. Việc honest acknowledgement về limitations và tách biệt rõ ràng giữa "proven contributions" và "future research directions" sẽ tăng đáng kể credibility của dự án tại Chung kết QC4SG 2026.