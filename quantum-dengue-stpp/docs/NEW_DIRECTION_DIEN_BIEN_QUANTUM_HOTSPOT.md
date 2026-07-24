# Quantum Epidemiology for Dien Bien Province: A Paradigm Shift from Benchmarking to Real-World Disease Hotspot Detection

**Author:** Quantum Dengue-STPP Research Team  
**Version:** v19 - New Direction  
**Date:** July 2026  

---

## 1. Tầm Nhìn Mới: Từ "Đo Speedup" sang "Trả Lời Câu Hỏi Dịch Tễ"

### 1.1 Tại Sao Cách Tiếp Cận Cũ Chưa Đủ

Các báo cáo trước đây trong repo này tập trung vào việc đo lường **quantum speedup** thông qua benchmark: Grover vs. scan cổ điển, QAOA vs. greedy, ESN vs. quantum reservoir. Đây là những bước đầu cần thiết, nhưng chúng ta đang ở thời điểm cần một **bước nhảy về chất**.

Lý do cốt lõi: việc đo speedup trên synthetic grid 10×10 hoàn toàn KHÁC với việc trả lời câu hỏi thực tế: *"Xã nào ở Điện Biên có nguy cơ bùng phát cao nhất trong 2 tuần tới?"*

### 1.2 Quantum Epidemiology Thực Sự

**Định nghĩa mới:** Quantum epidemiology không phải là "chạy thuật toán quantum trên dữ liệu dịch tễ" — mà là thiết kế **representation và algorithm** tận dụng cấu trúc computational của bài toán dịch tễ.

Điện Biên là case study lý tưởng vì:
- **130 xã** với địa hình chia cắt mạnh bởi Pha Din, Keo Lôm
- **Mạng lưới giao thông bất đối xứng**: QL6, QL12, đường mòn biên giới
- **Dữ liệu lịch sử**: sốt rét, sốt xuất huyết đang có cảnh báo bùng phát

### 1.3 Positioning: Không Phải Toy Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM EPIDEMIOLOGY SPECTRUM                        │
├─────────────────────────────────────────────────────────────────────────┤
│  [Toy Grid] ───────→ [Synthetic Network] ───────→ [Real Province]       │
│       ↑                    ↑                        ↑                   │
│   Grover vs         QAOA vs Greedy           BÀI TOÁN THỰC TẾ          │
│   Classical           Benchmark              Dien Bien 130 xã           │
│                                                                         │
│   CÂU HỎI CŨ: "Có speedup không?"                                      │
│   CÂU HỎI MỚI: "Hotspot nào cần ưu tiên can thiệp?"                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kiến Trúc Đề Xuất: Hybrid Pipeline

### 2.1 Sơ Đồ Khối Tổng Quan

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID QUANTUM EPIDEMIOLOGY PIPELINE                  │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
    │   Classical   │      │     Graph    │      │      Quantum     │
    │      GIS      │ ───→ │  Extraction  │ ───→ │      Search      │
    │  Preprocessing│      │    (OSM)     │      │   (Dürr-Høyer)   │
    └──────────────┘      └──────────────┘      └──────────────────┘
          │                     │                       │
          ▼                     ▼                       ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
    │  - Population│      │  - Adjacency │      │  - Threshold     │
    │  - Elevation │      │    matrix    │      │    Oracle        │
    │  - Weather   │      │  - Edge      │      │  - Amplitude     │
    │  - Case hist │      │    weights   │      │    Encoding      │
    └──────────────┘      └──────────────┘      └──────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │                     OUTPUT: Prioritized Commune List             │
    │                    for Public Health Intervention                │
    └──────────────────────────────────────────────────────────────────┘
```

### 2.2 Tại Sao Cần Hybrid?

| Thành Phần | Classical giỏi | Quantum giỏi |
|------------|----------------|--------------|
| **Data collection** | GIS, database, API | — |
| **Graph construction** | OSM parsing, edge weighting | — |
| **Preprocessing** | Normalization, feature extraction | — |
| **Hotspot search** | Linear scan O(N) | **Quantum search O(√N)** |
| **Multi-hotspot** | Greedy clustering | **Lackadaisical Walk** |
| **Validation** | Statistical tests | — |

**Nguyên tắc:** Quantum chỉ replace phần mà nó thực sự giỏi — tìm kiếm trên không gian trạng thái lớn.

---

## 3. So Sánh 3 Phương Pháp Mã Hóa

### 3.1 Bảng So Sánh Chi Tiết

| Tiêu Chí | NEQR | Amplitude Encoding (QPIE) | Quantum Walk Graph (CTQW) |
|----------|------|---------------------------|--------------------------|
| **Số qubit cho 130 xã** | ⌈log₂(130)⌉×2 = 16 qubit (coord + state) | ⌈log₂(130)⌉ = 8 qubit | Tùy graph, ~15-20 qubit |
| **Độ chính xác** | Deterministic (binary basis) | Probabilistic (variational) | Deterministic (unitary) |
| **Khả năng encode severity** | ✓ Native (state qubits) | △ Encode vào amplitude | △ Phải design riêng |
| **Phù hợp địa hình Điện Biên** | ★★★☆☆ (grid-based, sai topology) | ★★★☆☆ (vector-based) | ★★★★★ (graph-native) |
| **Dürr-Høyer compatible** | ✓ Easy oracle | △ Phase-based | ✓✓ Native to graph |
| **Độ phức tạp Oracle** | O(N) | O(N) | O(N²) cho adjacency |
| **Hardware feasibility (NISQ)** | ★★★★☆ (shallow circuit) | ★★★☆☆ (deep variational) | ★★☆☆☆ (many-qubit) |

### 3.2 Đánh Giá Chi Tiết

#### NEQR (Novel Enhanced Quantum Representation)
- **Ưu điểm:** Deterministic, dễ implement, severity encode trực tiếp
- **Nhược điểm:** Dùng 2n qubit cho n features, basis encoding không reflect topology
- **Phù hợp:** Khi cần deterministic output và severity là binary

#### Amplitude Encoding (QPIE)
- **Ưu điểm:** Tối ưu số qubit, exponential compression
- **Nhược điểm:** Probabilistic, cần variational optimization, phase sensitivity
- **Phù hợp:** Khi data dimension cao, cần qubit-efficient

#### Quantum Walk Graph (CTQW/DTQW)
- **Ưu điểm:** **Topology-native** — đỉnh = xã, cạnh = đường giao thông
- **Nhược điểm:** Oracle phức tạp, nhiều qubit cho weighted graph
- **Phù hợp:** ★★★★★ **CHO ĐIỆN BIÊN** — vì địa hình chia cắt cần graph, không phải grid

### 3.3 Recommendation

> **Chọn Quantum Walk Graph + Dürr-Høyer** cho bài toán Dien Bien vì:
> 1. Topology phản ánh thực tế (thung lũng + đường đi)
> 2. Dürr-Høyer tìm maximum trên graph tự nhiên
> 3. Multi-hotspot bằng Lackadaisical Quantum Walk

---

## 4. Thuật Toán Cốt Lõi: Dürr-Høyer + Lackadaisical Quantum Walk

### 4.1 Dürr-Høyer Algorithm: Quantum Minimum/Maximum Finding

**Bài toán:** Tìm xã có chỉ số nguy cơ cao nhất trong 130 xã.

**Độ phức tạp:** O(√N) oracle queries so với O(N) classical scan.

#### Math Foundation

```math
\begin{aligned}
&\text{Gọi } f: \{0,1\}^n \rightarrow \mathbb{R} \text{ là hàm risk score cho mỗi xã} \\
&\text{Mục tiêu: tìm } x^* = \arg\max_{x \in \{0,1\}^n} f(x) \\
&\text{Quantum Oracle: } O_f|x\rangle|q\rangle = |x\rangle|q \oplus f(x)\rangle \\
&\text{Grover Diffusion: } D = 2|s\rangle\langle s| - I \\
\end{aligned}
```

#### Recursive Threshold Update

```
Dürr-Høyer(xác suất p, ngưỡng t):
    
    m = ceil(1.45 * (12/5)^k)  // số iterations tăng theo k
    
    FOR i = 1 to m:
        // 1. Superposition
        |ψ⟩ = H^{⊗n}|0⟩^{⊗n}
        
        // 2. Phase estimation để estimate f(x)
        // 3. Marking: flip phase nếu f(x) > t
        O_t|ψ⟩ = (-1)^{[f(x)>t]}|ψ⟩
        
        // 4. Diffusion để amplify marked states
        D(O_t|ψ⟩)
        
        // 5. Measure và update t
        x_measured = measure()
        t = max(t, f(x_measured))
    
    RETURN t, x_measured
```

**Điểm then chốt:** Thay vì tìm trực tiếp maximum, ta tìm **ngưỡng tối thiểu** mà có thể exceed với xác suất p, sau đó tăng dần.

### 4.2 Lackadaisical Quantum Walk cho Multi-Hotspot

**Vấn đề:** Dürr-Høyer tìm được 1 hotspot. Thực tế có thể có 3-5 hotspot ở Điện Biên.

**Giải pháp:** Lackadaisical Quantum Walk (LQW) — thêm self-loops với weight l.

```math
\begin{aligned}
&\text{Laplacian với self-loops: } L_{lz} = L + lI \\
&\text{Coin operator: } S = (2\pi/2) \otimes H \text{ với self-loop probability } \alpha = l/(d+l) \\
&\text{Evolution: } U = S \cdot e^{-i\gamma L_{lz} t} \\
\end{aligned}
```

**Kết quả:** LQW phân tán probability vào nhiều đỉnh gần nhau → phát hiện cluster của hotspots thay vì 1 đỉnh cô lập.

### 4.3 Pseudocode: Combined Algorithm

```python
def quantum_epidemiology_hotspot(communes, graph, case_data, weather):
    # Phase 1: Classical Preprocessing
    risk_scores = compute_risk_score(communes, case_data, weather)
    normalized_scores = normalize(risk_scores)
    
    # Phase 2: Graph Construction (Classical)
    adjacency = build_graph(communes, roads=OSM_data)
    weights = compute_edge_weights(adjacency, elevation)
    
    # Phase 3: Quantum Search
    n_qubits = ceil(log2(130))  # = 8 qubits
    
    # 3a. Amplitude encoding của risk scores
    amplitudes = encode_amplitudes(normalized_scores)  # size 2^8
    
    # 3b. Dürr-Høyer với dynamic threshold
    p = 2/3
    t = 0
    for k in range(1, 5):
        m = ceil(1.45 * (12/5)^k)
        for _ in range(m):
            # Apply quantum walk on graph
            state = apply_CTQW(amplitudes, adjacency, weights)
            # Mark and diffuse
            state = mark_above_threshold(state, t)
            state = diffuse(state)
        
        t = max(t, measure_and_update(state, risk_scores))
    
    # Phase 4: LQW cho multi-hotspot
    clusters = apply_LQW(amplitudes, adjacency, self_loop_weight=0.1)
    
    RETURN rank_communes_by_risk(clusters, t)
```

---

## 5. Case Study: 130 Xã Điện Biên

### 5.1 Dataset Requirements

| Data Type | Source | Resolution | Use Case |
|-----------|--------|------------|----------|
| **Dân số** | GSO Vietnam | Xã-level | Risk weighting |
| **Ca bệnh lịch sử** | DPI (Điện Biên CDC) | 2019-2025 | Training baseline |
| **Thời tiết** | NCHMF | Daily, xã-level | Fever correlation |
| **Địa hình** | SRTM/DEM | 30m resolution | Elevation features |
| **Mạng lưới đường** | OSM | Vector | Graph construction |
| **Di chuyển dân** | Telecom CDR | Aggregate | Mobility matrix |

### 5.2 Quantum Resource Estimate

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM RESOURCE ESTIMATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem size: 130 communes                                            │
│  Hilbert space: 2^8 = 256 dimensions (8 qubits)                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    QUBIT BUDGET                                 │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  Data qubits:        8 qubits (amplitude encoding)            │    │
│  │  Ancilla (phase est): 4 qubits (iterative)                    │    │
│  │  Measurement qubit:   1 qubit                                   │    │
│  │  ────────────────────────────────────────────────────────────   │    │
│  │  TOTAL (NISQ):      ~13-15 qubits                              │    │
│  │  TOTAL (FTQC):      ~50-100 qubits (error correction)         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Circuit depth estimate:                                               │
│  - Amplitude encoding:  O(n)  ≈ 50-100 gates                           │
│  - Grover diffusion:   O(n)  ≈ 100-200 gates                          │
│  - Phase estimation:   O(n²) ≈ 500-1000 gates                          │
│  ──────────────────────────────────────────────────────────────────     │
│  TOTAL: ~1000-2000 gates (within NISQ capability)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 So Sánh với Classical Epidemic Forecasting

| Phương pháp | Ưu điểm | Nhược điểm | Độ phức tạp |
|-------------|---------|------------|-------------|
| **SEIR Compartmental** | Well-studied, interpretable | Uniform mixing assumption | O(N) params |
| **Agent-Based Model** | Realistic mobility | Computationally heavy | O(N×T) |
| **Quantum Hotspot (OURS)** | O(√N) search, graph-native | Requires quantum hardware | O(√N) queries |
| **Classical GIS Scan** | Simple, robust | O(N) scan, ignores topology | O(N) |

**Key insight:** Với 130 xã, O(√N) = O(11) vs O(N) = O(130) — quantum advantage **nhỏ nhưng measurable**. Lợi ích thực sự là **graph representation** phản ánh topology địa hình.

---

## 6. Đổi Mới Về Tư Duy (Paradigm Shifts)

### 6.1 5 Paradigm Shifts

| # | TỪ | ĐẾN | Lý do |
|---|-----|-----|-------|
| 1 | "Quantum advantage = speedup factor" | "Quantum advantage = actionable insight" | Speedup không matter nếu output không action được |
| 2 | "Uniform grid representation" | "Irregular graph ( topology-aware)" | Điện Biên không phải grid phẳng |
| 3 | "Single hotspot detection" | "Multi-hotspot clustering với LQW" | Thực tế có 3-5 cluster đồng thời |
| 4 | "Binary oracle (mark/unmark)" | "Dynamic threshold Oracle (Dürr-Høyer)" | Risk là continuous, không phải binary |
| 5 | "Benchmark trên synthetic data" | "Deploy-ready cho 1 tỉnh cụ thể" | Từ research → impact thực sự |

### 6.2 Tại Sao Những Thay Đổi Này Quan Trọng?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TẠI SAO PARADIGM SHIFT?                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  OLD PARADIGM (Grover vs Classical):                                   │
│  ─────────────────────────────────────                                  │
│  "Chúng tôi tìm hotspot O(√N) thay vì O(N)"                            │
│  → Benchmark: irrelevant với public health workers                      │
│                                                                         │
│  NEW PARADIGM (Quantum Epidemiology):                                   │
│  ─────────────────────────────────────                                  │
│  "Xã Mường Pồn, xã Pa Thơm, xã Chung Chải có nguy cơ cao nhất"        │
│  → Actionable: gửi team y tế đến đúng nơi cần                         │
│                                                                         │
│  Quantum speedup TRỞ THÀNH CÔNG CỤ, không phải MỤC TIÊU                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Roadmap Thực Hiện

### Phase Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    8-WEEK IMPLEMENTATION ROADMAP                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Week 1-2: Data Collection                                             │
│  ├── Thu thập dữ liệu dân số 130 xã (GSO)                             │
│  ├── Thu thập ca bệnh lịch sử 2019-2025 (DPI CDC)                     │
│  └── Parse OSM data cho mạng lưới đường                                │
│                                                                         │
│  Week 3: Graph Construction                                            │
│  ├── Build adjacency matrix từ OSM                                     │
│  ├── Weight edges bằng elevation + distance                            │
│  └── Validate graph properties (connected, weighted)                   │
│                                                                         │
│  Week 4-6: Quantum Algorithm Implementation                            │
│  ├── Implement Dürr-Høyer trên PennyLane                               │
│  ├── Implement CTQW/DTQW cho graph                                     │
│  ├── Implement LQW cho multi-hotspot                                   │
│  └── Test trên simulator (default.qubit)                               │
│                                                                         │
│  Week 7: Validation                                                    │
│  ├── So sánh với SEIR baseline                                         │
│  ├── Backtest trên historical outbreak (2023)                          │
│  └── Statistical significance testing                                   │
│                                                                         │
│  Week 8: Visualization & Deployment                                     │
│  ├── Dashboard cho public health workers                               │
│  ├── Bản đồ risk score 130 xã                                         │
│  └── Documentation + User guide                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Deliverables Mỗi Phase

| Phase | Deliverable | Success Criteria |
|-------|-------------|------------------|
| 1 | Dataset file (JSON/CSV) | 130 xã × 10 features |
| 2 | Graph file (GraphML) | 130 nodes, edges > 200 |
| 3 | Quantum circuit (PennyLane) | Chạy được trên simulator |
| 4 | Validation report | p-value < 0.05 vs baseline |
| 5 | Interactive dashboard | < 5 clicks đến kết quả |

---

## 8. Limitations & Honest Assessment

### 8.1 Hardware Limitations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HONEST ASSESSMENT: WHAT WE CAN'T DO YET              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ✗ Fault-Tolerant Quantum Computing                                    │
│    → Hiện tại chỉ chạy được trên simulator                             │
│    → NISQ devices: noise sẽ degrade kết quả                           │
│    → FTQC cần ~1000 logical qubits (hiện chỉ có ~100)                  │
│                                                                         │
│  ✗ Large Problem Sizes                                                 │
│    → 130 xã: quá nhỏ để thấy rõ quantum advantage                      │
│    → Cần ~10,000+ nodes để O(√N) vs O(N) thực sự khác biệt             │
│                                                                         │
│  ✗ Deterministic Output                                                │
│    → Quantum measurement inherently probabilistic                      │
│    → Cần nhiều shots để có confidence                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Scientific Limitations

| Limitation | Mức độ | Mitigation |
|------------|--------|------------|
| Classical baseline mạnh | High | Agent-based model + SEIR ensemble |
| Risk score là proxy | Medium | Dùng actual case data để validate |
| Graph construction có thể sai | Medium | Expert validation từ DPI |
| Overfitting vào historical | Low | Cross-validation trên nhiều năm |

### 8.3 Risk Assessment

> **Risk cao nhất:** Over-engineering cho bài toán có thể giải bằng classical GIS scan đơn giản.
>
> **Mitigation:** Nếu quantum + graph không outperform simple spatial autocorrelation, quay lại classical approach. Quantum là means, không phải ends.

---

## 9. Kết Luận

### Tóm Tắt Đóng Góp

1. **Paradigm shift:** Từ "benchmark quantum speedup" → "real-world quantum epidemiology"
2. **Topology-aware:** Graph encoding phản ánh địa hình thực Điện Biên, không phải grid giả tạo
3. **Multi-hotspot:** Lackadaisical Quantum Walk phát hiện cluster, không chỉ 1 đỉnh
4. **Actionable output:** Ưu tiên xã cụ thể cho can thiệp y tế

### Tại Sao Hướng Này Đáng Theo Đuổi?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHY THIS DIRECTION MATTERS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Điện Biên có 130 xã với địa hình chia cắt — GRID SAI, GRAPH ĐÚNG   │
│                                                                         │
│  2. Sốt xuất huyết đang bùng phát Gia Lai, phá vỡ mọi quy luật —     │
│     cần CÔNG CỤ MỚI, không phải classical rules                         │
│                                                                         │
│  3. Quantum không phải magic bullet, nhưng GRAPH NATIVE SEARCH         │
│     có thể outperform classical scan khi topology phức tạp             │
│                                                                         │
│  4. 8-13 qubits cho 130 xã — TRONG TẦM NISQ, deploy được ngay          │
│                                                                         │
│  5. Nếu thành công: mở ra hướng quantum epidemiology cho 63 tỉnh      │
│     thành khác của Việt Nam                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bước Tiếp Theo

**Ngay lập tức:**
1. Liên hệ DPI CDC để request historical dengue data
2. Parse OSM cho Điện Biên (đã có QL6, QL12, đường biên)
3. Implement baseline: classical GIS scan + SEIR model

**Sau 4 tuần:**
1. Benchmark: Quantum vs Classical trên cùng dataset
2. Nếu quantum không outperform → document và pivot

---

## References

1. Dürr, C., & Høyer, P. (1996). A quantum algorithm for finding the minimum. *arXiv:quant-ph/9607014*

2. Childs, A. M., & Goldstone, J. (2004). Spatial search by quantum walk. *Physical Review A, 70(4)*, 042314.

3. Ambainis, A. (2004). Quantum walk algorithm for element distinctness. *SIAM Journal on Computing, 37*(1), 210-239.

4. Portugal, R. (2013). *Quantum Walk and Search Algorithms* (2nd ed.). Springer.

5. Janmark, H., Meyer, D. A., & Wong, T. G. (2014). Global symmetry is unnecessary for fast quantum searching. *Physical Review Letters, 112*(21), 210502.

6. Wong, T. G. (2015). Approaching the optimal cost of quantum search. *Quantum Information Processing, 14*(6), 2019-2025.

7. OSM Contributors. (2024). OpenStreetMap Vietnam. *OpenStreetMap Foundation*.

8. Scarpino, S. V., & Petri, G. (2019). Limitations of predictability in epidemic dynamics. *arXiv:1912.07736*.

9. GSO Vietnam. (2023). *General Statistics Office of Vietnam - Dien Bien Province Data*.

10. Vietnam Ministry of Health. (2025). *Dengue Fever Surveillance Report, Northern Highlands Region*.

---

*Document version: v19 - New Direction*  
*Project: Quantum Dengue-STPP*  
*Track: QC4SG 2026*
