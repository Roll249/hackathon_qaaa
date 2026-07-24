# Paradigm Shift: Tại sao dự đoán lan truyền dịch là BẤT KHẢ (và tại sao lượng tử giải quyết được bài toán NGƯỢC LẠI)

## 1. Bài toán "lan truyền dịch sang vùng lân cận" — tại sao luôn thất bại?

### 1.1 Cách tiếp cận cũ (đã đổ vỡ nhiều lần)

```python
# Ý tưởng: "Xã A có dịch → xã B lân cận có dịch không?"
predictions = model.predict_propagation(historical_data)
# → Accuracy thấp, Lyapunov chaos, FPT < 2 weeks
```

### 1.2 Ba lý do thất bại

| # | Lý do | Hệ quả |
|---|-------|--------|
| 1 | **Lyapunov exponent dương** (chaos) | Sai số initial conditions tăng exponential |
| 2 | **Information-theoretic limits** | Heterogeneity của social network giới hạn predictability |
| 3 | **Stochastic peak position** | Uncertainty tối đa quanh peak — không thể predict chính xác vị trí |

### 1.3 Bài học xương máu

> **Nếu cố dự đoán "từ dữ liệu cũ, các khu vực lân cận khác có dịch không", ta đang cố vượt qua chaos theory.** Bất khả.

---

## 2. Bản chất lượng tử giải quyết bài toán NGƯỢC LẠI

### 2.1 Thay vì dự đoán lan truyền → tìm peaks từ chính dữ liệu lịch sử

**Câu hỏi mới:** "Với dữ liệu lịch sử đã có, đâu là **điểm nóng nhất** ngay tại thời điểm hiện tại?"

→ Đây là bài toán **Quantum Maximum Finding** (Dürr-Høyer), không phải prediction.

### 2.2 Vì sao lượng tử tự nhiên phù hợp?

**Cơ chế phase kickback / amplitude amplification:**

```
Initial state (uniform superposition):
  |ψ_0⟩ = (1/√N) Σ_i |i⟩

Sau Grover iterations khoảng √N lần:
  - Amplitude ở peaks (high risk) → KHUẾCH ĐẠI
  - Amplitude ở valleys (low risk) → TRIỆT TIÊU
  - Measurement → xác suất rơi vào peaks ≈ 1
```

**Đây chính là "cá leo cây" bro nói:**
- Peaks = cây (high risk communes)
- Valleys = mặt nước (low risk, không quan tâm)
- Lượng tử **TỰ ĐỘNG** triệt tiêu noise, không cần explicit prediction!

### 2.3 So sánh hai paradigm

| | Cũ (failed) | Mới (quantum-native) |
|---|---|---|
| **Câu hỏi** | "Xã B có dịch không?" | "Xã nào có chỉ số cao nhất?" |
| **Phương pháp** | Time-series forecasting | Maximum finding trên snapshot |
| **Quantum role** | Không có / forcé | **Native**: triệt tiêu noise tự nhiên |
| **Đầu vào** | Time series | Risk intensity map (single snapshot) |
| **Đầu ra** | Future probability | Top-K communes NGAY BÂY GIỜ |
| **Khả thi?** | ❌ Chaos, FPT < 2w | ✅ O(√N) queries, deterministic |

---

## 3. Kiến trúc pipeline mới

### 3.1 Tổng quan 5 modules

```
┌─────────────────────────────────────────────────────────────┐
│         q_dengue_epidemiology PIPELINE (v19 NEW)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] DATA LAYER                                             │
│       └─ Historical dengue cases × 130 communes             │
│       └─ Static risk scores (mật độ ca, vector index, ...)  │
│                                                             │
│  [2] GRAPH LAYER (Classical GIS preprocessing)              │
│       └─ Build adjacency matrix từ OSM roads                │
│       └─ Weight bằng travel distance (Pha Din, QL6, QL12)  │
│       └─ Output: weighted graph G = (V, E, W)               │
│                                                             │
│  [3] QPIE ENCODER                                           │
│       └─ Embed risk intensity vào amplitudes                │
│       └─ Normalize → statevector với peaks = hotspots       │
│       └─ Cost: log₂(N) qubits                               │
│                                                             │
│  [4] QUANTUM MAX FINDER (Dürr-Høyer)                       │
│       └─ Recursive threshold update                         │
│       └─ Oracle marks anything ABOVE dynamic threshold      │
│       └─ Grover amplification → finds MAX in O(√N)         │
│                                                             │
│  [5] LACKADAISICAL QUANTUM WALK (multi-hotspot)            │
│       └─ Find TOP-K peaks, không phải 1                     │
│       └─ Self-loop parameter l = K/N                        │
│       └─ Coin operator C_i = diag(√l, √(1-l)) on marked    │
│                                                             │
│  OUTPUT: List of top-K communes + their risk scores         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Hybrid philosophy

- **Classical** làm cái classical giỏi: GIS, graph extraction, data normalization
- **Quantum** làm cái quantum giỏi: tìm maximum trong O(√N)
- **KHÔNG ép quantum vào prediction** (đã thất bại)
- **KHÔNG ép classical vào exponential search**

---

## 4. Tại sao cách tiếp cận này KHÔNG phải "trick"

### 4.1 Không phải benchmark suông

- Có **real dataset** (130 communes Điện Biên)
- Có **real decision** (can thiệp ở đâu)
- Có **honest assessment** (chỉ O(√N) queries, không claim wall-clock speedup)

### 4.2 Không phải quantum forcing

- Bài toán **maximum finding** là quantum-native (Dürr-Høyer)
- QPIE encoding vừa tiết kiệm qubit, vừa khớp với Grover
- Graph encoding phản ánh **cấu trúc thật** của epidemic spread

### 4.3 Bám sát thực tế

| Tỉnh thành VN | Số xã/phường | Số qubit cần |
|---------------|--------------|---------------|
| Điện Biên | 130 | ⌈log₂(130)⌉ = 8 |
| Hà Nội | 579 | ⌈log₂(579)⌉ = 10 |
| TP.HCM | 312 phường | ⌈log₂(312)⌉ = 9 |
| Cả nước (63 tỉnh) | ~10,000 | ⌈log₂(10k)⌉ = 14 |

→ Tất cả đều **trong tầm tay của NISQ devices** (IBM, IonQ đã có 100+ qubits).

---

## 5. Kết luận

> **"Bắt cá leo cây"** = Lợi dụng **đặc tính triệt tiêu noise tự nhiên** của amplitude amplification để tìm peaks, thay vì cố predict lan truyền.

Pipeline này:
1. **Honest**: chỉ claim O(√N) queries, không wall-clock advantage
2. **Practical**: áp dụng được cho 63 tỉnh thành VN
3. **Quantum-native**: dùng algorithm mà classical không làm được (max finding in O(√N))
4. **Real-world**: trả lời câu hỏi dịch tễ thực tế

Implementation ở `q_dengue_epidemiology/`.
