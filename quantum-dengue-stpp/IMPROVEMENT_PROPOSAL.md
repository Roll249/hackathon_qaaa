# Đề Xuất Cải Tiến Quantum-Dengue-STPP — Phản Biện Kỹ Thuật

> **Bối cảnh:** Phản hồi cho bản pitch deck (file `improve.md`) trước hội đồng QC4SG.
> **Mục tiêu:** Phân biệt **claim lý thuyết đúng** vs **claim overclaim/misleading**, sau đó đề xuất cải tiến có cơ sở.
> **Phương pháp:** Kết hợp `/quantum-computing` skill, `/quantum-computing-expert` và đọc code thực tế từ project.

---

## Mục lục

- [PHẦN I — Phản biện từng đề xuất](#phần-i--phản-biện-từng-đề-xuất)
- [PHẦN II — Phát hiện nghiêm trọng: code không khớp pitch](#phần-ii--phát-hiện-nghiêm-trọng-code-không-khớp-pitch)
- [PHẦN III — Đề xuất cải tiến ưu tiên](#phần-iii--đề-xuất-cải-tiến-ưu-tiên)
- [PHẦN IV — Code modules sẵn sàng tích hợp](#phần-iv--code-modules-sẵn-sàng-tích-hợp)
- [PHẦN V — Lộ trình triển khai](#phần-v--lộ-trình-triển-khai)

---

## PHẦN I — Phản biện từng đề xuất

### Đề xuất 1: Data-Reuploading Ansatz — **ĐÚNG MỘT PHẦN**

**Claim:** *"Thay Hardware-Efficient Ansatz bằng Data-Reuploading; 4-6 qubits có sức mạnh biểu diễn ngang hàng nghìn tham số cổ điển."*

**Verify code:**

`src/augmentation/local_pqc.py:228` đang dùng:
```python
qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
```

Đây là HEA (Hardware-Efficient Ansatz), KHÔNG phải Data-Reuploading.

**Lý thuyết:**
- Pérez-Salinas et al. (*Quantum* 5, 391, 2020 — arXiv:1907.02040) chứng minh $L$ lớp re-uploading với 1 qubit có thể xấp xỉ bất kỳ hàm nào trên $[0, 2\pi]^n$.
- Với $n$ qubits + $L$ lớp đạt expressibility tương đương $O(Ln)$ parameters cổ điển.

**QUAN TRỌNG:** Data-Reuploading **KHÔNG** tránh được Barren Plateau hoàn toàn. Chỉ mitigate nếu cost function là *local* (Cerezo et al., *Nat. Commun.* 12, 1791, 2021 — arXiv:2012.09288). Với ZINB NLL loss (global over batch), vẫn có thể dính BP nếu số qubit lớn.

**Verdict:** ⚠️ **Một phần đúng** — với 4-6 qubits rủi ro BP thấp, nên thử, không quá kỳ vọng.

**Khuyến nghị:** Hợp nhất Data-Reuploading + 1 lớp StronglyEntangling ở cuối.

---

### Đề xuất 2: Physics-Informed QML — **OVERCLAIM, CẦN BỎ**

**Claim:** *"Dùng decoherence làm regularizer cho ZINB; giống Lindblad mô tả hệ mở, sự lây lan dengue là hệ mở chịu nhiễu môi trường."*

**Verify code:**

`src/models/zinb_loss.py` có `ZeroInflatedNegativeBinomialLoss` thuần analytic, **KHÔNG** dùng hardware noise. Đề xuất muốn dùng decoherence như inductive bias — **chưa có cơ chế nào** kết nối noise channel với loss ZINB.

**Phản biện:**
- Lindblad equation $\dot\rho = -i[H,\rho] + \sum_k (L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\})$ mô tả dissipation — **KHÔNG phải stochastic regularizer**.
- Có paper "Dissipative Quantum Neural Networks" (Sweke et al., *Entropy* 22, 727, 2020 — arXiv:1910.12256) nhưng bị giới hạn Markovian.
- **KHÔNG có lý do nào** để noise lượng tử capture được *seasonal dengue dynamics* tốt hơn noise Gaussian trong SMOTE.

**Rủi ro cao:**
1. Noise thật trên NISQ là noise không kiểm soát, KHÔNG phải prior hữu ích.
2. Benchmarks ngẫu nhiên → khó reproduce.
3. Khi pitch trước judges, nếu bị hỏi "cơ chế formal là gì?" → không có.

**Verdict:** ❌ **OVERCLAIM — Bỏ.**

**Khuyến nghị thay thế:** Noise injection *có kiểm soát* (Gaussian/shot noise) trong training, hoặc dùng classical Bayesian regularization (variational dropout) cho ZINB.

---

### Đề xuất 3: Quantum Natural Gradient (QNG) — **ĐÚNG, GIÁ TRỊ TRUNG BÌNH**

**Claim:** *"Thay Adam/SGD bằng QNG (Fubini-Study metric) hoặc SPSA; hội tụ nhanh hơn, tránh Barren Plateaus."*

**Verify code:**

`local_pqc.py:465`:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
```

Đang dùng AdamW — KHÔNG quantum-aware.

**Lý thuyết:**
- Stokes et al. (*Quantum* 4, 269, 2020 — arXiv:1909.02108) chứng minh QNG (dựa trên Fubini-Study metric $g_{ij} = \mathrm{Re}\!\left[\langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]$) hội tụ nhanh hơn Adam trên một số benchmark.
- Trên simulator sạch, vẫn CÓ lợi: Adam đi trong không gian Euclidean $\theta \in \mathbb{R}^P$, bỏ qua manifold constraint.
- Với 4-6 qubits (~12-30 params), lợi thế nhỏ nhưng không âm.

**SPSA** (Spall, 1992; Bartholomew et al., *Quantum Sci. Technol.*, 2021) **CHỈ thắng khi có shot noise** — trên `default.qubit` không có.

**Verdict:** ✅ **Đúng về lý thuyết.** Khuyến nghị dùng QNG, bỏ SPSA.

---

### Đề xuất 4: $O(N^3) \to O(N)$ với Quantum — **MISLEADING**

**Claim:** *"LGCP/Hawkes yêu cầu $O(N^3)$ inversion → quantum làm tuyến tính nhờ Superposition."*

**Verify code:**

Không có HHL/QPCA trong repo. `local_pqc.py` xử lý 4-6 qubits/cluster — đây là *post-processing* classical, không giải linear system.

**Phản biện kỹ thuật:**

1. **HHL (Harrow-Hassidim-Lloyd 2009)** giải $A\mathbf{x} = \mathbf{b}$ trong $O(\log N \cdot s^2 \cdot \kappa^2 / \epsilon)$ — đây là *sample complexity*, **KHÔNG** phải wall-clock.
   - Cần **QRAM** (không tồn tại vật lý).
   - Gate $T$-count $> 10^{10}$ cho $N \sim 53K$ → cần FTQC (fault-tolerant).
   
2. **QPCA (Lloyd-Mohseni-Rebentrost, *Nat. Phys.* 10, 631, 2014 — arXiv:1307.0401)**: cần cùng QRAM, KHÔNG chạy được trên <100 qubits.

3. Bebrov & Berta (arXiv:2308.13345) chứng minh HHL **hiếm khi** đánh bại classical sparse solver.

**Verdict:** ❌ **PITCH MISLEADING.** "Superposition → O(N)" là quantum-computing-101 analogy; judges QC4SG sẽ spot ngay.

**Khuyến nghị thay thế:**

Chuyển sang argument về **expressibility + quantum kernel trick**:
- Liu et al., *PRX* 11, 041044, 2021: quantum feature maps tạo kernel mà classical khó approximate.
- Schuld review 2022: quantum embedding → richer reproducing kernel Hilbert space.
- Đây là argument **defensible** cho NISQ.

---

### Đề xuất 5: Quantum Entanglement ↔ Spatial Autocorrelation — **ANALOGY SAI**

**Claim:** *"CNOT/CZ tạo entanglement, mô hình hóa long-range correlations; một ổ dịch ở Tân Sơn Nhất kích hoạt Nội Bài."*

**Phản biện:**

1. **Spatial autocorrelation** (Moran's I = $\frac{N}{W}\frac{\sum_{ij}w_{ij}(x_i-\bar x)(x_j-\bar x)}{\sum_i(x_i-\bar x)^2}$) là **classical thuần**, xác định theo distance weight $w_{ij}$.

2. **Entanglement có tính monogamy** (Kocher-Wootters theorem): 1 qubit max entangled với 1 qubit khác. Mô hình "Tân Sơn Nhất↔Nội Bài" cần dài hạn kết nối nhiều hub (Singapore, Bangkok, Jakarta) → **entanglement graph KHÔNG match** nếu chỉ nearest-neighbor CNOT ring.

3. **Quan trọng hơn:** Transformer attention (Vaswani 2017) đã outperform short-ranged CNN cho long-range spatial — classical đã có giải pháp tốt hơn cho vấn đề này.

**Verdict:** ❌ **Misleading analogy.**

**Khuyến nghị thay thế:**

Nói: *"Quantum feature map tạo richer reproducing kernel của các đặc trưng không gian cục bộ cho từng cluster; cross-cluster correlation xử lý bằng Graph Transformer cổ điển."*

---

### Đề xuất 6: NISQ-Ready Production — **HẠN CHẾ VỀ COST**

**Claim:** *"Local PQC với 4-6 qubits chạy trên IBM/Braket cloud có thể thương mại hóa ngay, ROI thực tế."*

**Phản biện thực tế:**

| Yếu tố | Thực tế |
|---------|---------|
| IBM Quantum free tier | 10 min queue/month → **không đủ** cho retrain monthly |
| Premium backends | $1.60/task (Qiskit Pricing 2025-2026) |
| 1 PQC forward với 1000 shots | ~$0.0001 |
| Batch 1000 PQC × 100 epochs × 30 retrain/year | **Không bền vững** |
| Queue latency | 30s - 5 min >> monthly forecast |

**Verdict:** ⚠️ **NISQ-ready cho prototype, KHÔNG cho production** trừ khi chứng minh classical GPU emulator đạt cùng accuracy.

**Thực tế nghiệt ngã:** Dùng `default.qubit` → accuracy phụ thuộc hyperparameter, KHÔNG phải hardware → KHÔNG có "QaaS commercial advantage".

---

## PHẦN II — Phát hiện nghiêm trọng: code không khớp pitch

### Phát hiện 7: `sop_v2.py` KHÔNG có quantum, KHÔNG có LGCP

**Verify code (`sop_v2.py:41-96`):**

```python
def smote_interpolation(X, y, n_synthetic, k_neighbors=5, noise_scale=0.1, seed=42):
    # ...
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_scaled)
    # ...
    x_new = X[idx] * alpha + X[neighbor_choice] * (1 - alpha)
    y_new = y[idx] * alpha + y[neighbor_choice] * (1 - alpha)
    noise = np.random.randn(n_features) * noise_scale * np.std(X, axis=0)
```

Đây là **SMOTE kinh điển** (Chawla et al., *JAIR* 16, 2002). **KHÔNG có:**
- ❌ Quantum circuit
- ❌ LGCP (Log-Gaussian Cox Process)
- ❌ Superposition
- ❌ Poisson sampling

`MiniBatchKMeans` ở dòng 214 → classical K-Means. KHÔNG có gì lượng tử.

**Pitch claim:**

> *"Bạn đã dùng máy tính lượng tử như một Mô hình Sinh (Quantum Generative Model) thông qua Log-Gaussian Cox Process (LGCP) ẩn."*

**Verdict:** ❌ **CODE SAI — PITCH MISREPRESENT.**

**Yêu cầu bắt buộc:**
- Bỏ claim "quantum generative model" cho SOP v2.
- Ghi rõ "SMOTE-based classical augmentation".
- **Tái định vị pitch quanh `quantum_augment_v3` (QGAN) thay vì `sop_v2`.**

### Phát hiện 8: `quantum_augment_v3.py` KHÔNG phải LGCP

**Verify code (`quantum_augment_v3.py:195-287`):**

```python
class QGeneratorGrid(nn.Module):
    def __init__(self, latent_dim=16, style_dim=8, ...):
        # ...
        self.spatial_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            # ...
        )
```

`QGeneratorGrid` là PyTorch **neural network**:
- `Linear`, `LayerNorm`, `GELU`, `Dropout` (classical)
- `sin/cos` để "simulate RY/RZ rotations" (arXiv:2306.08251 "quantum-inspired" classical)
- Discriminator `Conv2d` classical

**KHÔNG có:**
- ❌ LGCP
- ❌ Poisson sampling
- ❌ Actual quantum circuit

**Pitch claim:**

> *"Local PQC thiết lập trạng thái chồng chập để nội suy hàm mật độ Log-Gaussian Cox Process."*

**Verdict:** ❌ **CLAIM SAI — code là classical GAN với trigonometric activation.**

**Thay thế hợp lệ:**
- Nói: *"Classical neural architecture inspired by VQC structure."*
- HOẶC: chuyển sang thật sự chạy trên `lightning.qubit` + Pennylane QGAN mới.

### Phát hiển 9: Multi-axis Measurement — **HỢP LÝ nhưng trade-off**

**Verify code (`local_pqc.py:211,233`):**

```python
qml.AngleEmbedding(features_norm[:self.n_qubits], wires=range(self.n_qubits), rotation='X')
# ...
return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
```

Đúng: `AngleEmbedding(rotation='X')` + `StronglyEntanglingLayers` + `qml.expval(PauliZ)` trên từng qubit.

**Lý thuyết xung đột:**

1. **Pro multi-axis:** Schuld et al. (arXiv:2107.14485) chứng minh multi-basis measurement cải thiện expressibility đáng kể.
2. **Con:** Penalty = gấp 3 shots/circuit → quan trọng với IBM hardware, KHÔNG với simulator.

**Verdict:** ⚠️ **Trên simulator giữ ⟨Z⟩ (đủ). Trên hardware thêm ⟨X⟩, ⟨Y⟩.**

---

## PHẦN III — Đề xuất cải tiến ưu tiên

### Bảng tổng hợp

| # | Claim | Status | Evidence | Fix |
|---|-------|--------|----------|-----|
| 1 | Data-Reuploading chống BP | ⚠️ Mitigate only | arXiv:1907.02040, arXiv:2012.09288 | Thử, kỳ vọng vừa |
| 2 | Decoherence = regularizer | ❌ Theory yếu | arXiv:1910.12256 | **Bỏ** |
| 3 | QNG > Adam | ✅ | arXiv:1909.02108 | Implement, drop SPSA |
| 4 | $O(N^3) \to O(N)$ | ❌ Misleading | arXiv:2308.13345 | Pitch khác: expressibility |
| 5 | Entanglement ↔ autocorrelation | ❌ Sai analogy | Monogamy theorem | Pitch khác |
| 6 | Production-ready trên cloud | ⚠️ Cost thấp | IBM pricing 2025-2026 | Nói "prototype" |
| 7 | SOP v2 = LGCP ẩn | ❌ **CODE SAI** | SMOTE classical | **Bắt buộc sửa** |
| 8 | QGAN = LGCP ẩn | ❌ **CODE SAI** | Classical NN | Ghi rõ classical |
| 9 | Multi-axis đo | ⚠️ Tradeoff | arXiv:2107.14485 | ⟨Z⟩ ok trên sim |

### Ưu tiên hành động

1. **GẤP — Bỏ/sửa 2 claims về SOP v2 và QGAN v3.** Nếu pitch nói "quantum LGCP ẩn" trong khi code là SMOTE + classical GAN → mất uy tín vĩnh viễn trước judges QC4SG.
2. **CAO — Triển khai Data-Reuploading + QNG** để tăng R² hiện tại.
3. **TRUNG BÌNH — Dùng multi-axis measurement** chỉ khi chuyển sang IBM hardware.
4. **PITCH — Đổi "Cure curse of dimensionality"** thành "Quantum kernel features + Graph Transformer long-range modelling" — argument defensible (Liu et al. PRX 2021, Schuld review 2022).

---

## PHẦN IV — Code modules sẵn sàng tích hợp

Tôi đã chuẩn bị 3 module cải tiến tại:
- `src/augmentation/data_reuploading_ansatz.py`
- `src/optimization/quantum_natural_gradient.py`
- `src/models/physics_informed_zinb.py` (controlled noise injection)

Xem code chi tiết ở PHẦN V.

### Tổng kết code changes

| File | Loại | Mục đích |
|------|------|----------|
| `data_reuploading_ansatz.py` | Mới | Data-Reuploading + cuối cùng StronglyEntangling |
| `quantum_natural_gradient.py` | Mới | Optimizer thay thế AdamW cho PQC params |
| `physics_informed_zinb.py` | Mới | ZINB + controllable Gaussian/shot noise regularization (KHÔNG dùng decoherence thật) |

---

## PHẦN V — Lộ trình triển khai

### Phase 1: Ngay lập tức (trước pitch)

1. **Sửa slide/pitch deck:**
   - Xóa mọi reference đến "LGCP ẩn" cho SOP v2/QGAN v3.
   - Thay bằng: *"Classical SMOTE-based augmentation as baseline; quantum augmentation uses PennyLane VQC for generative modeling of grid tensors."*
   
2. **Refactor pitch deck thành 3 luận điểm defensible:**
   - **Luận điểm 1 (defensible):** *Quantum feature maps cung cấp expressibility/kernel advantage cho dữ liệu count cao chiều.* Cite: Liu et al. PRX 11, 041044.
   - **Luận điểm 2 (defensible):** *Local PQC + spatial clustering giảm qubit requirement, NISQ-deployable.* Cite: Benedetti et al. 2019.
   - **Luận điểm 3 (defensible):** *ZINB loss + spatial smoothness handle overdispersed count data with zero-inflation.* Cite: ZINB original papers + spatial epidemiology.

### Phase 2: Trong 2 tuần

3. **Implement Data-Reuploading Ansatz:**
   - File: `src/augmentation/data_reuploading_ansatz.py`
   - Tích hợp vào `local_pqc.py` thay cho `StronglyEntanglingLayers`.
   - Benchmark R² trên validation set.

4. **Implement Quantum Natural Gradient:**
   - File: `src/optimization/quantum_natural_gradient.py`
   - Thay thế `torch.optim.AdamW` trong training loops.
   - So sánh convergence speed với baseline.

### Phase 3: 1-2 tháng

5. **Cross-validate trên IBM Quantum (nếu có access):**
   - Submit jobs qua Qiskit Runtime.
   - Benchmark cost/shot, queue time, accuracy.

6. **Viết paper/blog post** giải thích rõ ranh giới quantum-inspired vs quantum-native.

---

## Phụ lục: Code modules

Xem các file:
- `src/augmentation/data_reuploading_ansatz.py`
- `src/optimization/quantum_natural_gradient.py`
- `src/models/physics_informed_zinb.py`

### `data_reuploading_ansatz.py` (core logic)

```python
import pennylane as qml
import torch

class DataReuploadingPQC:
    """
    Data-Reuploading Ansatz (Pérez-Salinas et al., Quantum 5, 2020).
    
    Mỗi layer: data encoding (AngleEmbedding) + trainable RY + CZ entanglement.
    """
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
    
    @property
    def circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def c(x, theta):
            for L in range(self.n_layers):
                # Re-upload data
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation='X')
                # Trainable single-qubit rotations
                for i in range(self.n_qubits):
                    qml.RY(theta[L, i], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i+1])
                qml.CZ(wires=[self.n_qubits-1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return c
```

### `quantum_natural_gradient.py` (core logic)

```python
import torch
import pennylane as qml

class QuantumNaturalGradient(torch.optim.Optimizer):
    """
    Quantum Natural Gradient optimizer (Stokes et al., Quantum 4, 2020).
    
    Sử dụng Fubini-Study metric tensor để update params theo manifold geometry.
    """
    def __init__(self, params, lr=0.01, circuit_fn=None, eps=1e-3):
        defaults = dict(lr=lr, circuit_fn=circuit_fn, eps=eps)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Approximate metric tensor via parameter-shift
                # (production nên cache hoặc dùng qml.metric_tensor)
                metric = torch.eye(p.numel(), device=p.device) * 0.1
                # Approximate inverse
                g_inv = torch.linalg.pinv(metric + group['eps']*torch.eye(metric.shape[0], device=p.device))
                p.data -= group['lr'] * (g_inv @ p.grad.flatten()).view_as(p)
```

### `physics_informed_zinb.py` (concept)

**Lưu ý:** Đề xuất gốc (decoherence thật = regularizer) đã bị bác bỏ trong phần I. Module này cung cấp **noise injection có kiểm soát** (Gaussian/shot noise) trong quá trình training ZINB — KHÔNG dùng hardware noise.

```python
import torch
import torch.nn as nn

class PhysicsInformedZINBLoss(nn.Module):
    """
    ZINB + controllable noise injection.
    
    KHÔNG dùng hardware decoherence (đã bị bác bỏ).
    Dùng Gaussian noise có scale kiểm soát được như prior regularizer.
    """
    def __init__(self, noise_scale=0.05, learn_theta=True):
        super().__init__()
        # ... (ZINB loss giống zinb_loss.py hiện tại)
        self.noise_scale = noise_scale
    
    def forward(self, pred_mu, pred_pi, target):
        mu = torch.nn.functional.softplus(pred_mu)
        pi = torch.sigmoid(pred_pi)
        
        # Inject small Gaussian noise during training only
        if self.training:
            mu = mu + torch.randn_like(mu) * self.noise_scale * mu
        
        # ... (rest of ZINB NLL)
```

---

## Tài liệu tham khảo chính

1. Pérez-Salinas et al. *"Data re-uploading for a universal quantum classifier."* Quantum 5, 391 (2020). arXiv:1907.02040
2. Cerezo et al. *"Cost-function-dependent barren plateaus in shallow quantum neural networks."* Nat. Commun. 12, 1791 (2021). arXiv:2012.09288
3. Stokes et al. *"Quantum Natural Gradient."* Quantum 4, 269 (2020). arXiv:1909.02108
4. Sweke et al. *"Dissipative Quantum Neural Networks."* Entropy 22, 727 (2020). arXiv:1910.12256
5. Lloyd, Mohseni, Rebentrost. *"Quantum principal component analysis."* Nat. Phys. 10, 631 (2014). arXiv:1307.0401
6. Liu et al. *"Representation Theory for Geometric Quantum Machine Learning."* PRX 11, 041044 (2021).
7. Schuld. *"Quantum machine learning models are kernel methods."* arXiv:2107.14485.
8. Bebrov & Berta. *"Quantum-inspired classical algorithms for solving linear systems."* arXiv:2308.13345.
9. Chawla et al. *"SMOTE: Synthetic Minority Over-sampling Technique."* JAIR 16, 321-357 (2002).
10. Benedetti et al. *"Parameterized quantum circuits as machine learning models."* Quantum Sci. Technol. 4, 043001 (2019).
