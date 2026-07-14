# Phân tích Quantum-Dengue-STPP — Code Review & Codebase Design

> **Project:** Quantum-Enhanced Spatio-Temporal Point Process (STPP) for Dengue Fever Prediction in Southeast Asia
> **Repo:** `quantum-dengue-stpp/`
> **Phương pháp:** Kết hợp `/code-review` (phân tích kỹ thuật) và `/codebase-design` (thiết kế module hóa)

---

## Mục lục

- [PHẦN 1 — Code Review](#phần-1--code-review)
  - [1.1. Các biện pháp thống kê](#11-các-biện-pháp-thống-kê)
  - [1.2. Thuật toán dự đoán](#12-thuật-toán-dự-đoán)
  - [1.3. Phương pháp Quantum Computing cho sinh dữ liệu](#13-phương-pháp-quantum-computing-cho-sinh-dữ-liệu)
- [PHẦN 2 — Bản chất toán học của vấn đề](#phần-2--bản-chất-toán-học-của-vấn-đề)
- [PHẦN 3 — Codebase Design: Module hóa & mở rộng](#phần-3--codebase-design-module-hóa--mở-rộng)
- [Tổng kết](#tổng-kết)

---

## PHẦN 1 — Code Review

### 1.1. Các biện pháp thống kê

#### A. Thống kê không gian (Spatial Statistics)

**1. Ripley's K-function** — `src/evaluation/spatial_stats.py:31-75`

```python
K(r) = (1/λ) × E[events within distance r of a randomly chosen event]
```

- Đo lường mức độ phân cụm không gian tại các bán kính `r` khác nhau.
- Nếu `K(r) > K_CSR(r)` (Complete Spatial Randomness) → có clustering.
- Implementation dùng `haversine_distance` cho tọa độ địa lý, normalize theo `λ = n/A`.

**2. L-function (chuẩn hóa)** — `src/evaluation/spatial_stats.py:78-80`

```python
L(r) = √(K(r)/π) - r
```

- Đường chuẩn `L(r) = 0` cho CSR; `L > 0` = clustering, `L < 0` = regularity.
- Kết quả thực tế trong project:
  - Indonesia: `L = +169` (strongly clustered)
  - Malaysia:  `L = +131` (clustered)
  - Vietnam:   `L = +72`  (clustered)
  - Singapore: `L = -200` (regular pattern)

**3. Pair Correlation Function g(r)** — `spatial_stats.py:83-112`

```python
g(r) = histogram_count / (2π·r·dr·λ·n²)
```

- Ước lượng mật độ xác suất có điều kiện giữa hai điểm cách nhau khoảng `r`.
- Phân biệt được clustering ở các scale khác nhau (multi-scale analysis).

**4. Moran's I (Global Spatial Autocorrelation)** — `spatial_stats.py:115-147`

```python
I = (n / Σy²) × (y^T W y) / (Σy²)
```

- KNN-based spatial weights matrix (k = 5).
- `I > 0` → tương quan dương (clustering); `I < 0` → tương quan âm.
- Z-test cho ý nghĩa thống kê.

#### B. Thống kê thời gian (Temporal Statistics)

**5. ACF/PACF** — `spatial_stats.py:150-176`

- Tự tương quan và tự tương quan riêng phần với lag tối đa 12 tháng.
- PACF dùng đệ quy Durbin-Levinson:
  `φ_kk = (ρ_k − Σφ_{k−1,j}·ρ_{k−j}) / (1 − Σφ_{k−1,j}·ρ_j)`

**6. Seasonal Decomposition** — `spatial_stats.py:179-201`

- Phân tách chuỗi thành `Trend + Seasonal + Residual`.
- Dùng moving average cho trend (window = period/2).
- Mùa vụ dengue thường là chu kỳ 12 tháng.

**7. Zero-Inflation Ratio & Overdispersion** — `spatial_stats.py:204-223`

- `zero_ratio = mean(y == 0)` — quan trọng cho dữ liệu dengue (>30% zero ở các vùng Việt Nam).
- `dispersion = Var(Y) / E(Y)` — dấu hiệu cần dùng NB/ZINB thay vì Poisson (>1 nghĩa là overdispersion).

#### Đánh giá theo smell baseline

| Smell | Mức độ | Bằng chứng |
|-------|--------|------------|
| **Duplicated Code** | Trung bình | `_vectorized_haversine` (`spatial_stats.py:16-28`) gần giống `haversine_distance` (dòng 6-13) |
| **Primitive Obsession** | Thấp | `pair_alpha_idx = i_regs * n + j_regs` ở `hawkes.py:87` — ma trận alpha biểu diễn bằng 1D index, có thể tạo type `AlphaMatrix` |
| **Mysterious Name** | Thấp | `g_adv`, `g_rec`, `g_div` trong `quantum_augment_v3.py:433` — tên viết tắt không tường minh |

---

### 1.2. Thuật toán dự đoán

#### A. Hawkes Process — Quá trình tự kích thích đa chiều

**File:** `src/models/hawkes.py`

```python
λ_i(t) = μ_i + Σ_j Σ_{t_k<t, d_k=j} α_ij · β · exp(-β(t-t_k)) · w_k
```

**Cơ chế toán học:**

- Mục đích: mô hình hóa hiện tượng tự kích thích — khi có ca bệnh ở vùng này thì làm tăng xác suất xuất hiện ca bệnh ở vùng lân cận.
- **EM algorithm** (dòng 41-154): vectorized với `np.bincount` thay cho Python loops.
- Hằng số `MAX_N = 1000` events để tránh bùng nổ `O(n²)` (vì `N(N-1)/2 ≈ 500K` pairs).

**Các tham số:**

- `μ_i`: background intensity (tỷ lệ nền của region i).
- `α_ij`: ma trận lan truyền không gian (region j → region i).
- `β`: tốc độ phân rã theo thời gian (1/month).
- Convergence criterion: `change < 1e-4` sau iter > 5.

**Vấn đề tiềm ẩn (smell baseline):**

- **Data Clumps:** `i_regs`, `j_regs`, `dt_flat`, `pair_alpha_idx` ở `hawkes.py:80-87` luôn đi cùng nhau → có thể gói thành `PairIndexSet`.
- **Long Method:** hàm `fit()` từ dòng 41-154 quá dài (~113 dòng) → tách thành `_e_step()`, `_m_step()`, `_update_beta()`.

#### B. NEST — Neural Spatio-Temporal Point Process

**File:** `src/models/nest.py`

```python
λ(s, t) = exp(w^T · h(s, t) + b)
```

**Kiến trúc:**

```
Input (B, T, H, W)
  → Spatial Encoder: 2× Conv2d(1→64) + AdaptiveAvgPool → (B, 64, 4, 4)
  → Temporal: 2-layer LSTM (64 hidden)
  → Intensity Head: Linear(64 → gs²) + Softplus
  → Output (B, gs, gs) — intensity mỗi cell
```

**Hàm mất mát:** Poisson NLL — `criterion = lambda pred, target: (pred.exp() - target * pred).mean()`

**Điểm mạnh:**

- Hàm loss đúng phân phối cho count data.
- Softplus đảm bảo `λ > 0` luôn (mathematical correctness).
- Poisson NLL ≈ `λ − count·log(λ)`.

**Smell đáng chú ý:**

- **Refused Bequest:** `NESTForecaster` ở `nest.py:103` chỉ dùng Poisson NLL, nhưng không tận dụng được ZINB (mặc dù `zinb_loss.py` đã có sẵn) → đáng lý phải là strategy pattern.
- **Speculative Generality:** `output_activation='exp'` ở `nest.py:30` nhưng trong code chỉ dùng `'softplus'`.

#### C. CNN-LSTM — Mô hình hybrid

**File:** `src/models/cnn_lstm.py`

**Kiến trúc:**

```
Input (B, T, H, W)
  → Conv2d(1→32) + BatchNorm + ReLU + MaxPool
  → Conv2d(32→64) + ...
  → LSTM(2 layers, 128 hidden) theo temporal
  → Linear head → Softplus
```

**Kết quả thực nghiệm:**

| Method | Val RMSE | R² | Pearson r |
|--------|----------|-----|-----------|
| Hawkes | 2,065 | — | — |
| CNN-LSTM (No Aug) | 2.48 | 0.855 | 0.935 |
| **CNN-LSTM + Quantum** | **2.46** | **0.858** | **0.967** |
| CNN-LSTM + SOP | 4.32 | 0.560 | 0.937 |

#### D. ZINB Loss — Zero-Inflated Negative Binomial

**File:** `src/models/zinb_loss.py`

**Phân phối toán học:**

```text
P(Y=0) = π + (1-π)·(1 + μ/θ)^(-θ)              ← structural zeros
P(Y=k) = (1-π)·NB(k | μ, θ)                     ← overdispersed counts
```

**Bốn thành phần:**

1. **π (zero-inflation)** — xác suất structural zero (output: `sigmoid`).
2. **μ (mean count)** — tỷ lệ trung bình (output: `softplus`).
3. **θ (dispersion)** — tham số phân tán, `exp(log_θ)` luôn dương.
4. **Spatial smoothness** — Total Variation penalty `‖∇μ‖²` để đảm bảo mượt không gian.

**Log-prob:**

```text
log P(Y=y) = 𝟙(y=0)·log(π + (1-π)(1+μ/θ)^(-θ))
           + 𝟙(y>0)·[log(1-π) + lgamma(y+θ) - lgamma(y+1) - lgamma(θ)
                       + y·log(θ/(θ+μ)) + θ·log(θ/(θ+μ))]
```

**Smell đáng chú ý:**

- **Shotgun Surgery:** thay đổi grid size ảnh hưởng đến nhiều file (`cnn_lstm.py`, `nest.py`, `quantum_augment_v3.py`) — đáng lý nên có một config object duy nhất.
- **Middle Man:** `MultiHawkesExpKern` ở `hawkes.py:224-326` ủy quyền gần như toàn bộ cho `scipy.minimize` — không có logic riêng, có thể xóa.

---

### 1.3. Phương pháp Quantum Computing cho sinh dữ liệu

#### A. Quantum Born Machine v3 (QBM v3)

**File:** `src/augmentation/quantum_augment_v3.py:34-188`

**Mạch lượng tử:**

```python
@qml.qnode(self.qdev, diff_method="backprop")
def circuit(params_flat):
    for layer in range(n_layers):
        for i in range(n_q):
            qml.RY(params[layer, i], wires=i)        # RY rotation
        for i in range(n_q - 1):
            qml.CNOT(wires=[i, i + 1])                # Linear entanglement
        if n_q > 2:
            qml.CNOT(wires=[n_q - 1, 0])              # Ring closure
    return qml.probs(wires=range(n_q))
```

**Cơ chế:**

- **Không gian Hilbert 2^n:** phân phối xác suất trên `2^n` trạng thái cơ sở.
- **n_qubits = min(n_patterns, 12)** — giới hạn bởi classical tractability.
- **Mục đích:** học phân phối probability mask cho 16 regions (4×4 grid blocks).
- **Training:** MMD loss qua `target_dist` được xây từ `spatial_binary.mean()`.

**Hạn chế thực tế:**

- `n_qubits = 12` thực tế chỉ là classical simulation (không phải quantum hardware thật).
- Không có quantum advantage thực sự — chỉ là "quantum-inspired" classical.

#### B. QGAN v3 — Quantum-Inspired Generator

**File:** `src/augmentation/quantum_augment_v3.py:195-502`

**Kiến trúc generator:**

```python
# Quantum-inspired transformation
z_enc = z * π + vqc_ry                    # RY rotation angle
z_ang = sin(z_enc)                         # Non-linear encoding
s_enc = style * π + vqc_rz                 # RZ modulation
s_ang = cos(s_enc)

# Entanglement (classical analog of CNOT)
z_entangled = z_ang @ tanh(entangle_weights)

# Decoder → grid tensor (seq_len × H × W)
grid = softplus(decoder([z_feat, s_feat]))
```

**Loss function:**

```python
g_loss = g_adv + 0.3·g_rec + g_div
# g_adv : binary cross-entropy với discriminator
# g_rec : MSE so với real (đảm bảo giữ đặc trưng)
# g_div : -MSE(mean along time)  ← khuyến khích diversity
```

**Đặc điểm:**

- Gradient penalty `λ_gp = 10.0` (WGAN-GP style).
- Sin/cos encoding tương ứng với góc quay qubit (quantum-inspired, không phải quantum thật).
- Output: `(batch, seq_len, grid_h, grid_w)` — full grid tensors.

#### C. Local PQC — Parameterized Quantum Circuit cục bộ

**File:** `src/augmentation/local_pqc.py`

**Pipeline:**

```
Events (lat, lon)
  → SpatialClusterer (DBSCAN/K-Means/Ripley-KMeans) → k clusters
  → For each cluster:
      LocalPQC(4-6 qubits) — AngleEmbedding + StronglyEntanglingLayers
  → Aggregation: weighted / mean / sum
```

**Mạch lượng tử mỗi cluster:**

```python
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(features, weights):
    # 1. AngleEmbedding (data → qubits)
    qml.AngleEmbedding(features, wires=range(n_qubits), rotation='X')

    # 2. Variational layers (trainable)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # 3. Measurement: ⟨Z_i⟩ expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Điểm sáng tạo:**

- **Quantum Fisher Information (QFI):** đo lượng thông tin quantum — proxy cho expressibility.
- **Haar Expressibility:** so sánh variance với `1/2^n` của Haar-random states.
- **Ripley-guided clustering:** số cluster được tự động chọn từ K-function elbow.

#### Đánh giá tổng thể về Quantum Components

**Hạn chế quan trọng:**

1. Tất cả circuit chạy trên `default.qubit` (simulator classical) — chưa có quantum advantage thực sự.
2. `N_qubits ≤ 12` — quá nhỏ cho các bài toán quantum thực sự.
3. "Quantum-inspired ≠ Quantum": `QGeneratorGrid` dùng `sin/cos` thay vì quantum gates thật.
4. Cost function không exploit quantum — dùng MMD/MSE thay vì quantum-native metrics.

**Theo smell baseline:**

| Smell | Mức độ | Vị trí |
|-------|--------|--------|
| **Speculative Generality** | **Cao** | `true_quantum.py` import 5 backends nhưng code chỉ chạy simulator |
| **Divergent Change** | Cao | `quantum_augment_v1/v2/v3.py` cùng mục đích nhưng 3 phiên bản khác nhau |
| **Middle Man** | Trung bình | `QuantumFisherInformation.estimate_haar_expressibility` chỉ là wrapper cho `np.var` |
| **Data Clumps** | Trung bình | `(coords, features, targets, cluster_ids)` đi cùng nhau trong `create_local_pqc_training_pipeline` → cần `LocalPQCInputs` dataclass |

---

## PHẦN 2 — Bản chất toán học của vấn đề

### 2.1. Bài toán gốc (Mathematical Formulation)

Cho dữ liệu quan sát `{(s_i, t_i, n_i)}_{i=1}^N`:

- `s_i ∈ ℝ²`: vị trí (lat, lon).
- `t_i ∈ [0, T]`: thời điểm.
- `n_i ∈ ℕ`: số ca bệnh.

**Bài toán ước lượng cường độ (Intensity Estimation):**

```text
Tìm hàm λ(s, t): ℝ² × [0,T] → ℝ⁺
sao cho: P(sự kiện xảy ra tại (s,t)) = λ(s,t) · ds · dt
```

**Bài toán dự đoán (Forecasting):**

```text
Cho lịch sử H_t = {(s_k, t_k, n_k): t_k < t}
Dự đoán: E[N(t+Δ) | H_t] = ∫_A λ(s, t+Δ | H_t) ds
```

### 2.2. Các giả định thống kê tiềm ẩn

| Giả định | Biểu hiện trong code | Vi phạm nếu |
|----------|---------------------|-------------|
| **Stationarity** | Hawkes β constant | Dịch bệnh theo mùa |
| **Markov property** | LSTM context window | Tương tác dài hạn giữa các vùng |
| **Poisson assumption** | CNN-LSTM + softplus, Poisson NLL | Overdispersion → cần ZINB |
| **Spatial homogeneity** | K-Means fixed clusters | Cluster động theo mùa |
| **Independence** | MSE loss giữa các cells | Spatial autocorrelation cao |

### 2.3. Toàn cảnh lý thuyết

```
┌──────────────────────────────────────────────────────────┐
│           INPUT SPACE (Stochastic Process)              │
│  X(t) = Σ_i 𝟙(event i trong region s_i, time t_i)     │
│           ↓                                              │
│  Spatial Discretization: ℝ² → Grid {0,1,...,G-1}²        │
│           ↓                                              │
│  Temporal Sequence: G(t) ∈ ℝ^{G×G}, t=1,...,T            │
│           ↓                                              │
│  Tensor Form: X ∈ ℝ^{T×H×W}                              │
│           ↓                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Model layer (parallel hypothesis classes):        │   │
│  │  • Parametric:  Hawkes λ(t) = μ + Σ αe^(-βΔt)    │   │
│  │  • Neural:      λ(s,t) = exp(f_θ(s,t))           │   │
│  │  • Hybrid:      Quantum encoding → Classical net  │   │
│  └──────────────────────────────────────────────────┘   │
│           ↓                                              │
│  LOSS (negative log-likelihood):                         │
│  L = -Σ log p(y_t | λ_t)                                 │
│       + λ_tv · ‖∇μ‖²  (spatial smoothness)               │
│       + λ_zinb · 𝟙(ZINB case)                            │
└──────────────────────────────────────────────────────────┘
```

---

## PHẦN 3 — Codebase Design: Module hóa & mở rộng

### 3.1. Nguyên tắc thiết kế

**Thuật ngữ chuẩn:**

- **Module:** bất kỳ thứ gì có interface + implementation.
- **Interface:** mọi thứ caller phải biết (signature + invariants + error modes).
- **Adapter:** concrete thing tại một seam.
- **Seam:** nơi có thể thay đổi behavior mà không edit tại chỗ.
- **Deep module:** interface nhỏ + implementation dày (nhiều behavior sau ít method).

### 3.2. Backbone module đề xuất — STPP-Core

**Triết lý:** tách biệt **5 module sâu** (deep modules), mỗi module có interface nhỏ + implementation dày.

```
┌─────────────────────────────────────────────────────────────────┐
│                     STPP-CORE BACKBONE                          │
│   (Spatio-Temporal Point Process Framework)                    │
└─────────────────────────────────────────────────────────────────┘
         │                │                │              │
         ▼                ▼                ▼              ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Module  │    │  Module  │    │  Module  │    │  Module  │
   │    A     │    │    B     │    │    C     │    │    D     │
   │ Spatial  │    │ Temporal │    │Intensity │    │Quantum   │
   │ Engine   │    │Dynamics  │    │  Model   │    │Sampler   │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 3.3. Thiết kế 6 module cốt lõi

#### Module A: `SpatialEngine` (Interface: 5 phương thức)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class SpatialPoint:
    """Một observation — domain-agnostic."""
    coords: tuple[float, ...]   # 2D, 3D, hoặc higher
    timestamp: float
    intensity: float           # có thể là count, weight, magnitude

class SpatialEngine(ABC):
    """Deep module: ẩn toàn bộ spatial statistics phía sau interface nhỏ."""

    @abstractmethod
    def compute_clustering(self, points: list[SpatialPoint]) -> 'ClusteringResult':
        """Trả về K-function, L-function, g(r), Moran's I — unified."""

    @abstractmethod
    def cluster(self, points: list[SpatialPoint], method: str = 'auto') -> list[int]:
        """Trả về cluster assignments (DBSCAN/K-Means/Ripley-guided)."""

    @abstractmethod
    def to_grid(self, points: list[SpatialPoint], grid_size: tuple[int, ...]) -> 'Grid':
        """Discretize không gian liên tục → tensor."""

    @abstractmethod
    def normalize(self, points: list[SpatialPoint]) -> tuple[list[SpatialPoint], 'Normalizer']:
        """Normalize về [0,1]^d cho embedding (quantum/ML)."""

    @abstractmethod
    def inverse_normalize(self, points: list[SpatialPoint], normalizer: 'Normalizer') -> list[SpatialPoint]:
        """Ánh xạ ngược về không gian gốc."""
```

**Adapter implementations:**

- `GeoSpatialEngine(lat, lon)` — cho dịch tễ, khí hậu.
- `FinancialSpatialEngine(price, volume, ticker_index)` — cho tài chính.
- `TrafficSpatialEngine(road_id, lat, lon)` — cho giao thông.
- `NeuralSpatialEngine(brain_region_coords)` — cho neuroscience.

**Depth justification:** caller chỉ cần biết 5 method, nhưng có thể tính K/L/g/Moran, clustering, grid discretization, normalization — leverage rất cao.

#### Module B: `TemporalDynamics`

```python
class TemporalDynamics(ABC):
    """Ẩn ACF/PACF/seasonality/decomposition."""

    @abstractmethod
    def fit(self, series: np.ndarray, period_hint: int | None = None) -> 'TemporalModel':
        """Auto-detect period, decompose."""

    @abstractmethod
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast next h steps."""

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Sinh time series mới giữ pattern."""
```

**Adapter:** `MonthlyDengueDynamics`, `DailyTrafficDynamics`, `HighFreqFinancialDynamics`.

#### Module C: `IntensityModel` — core của backbone

```python
class IntensityModel(ABC):
    """λ(s, t) estimator — parametric, neural, hoặc quantum-hybrid."""

    @abstractmethod
    def intensity(self, points: list[SpatialPoint]) -> np.ndarray:
        """λ(s,t) tại mỗi point."""

    @abstractmethod
    def conditional_intensity(self, history: list[SpatialPoint],
                              query: SpatialPoint) -> float:
        """λ(s_q, t_q | H) — cho forecasting."""

    @abstractmethod
    def negative_log_likelihood(self, observations: list[SpatialPoint]) -> float:
        """Training objective."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialization."""
```

**Adapter (3 strategy implementations):**

| Adapter | Khi nào dùng | Ưu điểm |
|---------|--------------|---------|
| `HawkesIntensity` | Self-exciting clear | Interpretable, fast |
| `NeuralIntensity` (NEST) | Pattern phức tạp | Flexible, deep |
| `HybridQuantumIntensity` | Limited data + structure preservation | Multi-scale correlation |

**The deletion test:** xóa `HawkesIntensity` → caller phải tự code EM (mất 200+ dòng) → module đang "earn its keep".

#### Module D: `QuantumSampler` (Seam cho quantum augmentation)

```python
class QuantumSampler(ABC):
    """Sinh dữ liệu giả lập bảo toàn spatio-temporal structure."""

    @abstractmethod
    def fit(self, real_data: 'Grid') -> 'QuantumSampler':
        """Train trên data thật (CHỈ train set)."""

    @abstractmethod
    def sample(self, n: int, style: np.ndarray | None = None) -> 'Grid':
        """Generate synthetic data."""

    @abstractmethod
    def fidelity_to_real(self, n_samples: int = 1000) -> dict[str, float]:
        """Đo lường K-fn, L-fn, MMD, correlation."""
```

**Adapter (3 strategies):**

```python
class QBMSampler(QuantumSampler): ...        # Quantum Born Machine
class QGANSampler(QuantumSampler): ...       # Hybrid QGAN
class SOPSampler(QuantumSampler): ...        # Classical baseline (Second-Order Preserving)
class LocalPQCSampler(QuantumSampler): ...   # Clustered PQC
```

**Seam rule:** `MultiDimensionalHawkes`, `SpatialClusterer` là **adapters** tại các seam khác nhau — không lẫn lộn với sampler.

#### Module E: `LossFactory`

```python
class LossFactory:
    """Strategy cho count data với các đặc tính khác nhau."""

    @staticmethod
    def build(data: 'Grid') -> 'LossFn':
        if data.zero_ratio < 0.1:
            return PoissonNLL()
        elif data.dispersion < 2.0:
            return GaussianMSELoss()
        else:
            return ZINBLoss(learn_theta=True, spatial_smooth_weight=0.1)
```

#### Module F: `DataLeakageGuard`

```python
class DataLeakageGuard:
    """Bảo vệ temporal integrity — universal cho mọi lĩnh vực time-series."""

    def __init__(self, time_col: str):
        self.time_col = time_col

    def split(self, df: pd.DataFrame, ratios: tuple[float, ...]) -> tuple[pd.DataFrame, ...]:
        """Sort theo time, split strictly chronological."""

    def validate(self, train_df, val_df, test_df) -> None:
        """Assert: min(val) ≥ max(train), min(test) ≥ max(val)."""
```

### 3.4. Cấu trúc thư mục đề xuất (Backbone)

```
stpp_core/                      # ← backbone Python package
├── __init__.py
├── spatial/
│   ├── engine.py              # SpatialEngine ABC
│   ├── geo_adapter.py         # GeoSpatialEngine (haversine-based)
│   ├── financial_adapter.py   # FinancialSpatialEngine
│   ├── clusterer.py           # SpatialClusterer
│   └── gridder.py             # AdaptiveGridder
├── temporal/
│   ├── dynamics.py            # TemporalDynamics ABC
│   └── decomposer.py          # SeasonalDecomposer
├── intensity/
│   ├── model.py               # IntensityModel ABC
│   ├── hawkes.py              # HawkesIntensity adapter
│   ├── neural.py              # NeuralIntensity adapter (NEST)
│   └── quantum_hybrid.py      # HybridQuantumIntensity adapter
├── quantum/
│   ├── sampler.py             # QuantumSampler ABC
│   ├── qbm.py                 # QBMSampler
│   ├── qgan.py                # QGANSampler
│   ├── local_pqc.py           # LocalPQCSampler
│   └── sop.py                 # SOPSampler (classical baseline)
├── losses/
│   ├── factory.py             # LossFactory
│   ├── poisson.py
│   ├── zinb.py
│   └── gaussian.py
├── guards/
│   └── leakage.py             # DataLeakageGuard
└── evaluation/
    ├── metrics.py             # RMSE/MAE/R²
    └── spatial_stats.py       # K/L/g/Moran

domains/                        # ← domain-specific thin adapters
├── dengue/
│   ├── data_loader.py         # implements GeoSpatialEngine
│   ├── pipeline.py            # wires modules
│   └── api.py
├── traffic/
│   ├── data_loader.py
│   ├── pipeline.py
│   └── api.py
└── finance/
    └── ...
```

### 3.5. Ứng dụng mở rộng — 6 lĩnh vực gợi ý

| Lĩnh vực | Spatial adapter | Temporal adapter | Intensity adapter | Quantum role |
|----------|-----------------|------------------|-------------------|--------------|
| **Dịch tễ (hiện tại)** | `GeoSpatialEngine` | `MonthlyDengueDynamics` | `Hawkes`/`Neural` | Augment rare outbreaks |
| **Giao thông đô thị** | `TrafficSpatialEngine` (road graph) | `DailyTrafficDynamics` | `NeuralIntensity` | Forecast congestion events |
| **Tài chính** (jump detection) | `FinancialSpatialEngine` (correlations) | `HighFreqFinancialDynamics` | `Hawkes` (self-exciting jumps) | Synthetic stress scenarios |
| **Khí hậu cực đoan** | `GeoSpatialEngine` | `ClimateSeasonalDynamics` | `NeuralIntensity` | Extreme event upsampling |
| **Động đất** | `GeoSpatialEngine` (fault lines) | `SeismicDynamics` | `ETAS` (= Hawkes extension) | Synthetic catalogs |
| **Crime hotspots** | `UrbanSpatialEngine` | `WeeklyCrimeDynamics` | `NeuralIntensity` | Data augmentation cho cold-start areas |

### 3.6. Pattern sử dụng (Caller code)

```python
from stpp_core import SpatialEngine, QuantumSampler, LossFactory, DataLeakageGuard
from domains.dengue import DengueDataLoader

# 1. Load domain-specific data
loader = DengueDataLoader("sea_dengue_admin1_month.csv")
events = loader.load()

# 2. Universal preprocessing
guard = DataLeakageGuard(time_col="timestamp")
train, val, test = guard.split(events, ratios=(0.7, 0.15, 0.15))
grid = loader.to_grid(train)

# 3. Quantum augmentation (seam: swap with classical SOP)
sampler = QuantumSampler.create("qgan", n_qubits=8)
sampler.fit(grid)
augmented = sampler.sample(n=500, style=grid.temporal_features())

# 4. Train intensity model
loss_fn = LossFactory.build(grid)             # auto-chọn ZINB
model = NeuralIntensity(input_dim=grid.feature_dim)
model.fit(augmented, loss_fn=loss_fn)

# 5. Forecast & evaluate
predictions = model.conditional_intensity_batch(val)
metrics = evaluate(predictions, val, metrics=["rmse", "k_function"])
```

### 3.7. Test pattern — Locality preserved

```python
# Test cho QuantumSampler — chỉ cần interface
def test_qbm_sampler_preserves_spatial_structure():
    sampler = QBMSampler(n_qubits=4)
    sampler.fit(real_grid)
    synthetic = sampler.sample(1000)

    real_K = spatial_stats(real_grid)
    synth_K = spatial_stats(synthetic)

    assert np.allclose(real_K, synth_K, atol=0.1)   # K-function preserved
```

### 3.8. Các smell hiện tại cần sửa khi tái cấu trúc

| Smell | Vị trí hiện tại | Refactor đề xuất |
|-------|-----------------|------------------|
| **Shotgun Surgery** | Grid size 16/32/48 xuất hiện ở 8+ files | Tạo `GridConfig` dataclass, truyền qua |
| **Divergent Change** | `quantum_augment_v1/v2/v3.py` cùng mục đích | Hợp nhất thành `quantum/sampler.py` với versioning |
| **Speculative Generality** | `true_quantum.py` import 5 backends | Chỉ giữ 1 active backend, dùng `QuantumBackend` adapter |
| **Middle Man** | `MultiHawkesExpKern` (`hawkes.py:224`) | Xóa — chỉ là wrapper scipy |
| **Duplicated Code** | `haversine_distance` & `_vectorized_haversine` | Một hàm vectorized duy nhất |
| **Data Clumps** | `(coords, features, targets)` params | Gói thành `STPPDataset` dataclass |

---

## Tổng kết

| Trụ cột | Điểm mạnh | Điểm yếu cần cải thiện |
|---------|-----------|------------------------|
| **Thống kê** | Đầy đủ K/L/g/Moran + ACF/PACF | `_vectorized_haversine` duplicate |
| **Dự đoán** | 3 paradigms (Hawkes/NEST/CNN-LSTM) + ZINB | `MultiHawkesExpKern` là middle man |
| **Quantum** | 4 phương pháp (QBM/QGAN/Local PQC/SOP) | Tất cả chạy simulator — chưa có quantum advantage thật |
| **Bản chất toán học** | Rõ ràng — intensity estimation với Poisson | Giả định stationarity có thể vi phạm |
| **Khả năng mở rộng** | Có thể module hóa thành STPP-Core | Cần tách domain-specific khỏi backbone |

### Khuyến nghị ưu tiên

1. **Ngắn hạn:** tách `stpp_core/` package với 6 module ABC + adapter pattern.
2. **Trung hạn:** chạy quantum circuits trên IBM Q hoặc IonQ thật để có quantum advantage thực sự (hiện tại 100% là simulator).
3. **Dài hạn:** mở rộng sang 6 lĩnh vực (giao thông, tài chính, khí hậu, động đất, crime, neuroscience) chỉ bằng cách viết adapter mới — backbone không cần đổi.

### Worst issue per axis

- **Standards (smell baseline):** *Speculative Generality* cao — `true_quantum.py` import 5 quantum backends nhưng code chỉ chạy simulator, gây hiểu lầm về quantum capability.
- **Spec (research hypothesis):** *Quantum advantage chưa chứng minh được* — toàn bộ "quantum" trong project chạy trên classical simulator, nên kết luận "CNN-LSTM + Quantum tốt hơn No-Aug" thực chất so sánh quantum-inspired-classical với classical.