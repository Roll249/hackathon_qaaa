# Phân Tích & Hướng Dẫn: Statistical Learning for Spatio-Temporal Point Processes

> **Nguồn:** Bài trình bày của GS. **Jorge Mateu** (University Jaume I, Castelló, Tây Ban Nha)
> tại **14th European Congress for Stereology and Image Analysis (ECSIA)**, Prague, 16/09/2025.
> **Tài liệu gốc:** `S7-ECSIA-2025-Prague.pdf`
>
> Tài liệu này được viết theo phong cách **giảng dạy** — mỗi phương pháp sẽ được "mổ xẻ" từ
> ý tưởng → toán học → kiến trúc → ưu/nhược điểm. Đọc tuần tự sẽ hiểu sâu nhất.

---

## Mục lục

1. [Bối cảnh & bài toán](#1-bối-cảnh--bài-toán)
2. [Bản đồ tổng quan 5 phương pháp](#2-bản-đồ-tổng-quan-5-phương-pháp)
3. [Phương pháp 1 — Siamese CNN cho phân biệt mẫu điểm](#3-phương-pháp-1--siamese-cnn-cho-phân-biệt-mẫu-điểm)
4. [Phương pháp 2 — SOP (Second-Order Preserving) Permutations](#4-phương-pháp-2--sop-second-order-preserving-permutations)
5. [Phương pháp 3 — STNPP (Spatio-Temporal-Network Point Process)](#5-phương-pháp-3--stnpp-spatio-temporal-network-point-process)
6. [Phương pháp 4 — Non-stationary deep STPP (COVID Cali)](#6-phương-pháp-4--non-stationary-deep-stpp-covid-cali)
7. [Phương pháp 5 — Neural Likelihood Inference](#7-phương-pháp-5--neural-likelihood-inference)
8. [Tổng hợp điểm mạnh & điểm yếu](#8-tổng-hợp-điểm-mạnh--điểm-yếu)
9. [Hướng ứng dụng cho bạn](#9-hướng-ứng-dụng-cho-bạn)

---

## 1. Bối cảnh & bài toán

### 1.1 Quá trình điểm (point process) là gì?

> **Quá trình điểm không gian** = một cách mô hình hoá sự xuất hiện ngẫu nhiên
> của các **điểm (sự kiện)** trong một vùng quan sát \(W \subset \mathbb{R}^2\)
> (không gian) hoặc \(W \subset \mathbb{R}^3\) (không gian-thời gian).

Ví dụ đời thường:

| Bài toán | Điểm (sự kiện) | Vùng W |
|----------|------------------|--------|
| Dịch tễ | ca nhiễm COVID | bản đồ thành phố × thời gian |
| Tội phạm | vụ trộm/cướp | thành phố Valencia × 5 năm |
| Sinh thái | vị trí cây gỗ | rừng BCI × 8 lần điều tra |
| Địa chấn | chấn động | vùng đứt gãy × năm |

### 1.2 Vì sao cần "học thống kê" (statistical learning)?

Các phương pháp cổ điển (Poisson, Strauss, Matérn, LGCP, Thomas...) có **mô hình sinh xác suất rõ ràng**, nhưng:

- **Likelihood không tính được** (hằng số chuẩn hoá \(Z(\theta)\) là tích phân trên không gian vô hạn chiều).
- Khó mở rộng khi dữ liệu phức tạp (mark, network, không dừng...).
- Khó xử lý dữ liệu lớn.

→ Mateu & cộng sự đề xuất **5 phương pháp học thống kê** (CNN, Siamese Net, GNN, GAT, non-stationary kernel) để giải quyết 5 bài toán khác nhau trên cùng một khung lý thuyết.

### 1.3 Bộ dữ liệu minh hoạ chính

- **BCI**: 130 loài cây × 8 lần điều tra × khu rừng 1000×500 m → **1040 mẫu điểm, 1.808.725 cây sống**.
- **Valencia crime**: 47.125 vụ án trong 5 năm (2015–2019) + 1975 địa danh (landmark).
- **Cali COVID**: 38.611 ca từ 15/03/2020 đến 30/09/2020 (Colombia).
- **Mô phỏng**: 10 quá trình mẫu (Poisson, LGCP, Thomas, Matérn, VarGamma, Cauchy, dppG, dppC, Strauss, AreaInter).

---

## 2. Bản đồ tổng quan 5 phương pháp

```
┌────────────────────────────────────────────────────────────────────┐
│                    BÀI TOÁN GỐC: Quá trình điểm STPP              │
└────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1. SIAMESE CNN    Hai mẫu điểm x, x' có cùng "quá trình sinh" hay │
│                   không? (Phân biệt / gộp nhóm / one-shot)         │
│ 2. SOP PERMUTATION Tăng cường dữ liệu bằng hoán vị giữ K-function│
│                   cho dự báo CNN-LSTM 1 ngày                        │
│ 3. STNPP + GAT    Mô hình Hawkes trên mạng lưới đường + địa danh  │
│                   cho tội phạm Valencia                            │
│ 4. NEURAL KERNEL  Quá trình KHÔNG DỪNG (non-stationary) cho       │
│                   COVID-19 ở Cali, dùng covariance thay đổi        │
│                   theo không gian                                   │
│ 5. NEURAL LIKELIHOOD Bỏ qua likelihood khó — dùng CNN/GNN làm     │
│                   "proxy likelihood" cho tham số θ                  │
└────────────────────────────────────────────────────────────────────┘
```

Mỗi phương pháp giải một bài toán khác nhau, dùng **kiến trúc học sâu khác nhau**:

| # | Tên | Bài toán | Mạng chính | Tác giả chính |
|---|------|----------|------------|----------------|
| 1 | Siamese CNN | Pattern discrimination | CNN + Siamese | Jalilian & Mateu (2023) |
| 2 | SOP | Data augmentation | CNN-LSTM | Mohler & Mateu (2024) |
| 3 | STNPP | Crime modeling | Hawkes + GAT | Dong, Zhu, Xie, Mateu (2025) |
| 4 | Neural kernel | Non-stationary STPP | DNN feature mapping | Dong, Mateu, Xie (2023) |
| 5 | Neural likelihood | Likelihood-free inference | CNN / GNN | Platero, Walchessen, Kuusela, Mateu (2025) |

---

## 3. Phương pháp 1 — Siamese CNN cho phân biệt mẫu điểm

> **Mục tiêu:** Cho hai mẫu điểm quan sát \(x, x'\). Hỏi chúng có được sinh ra từ cùng một quá trình hay không?
> Ứng dụng: phân loại loài cây rừng, phân biệt cụm bệnh dịch, one-shot learning.

### 3.1 Ý tưởng cốt lõi

#### Bước 1 — Hàm "tóm tắt" mẫu điểm (summary statistic)

Cổ điện dùng K-function \(K(r)\), g-function, F-function, J-function... Nhưng Mateu thay bằng **CNN feature vector** \(\mathcal{G}(x)\) — mạng học tóm tắt mẫu điểm.

**Hàm bất đồng điều kiện lý tưởng:**

$$
D(x, x') = \begin{cases}
\approx 0 & \text{nếu } x, x' \text{ cùng loài (cùng quá trình)} \\
\text{lớn} & \text{nếu } x, x' \text{ khác loài}
\end{cases}
$$

#### Bước 2 — Rời rạc hoá mẫu điểm thành "ảnh"

Chia cửa sổ quan sát \(W \subset \mathbb{R}^2\) thành lưới \(d_1 \times d_2\) ô.
Mỗi ô \(B_{ij}\) chứa **số điểm** \(n(x \cap B_{ij})\). Kết quả là ma trận \(\tilde{x} \in \mathbb{R}^{d_1 \times d_2}\).

> Trong thí nghiệm với BCI: lưới \((256, 512)\) → mỗi mẫu điểm trở thành **ảnh 256×512**.

#### Bước 3 — Tích chập (convolution) với kernel

Với mỗi kernel \(F^{(1,k)}\) kích thước \(d_1^{(1)} \times d_2^{(1)}\):

$$
(\tilde{x} \ast F^{(1,k)})_{ij} = \sum_{i',j'} x_{i+i', j+j'} \cdot F^{(1,k)}_{i'j'}
$$

**Ví dụ kernel 3×3 "đường viền":**

$$
\begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$

→ Phát hiện **các điểm nằm tách biệt** với lân cận (tương tự Sobel edge detector).

#### Bước 4 — Bias + Activation = Feature map đầu tiên

$$
H^{(1,k)}_{ij} = f_1\left(b^{(1,k)}_0 + \sum_{u \in x} A^{(k)}_{ij}(u)\right)
$$

- **Bias** \(b^{(1,k)}_0\) cho phép dịch đầu ra.
- **Activation** \(f_1\) = ReLU \(\max(0, x)\) cho phép phi tuyến.
- **Diễn giải xác suất** (một điểm rất hay trong bài):

$$
P\!\left(H^{(1,k)}_{ij} \le z\right) = \sum_{n=0}^{\infty} \frac{e^{-|W|}}{n!}
\int_{W^n} \mathbf{1}\!\left[f_1\!\left(b^{(1,k)}_0 + \sum_{l=1}^{n} A^{(k)}_{ij}(u_l)\right) \le z\right] \prod f_X\, du_l
$$

Nghĩa là: feature map là một **thống kê tóm tắt mới** với phân phối xác suất chính xác.

#### Bước 5 — Pooling (max / mean / sum)

Chia \(H^{(1,k)}\) thành các khối \(p_1^{(1)} \times p_2^{(1)}\), gộp mỗi khối thành 1 số (thường là **max**).
- Giảm kích thước.
- **Bền vững hơn** trước nhiễu nhỏ trong rời rạc hoá.

#### Bước 6 — Lặp lại L lớp, đến lớp perceptron

Sau L-1 lớp convolution, lớp cuối là **fully-connected perceptron**:

$$
g_{k'} = f_L\!\left(b^{(L,k')}_0 + \sum_{i,j,k} H^{(L-1,k)}_{ij} w^{(k,k')}_{ij}\right), \quad k' = 1, \ldots, \ell_L
$$

Véc tơ \(\mathcal{G} = (g_1, \ldots, g_{\ell_L}) \in [0,1]^{\ell_L}\) là **đặc trưng cuối cùng**.

> **Tham số** \(\vartheta = \{b^{(l,k)}_0, F^{(l,k,k')}_{ij}, w^{(k,k')}_{ij}\}\) — có thể lên tới
> **73 triệu tham số** với mạng BCI.

### 3.2 Siamese Network — gắn 2 CNN giống nhau

```
        Mẫu x          Mẫu x'
          │                │
          ▼                ▼
      CNN (Gθ)         CNN (Gθ)       ← Hai mạng CHIA SẺ TRỌNG SỐ
          │                │
          ▼                ▼
      G(x)              G(x')
             \         /
              \       /
               |hiệu|    Δ = G(x) - G(x')
                  │
                  ▼
   pθ(x, x') = f_{L+1}(β₀ + Σ βₖ · Δₖ)    ← Logistic output
```

**Discriminant model cuối cùng:**

$$
p_\theta(x, x') = f_{L+1}\!\left(\beta_0 + \sum_{k=1}^{\ell_L} \beta_k \left[G_\vartheta(x) - G_\vartheta(x')\right]_k\right)
$$

- \(f_{L+1}\) = logistic sigmoid → đầu ra ∈ [0,1].
- \(\theta = (\vartheta, \beta_0, \ldots, \beta_{\ell_L})\) — mở rộng tham số.
- **Dissimilarity** \(D(x,x') = 1 - p_\theta(x, x')\).

### 3.3 Huấn luyện — Composite Bernoulli Likelihood

Tách dữ liệu thành **train** và **valid** (ví dụ \(T_{valid}/T \approx 0.3\)).

**Loss:**

$$
\ell(\theta; \mathcal{D}_{train}) = \sum_{\{x,x'\} \subset \mathcal{D}_{train}} \left[ y(x,x') \log p_\theta(x,x') + (1 - y(x,x')) \log(1 - p_\theta(x,x')) \right]
$$

- \(y = 1\) nếu cùng loài, \(y = 0\) nếu khác loài.
- Đây là **composite likelihood Bernoulli** (gần với logistic regression trên cặp).

### 3.4 Kiến trúc cụ thể

| Tập dữ liệu | Đầu vào | L1 | L2 | L3 | L4 | Output | Tham số |
|--------------|---------|----|----|----|----|--------|---------|
| Mô phỏng (10 quá trình) | 128×128 | 8, kernel 9×9, pool 3×3 | 16, 5×5, 3×3 | 32, 3×3, 2×2 | — | 256 | 213,825 |
| BCI (130 loài) | 256×512 | 64, 20×10, 2×2 | 128, 7×14, 2×2 | 128, 4×8, 2×2 | 256, 4×8, 2×2 | 2048 | 72,956,449 |

### 3.5 Kết quả

- **Mô phỏng:** Siamese CNN đạt **≥10% cao hơn** K-function dissimilarity, và ≥30% cao hơn ngẫu nhiên.
- **BCI:** Chia 130 loài thành **7 cụm sinh thái** rõ ràng — phù hợp giả thuyết "rừng mưa nhiệt đới có **bất đồng nhất không gian lớn** (inhomogeneity) quan trọng hơn tương tác cây-cây".

### 3.6 ✅ Điểm mạnh

1. **Tổng quát hoá tốt** cho các summary statistic cổ điển (K, F, G, J) — CNN học được feature tốt hơn.
2. **One-shot learning** — chỉ cần vài mẫu/lớp là phân loại được.
3. **Học được nhiều mức**: cụm bộ (pair), mẫu (pattern), quá trình (process).
4. Cho phép so sánh **cả pattern matching** lẫn **process matching**.
5. Có **lý thuyết xác suất chính xác** cho feature map (công thức integral ở trên).

### 3.7 ❌ Điểm yếu

1. **Cần lượng lớn dữ liệu**: BCI cần mạng 73 triệu tham số.
2. **Siêu tham số khó chọn**: số lớp L, số node \(\ell_l\), kích thước kernel/pool — hoàn toàn heuristic.
3. **Phụ thuộc rời rạc hoá**: chọn \((d_1, d_2)\) ảnh hưởng lớn đến kết quả.
4. **Không diễn giải được**: không biết CNN học "đặc trưng gì" của quá trình điểm (black-box).
5. **Không tổng quát cho mark phức tạp**: phương pháp chỉ dùng vị trí, chưa tích hợp mark (loại sự kiện).

---

## 4. Phương pháp 2 — SOP (Second-Order Preserving) Permutations

> **Mục tiêu:** Trong kiểm định tương tác giữa 2 quá trình điểm, ta cần "đảo" dữ liệu gốc để tạo **null distribution**. Nhưng hoán vị ngẫu nhiên thường **phá vỡ** cấu trúc K-function. SOP khắc phục bằng MCMC.

### 4.1 Bài toán

Kiểm định: *"Hai quá trình điểm có tương tác không?"*

Trong randomization test:
1. Cố định toạ độ không gian \(x_i\).
2. **Hoán vị thời gian** \(\tilde{t}_i\) ngẫu nhiên.
3. So sánh quá trình hoán vị với quá trình thứ 2.

**Vấn đề:** Quá trình hoán vị KHÔNG còn cùng K-function với dữ liệu → kiểm định **sai lệch**.

Ví dụ: Hawkes process có clustering không-thời gian mạnh → hoán vị thời gian ngẫu nhiên chỉ giữ clustering không gian.

### 4.2 Ý tưởng SOP

Tạo ra \(M\) hoán vị sao cho **L-function** (sqrt của K-function) của chúng khớp phân phối L-function của dữ liệu gốc:

$$
K(r) = \frac{1}{|N|^2} \sum_{i,j} \mathbf{1}\{\|z_i - z_j\| < r\}, \quad L(r) = \sqrt{K(r)}
$$

### 4.3 Thuật toán 2 giai đoạn

```
GIAI ĐOẠN 1:
  for k = 1..M:
      z̃_k = (x_i, t̃_k_i)   ← hoán vị thời gian ngẫu nhiên
      L_k(r) ← K-function của z̃_k
  μ(r) = (1/M) Σ L_k(r)
  ε_k(r) = L_k(r) - μ(r)            ← "sai số" của mỗi hoán vị

GIAI ĐOẠN 2:  (MCMC chỉnh sửa)
  for k = 1..M:
      khởi tạo z̃_k từ giai đoạn 1
      repeat:
          q̃_k ← hoán đổi 2 thời điểm ngẫu nhiên (proposal)
          L_prop(r) ← K-function của q̃_k
          err_prop = ‖ L_prop(r) - L_data(r) - ε_k(r) ‖₂
          chấp nhận q̃_k nếu err_prop giảm
```

**Lưu ý:** Mỗi hoán vị \(z̃_k\) có "target sai số" \(\varepsilon_k(r)\) riêng → phân phối của L-functions khớp dữ liệu gốc.

### 4.4 Ứng dụng — Data augmentation cho CNN-LSTM

Thí nghiệm:
- Hawkes 3D: \([0,1]^2 \times [0,730]\), background \(\mu = 40\), \(\theta = 0.75\), exp kernel \(\omega=100\), Gaussian không gian \(\sigma=0.01\).
- Lưới \(25 \times 25\) không gian × 730 ngày → tensor binary \(14 \times 25 \times 25\) (cửa sổ trượt 14 ngày).
- **CNN-LSTM** dự báo 1 ngày tới.

**Kết quả:** SOP augmentation tăng dữ liệu × 10 → **AUC cao hơn đáng kể** so với không augmentation.

### 4.5 ✅ Điểm mạnh

1. **Giữ nguyên K-function** của quá trình gốc — khắc phục điểm yếu cốt lõi của random permutation.
2. Có **nền tảng lý thuyết vững** (L-function là thống kê đủ trong nhiều trường hợp).
3. **Generic** — áp dụng cho mọi quá trình Hawkes/self-exciting.
4. **Cải thiện rõ rệt** hiệu năng mô hình dự báo khi augmentation.

### 4.6 ❌ Điểm yếu

1. **Chỉ bảo toàn K-function** — không bảo toàn các bậc cao hơn (3rd-order, 4th-order).
2. **Chi phí tính toán lớn**: phải tính L-function mỗi bước MCMC, swap 2 điểm → hội tụ chậm.
3. **K-function đơn giản**: chỉ dùng 1D L(r), không tận dụng K(r) nhiều chiều.
4. **Chưa mở rộng** sang không gian phức tạp (network, mark space).
5. MCMC swap có thể **mắc kẹt** ở local minimum khi K-function có nhiều mode.

---

## 5. Phương pháp 3 — STNPP (Spatio-Temporal-Network Point Process)

> **Mục tiêu:** Mô hình tội phạm ở Valencia (47.125 vụ trong 5 năm) với **mạng lưới đường** + **địa danh đô thị** (1975 điểm thuộc 7 loại). Hawkes truyền thống dùng khoảng cách Euclidean — sai lầm vì tội phạm đi theo đường.

### 5.1 Dữ liệu

- **Sự kiện:** \((t_i, s_i, c_i)\) với \(t_i \in [0,T]\), \(s_i \in S \subset \mathbb{R}^2\), \(c_i \in \{1,2,3\}\) (Assault / Subtraction / Others).
- **Địa danh (landmark):** 7 loại — financial, industrial, market, nightclub, police, restaurant, taxi.
- **Mạng lưới đường:** 8.043 node, 12.309 cạnh có trọng số (chiều dài km), đã vô hướng hoá.

### 5.2 Xây dựng mark

```
Mark = c × ℓ(s)
   c  = loại tội phạm (3 loại)
   ℓ(s) = loại địa danh (7 loại) — xác định bằng kNN
```

→ **Mark space** \(C \times L\) có 21 nhãn.

### 5.3 Hawkes Process mở rộng

Cường độ điều kiện cho mỗi nhãn \(c \times l\):

$$
\lambda_{cl}(t, s \mid \mathcal{H}_t) = \mu_{cl} + \sum_{(t',s',c'\times l') \in \mathcal{H}_t} k(t', t, s', s, c'\times l', c \times l)
$$

**Influence kernel tách 3 phần:**

$$
k(t', t, s', s, c'\times l', c \times l) = f(t',t) \cdot g(s', s) \cdot h(c'\times l', c \times l)
$$

| Phần | Công thức | Diễn giải |
|------|-----------|-----------|
| Temporal \(f\) | \(\beta e^{-\beta(t-t')}\) | Suy giảm theo thời gian (exponential Hawkes) |
| Spatial \(g\) | \(\frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{d^2_{net}(s, s')}{2\sigma^2}\right)\) | Gaussian trên **khoảng cách mạng lưới** \(d_{net}\) — KHÔNG phải Euclidean |
| Mark \(h\) | \(\alpha_{cl, c'l'}\) | Hệ số tương tác giữa các nhãn |

### 5.4 GAT (Graph Attention Network) cho hệ số tương tác mark

Hệ số \(\alpha_{cl, c'l'}\) **phân tách thành**:

$$
\alpha_{cl, c'l'} = a_{cl, c'l'} \cdot p_{cl, c'l'}
$$

- \(a_{cl, c'l'} > 0\): **cường độ** tương tác (học trực tiếp).
- \(0 \le p_{cl, c'l'} \le 1\), \(\sum p = 1\): **xác suất** tương tác (học bằng GAT).

**Multi-head attention trong GAT:**

$$
e^r_{cl, c'l'} = \gamma(W^r X_{cl}, W^r X_{c'l'})
$$

$$
p^r_{cl, c'l'} = \frac{\exp(\text{LeakyReLU}(b_r^\top [W^r X_{cl} \| W^r X_{c'l'}]))}{\sum_{c',l'} \exp(\text{LeakyReLU}(b_r^\top [W^r X_{cl} \| W^r X_{c'l'}]))}
$$

$$
p_{cl, c'l'} = \frac{1}{R}\sum_{r=1}^{R} p^r_{cl, c'l'}
\]

→ GAT tự học **topology mạng giữa các nhãn mark**.

### 5.5 Ước lượng tham số

**Log-likelihood:**

$$
L(\theta) = \sum_{i=1}^n \log \lambda_{c_i l_i}(t_i, s_i) - \sum_{c,l} \int_0^T \int_S \lambda_{cl}(t,s)\, ds\, dt
$$

Tích phân thứ hai **không có dạng đóng** → dùng **SGD** thay vì EM (vì EM intractable).

### 5.6 Kết quả (Valencia)

So sánh với Persistent baseline, VAR, ETAS:

| Mô hình | MAE rare ↓ | MAE frequent ↓ | MAE total ↓ | Train LL ↑ | AIC ↓ |
|---------|-----------|----------------|-------------|------------|-------|
| Persistent | 0.998 | 5.736 | 31.538 | — | — |
| VAR | 0.906 | 3.680 | 21.940 | — | — |
| ETAS | 0.785 | 4.266 | 30.925 | -2.476 | 45039.27 |
| STNPP-GAT | 0.728 | 3.875 | 21.561 | -2.427 | 44173.39 |
| **STNPP** | **0.716** | **3.708** | **20.080** | **-2.413** | **44099.27** |

→ **STNPP thắng tất cả metrics** trên tập in-sample và out-of-sample.

### 5.7 ✅ Điểm mạnh

1. **Khoảng cách mạng lưới** thay Euclidean → mô hình đúng cấu trúc đô thị.
2. **Mark phong phú** (21 nhãn) → nắm được tương tác giữa loại tội phạm × loại địa danh.
3. **GAT tự học topology** → khám phá cộng đồng tội phạm (7 cộng đồng phát hiện).
4. **Vượt trội ETAS** trên mọi metric.

### 5.8 ❌ Điểm yếu

1. **Baseline intensity \(\mu_{cl}\) constant** — chưa phụ thuộc thời gian/không gian.
2. **Giả định tuyến tính cộng** (linear additive Hawkes) — không nắm được tương tác bậc cao.
3. **Phụ thuộc mạng lưới đường có sẵn** — cần dữ liệu GIS chất lượng.
4. **Tích phân không gian tính xấp xỉ** → gradient nhiễu.
5. **Chưa xử lý non-stationarity** (giờ cao điểm, mùa).

---

## 6. Phương pháp 4 — Non-stationary deep STPP (COVID Cali)

> **Mục tiêu:** Mô hình COVID-19 ở Cali, Colombia (38.611 ca). Quá trình **không dừng** theo không gian — khu trung tâm lan khác khu ngoại ô. Kernel cố định (Gaussian đẳng hướng) **không đủ**.

### 6.1 Vấn đề với kernel dừng

Kernel cổ điển \(k(t,t',s,s') = \nu(t,t') \cdot v(s, s')\) với \(v(s,s') = \langle \phi_s, \phi_{s'} \rangle\) — covariance **không đổi** theo vị trí.

Với COVID ở Cali: khu trung tâm thành phố có mật độ cao, lan xa; khu ngoại ô thưa thớt. **Một Gaussian không thể mô tả cả hai**.

### 6.2 Non-stationary neural kernel

Tổng quát hoá feature map:

$$
\phi_s(\cdot) = \sum_{r=1}^{R} w_{sr} \kappa_{sr}(\cdot)
$$

- \(\kappa_{sr}(\cdot)\): **Gaussian component** \(r\) tại vị trí \(s\) với covariance \(\Sigma_{sr}\).
- \(w_{sr}\): trọng số học được.

Kernel không gian trở thành:

$$
v(s, s') = \sum_{r_1, r_2} w_{sr_1} w_{s'r_2} \cdot \frac{1}{\sqrt{|2\pi(\Sigma_{sr_1} + \Sigma_{s'r_2})|}} \exp\!\left(-\frac{1}{2}(s-s')^\top (\Sigma_{sr_1} + \Sigma_{s'r_2})^{-1}(s-s')\right)
$$

→ **Tổng của nhiều Gaussian** với covariance **phụ thuộc vị trí**.

### 6.3 Tham số hoá covariance bằng ellipse

Mỗi \(\Sigma_s\) được biểu diễn qua **điểm focal** \(\psi_s\) của ellipse độ lệch chuẩn (diện tích cố định \(A\)):

$$
\Sigma_s = \tau_z^2 \begin{bmatrix} Q + \|\psi_s\|^2 \cos^2\alpha/2 & \|\psi_s\|^2 \sin(2\alpha)/2 \\ \|\psi_s\|^2 \sin(2\alpha)/2 & Q - \|\psi_s\|^2 \cos^2\alpha/2 \end{bmatrix}
$$

với \(\alpha = \arctan(\psi_{sy}/\psi_{sx})\), \(Q = 4A^2 + \|\psi_s\|^4\pi^2/(2\pi)\).

→ **Một-một** giữa \(\Sigma_s\) và focal point.

### 6.4 Ánh xạ bằng Deep Neural Network

Đặt \(\psi_s = \text{NeuralNet}(s)\). Mạng học ánh xạ **trực tiếp từ toạ độ → focal point → covariance**.

Ưu điểm: mượt, liên tục, dễ tích phân.

### 6.5 Tính toán hiệu quả

Tích phân \(\int_S \lambda(t,s)\, ds\) **không có dạng đóng**.

- **Phương pháp chuẩn**: numerical integration phức tạp \(\mathcal{O}(N^3)\) — 5 phút/epoch với \(N=38.611\).
- **Phương pháp đề xuất**: xấp xỉ \(\mathcal{O}(N)\) với **kiểm soát sai số** — 5 giây/epoch.

→ **Tăng tốc 60×** với sai số có giới hạn lý thuyết.

### 6.6 Kết quả (Cali COVID)

- **In-sample & out-of-sample prediction** tốt hơn stationary kernel.
- **Hạt nhân giải thích được**: trực quan hoá \(\kappa_{sr}\) tại các điểm khác nhau → phát hiện "hướng lan" của dịch.

### 6.7 ✅ Điểm mạnh

1. **Nắm bắt non-stationarity** — covariance thay đổi theo không gian.
2. **Diễn giải được** — focal point ellipse có ý nghĩa hình học.
3. **Hiệu quả tính toán** — giảm từ \(\mathcal{O}(N^3)\) xuống \(\mathcal{O}(N)\).
4. **Mượt** — DNN đảm bảo tính liên tục của covariance.

### 6.8 ❌ Điểm yếu

1. **Giả định temporal kernel dừng** \(\nu(t,t')\) — vẫn dùng Gaussian temporal.
2. **Phụ thuộc neural network** — cần dữ liệu lớn để huấn luyện.
3. **Không có mark** — bài toán COVID chỉ có thời gian-không gian.
4. **Ellipse diện tích cố định** \(A\) — chưa tổng quát.
5. **Chi phí kiểm chứng lý thuyết** — sai số xấp xỉ cần giả định chặt.

---

## 7. Phương pháp 5 — Neural Likelihood Inference

> **Mục tiêu:** Trong nhiều mô hình (LGCP, Thomas, Matérn, Strauss, Geyer, Area-interaction), likelihood chứa **hằng số chuẩn hoá** \(Z(\theta)\) — **không tính được**. Phương pháp này dùng **mạng học sâu làm proxy cho likelihood**.

### 7.1 Vì sao likelihood khó?

$$
L(\theta \mid x) = \frac{1}{Z(\theta)} f(x_1, \ldots, x_n)
$$

$$
Z(\theta) = \sum_{n=0}^{\infty} \frac{1}{n!} \int_{W^n} f(x_1, \ldots, x_n)\, dx_1 \cdots dx_n
$$

- Tích phân trên không gian **vô hạn chiều** (mọi số điểm, mọi vị trí).
- Áp dụng cho LGCP, Thomas, Matérn, Strauss, Geyer...

### 7.2 Các hướng tiếp cận truyền thống

| Phương pháp | Ý tưởng | Hạn chế |
|-------------|----------|---------|
| Pseudo-likelihood (Besag 1974) | Giả định độc lập có điều kiện | Sai số hệ thống cho tương tác mạnh |
| Composite likelihood | Tích của marginal đôi | Mất thông tin bậc cao |
| MCMC | Lấy mẫu từ posterior | Cực chậm với không gian lớn |
| ABC | So sánh thống kê tóm tắt | Cần chọn summary statistic |

### 7.3 Phương pháp Neural Likelihood (CNN-based)

#### Ý tưởng: classifier thay cho likelihood

**Bước 1 — Sinh dữ liệu huấn luyện:**

```python
for theta_i in thetas:                     # Latin Hypercube Sampling
    y_i1 ... y_in ~ p(y | theta_i)          # Mô phỏng realizations
    null_class = permute(y, theta)          # C2: cặp (y, theta') ngẫu nhiên
```

**Bước 2 — Huấn luyện binary classifier:**

- **Class 1 (positive)**: \((y, \theta) \sim p(y \mid \theta)\, p(\theta)\) — đúng cặp.
- **Class 0 (negative)**: \((y, \theta) \sim p(y)\, p(\theta)\) — sai cặp.

**Bước 3 — Kết nối với likelihood:**

$$
\hat{h}(y, \theta) = \frac{p(y \mid \theta) / p(y)}{p(y \mid \theta)/p(y) + 1} = \frac{L(\theta \mid y) / p(y)}{L(\theta \mid y)/p(y) + 1}
$$

→ Cho nhiều quan sát i.i.d.:

$$
L(\theta \mid y_1, \ldots, y_n) \propto \prod_{i=1}^n \frac{\hat{h}(y_i, \theta)}{1 - \hat{h}(y_i, \theta)}
$$

**Classifier chính là likelihood proxy** — bỏ qua hoàn toàn \(Z(\theta)\).

### 7.4 Kiến trúc CNN

| Layer | Output | Filters | Kernel | Activation | Weights |
|-------|--------|---------|--------|------------|---------|
| Conv2D | (48,48,128) | 128 | 3×3 | ReLU | 1,280 |
| MaxPool | (24,24,128) | — | 2×2 | — | 0 |
| Conv2D | (22,22,128) | 128 | 3×3 | ReLU | 147,584 |
| MaxPool | (11,11,128) | — | 2×2 | — | 0 |
| Conv2D | (9,9,128) | 128 | 3×3 | ReLU | 147,584 |
| MaxPool | (5,5,128) | — | 2×2 | — | 0 |
| Conv2D | (3,3,16) | 16 | 3×3 | ReLU | 18,448 |
| MaxPool | (2,2,16) | — | 2×2 | — | 0 |
| Flatten | (64,) | — | — | — | 0 |
| Concat với θ | (66,) | — | — | — | 0 |
| Dense | (64,) | — | — | ReLU | 4,288 |
| Dense | (16,) | — | — | ReLU | 1,040 |
| Dense | (8,) | — | — | ReLU | 136 |
| Dense | (2,) | — | — | Softmax | 18 |

→ Trộn **đặc trưng không gian** (qua CNN) với **tham số θ** (qua concatenation).

### 7.5 Mở rộng với GNN — bảo toàn toạ độ gốc

CNN cần **rời rạc hoá** lưới → mất thông tin toạ độ chính xác.
**GNN** hoạt động trên **đồ thị kNN** giữa các điểm.

```
Mỗi điểm → node với d features (vị trí, mật độ cục bộ)
3 GCN layers (32 chiều ẩn)
Global mean pooling → vector 32-d
Linear → θ_hat
```

**Ưu điểm GNN:** không cần chọn lưới, bảo toàn cấu trúc hình học.

### 7.6 Kết quả (tốc độ)

| Mô hình | N điểm | CNN (s) | Composite-Lik (s) | Speedup |
|---------|--------|---------|--------------------|---------|
| Matérn (1) | 2000 | 1.66 | 0.98 | ×0.59 (chậm hơn) |
| Matérn (2) | 9000 | 1.81 | 17.72 | ×9.79 |
| LGCP | 3000 | 1.69 | 2.21 | ×1.31 |

| Mô hình | N điểm | GNN (s) | Truyền thống (s) | Speedup |
|---------|--------|---------|-------------------|---------|
| Matérn | 1000 | 0.003 | 0.917 | **×291** |
| Strauss | 100 | 0.001 | 0.009 | ×7.6 |

→ GNN **nhanh hơn 291 lần** cho Matérn — vẫn giữ độ chính xác tương đương.

### 7.7 ✅ Điểm mạnh

1. **Bypass hoàn toàn** \(Z(\theta)\) — không cần tính hằng số chuẩn hoá.
2. **Tốc độ cực nhanh** với GNN (291×).
3. **Áp dụng được cho mọi mô hình** (Gibbs, Cox, cluster...).
4. **Likelihood có thể dùng cho AIC, BIC, posterior** — đầy đủ khuôn khổ suy luận Bayes.
5. **Không cần prior** (khác với MCMC).

### 7.8 ❌ Điểm yếu

1. **Cần dữ liệu mô phỏng lớn** — phải sinh hàng ngàn mẫu với nhiều giá trị θ.
2. **Coverage không đạt 95%** trong một số cấu hình → posterior ước lượng chưa đủ tin cậy.
3. **CNN phụ thuộc lưới** — chọn resolution là heuristic.
4. **Chỉ ước lượng θ** — chưa tổng quát cho nhiều mô hình cạnh tranh (model selection).
5. **Không diễn giải được**: classifier học "đặc trưng gì" của likelihood là black-box.

---

## 8. Tổng hợp điểm mạnh & điểm yếu

### 8.1 Ma trận so sánh

| Tiêu chí | 1. Siamese CNN | 2. SOP | 3. STNPP | 4. Neural kernel | 5. Neural likelihood |
|----------|----------------|--------|----------|------------------|----------------------|
| Bài toán chính | Phân biệt mẫu | Augmentation | Crime Hawkes | Non-stationary STPP | Likelihood-free |
| Mạng | CNN + Siamese | MCMC | Hawkes + GAT | DNN feature | CNN / GNN |
| Khả năng tổng quát | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Tốc độ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Diễn giải được | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Xử lý mark | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| Xử lý network | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| Xử lý non-stationary | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Đã có cơ sở lý thuyết | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Dữ liệu cần | Lớn | Trung bình | Lớn | Lớn | Lớn (mô phỏng) |

### 8.2 Điểm mạnh chung của dòng nghiên cứu Mateu

1. **Đa dạng bài toán**: 5 phương pháp → 5 bài toán khác nhau nhưng **cùng khung lý thuyết** point process.
2. **Kết hợp chặt chẽ giữa lý thuyết xác suất và học sâu** — không phải "deep learning chạy bừa".
3. **Có ứng dụng thực tế mạnh** (BCI rừng, Valencia crime, Cali COVID) — không phải benchmark suông.
4. **Đều có paper ở journal hàng đầu** (JRSS C, Stat, ADAC) — peer-review nghiêm túc.

### 8.3 Điểm yếu chung

1. **Phụ thuộc dữ liệu lớn** — 5/5 phương pháp đều cần dataset quy mô (BCI 130 loài, Valencia 47k vụ, Cali 38k ca).
2. **Khả năng diễn giải hạn chế** — chỉ 2/5 (STNPP, Neural kernel) có thể "mở hộp đen" ra xem.
3. **Hyperparameter heuristic** — kiến trúc mạng được chọn tay.
4. **Chưa xử lý tốt scale cực lớn** (>1M điểm).
5. **Chưa có benchmark chuẩn** so sánh với baselines (như transformer, diffusion models trong STPP).

---

## 9. Hướng ứng dụng cho bạn

### 9.1 Nếu bạn làm về dịch tễ (COVID, sốt xuất huyết, ...)

- **Bắt đầu từ Phương pháp 4** (Neural kernel) — đã validate trên Cali COVID.
- Mở rộng thêm **temporal kernel phi tuyến** (hiện đang Gaussian — chưa thực tế cho seasonality).
- Kết hợp với **Phương pháp 1** (Siamese CNN) để phát hiện **sớm** khi một vùng chuyển sang "chế độ" mới.

### 9.2 Nếu bạn làm về tội phạm / giao thông

- **Phương pháp 3** (STNPP) là lựa chọn tốt nhất — đã thắng ETAS ở Valencia.
- Cần dữ liệu **mạng lưới đường** (OpenStreetMap) + **POI** (Google Places API hoặc tương đương).
- Mở rộng: thêm **thời gian trong ngày** (giờ cao điểm), **ngày trong tuần**.

### 9.3 Nếu bạn làm về sinh thái / nông nghiệp

- **Phương pháp 1** (Siamese CNN) + BCI dataset là bài toán kinh điển.
- Có thể dùng để **phân loại mẫu đất, mẫu cây trồng** theo vùng địa lý.

### 9.4 Nếu bạn cần likelihood để Bayesian inference

- **Phương pháp 5** (Neural likelihood với GNN) — tốc độ 291×, áp dụng được cho mọi mô hình khó.
- Cần sinh trước **nhiều mẫu mô phỏng** với θ đa dạng (Latin Hypercube).

### 9.5 Nếu dữ liệu hạn chế

- **Phương pháp 2** (SOP) — augmentation hiệu quả cho mọi Hawkes process.

### 9.6 Stack công nghệ gợi ý

| Phương pháp | Framework | GPU cần |
|-------------|-----------|---------|
| 1. Siamese CNN | PyTorch + `torchvision` | Có |
| 2. SOP | NumPy + custom MCMC | Không |
| 3. STNPP | PyTorch + `torch-geometric` (GAT) | Có |
| 4. Neural kernel | PyTorch + custom covariance | Có |
| 5. Neural likelihood (CNN) | PyTorch + `torchvision` | Có |
| 5. Neural likelihood (GNN) | PyTorch + `torch-geometric` | Có |

### 9.7 Bước tiếp theo để học sâu

1. **Đọc trực tiếp 5 papers** trong slide 2 — tất cả đều có mã nguồn minh hoạ trong `Sppm` R package hoặc `spatstat`.
2. **Chạy lại thí nghiệm Siamese CNN** trên BCI — bạn có thể tìm dataset ở [`spatstat.data`](https://spatstat.org/) hoặc R `spatstat` package.
3. **So sánh Neural Likelihood (CNN vs GNN)** trên một mô hình đơn giản (Matérn) trước khi mở rộng.
4. **Áp dụng STNPP cho bài toán của bạn** — nếu có dữ liệu mạng lưới + mark.

---

## Tài liệu tham khảo chính

> Trích từ slide 2 của bài trình bày.

1. **Jalilian, A. & Mateu, J.** (2023). *Assessing similarities between spatial point patterns with a Siamese Neural Network discriminant model.* Advances in Data Analysis and Classification, **17**, 21–42.
2. **Dong, Z., Zhu, S., Xie, Y., Mateu, J. & Rodriguez-Cortes, F.** (2023). *Non-stationary spatio-temporal point process modeling for high-resolution COVID-19 data.* Journal of the Royal Statistical Society C, **72**, 368–386.
3. **Mohler, G. & Mateu, J.** (2024). *Second order preserving point process permutations.* Stat. doi: 10.1002/sta4.558.
4. **Dong, Z., Mateu, J. & Xie, Y.** (2025). *Spatio-temporal-network point processes for modeling crime incidents with landmarks.* Submitted.
5. **Platero, J., Walchessen, J., Kuusela, M. & Mateu, J.** (2025). *Neural likelihood inference for complex spatial points processes.* Submitted.

---

> 📝 **Ghi chú cuối:** Đây là bản dịch & tổng hợp từ slide trình bày ECSIA 2025. Nếu bạn cần đào sâu bất kỳ phương pháp nào (ví dụ: implement lại Siamese CNN, hoặc chạy lại STNPP trên dữ liệu của bạn), hãy cho tôi biết — tôi có thể đi tiếp vào code.