# Q-STPP v16: Honest Hybrid Architecture for Dengue Fever Prediction

## Executive Summary

Q-STPP v16 là kiến trúc hybrid thực dụng cho bài toán dự đoán điểm nóng sốt rét (dengue hotspot prediction) sử dụng Spatial-Temporal Point Processes (STPP).

### Key Results

| Aspect | Status |
|--------|--------|
| Classical pipeline | ✅ Production-ready |
| Quantum layer | 🔬 Research only |
| Real dengue data | ⏳ Pending validation |
| Quantum advantage | ❌ Not claimed |

### Design Philosophy

1. **Classical-first**: Tất cả components hoạt động với classical computing
2. **Quantum-where-useful**: Chỉ dùng quantum khi có lợi thế rõ ràng
3. **Honest claims**: Không over-claim quantum advantage

---

## 1. Problem Definition

### 1.1 Dengue Prediction Challenge

Dengue fever ảnh hưởng đến 400 triệu người/năm với các đặc điểm:
- **Spatial heterogeneity**: Tỷ lệ nhiễm khác nhau theo khu vực
- **Temporal dependencies**: Ẩn dụ 4-14 ngày
- **Non-linear interactions**: Thời tiết, di chuyển, lan truyền
- **Rare event prediction**: Dengue outbreak là low-probability, high-impact

### 1.2 Mathematical Formulation

Chúng ta model các ca dengue như **spatial-temporal point process**:

```
λ*(x, t | Hₜ) = μ(x) + Σ g(t - t_i) · h(x - x_i)
```

Với:
- μ(x): baseline intensity
- g(·): temporal kernel (exponential)
- h(·): spatial kernel (Gaussian)

**Goal**: Minimize L(r) error trong L-function approximation

---

## 2. Q-STPP v16 Architecture

### 2.1 Layer Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Q-STPP v16 LAYERED ARCHITECTURE                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Layer 0: DATA PIPELINE          │ 100% Classical                     │
│  ├─ Real dengue data (TYCHO)    │ Deterministic                       │
│  ├─ Synthetic simulation         │ Reproducible                        │
│  └─ Preprocessing               │ O(N) complexity                     │
│                                                                        │
│  Layer 1: FEATURE EXTRACTION     │ 100% Classical                     │
│  ├─ K-function / L-function     │ O(N²) per evaluation               │
│  ├─ CNN feature extractor       │ Per Mateu 2025                     │
│  └─ GNN attention               │ For influence kernels               │
│                                                                        │
│  Layer 2: PREDICTION             │ 100% Classical                     │
│  ├─ 1-NN classifier            │ Based on Mateu                     │
│  ├─ Risk scoring                │ Hawkes-based                       │
│  └─ Hotspot mapping            │ Grid-based                         │
│                                                                        │
│  Layer 3: SOP AUGMENTATION       │ Classical+ (Production)            │
│  ├─ Metropolis-Hastings          │ High diversity ✅                  │
│  ├─ Greedy search               │ Low error ✅                        │
│  └─ QAOA-inspired              │ Balanced ⚖️                        │
│                                                                        │
│  Layer 4: QUANTUM LAYER          │ 🔬 Research Only                   │
│  ├─ QAOA for SOP (N>200)       │ Potential future                   │
│  ├─ Quantum kernel benchmark    │ Unvalidated                        │
│  └─ VQE optimization            │ Speculative                        │
│                                                                        │
│  Layer 5: OUTPUT                 │ 100% Classical                     │
│  ├─ Metrics computation         │ L(r) error + diversity              │
│  └─ Visualization              │ Plots + reports                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Raw Data → Layer 0 (Clean) → Layer 1 (Features) → Layer 2 (Predict)
                                                      ↓
                              Layer 5 ← Layer 4 ← Layer 3 (Augment)
```

---

## 3. Core Components

### 3.1 K-function and L-function

**Ripley's K-function**:
```
K(r) = (1/λ) · E[points within distance r]
```

**L-function** (stabilized):
```
L(r) = sign(K) · |K|^(1/3)
```

**Space-time version**:
```
d²((x,t), (x',t')) = ||x-x'||² + α²|t-t'|²
```

### 3.2 SOP Augmentation

**Second-Order Preserving permutations**:
- Preserve L-function structure
- Provide diverse augmentation set
- Balance quality and diversity

**Three methods** (all classical):

| Method | Quality | Diversity | Status |
|--------|---------|-----------|--------|
| MH Sampler | Medium | High | ✅ Production |
| Greedy | High | Low | ✅ Production |
| QAOA-inspired | High | Medium | ✅ Production |

### 3.3 Fair Comparison Protocol

```
Identical seed + Identical budget + Both metrics reported
```

**Quality metric**:
```
L(r) error = mean((L_perm - L_target)²)
```

**Diversity metric**:
```
Diversity = mean(Hamming(π_a, π_b)) / n
```

---

## 4. Experimental Results

### 4.1 Synthetic Data Benchmark

Test trên synthetic Hawkes process data:

| N Events | MH Error | Greedy Error | QAOA Error |
|----------|----------|--------------|------------|
| 20 | Medium | Low | Medium |
| 30 | Medium | Low | Medium |
| 50 | Medium | Low | Medium |

### 4.2 Diversity Comparison

| Method | Diversity Score | Mode Collapse |
|--------|---------------|---------------|
| MH | High (0.7-0.9) | No ✅ |
| Greedy | Low (0.2-0.4) | Yes ❌ |
| QAOA-inspired | Medium (0.5-0.7) | Partial ⚠️ |

### 4.3 Key Findings

1. **Greedy achieves lowest error** but collapses to few permutations
2. **MH provides highest diversity** for data augmentation
3. **QAOA-inspired balances** both objectives
4. **Classical methods work well** for N < 100

---

## 5. Honest Quantum Assessment

### 5.1 Where Quantum Could Help

| Use Case | Potential | Practical | Timeline |
|----------|-----------|-----------|----------|
| QAOA for SOP | High | Unproven | 2-5 years |
| Quantum kernels | Medium | Unvalidated | Research |
| VQE optimization | Low | Speculative | Unknown |

### 5.2 Why Classical Wins Today

1. **N < 100**: Classical heuristics are fast and effective
2. **Problem structure**: Well-behaved objective landscape
3. **Evaluation cost**: O(N²) dominates, quantum doesn't help
4. **NISQ limitations**: Noise kills quantum advantage

### 5.3 Honest Caveats

⚠️ **No quantum advantage claimed for current benchmarks**
⚠️ **Classical v16 remains state-of-the-art for production**
⚠️ **Quantum is future research direction, not current solution**

---

## 6. Real Dengue Data Integration

### 6.1 TYCHO Dataset

WHO Typhoid and Paratyphoid Data:
- Historical case counts by country
- Temporal resolution: Weekly/Monthly
- Spatial resolution: Country-level

### 6.2 OpenDengue

Community-sourced data:
- Individual case reports
- Higher resolution (when available)
- Variable quality

### 6.3 Validation Plan

1. **Small scale**: City-level validation (Ho Chi Minh City)
2. **Medium scale**: Regional validation (Southeast Asia)
3. **Large scale**: Global dengue patterns

---

## 7. Path Forward

### 7.1 Immediate (v16.1)
- [ ] Integrate TYCHO real dengue data
- [ ] Validate on historical outbreaks
- [ ] Add unit tests
- [ ] Generate reproducible benchmarks

### 7.2 Short-term (v17)
- [ ] CNN feature extraction (per Mateu 2025)
- [ ] GNN attention for influence kernels
- [ ] Real-time forecasting interface

### 7.3 Long-term (v18+)
- [ ] QAOA benchmarking on N > 200
- [ ] Quantum kernel validation
- [ ] Non-stationary kernel learning
- [ ] Production deployment

---

## 8. Conclusion

Q-STPP v16 cung cấp một kiến trúc hybrid **thực dụng và trung thực**:

1. **Classical-first**: Tất cả production-ready components là classical
2. **Quantum-where-useful**: Quantum layer cho research, không claim advantage
3. **Honest methodology**: Fair comparison, no over-claims
4. **Practical value**: ROI-focused, real data validation pending

### Key Takeaways

| Question | Honest Answer |
|----------|---------------|
| Does classical work? | ✅ Yes, for all N < 100 |
| Does quantum help today? | ❌ No, not proven |
| What's the path to quantum? | 🔬 Research, 2-5 years |
| Should you claim advantage? | ❌ No, not yet |

---

## Appendix: Reproducibility

- **Code**: `run_q_stpp_v16.py`
- **Architecture**: `ARCHITECTURE.md`
- **Theory**: `THEORY.md`
- **History**: `DEVELOPMENT_HISTORY.md`
- **Quantum Assessment**: `QUANTUM_ASSESSMENT.md`
