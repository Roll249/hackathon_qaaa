# Q-STPP v16 Architecture

## Executive Summary

**Q-STPP v16** là kiến trúc hybrid thực dụng cho bài toán dự đoán điểm nóng sốt rét (dengue fever hotspot prediction) sử dụng Spatial-Temporal Point Processes (STPP).

### Design Philosophy
1. **Classical-first**: Tất cả components chạy được bằng classical → implement bằng classical
2. **Quantum-where-useful**: Chỉ dùng quantum khi có lợi thế rõ ràng, với honest caveats
3. **Honest claims**: Không over-claim quantum advantage

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Q-STPP v16: HYBRID ARCHITECTURE                            │
│                         Classical-First, Quantum-Where-Useful                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────────────────────┐
                              │          INPUT DATA                 │
                              │  • Real dengue data (TYCHO)        │
                              │  • Synthetic Hawkes (validation)    │
                              │  • Weather covariates (optional)    │
                              └──────────────────┬──────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
        ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
        │    LAYER 0        │      │    LAYER 1        │      │    LAYER 2        │
        │  DATA PIPELINE    │      │ FEATURE EXTRACT   │      │    PREDICTION     │
        │    (Classical)    │      │    (Classical)    │      │    (Classical)    │
        │                   │      │                   │      │                   │
        │ • Data loading    │      │ • K/L-function    │      │ • 1-NN class.     │
        │ • Preprocessing   │─────▶│ • CNN features    │─────▶│ • Risk scoring    │
        │ • Discretization  │      │ • GNN attention   │      │ • Hotspot maps    │
        │ • Normalization   │      │ • Non-stat. kernel│      │ • Forecast        │
        └───────────────────┘      └───────────────────┘      └───────────────────┘
                    │                            │                            │
                    │                            │                            │
                    └────────────────────────────┼────────────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
        ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
        │    LAYER 3        │      │    LAYER 4        │      │    LAYER 5        │
        │  SOP AUGMENTATION │      │  QUANTUM LAYER    │      │    OUTPUT         │
        │    (Classical+)   │      │  (Future/Benchmark)│      │    (Classical)    │
        │                   │      │                   │      │                   │
        │ • MH sampler      │      │ • QAOA SOP (N>200)│      │ • R² metrics      │
        │ • Greedy search   │      │ • Q kernel (bench)│      │ • L(r) error      │
        │ • QAOA-inspired   │─────▶│ • VQE optim      │      │ • Visualizations  │
        │ • [Future: QAOA]  │      │ • Honest caveats  │      │ • Reports         │
        └───────────────────┘      └───────────────────┘      └───────────────────┘
```

---

## 2. Layer Specifications

### 2.1 Layer 0: Data Pipeline (Classical)

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 0: DATA PIPELINE                       │
│                     100% Classical - No quantum needed           │
└─────────────────────────────────────────────────────────────────┘

DataSource
├── RealData
│   ├── TYCHO (WHO) - historical dengue cases
│   ├── OpenDengue - community data
│   └── Format: (lat, lon, timestamp, case_count)
│
├── SyntheticData (for validation)
│   ├── HawkesProcess - self-exciting patterns
│   ├── PoissonProcess - baseline
│   └── LGCP - log-Gaussian Cox process
│
└── Preprocessor
    ├── spatial_discretize(points, grid_size=12)
    │   └── Maps (x,y) → cell index (i,j)
    │
    ├── temporal_binning(times, window='1D')
    │   └── Aggregates events by time window
    │
    └── normalize_coordinates(coords)của
        └── Ensures consistent spatial scale
```

**Key operations**:
- O(N) for data loading
- O(N) for discretization
- All classical, deterministic

### 2.2 Layer 1: Feature Extraction (Classical)

```
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1: FEATURE EXTRACTION                     │
│            Based on Mateu 2025 (ECSIA) methodology               │
└─────────────────────────────────────────────────────────────────┘

FeatureExtractor
│
├── KFunction (Second-Order Statistics)
│   │
│   ├── compute_K(r, events)
│   │   └── K(r) = (1/λ) × (1/N²) × Σ 𝟙(dij < r)
│   │
│   └── compute_L(r)
│       └── L(r) = sign(K) × |K|^(1/3)  [stabilized transform]
│
├── LFunction (Ripley's L)
│   │
│   ├── compute_L_stpp(t, x, y, r_values)
│   │   └── Space-time distance: d² = ||x-x'||² + α²|t-t'|²
│   │
│   └── l_error(L_perm, L_target)
│       └── MSE(L_perm - L_target) → lower is better
│
├── CNNFeatureExtractor (per Mateu slides 17-19)
│   │
│   ├── discretize_to_grid(points, d1=12, d2=12)
│   │   └── Creates binary grid from point pattern
│   │
│   ├── SiameseCNN(input_shape=(d1,d2))
│   │   ├── Conv2D(8, kernel=3×3) + ReLU
│   │   ├── MaxPool(2×2)
│   │   ├── Conv2D(16, kernel=3×3) + ReLU
│   │   ├── MaxPool(2×2)
│   │   ├── Flatten
│   │   └── Dense(64) → feature vector
│   │
│   └── extract_features(grid) → φ(x)
│
└── GNNAttention (for influence kernel learning)
    │
    ├── GraphAttentionNetwork(mark_dim, hidden_dim=32)
    │   ├── Multi-head self-attention
    │   └── Learns α_cl,c'l' coefficients
    │
    └── compute_influence_kernel(θ) → k(t',t,s',s,c×l,c'×l')
```

**Key insights from Mateu 2025**:
- K-function baseline outperforms CNN on small data (slide 47)
- CNN advantage emerges with more data
- GNN captures mark-space interactions

### 2.3 Layer 2: Prediction (Classical)

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAYER 2: PREDICTION                         │
│                    Pattern Classification & Risk                   │
└─────────────────────────────────────────────────────────────────┘

Predictor
│
├── PatternClassifier
│   │
│   ├── OneNNClassifier(features, labels)
│   │   └── 1-Nearest Neighbor with L2 distance
│   │
│   └── predict(new_pattern, k=1)
│       └── Returns predicted class + probability
│
├── RiskScorer
│   │
│   ├── compute_intensity(events, kernel_params)
│   │   └── λ(x,t) = μ + Σ k(t',t,s',s)
│   │
│   ├── compute_hotspot_prob(grid, intensity)
│   │   └── P(hotspot| intensity) via threshold
│   │
│   └── generate_risk_map(spatial_grid)
│       └── Risk score per grid cell
│
└── ForecastEngine
    │
    ├── predict_future(current_events, horizon)
    │   └── Uses fitted Hawkes parameters
    │
    └── generate_alert(forecast, threshold)
        └── Returns alert level (1-5)
```

### 2.4 Layer 3: SOP Augmentation (Classical-First)

```
┌─────────────────────────────────────────────────────────────────┐
│               LAYER 3: SOP AUGMENTATION                          │
│         Second-Order Preserving Permutations (Mohler-Mateu 2024)  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  PURPOSE: Data augmentation for training ML models          │ │
│  │  CRITERIA:                                                 │ │
│  │    1. Preserve L(r) structure (LOW error)                 │ │
│  │    2. Produce DIVERSE set of permutations                  │ │
│  │    3. Same computational budget for fair comparison         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

SOPSearch
│
├── Method 1: Metropolis-Hastings (Classical)
│   │
│   ├── proposal_swap(perm)
│   │   └── Randomly swap two time indices
│   │
│   ├── accept_reject(cand_err, cur_err, temperature)
│   │   └── P(accept) = min(1, exp(-ΔE/T))
│   │
│   └──优点: High diversity, ✅ proven
│   └──缺点: May not reach lowest error
│
├── Method 2: Greedy Search (Classical)
│   │
│   ├── greedy_swap(perm, n_swaps=1)
│   │   └── Only accept improving swaps
│   │
│   └──优点: Lowest error, fast
│   └──缺点: Low diversity, mode collapse
│
├── Method 3: QAOA-Inspired Multi-Swap (Classical)
│   │
│   ├── multi_swap_proposal(perm, n_swaps)
│   │   └── Propose multiple swaps simultaneously
│   │
│   └──优点: Balances error and diversity
│   └──缺点: Still classical heuristic
│
└── [FUTURE] Method 4: Genuine QAOA (Quantum)
    │
    ├── XYMixerHamiltonian
    │   └── H_M = Σ (X_i X_j + Y_i Y_j)
    │
    ├── CostHamiltonian
    │   └── H_C = Σ |L(π_i) - L_target|²
    │
    └──优点: Theoretical quantum advantage for N>200
    └──缺点: NISQ limitations, needs validation
```

**FAIR COMPARISON PROTOCOL**:
- Same random seed
- Same evaluation budget (L-function calls)
- Both quality (L(r) error) AND diversity reported

### 2.5 Layer 4: Quantum Layer (Honest Benchmark)

```
┌─────────────────────────────────────────────────────────────────┐
│                       LAYER 4: QUANTUM LAYER                    │
│                      For Research & Benchmarking                  │
│                                                                   │
│  ⚠️ HONEST CAVEATS:                                              │
│  • No quantum advantage claimed for current problem sizes         │
│  • NISQ hardware too noisy for practical benefit                 │
│  • Quantum = future research direction, not current solution     │
│  • Classical v15 methods remain state-of-the-art                  │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

QuantumLayer
│
├── Component A: QAOA for SOP (Future Research)
│   │
│   ├── xy_mixer_circuit(p, β, γ)
│   │   └── U(β,γ) = exp(-iβ B) exp(-iγ C)
│   │
│   ├── CostFunction
│   │   └── C(π) = ||L(π) - L_target||²
│   │
│   └── StateVectorSimulation
│       └── For small N (≤20) on classical simulator
│
│   **When useful**: N > 200, specific problem structures
│   **Honest claim**: Potential future advantage, not proven
│
├── Component B: Quantum Kernel (Benchmark)
│   │
│   ├── IQPEmbedding(features)
│   │   └── φ(x) → |ψ⟩ via IQP circuit
│   │
│   ├── QuantumKernel(x, x')
│   │   └── k(x,x') = |⟨ψ(x)|ψ(x')⟩|²
│   │
│   └── ClassicalSimulation
│       └── 2^n statevector for n ≤ 15 qubits
│
│   **When useful**: Specific pattern families (periodic vs cluster)
│   **Honest claim**: Method exploration, not demonstrated advantage
│
└── Component C: VQE for Kernel Optimization (Speculative)
    │
    ├── VariationalForm(params)
    │   └── Parametrized quantum circuit
    │
    ├── ObjectiveFunction(θ)
    │   └── Negative log-likelihood
    │
    └── Optimizer
        └── COBYLA/SPSA gradient descent
    │
    **When useful**: Very large parameter spaces
    **Honest claim**: Research direction, unproven
```

### 2.6 Layer 5: Output (Classical)

```
┌─────────────────────────────────────────────────────────────────┐
│                       LAYER 5: OUTPUT                            │
│                     Metrics & Visualization                      │
└─────────────────────────────────────────────────────────────────┘

OutputGenerator
│
├── MetricsComputer
│   │
│   ├── compute_l_error(L_perm, L_target)
│   │   └── Primary quality metric
│   │
│   ├── compute_diversity(permutations)
│   │   └── Mean pairwise Hamming distance
│   │
│   └── compute_r2_score(predicted, actual)
│       └── Classification accuracy
│
├── VisualizationEngine
│   │
│   ├── plot_l_function_curves(L_curves, r_values)
│   │   └── Shows L(r) preservation
│   │
│   ├── plot_error_vs_diversity(methods)
│   │   └── Trade-off visualization
│   │
│   ├── plot_hotspot_map(predictions, grid)
│   │   └── Risk heatmap
│   │
│   └── plot_quantum_comparison(classical, quantum)
│       └── Honest side-by-side (with caveats)
│
└── ReportGenerator
    │
    ├── generate_summary_table(metrics)
    │   └── Markdown table with all results
    │
    └── generate_markdown_report()
        └── Full technical report
```

---

## 3. Data Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                 │
└────────────────────────────────────────────────────────────────────────────┘

INPUT
  │
  ▼
┌─────────────────┐
│  Layer 0        │  Raw Dengue Data
│  Data Pipeline   │  ↓
│  (Classical)    │  Cleaned & Discretized Events
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Layer 1        │  Spatial-Temporal Features
│  Feature Extract │  ├─ K/L-function summaries
│  (Classical)    │  ├─ CNN embeddings
│                 │  └─ GNN attention weights
└────────┬────────┘
         │
         ├──────────────────────────────────────────────┐
         │                                              │
         ▼                                              ▼
┌─────────────────┐                           ┌─────────────────┐
│  Layer 2        │                           │  Layer 3        │
│  Prediction     │                           │  SOP Augment    │
│  (Classical)    │                           │  (Classical+)   │
│                 │                           │                 │
│  ├─ 1-NN class │                           │  ├─ MH sampler  │
│  ├─ Risk score │                           │  ├─ Greedy      │
│  └─ Hotspot map│                           │  └─ QAOA-inspired
└────────┬────────┘                           └────────┬────────┘
         │                                              │
         │         ┌────────────────────────────────────┘
         │         │
         ▼         ▼
┌─────────────────────────────────┐
│         Layer 5                  │
│         Output                   │
│                                 │
│  ┌─────────────────────────────┐ │
│  │  Metrics:                  │ │
│  │  • L(r) error             │ │
│  │  • Diversity score         │ │
│  │  • R² / Accuracy          │ │
│  └─────────────────────────────┘ │
│                                 │
│  ┌─────────────────────────────┐ │
│  │  Visualizations:           │ │
│  │  • L-function curves       │ │
│  │  • Hotspot maps            │ │
│  │  • Comparison plots        │ │
│  └─────────────────────────────┘ │
└─────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │  Layer 4 (Optional/Future):              │
  │  Quantum Benchmarks — same output format │
  └─────────────────────────────────────────┘
```

---

## 4. File Structure

```
quantum-dengue-stpp/
│
├── run_q_stpp_v16.py              # Main v16 pipeline
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py              # Layer 0: Data loading
│   │   ├── preprocessor.py        # Layer 0: Preprocessing
│   │   └── synthetic.py           # Layer 0: Hawkes/Poisson simulation
│   │
│   ├── features/
│   │   ├── k_function.py         # Layer 1: K-function
│   │   ├── l_function.py          # Layer 1: L-function
│   │   ├── cnn_extractor.py       # Layer 1: CNN features
│   │   └── gnn_attention.py       # Layer 1: GNN attention
│   │
│   ├── prediction/
│   │   ├── classifier.py          # Layer 2: 1-NN classifier
│   │   ├── risk_scorer.py         # Layer 2: Risk scoring
│   │   └── forecaster.py          # Layer 2: Hawkes forecast
│   │
│   ├── augmentation/
│   │   ├── sop_search.py          # Layer 3: Base SOP
│   │   ├── metropolis_hastings.py # Layer 3: MH method
│   │   ├── greedy_search.py       # Layer 3: Greedy method
│   │   └── qaoa_inspired.py       # Layer 3: QAOA-inspired
│   │
│   ├── quantum/
│   │   ├── qaoa_sop.py            # Layer 4: Genuine QAOA (future)
│   │   ├── quantum_kernel.py       # Layer 4: Q kernel benchmark
│   │   └── vqe_optim.py           # Layer 4: VQE research
│   │
│   └── output/
│       ├── metrics.py             # Layer 5: Metrics
│       ├── plots.py               # Layer 5: Visualization
│       └── report.py              # Layer 5: Report generation
│
├── output_result/
│   └── q_stpp_v16/
│       ├── fair_comparison_results.json
│       └── quantum_benchmark_results.json
│
├── docs/
│   ├── ARCHITECTURE.md            # This file
│   ├── THEORY.md                 # Mathematical foundations
│   ├── QUANTUM_ASSESSMENT.md     # Honest quantum analysis
│   ├── Q_STPP_V16_REPORT.md      # Technical report
│   └── DEVELOPMENT_HISTORY.md    # Version history
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_augmentation.py
│   └── test_quantum_layer.py
│
├── requirements.txt
└── README.md
```

---

## 5. Complexity Analysis

| Component | Time Complexity | Space Complexity | Classical/Quantum |
|-----------|-----------------|------------------|------------------|
| Data loading | O(N) | O(N) | ✅ Classical |
| K-function | O(N²) | O(N²) | ✅ Classical |
| L-function | O(N²) | O(N²) | ✅ Classical |
| CNN features | O(N × D) | O(D) | ✅ Classical |
| GNN attention | O(N² × H) | O(N²) | ✅ Classical |
| MH sampler | O(N² × S) | O(N) | ✅ Classical |
| Greedy search | O(N² × S) | O(N) | ✅ Classical |
| QAOA-inspired | O(N² × S × K) | O(N) | ✅ Classical |
| **Genuine QAOA** | O(2ⁿ) | O(2ⁿ) | ⚠️ Quantum (N≤20) |
| **Q kernel** | O(2ⁿ) | O(2ⁿ) | ⚠️ Quantum (N≤15) |

Where:
- N = number of events
- S = number of swaps
- K = multi-swap count
- H = attention heads
- D = feature dimension
- n = number of qubits

---

## 6. Honest Claims Summary

### What Works (Classical)

| Method | N Range | Quality | Diversity | Status |
|--------|---------|---------|-----------|--------|
| MH Sampler | All | Medium | High | ✅ Production-ready |
| Greedy | All | High | Low | ✅ Production-ready |
| QAOA-inspired | All | High | Medium | ✅ Production-ready |

### What Could Work (Quantum - Research)

| Method | N Range | Theoretical | Practical | Status |
|--------|---------|-------------|-----------|---------|
| QAOA SOP | N > 200 | Potential | Unproven | 🔬 Research |
| Q Kernel | N ≤ 15 | Possible | Unvalidated | 🔬 Research |
| VQE Optim | Large | Speculative | Unproven | 🔬 Research |

### Key Messages

1. **Classical v16 methods are production-ready** for current problem sizes
2. **Quantum is a future research direction**, not current solution
3. **No quantum advantage claimed** for any current benchmark
4. **Honest comparison** between classical methods only

---

## 7. Dependencies

```
# Core
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0

# ML (optional)
torch>=2.0.0          # For CNN/GNN features
torch_geometric>=2.0  # For GNN

# Quantum (optional - for Layer 4 benchmarks)
pennylane>=0.30.0    # Quantum machine learning
qiskit>=0.45.0       # QAOA implementation

# Data
pandas>=1.5.0         # Data manipulation

# Visualization
seaborn>=0.12.0       # Statistical plots

# Testing
pytest>=7.0.0         # Unit tests
```

---

## 8. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run classical pipeline
python run_q_stpp_v16.py --mode classical

# Run with augmentation comparison
python run_q_stpp_v16.py --mode fair-comparison

# Run quantum benchmarks (requires PennyLane)
python run_q_stpp_v16.py --mode quantum-benchmark

# Generate report
python run_q_stpp_v16.py --mode report
```
