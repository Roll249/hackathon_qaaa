# Q-STPP v16: Pipeline Tasks & Research Assignments

## The Vision

**"Cứu người NGAY LẬP TỨC, không phải đợi 5-10 năm"**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAPID-DENGUE vs PHARMTOM LABS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pharmtom Labs (SEA Quantathon 2025 Winner)                      │
│  • Drug discovery via VQE                                        │
│  • Timeline: 5-10 năm cho thuốc mới                           │
│  • Impact: Long-term, không giúp được người hôm nay            │
│                                                                  │
│  RAPID-DENGUE (Our Project)                                      │
│  • Real-time hotspot prediction                                   │
│  • Timeline: Deploy được TRONG TUẦN NÀY                         │
│  • Impact: IMMEDIATE - cứu người ngay lập tức                 │
│                                                                  │
│  ✅ OUR ADVANTAGE: Speed to Impact                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**OUTPUT cuối cùng: DỰ ĐOÁN điểm nóng dengue (hotspot prediction)**

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Raw Dengue Data                                 │
│                    (cases, weather, mobility, population)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: DATA PIPELINE (Layer 0)                                             │
│  Input: Raw data → Output: Clean, discretized events                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 2: FEATURE EXTRACTION (Layer 1)                                        │
│  Input: Events → Output: Feature vectors (K/L-function, CNN, GNN)               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: PREDICTION (Layer 2)                                                │
│  Input: Features → Output: DỰ ĐOÁN (hotspot probability, risk scores)           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: SOP AUGMENTATION (Layer 3)                                          │
│  Input: Events → Output: Augmented event set for training                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 5: QUANTUM LAYER (Layer 4) - Research Only                            │
│  Research: Where can quantum actually help?                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 6: OUTPUT & METRICS (Layer 5)                                         │
│  Output: DỰ ĐOÁN cuối cùng + evaluation metrics                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## TASK ASSIGNMENTS

## ═══════════════════════════════════════════════════════════════
## MODULE 1: DATA PIPELINE
## ═══════════════════════════════════════════════════════════════

### Task 1.1: Data Loading & Preprocessing
**Owner**: [ASSIGN]
**Quantum Opportunity**: LOW (mostly I/O)
**Research Question**: Can quantum memory (QRAM) speed up data loading for massive datasets?

```python
# Key functions to research:
- load_real_dengue_data(source='TYCHO')
- load_synthetic_hawkes(n_events, seed)
- normalize_coordinates(coords)
- handle_missing_values(df)
```

### Task 1.2: Spatial Discretization
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Research Question**: Quantum sampling for optimal grid assignment?

```python
# Key functions:
- discretize_to_grid(points, d1=12, d2=12)
- assign_to_cells(coords, grid_size)
- compute_cell_counts(grid, events)
```

**Quantum Research**:
- Can QAOA optimize grid size selection?
- Can quantum annealing find better discretization?

### Task 1.3: Temporal Binning
**Owner**: [ASSIGN]
**Quantum Opportunity**: LOW
**Research Question**: Classical is likely optimal here

```python
# Key functions:
- bin_by_time(events, window='1D')
- aggregate_by_period(times, window)
- create_time_series(events)
```

---

## ═══════════════════════════════════════════════════════════════
## MODULE 2: FEATURE EXTRACTION (Layer 1)
## ═══════════════════════════════════════════════════════════════

### Task 2.1: K-function Computation ⭐ QUANTUM OPPORTUNITY
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH (for large N)
**Complexity**: O(N²) - THE BOTTLENECK

```python
def compute_K(r, events):
    """
    K(r) = (1/λ) × (1/N²) × Σ 𝟙(dij < r)
    
    THIS IS THE BOTTLENECK!
    - O(N²) pairwise distance computation
    - Called thousands of times in SOP search
    """
    pass
```

**Research Questions**:
1. Can HHL algorithm speed up linear systems in K-function computation?
2. Can quantum amplitude estimation improve convergence?
3. Can tensor networks (MPS/TT) accelerate kernel computation?

**Key Papers to Find**:
- [ ] "Quantum speedup for kernel methods" - any recent work?
- [ ] "HHL algorithm for linear systems" - applications to statistics
- [ ] "Tensor networks for spatial statistics" - classical acceleration

### Task 2.2: L-function Computation
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM (depends on K-function)
**Complexity**: O(N²)

```python
def compute_L(r, events):
    """
    L(r) = sign(K) × |K|^(1/3)  [stabilized transform]
    
    Depends on K-function - quantum advantage comes from Task 2.1
    """
    pass
```

**Research Questions**:
- Parallel L-function computation
- GPU acceleration for L-function

### Task 2.3: Space-Time Distance Matrix
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Complexity**: O(N²) - can we do O(log N) with quantum?

```python
def compute_spacetime_distances(t, x, y, alpha=1.0):
    """
    d²((x,t), (x',t')) = ||x-x'||² + α²|t-t'|²
    
    THIS IS THE CORE O(N²) OPERATION!
    """
    pass
```

**Research Questions**:
- Quantum distance calculation
- Quantum RAM (QRAM) for distance oracle
- Grover's algorithm for nearest neighbor search

### Task 2.4: CNN Feature Extractor (per Mateu 2025)
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Research**: Quantum CNN for point pattern features

```python
class SiameseCNN:
    """
    Based on Mateu 2025 slides 17-19
    - Discretize point pattern to grid
    - Conv2D layers + pooling
    - Siamese architecture for comparison
    """
    pass
```

**Research Questions**:
- Quantum Convolutional Neural Networks (QCNN)
- Variational Quantum Circuits for feature extraction
- Quantum Transfer Learning

### Task 2.5: GNN Attention for Influence Kernels
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Research**: Quantum Graph Neural Networks

```python
class GNNAttention:
    """
    Per Mateu 2025 - Graph Attention Network
    - Multi-head self-attention
    - Learns α_cl,c'l' coefficients
    """
    pass
```

**Research Questions**:
- Quantum Graph Neural Networks (QGNN)
- Quantum Attention Mechanisms
- Graph isomorphism networks on quantum hardware

### Task 2.6: Non-Stationary Kernel Learning
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Research**: VQE for kernel parameter optimization

```python
def learn_nonstationary_kernel(events, spatial_grid):
    """
    Per Mateu 2025 - neural kernel
    v(s, s') = <φ_s, φ_s'>
    
    Can VQE optimize these parameters?
    """
    pass
```

**Research Questions**:
- VQE for kernel parameter optimization
- Quantum Bayesian optimization for kernel hyperparameters
- Quantum natural gradient descent

---

## ═══════════════════════════════════════════════════════════════
## MODULE 3: PREDICTION (Layer 2) - OUTPUT IS DỰ ĐOÁN!
## ═══════════════════════════════════════════════════════════════

### Task 3.1: 1-NN Classification
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Output**: DỰ ĐOÁN class label

```python
class OneNNClassifier:
    """
    Per Mateu 2025 - 1-Nearest Neighbor
    Output: DỰ ĐOÁN pattern class (e.g., outbreak type)
    """
    def predict(new_pattern, features, labels):
        # Find nearest neighbor
        # Return DỰ ĐOÁN
        pass
```

**Research Questions**:
- Grover's algorithm for nearest neighbor search: O(√N) vs O(N)
- Quantum distance calculation
- Quantum metric learning

### Task 3.2: Risk Scoring
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Output**: DỰ ĐOÁN risk score (0-1)

```python
class RiskScorer:
    """
    Compute hotspot probability
    Output: DỰ ĐOÁN risk level per location
    """
    def compute_risk(features):
        # λ(x,t) = μ + Σ k(t',t,s',s)
        # Return DỰ ĐOÁN risk
        pass
```

**Research Questions**:
- Quantum optimization for risk threshold
- QAOA for multi-objective risk scoring

### Task 3.3: Hotspot Prediction ⭐ MAIN OUTPUT
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Output**: DỰ ĐOÁN hotspot map

```python
class HotspotPredictor:
    """
    MAIN OUTPUT MODULE!
    Returns: DỰ ĐOÁN hotspot locations for next time period
    """
    def predict(events, horizon=7):
        """
        Dự đoán điểm nóng trong 7 ngày tới
        """
        pass
```

**Research Questions**:
- QAOA for spatial optimization
- Quantum annealing for hotspot assignment
- Quantum Monte Carlo for prediction uncertainty

### Task 3.4: Temporal Forecasting
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Output**: DỰ ĐOÁN time series

```python
class ForecastEngine:
    """
    Predict future case counts
    Output: DỰ ĐOÁN time series
    """
    def forecast(current_events, horizon=30):
        pass
```

**Research Questions**:
- Quantum LSTM/Transformer for time series
- Variational Quantum Circuits for forecasting
- Quantum state representation for temporal data

---

## ═══════════════════════════════════════════════════════════════
## MODULE 4: SOP AUGMENTATION (Layer 3)
## ═══════════════════════════════════════════════════════════════

### Task 4.1: Metropolis-Hastings Sampler
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Purpose**: Data augmentation

```python
def metropolis_hastings(times, coords, L_target, n_perms):
    """
    MH sampler for SOP permutation search
    - O(N²) per evaluation
    - Called many times
    """
    pass
```

**Research Questions**:
- Quantum Metropolis-Hastings
- Quantum walks for sampling
- QAOA as proposal distribution

### Task 4.2: Greedy Search
**Owner**: [ASSIGN]
**Quantum Opportunity**: MEDIUM
**Purpose**: Find best permutation (lowest error)

```python
def greedy_search(times, coords, L_target, n_swaps):
    """
    Greedy swap optimization
    - O(N² × S) where S = number of swaps
    """
    pass
```

**Research Questions**:
- QAOA for combinatorial optimization
- Grover's algorithm for swap selection

### Task 4.3: QAOA-Inspired Multi-Swap ⭐ MAIN RESEARCH
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Purpose**: Balance quality and diversity

```python
def qaoa_multi_swap(times, coords, L_target, n_perms, p=1):
    """
    QAOA-inspired permutation search
    
    Research: Can GENUINE QAOA outperform classical?
    - XY-mixer Hamiltonian: H_M = Σ (X_i X_j + Y_i Y_j)
    - Cost Hamiltonian: H_C = ||L(π) - L_target||²
    """
    pass
```

**Research Questions**:
- QAOA performance on SOP problem
- Optimal p (depth) for QAOA
- Classical simulation vs real quantum hardware
- When does QAOA beat greedy? (N > ?)

### Task 4.4: L(r) Error Evaluation
**Owner**: [ASSIGN]
**Quantum Opportunity**: HIGH
**Complexity**: O(N²)

```python
def l_error(L_perm, L_target):
    """
    MSE(L_perm - L_target)
    
    THIS IS CALLED THOUSANDS OF TIMES!
    - Each SOP evaluation calls this
    - O(N²) per call
    """
    pass
```

**Research Questions**:
- Can we precompute L-target more efficiently?
- Quantum distance calculation
- Caching strategies

---

## ═══════════════════════════════════════════════════════════════
## MODULE 5: QUANTUM LAYER (Layer 4) - Research Only
## ═══════════════════════════════════════════════════════════════

### Task 5.1: Genuine QAOA Implementation
**Owner**: [ASSIGN]
**Purpose**: Benchmark against classical

```python
def genuine_qaoa_sop(events, p=1, shots=1000):
    """
    ACTUAL QAOA CIRCUIT - not "QAOA-inspired"
    
    Requirements:
    - PennyLane or Qiskit implementation
    - Statevector simulation for small N (N ≤ 10)
    - Qubit simulation for N ≤ 20
    - Real hardware for N ≤ 127 (IBM)
    """
    pass
```

**Research Questions**:
- Benchmark: QAOA vs Greedy vs MH at various N
- Optimal circuit depth (p) for SOP
- Noise models and error mitigation

### Task 5.2: Quantum Kernel Methods
**Owner**: [ASSIGN]
**Purpose**: Pattern classification with quantum features

```python
def quantum_kernel_classification(training_data, test_data):
    """
    Per Schuld et al. - Quantum kernel for ML
    
    k(x, x') = |⟨φ(x)|φ(x')⟩|²
    
    where |φ(x)⟩ is quantum state encoding x
    """
    pass
```

**Research Questions**:
- IQP kernel implementation
- Expressibility of quantum kernels
- Comparison with classical RFF/Gaussian kernels

### Task 5.3: VQE for Kernel Optimization
**Owner**: [ASSIGN]
**Purpose**: Optimize Hawkes kernel parameters

```python
def vqe_kernel_optimization(events):
    """
    Use VQE to minimize:
    L(θ) = -log-likelihood(θ | events)
    
    Variational form: ansatz circuit
    Optimizer: COBYLA, SPSA, gradient descent
    """
    pass
```

**Research Questions**:
- Variational forms for kernel parameters
- Barren plateau avoidance
- Classical vs VQE performance

### Task 5.4: Quantum Amplitude Estimation
**Owner**: [ASSIGN]
**Purpose**: Estimate tail probabilities

```python
def quantum_amplitude_estimation(problem):
    """
    QAE for rare event probability
    
    Classical: O(1/N) precision
    Quantum: O(1/√N) precision (quadratic speedup)
    
    Application: P(L(r) < threshold)
    """
    pass
```

**Research Questions**:
- QAE for L-function distribution tails
- Maximum likelihood estimation with QAE
- Practical speedup for N = ?

---

## ═══════════════════════════════════════════════════════════════
## MODULE 6: OUTPUT & METRICS (Layer 5)
## ═══════════════════════════════════════════════════════════════

### Task 6.1: Prediction Visualization
**Owner**: [ASSIGN]
**Output**: DỰ ĐOÁN hotspot map (image)

### Task 6.2: Metrics Computation
**Owner**: [ASSIGN]
**Output**: Evaluation metrics

```python
metrics = {
    'l_error': ...,          # Quality
    'diversity': ...,        # Augmentation quality  
    'accuracy': ...,          # DỰ ĐOÁN accuracy
    'precision': ...,        # DỰ ĐOÁN precision
    'recall': ...,           # DỰ ĐOÁN recall
    'auc_roc': ...           # DỰ ĐOÁN AUC
}
```

### Task 6.3: Report Generation
**Owner**: [ASSIGN]
**Output**: Technical report with DỰ ĐOÁN results

---

## ═══════════════════════════════════════════════════════════════
## RESEARCH PRIORITY MATRIX
## ═══════════════════════════════════════════════════════════════

| Module | Task | Quantum Opportunity | Difficulty | Priority |
|--------|------|-------------------|-----------|----------|
| 2.1 | K-function | ⭐⭐⭐ HIGH | Hard | P1 |
| 2.3 | Space-time distance | ⭐⭐⭐ HIGH | Hard | P1 |
| 3.1 | 1-NN prediction | ⭐⭐⭐ HIGH | Medium | P1 |
| 3.3 | Hotspot prediction | ⭐⭐⭐ HIGH | Medium | P1 |
| 4.3 | QAOA SOP | ⭐⭐⭐ HIGH | Hard | P1 |
| 5.1 | Genuine QAOA | ⭐⭐⭐ HIGH | Hard | P2 |
| 5.2 | Quantum kernels | ⭐⭐ MEDIUM | Medium | P2 |
| 4.4 | L(r) error | ⭐⭐ MEDIUM | Medium | P2 |
| 2.6 | VQE kernel | ⭐⭐ MEDIUM | Hard | P3 |
| 2.4 | CNN features | ⭐⭐ MEDIUM | Medium | P3 |
| 3.4 | Temporal forecast | ⭐⭐ MEDIUM | Medium | P3 |
| 5.3 | VQE optimization | ⭐ LOW | Hard | P4 |
| 1.2 | Grid discretization | ⭐ LOW | Easy | P4 |

---

## HOW TO USE THE DOCUMENTATION FILES

### 1. ARCHITECTURE.md
```
📁 Contains: Full system architecture
📖 Read: Sections 1-2 for pipeline overview
🔍 Focus: Layer specifications (2.1-2.6)
🎯 Use: Understand where each module fits
```

### 2. DEVELOPMENT_HISTORY.md
```
📁 Contains: What failed, what worked
📖 Read: Section 3 (Honest Assessment)
🔍 Focus: "What Doesn't Work (Yet)"
🎯 Use: Avoid repeating past mistakes
```

### 3. Q_STPP_V16_REPORT.md
```
📁 Contains: Technical report
📖 Read: Sections 2-4 for methodology
🔍 Focus: "Honest Quantum Assessment" (Section 5)
🎯 Use: Understand current capabilities
```

### 4. THEORY.md
```
📁 Contains: Mathematical foundations
📖 Read: Sections 1-3 for STPP basics
🔍 Focus: Section 5 (Quantum-Inspired)
🎯 Use: Research quantum methods
```

---

## EXPECTED OUTPUT FORMAT

For each task, the assigned person should deliver:

```
📄 [Task ID] - [Task Name]
   
   1. Literature Review
      - Key papers found
      - Summary of methods
      - Relevance to task
   
   2. Implementation Plan
      - Algorithm steps
      - Complexity analysis
      - Expected quantum speedup
   
   3. Benchmark Design
      - How to compare classical vs quantum
      - Metrics to track
      - Fair comparison protocol
   
   4. Code Sketch (if applicable)
      - Pseudocode
      - Key functions
      - Integration points
   
   5. Deliverables
      - [ ] Research report
      - [ ] Prototype code (optional)
      - [ ] Benchmark results
      - [ ] Recommendations
```

---

## MEETING SCHEDULE

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | Research & Literature Survey | Report on 3-5 key papers per task |
| Week 2 | Implementation Planning | Detailed plan for quantum approach |
| Week 3 | Prototype Development | Working code for classical baseline |
| Week 4 | Benchmarking | Compare classical vs quantum |
| Week 5 | Integration | Integrate into full pipeline |
| Week 6 | Final Report | Complete DỰ ĐOÁN system |

---

## CONTACT & QUESTIONS

For questions about specific tasks:
- Check ARCHITECTURE.md first
- Check THEORY.md for math background
- Check DEVELOPMENT_HISTORY.md to avoid past mistakes
- Ask in group channel

---

**REMEMBER: The final output must be DỰ ĐOÁN (prediction)!**
Every module should contribute to better hotspot prediction.
