# TASK 3.1: 1-NN Classification - QUANTUM OPPORTUNITY

## Thông tin chung

| Field | Value |
|-------|-------|
| **Task ID** | MODULE-3.1 |
| **Module** | Layer 2: Prediction |
| **Priority** | P1 - HIGH |
| **Assigned to** | [ASSIGN] |
| **Due Date** | Week 4 |
| **OUTPUT** | **DỰ ĐOÁN pattern class** |

---

## 1. Mục tiêu

Implement 1-NN classification cho STPP pattern recognition, với research xem quantum có thể speedup nearest neighbor search không.

```
1-NN Algorithm:
1. Compute distance from test pattern to all training patterns
2. Find nearest neighbor
3. Assign test pattern to same class

Classical: O(N) distance computations
Quantum: O(√N) via Grover's algorithm
```

---

## 2. Input/Output

```
Input:  
  - Training patterns: features (K/L-function, CNN embeddings)
  - Training labels: pattern classes
  - Test pattern: feature vector
  
Output: 
  - DỰ ĐOÁN class label
  - Confidence score
  - Distance to nearest neighbor
```

---

## 3. Pipeline Context

```
┌─────────────────────────────────────────────────────────────┐
│  MODULE 2: Feature Extraction                             │
│  (Output: Feature vectors φ(x))                           │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 3.1: 1-NN Classification ⭐ DỰ ĐOÁN           │
│                                                             │
│  Uses features to predict outbreak class                    │
│                                                             │
│  Training: Learn (features, labels) pairs                   │
│  Testing: Given features → DỰ ĐOÁN label                  │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 3.3: Hotspot Prediction                          │
│  (DỰ ĐOÁN hotspot locations)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Tài liệu cần đọa

### 4.1 Bắt buộc
- [ ] `THEORY.md` - Section 3 (Pattern Classification)
- [ ] `ARCHITECTURE.md` - Section 2.3 (Prediction Layer)
- [ ] Mateu 2025 slides 30-34 (Siamese + 1-NN)
- [ ] `run_q_stpp_v15_fair.py` - No 1-NN yet (to implement)

### 4.2 Grover's Algorithm Papers
1. **Foundational**
   - Grover (1996) - "A fast quantum mechanical algorithm"
   - Grover (1997) - "Quantum mechanics helps in searching"

2. **Quantum ML**
   - "Quantum speedup for machine learning"
   - "Quantum nearest neighbor algorithms"
   - "Grover's algorithm in practice"

---

## 5. Research Questions

### 5.1 Can Grover speed up 1-NN?

```
Classical 1-NN:
- Compute distance to ALL N training points: O(N × D)
- Find minimum: O(N)
- Total: O(N × D)

Quantum 1-NN (using Grover):
- Load classical data into quantum memory (QRAM): O(N log D)
- Use Grover to find minimum: O(√N)
- Total: O(√N log D)

Speedup: O(N/√N) = O(√N)
```

### 5.2 QRAM Requirement

```
QRAM is REQUIRED for quantum speedup!
- QRAM: Quantum Random Access Memory
- Classical simulation: O(N) anyway
- Real advantage: Only with QRAM

Is QRAM available? NO - it's theoretical!
```

### 5.3 Alternatives without QRAM

| Method | Requirement | Speedup |
|--------|-----------|---------|
| Grover with full superposition | N qubits | O(√N) but O(2^N) to prepare |
| Variational quantum search | Hybrid QC | Unclear |
| Quantum distance oracle | QRAM | O(log N) |

---

## 6. Implementation Plan

### 6.1 Classical 1-NN (Baseline)

```python
class OneNNClassifier:
    """Classical 1-NN - baseline"""
    
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
    
    def predict(self, test_feature):
        # Compute distances
        distances = []
        for i, feat in enumerate(self.features):
            d = euclidean_distance(feat, test_feature)
            distances.append((d, self.labels[i]))
        
        # Find nearest
        min_dist, min_label = min(distances, key=lambda x: x[0])
        return min_label, min_dist
```

### 6.2 Feature Extraction for 1-NN

```python
# Per Mateu 2025, use these features:

features = {
    'k_function': compute_K(events, r_values),
    'l_function': compute_L(events, r_values),
    'cnn_embedding': cnn.extract(events),
    'gnn_attention': gnn.compute(events)
}
```

### 6.3 Distance Metrics

```python
# Different distance metrics to try:
distances = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'correlation': correlation_distance,
    'kl_divergence': kl_divergence  # For distributions
}
```

---

## 7. Benchmark Design

### 7.1 Test Data

```python
# Per Mateu 2025, use synthetic STPP:
patterns = {
    'poisson': poisson_process(n=100),
    'lgcp': lgcp_process(n=100),
    'thomas': thomas_process(n=100),
    'hawkes': hawkes_process(n=100)
}

# Create training/test split
train_size = 0.7
```

### 7.2 Metrics

```python
metrics = {
    'accuracy': 'Classification accuracy',
    'precision': 'Per-class precision',
    'recall': 'Per-class recall',  
    'f1': 'Harmonic mean',
    'auc_roc': 'Area under ROC',
    'time': 'Prediction time'
}
```

### 7.3 Comparison

| Method | Classical | Quantum | Notes |
|--------|-----------|---------|-------|
| 1-NN (euclidean) | O(N×D) | O(√N×D) with QRAM | Baseline |
| 1-NN (cosine) | O(N×D) | O(√N×D) with QRAM | Often better |
| k-NN | O(k×N×D) | O(√N×D) | k neighbors |

---

## 8. Quantum Analysis

### 8.1 Grover's Algorithm for Minimum

```python
def grover_minimum_search(values):
    """
    Find index of minimum value using Grover
    
    Oracle: f(i) = 1 if values[i] is minimum
    Grover: Amplify minimum state
    Complexity: O(√N)
    
    BUT: Need to encode values into quantum state!
    """
    pass
```

### 8.2 Quantum Distance

```python
def quantum_distance(phi_x, phi_y):
    """
    ||φ(x) - φ(y)||² via quantum circuits
    
    Can compute distance without QRAM!
    - Encode via variational circuit
    - Measure distance
    
    Complexity: O(poly(log N))
    """
    pass
```

### 8.3 Requirements

```
For quantum 1-NN:
- Qubits: N (for superposition)
- Circuit depth: poly(log N) per distance
- QRAM: Required for full speedup
- Fidelity: > 0.99 for reliable results
```

---

## 9. Implementation Checklist

### 9.1 Week 1: Literature
- [ ] Read Grover's original papers
- [ ] Study quantum ML papers
- [ ] Design feature extraction pipeline
- [ ] Write literature report

### 9.2 Week 2: Classical Baseline
- [ ] Implement feature extraction
- [ ] Implement 1-NN classifier
- [ ] Benchmark with synthetic data
- [ ] Validate with Mateu 2025 approach

### 9.3 Week 3: Real Data + Quantum Research
- [ ] Test with TYCHO data
- [ ] Research quantum distance methods
- [ ] Design quantum 1-NN (if feasible)
- [ ] Write recommendation

### 9.4 Week 4: Final
- [ ] Complete implementation
- [ ] Full benchmark
- [ ] Quantum analysis
- [ ] Final report

---

## 10. Expected Deliverables

### Week 1: Literature Report
```
📄 1-NN Quantum Research

1. Grover's algorithm for minimum
2. Quantum distance calculation
3. QRAM requirement analysis
4. Feasibility assessment
```

### Week 2: Classical Baseline
```
📊 1-NN Classifier - Classical

1. Feature extraction
2. Distance metrics comparison
3. Baseline accuracy
4. Per-class analysis
```

### Week 3-4: Final System
```
📄 1-NN + Quantum Analysis

1. Complete 1-NN system
2. Real data validation
3. Quantum opportunity analysis
4. Recommendations
```

---

## 11. Questions for Team Lead

1. What features to prioritize? (K-function, CNN, GNN?)
2. How many pattern classes?
3. What is acceptable accuracy target?
4. Should we include k > 1 neighbors?

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Assigned | | |
| Team Lead | | |
