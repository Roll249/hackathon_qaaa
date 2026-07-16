# TASK 3.3: Hotspot Prediction - MAIN OUTPUT

## Thông tin chung

| Field | Value |
|-------|-------|
| **Task ID** | MODULE-3.3 |
| **Module** | Layer 2: Prediction |
| **Priority** | P1 - HIGH |
| **Assigned to** | [ASSIGN] |
| **Due Date** | Week 4 |
| **OUTPUT** | **DỰ ĐOÁN hotspot map** |

---

## 1. Mục tiêu

**MAIN OUTPUT** của toàn bộ pipeline - dự đoán điểm nóng dengue.

```
Input: Historical events → Output: DỰ ĐOÁN hotspot locations
```

**Đây là cái người dùng cuối muốn thấy!**

---

## 2. Input/Output

```
Input:  
  - Historical dengue cases (t, x, y, count)
  - Weather data (temperature, humidity, rainfall)
  - Population density
  - Time horizon for prediction

Output: 
  - DỰ ĐOÁN hotspot probability map
  - DỰ ĐOÁN risk scores per grid cell
  - DỰ ĐOÁN temporal evolution
```

---

## 3. Pipeline Context

```
┌─────────────────────────────────────────────────────────┐
│  MODULE 2: Feature Extraction                          │
│  (Output: K/L-function, CNN features, GNN weights)    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 3.3: Hotspot Prediction ⭐ MAIN OUTPUT       │
│                                                         │
│  Output: DỰ ĐOÁN                                        │
│  - Hotspot probability grid                            │
│  - Risk scores (0-1)                                   │
│  - Alert levels (1-5)                                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  MODULE 6: Output & Metrics                           │
│  (Output: Visualization, Reports)                     │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Tài liệu cần đọc

### 4.1 Bắt buộc
- [ ] `ARCHITECTURE.md` - Section 2.3 (Prediction Layer)
- [ ] `THEORY.md` - Section 1.2-1.3 (Hawkes Process)
- [ ] `Q_STPP_V16_REPORT.md` - Section 1-2 (Problem Definition)
- [ ] `DEVELOPMENT_HISTORY.md` - Section 6 (What Works)

### 4.2 Research Papers cần tìm
1. **Hotspot Detection**
   - "Hotspot detection methods" surveys
   - "Epidemic hotspot prediction" recent papers
   - "Spatio-temporal clustering" methods

2. **Quantum Optimization for Spatial Problems**
   - "QAOA for facility location"
   - "Quantum annealing for clustering"
   - "Max-cut and spatial optimization"

3. **Quantum ML for Prediction**
   - "Quantum machine learning for time series"
   - "Variational quantum circuits for forecasting"
   - "Quantum neural networks"

---

## 5. Research Questions

### 5.1 Where can quantum help hotspot prediction?

```
Hotspot prediction involves:
1. Feature combination: λ(x,t) = μ + Σ k(t',t,s',s)
2. Risk scoring: normalize intensity to [0,1]
3. Spatial assignment: find hotspots
4. Threshold optimization: what cutoff?
```

### 5.2 Specific quantum opportunities

| Component | Classical | Quantum | Speedup? |
|-----------|-----------|---------|----------|
| Feature weighting | O(D) | ? | ? |
| Intensity calculation | O(N×M) | ? | ? |
| Hotspot assignment | NP-hard | ? | ? |
| Threshold optimization | Grid search | ? | ? |

### 5.3 DỰ ĐOÁN accuracy metrics

```
How to evaluate DỰ ĐOÁN:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)  
- AUC-ROC: Area under ROC curve
- F1-score: Harmonic mean
- Calibration: Reliability diagram
```

---

## 6. Implementation Checklist

### 6.1 Literature Survey (Week 1)
- [ ] Tìm 5-10 papers về hotspot prediction
- [ ] Đọc methods cho dengue/spatial epidemics
- [ ] Xác định quantum opportunity
- [ ] Write literature report

### 6.2 Classical Baseline (Week 2)
- [ ] Implement Hawkes intensity model
- [ ] Implement risk scoring
- [ ] Implement hotspot detection
- [ ] Benchmark với synthetic data

### 6.3 Real Data Integration (Week 2-3)
- [ ] Integrate TYCHO data
- [ ] Handle missing values
- [ ] Validate predictions
- [ ] Compare with baselines

### 6.4 Quantum Analysis (Week 3-4)
- [ ] Identify optimization subproblems
- [ ] Research QAOA for spatial optimization
- [ ] Estimate quantum advantage
- [ ] Write recommendation

---

## 7. Expected Deliverables

### Week 1: Literature Report
```
📄 Hotspot Prediction Research Report

1. Summary of methods
2. Classical approaches (baseline)
3. Quantum opportunity assessment
4. Recommended approach
```

### Week 2: Classical Baseline
```
📊 DỰ ĐOÁN System - Classical Baseline

1. Hawkes model implementation
2. Risk scoring algorithm
3. Hotspot detection
4. Benchmark results
```

### Week 3-4: Final System
```
📄 Hotspot Prediction - Complete DỰ ĐOÁN System

1. Full implementation
2. Real data validation
3. DỰ ĐOÁN accuracy metrics
4. Quantum optimization plan
```

---

## 8. DỰ ĐOÁN Output Format

### 8.1 Hotspot Probability Grid
```python
output = {
    'grid_shape': (100, 100),
    'probabilities': np.array,  # shape: (100, 100)
    'timestamps': [...],  # prediction times
    'locations': {
        'lat': [...],
        'lon': [...]
    }
}
```

### 8.2 Risk Scores
```python
risk_scores = {
    'high_risk': [(lat, lon, score), ...],
    'medium_risk': [...],
    'low_risk': [...]
}
```

### 8.3 Alerts
```python
alerts = {
    'level': 1-5,  # 5 = critical
    'message': 'Dengue outbreak predicted in District X',
    'locations': [...],
    'confidence': 0.85
}
```

---

## 9. Benchmark Design

### 9.1 Test Scenarios
```python
scenarios = {
    'synthetic_small': {'n_events': 50, 'horizon': 7},
    'synthetic_medium': {'n_events': 200, 'horizon': 14},
    'synthetic_large': {'n_events': 1000, 'horizon': 30},
    'real_tycho': {'source': 'TYCHO', 'horizon': 7}
}
```

### 9.2 Metrics
```python
metrics = {
    'precision': 'Hotspot accuracy',
    'recall': 'Missed hotspots',
    'auc_roc': 'Overall discrimination',
    'mae': 'Mean absolute error in risk',
    'calibration': 'Reliability'
}
```

---

## 10. Questions for Team Lead

1. What resolution for hotspot map? (district/city/block level?)
2. How far ahead to predict? (1 day / 1 week / 1 month?)
3. What is the false positive tolerance? (better recall or precision?)
4. Should we include weather as input?

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Assigned | | |
| Team Lead | | |
