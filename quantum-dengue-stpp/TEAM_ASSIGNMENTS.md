# Q-STPP v16: TEAM ASSIGNMENTS

## Team Structure

| Role | Name | Responsibilities |
|------|------|-----------------|
| Team Lead | [Khang] | Architecture, Integration, Final Review |
| Module Lead | [ASSIGN] | Data Pipeline |
| Module Lead | [ASSIGN] | Feature Extraction |
| Module Lead | [ASSIGN] | Prediction |
| Module Lead | [ASSIGN] | SOP Augmentation |
| Module Lead | [ASSIGN] | Quantum Research |

---

## TASK ASSIGNMENT MATRIX

```
Priority Legend:
⭐⭐⭐ P1 = Must complete
⭐⭐   P2 = Important
⭐     P3 = Nice to have
```

### MODULE 1: DATA PIPELINE

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| 1.1 | Data Loading & Preprocessing | ⭐⭐ | [ASSIGN] | [ ] |
| 1.2 | Spatial Discretization | ⭐⭐ | [ASSIGN] | [ ] |
| 1.3 | Temporal Binning | ⭐ | [ASSIGN] | [ ] |

### MODULE 2: FEATURE EXTRACTION

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| **2.1** | **K-function Computation** | **⭐⭐⭐** | **TASK-2.1** | [ ] |
| 2.2 | L-function Computation | ⭐⭐ | [ASSIGN] | [ ] |
| **2.3** | **Space-Time Distance** | **⭐⭐⭐** | [ASSIGN] | [ ] |
| 2.4 | CNN Feature Extractor | ⭐⭐ | [ASSIGN] | [ ] |
| 2.5 | GNN Attention | ⭐⭐ | [ASSIGN] | [ ] |
| 2.6 | Non-Stationary Kernel | ⭐⭐ | [ASSIGN] | [ ] |

### MODULE 3: PREDICTION (OUTPUT = DỰ ĐOÁN)

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| **3.1** | **1-NN Classification** | **⭐⭐⭐** | **TASK-3.1** | [ ] |
| 3.2 | Risk Scoring | ⭐⭐ | [ASSIGN] | [ ] |
| **3.3** | **Hotspot Prediction** | **⭐⭐⭐** | **TASK-3.3** | [ ] |
| 3.4 | Temporal Forecasting | ⭐⭐ | [ASSIGN] | [ ] |

### MODULE 4: SOP AUGMENTATION

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| 4.1 | Metropolis-Hastings | ⭐⭐ | [ASSIGN] | [ ] |
| 4.2 | Greedy Search | ⭐⭐ | [ASSIGN] | [ ] |
| **4.3** | **QAOA SOP** | **⭐⭐⭐** | **TASK-4.3** | [ ] |
| 4.4 | L(r) Error Evaluation | ⭐⭐ | [ASSIGN] | [ ] |

### MODULE 5: QUANTUM LAYER (Research Only)

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| 5.1 | Genuine QAOA | ⭐⭐ | [ASSIGN] | [ ] |
| **5.2** | **Quantum Kernels** | **⭐⭐** | **TASK-5.2** | [ ] |
| 5.3 | VQE Optimization | ⭐ | [ASSIGN] | [ ] |
| 5.4 | Amplitude Estimation | ⭐ | [ASSIGN] | [ ] |

### MODULE 6: OUTPUT

| Task | Description | Priority | Assigned | Status |
|------|------------|----------|----------|--------|
| 6.1 | Visualization | ⭐⭐ | [ASSIGN] | [ ] |
| 6.2 | Metrics Computation | ⭐⭐ | [ASSIGN] | [ ] |
| 6.3 | Report Generation | ⭐⭐ | [ASSIGN] | [ ] |

---

## PRIORITY TASKS (P1)

### Task 2.1: K-function Computation
- **Owner**: [ASSIGN]
- **File**: `tasks/MODULE-2.1_K_Function.md`
- **Deadline**: Week 4
- **Key Question**: Can quantum speed up O(N²) bottleneck?

### Task 3.1: 1-NN Classification
- **Owner**: [ASSIGN]
- **File**: `tasks/MODULE-3.1_OneNN_Classification.md`
- **Deadline**: Week 4
- **Key Question**: Can Grover speed up nearest neighbor?

### Task 3.3: Hotspot Prediction
- **Owner**: [ASSIGN]
- **File**: `tasks/MODULE-3.3_Hotspot_Prediction.md`
- **Deadline**: Week 4
- **Key Question**: DỰ ĐOÁN hotspots - MAIN OUTPUT

### Task 4.3: QAOA SOP
- **Owner**: [ASSIGN]
- **File**: `tasks/MODULE-4.3_QAOA_SOP.md`
- **Deadline**: Week 4
- **Key Question**: Can genuine QAOA beat classical heuristics?

---

## TASK DETAILS

### How to use task files:

1. Read `TASKS.md` - Overview of all tasks
2. Read your specific task file (e.g., `tasks/MODULE-3.3_Hotspot_Prediction.md`)
3. Read required documentation files
4. Follow implementation checklist
5. Deliver weekly progress

### Task files location:
```
quantum-dengue-stpp/
└── tasks/
    ├── MODULE-2.1_K_Function.md          # P1
    ├── MODULE-3.1_OneNN_Classification.md  # P1
    ├── MODULE-3.3_Hotspot_Prediction.md   # P1 - MAIN OUTPUT
    ├── MODULE-4.3_QAOA_SOP.md              # P1
    ├── MODULE-5.2_Quantum_Kernels.md       # P2
    └── task_template.md                      # Template for new tasks
```

---

## DOCUMENTATION FILES

| File | Content | When to Read |
|------|---------|--------------|
| `ARCHITECTURE.md` | Full system architecture | First, then reference |
| `THEORY.md` | Mathematical foundations | When doing research |
| `Q_STPP_V16_REPORT.md` | Technical report | Planning phase |
| `DEVELOPMENT_HISTORY.md` | What failed | Avoid past mistakes |
| `TASKS.md` | Task overview | Weekly reference |

---

## MEETING SCHEDULE

| Day | Time | Focus |
|-----|------|-------|
| Monday | 10:00 | Week planning |
| Wednesday | 10:00 | Progress check |
| Friday | 14:00 | Demo & feedback |

---

## COMMUNICATION

### Team Channel
- [Slack/Discord link]

### Documentation
- All docs in `quantum-dengue-stpp/docs/`
- Task files in `quantum-dengue-stpp/tasks/`

### Issues
- Use GitHub Issues
- Tag with module (e.g., `[MODULE-3.3]`)

---

## EXPECTED OUTCOMES

### Week 1
- [ ] All tasks assigned
- [ ] Literature survey complete
- [ ] Initial research reports

### Week 2
- [ ] Classical baselines implemented
- [ ] Benchmarks designed
- [ ] Quantum approaches identified

### Week 3
- [ ] Prototypes working
- [ ] Preliminary results
- [ ] Integration started

### Week 4
- [ ] Full integration
- [ ] DỰ ĐOÁN system working
- [ ] Final reports

### Week 5-6
- [ ] Performance optimization
- [ ] Real data validation
- [ ] Final demo

---

## SUCCESS CRITERIA

| Metric | Target |
|--------|--------|
| DỰ ĐOÁN Accuracy | > 70% |
| L(r) Error | < 0.001 |
| Diversity Score | > 0.8 |
| Code Coverage | > 80% |
| Documentation | Complete |

---

## REMINDERS

⚠️ **Remember**: The final OUTPUT is **DỰ ĐOÁN** (prediction)!
- Every task should contribute to better prediction
- Don't optimize for research if it hurts prediction

⚠️ **Be Honest**: 
- No quantum advantage without proof
- Classical first, quantum where useful

⚠️ **Fair Comparison**:
- Same seed, same budget
- Report both quality AND diversity

---

## Contact

**Team Lead**: Khang Le
**Project**: Q-STPP v16
**Goal**: DỰ ĐOÁN điểm nóng dengue

---

**Last Updated**: 2026-07-16
**Version**: v16
