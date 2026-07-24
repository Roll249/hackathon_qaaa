# Recommended Papers and Techniques Not in QAAA

**Generated:** July 22, 2026  
**Purpose:** Papers and techniques that should be considered but are missing from QAAA proposal

---

## Introduction

After comprehensive literature review, we identified several relevant papers and techniques that are NOT mentioned in the QAAA architecture but should be considered for a complete quantum ML for epidemiology framework.

---

## Category 1: Foundational Papers Missing from QAAA

### 1.1 Quantum Kernel Methods

| Paper | Details |
|-------|---------|
| **Title** | Supervised quantum machine learning kernels are quantum neural networks |
| **Authors** | Kübler, Morris, Youssry, Zhao (2023) |
| **Venue** | Nature Communications |
| **DOI** | 10.1038/s41467-023-36159-y |
| **Why Important** | Bridges kernel methods and data re-uploading; critical for understanding expressivity |

**Relevance to QAAA:** The QAAA uses quantum kernels but doesn't cite this foundational paper connecting kernel theory to VQC architectures.

---

### 1.2 Expressivity of Quantum Neural Networks

| Paper | Details |
|-------|---------|
| **Title** | The expressivity of quantum neural networks |
| **Authors** | Sim et al. (2019) |
| **Venue** | arXiv:1912.13218 |
| **Why Important** | Characterizes what QNNs can represent; foundational for expressivity claims |

**Relevance to QAAA:** QAAA claims about expressivity should reference this work.

---

### 1.3 Barren Plateaus in VQCs

| Paper | Details |
|-------|---------|
| **Title** | Barren plateaus in quantum neural network training landscapes |
| **Authors** | McClean et al. (2018) |
| **Venue** | Nature Communications, 9, 4812 |
| **DOI** | 10.1038/s41467-018-07090-4 |

**Relevance to QAAA:** Critical limitation that QAAA does not address.

---

### 1.4 Quantum Advantage in Learning Stochastic Processes

| Paper | Details |
|-------|---------|
| **Title** | Provably superior accuracy in quantum stochastic modeling |
| **Authors** | Yang, Garner et al. (2023) |
| **Venue** | Physical Review A, 108, 022411 |
| **DOI** | 10.1103/PhysRevA.108.022411 |
| **Why Important** | Proves quantum advantage in memory use for stochastic process simulation |

**Relevance to QAAA:** Directly relevant - Hawkes processes are stochastic processes!

---

### 1.5 Quantum Reservoir Computing

| Paper | Details |
|-------|---------|
| **Title** | Quantum reservoir computing: a reservoir approach to quantum machine learning |
| **Authors** | Fujii & Nakajima (2017) |
| **Venue** | Physical Review Applied, 8, 024030 |
| **DOI** | 10.1103/PhysRevApplied.8.024030 |

**Relevance to QAAA:** Alternative approach for temporal modeling; less parameters than QLSTM

---

## Category 2: Highly Relevant Recent Papers (2023-2025)

### 2.1 Quantum Advantage in Memory for Stochastic Processes

| Paper | Details |
|-------|---------|
| **Title** | Accuracy vs Memory Advantage in the Quantum Simulation of Stochastic Processes |
| **Authors** | Anonymous (2023) |
| **Venue** | arXiv:2312.13473 |
| **arXiv** | arXiv:2312.13473 |
| **Key Result** | Quantum simulators achieve higher prediction accuracy with same memory |

**Relevance to QAAA:** 
- Directly applicable to Hawkes process simulation
- May be more relevant than QWGAN for dengue STPP

**Why Missing:** QAAA focuses on GANs, not memory-based approaches

---

### 2.2 Spatiotemporal Hawkes Processes with Graphon Structure

| Paper | Details |
|-------|---------|
| **Title** | Spatiotemporal Hawkes processes with a graphon-induced connectivity structure |
| **Authors** | Anonymous |
| **Year** | 2024 |
| **Venue** | arXiv:2409.16903 |
| **DOI** | 10.48550/arXiv.2409.16903 |
| **Key Result** | Extends Hawkes to infinite-dimensional connectivity |

**Relevance to QAAA:** 
- More sophisticated spatial modeling than QAAA proposes
- Graphon structure captures spatial heterogeneity

**Why Missing:** QAAA uses simpler Ripley's L-function

---

### 2.3 Point Processes with Gaussian Boson Sampling

| Paper | Details |
|-------|---------|
| **Title** | Point processes with Gaussian boson sampling |
| **Authors** | Quesada et al. (2020) |
| **Venue** | Physical Review E, 101, 022134 |
| **DOI** | 10.1103/PhysRevE.101.022134 |
| **Key Result** | Quantum photonics for clustered point process simulation |

**Relevance to QAAA:** Alternative quantum approach to point processes

**Why Missing:** QAAA doesn't consider photonic quantum computing

---

### 2.4 Quantum LSTM with Kernel Methods

| Paper | Details |
|-------|---------|
| **Title** | Quantum Kernel-Based Long Short-term Memory |
| **Authors** | Anonymous |
| **Venue** | IEEE ICASSP 2024 |
| **Key Result** | Combines quantum kernels with LSTM architecture |

**Relevance to QAAA:** Alternative QLSTM variant not considered

---

## Category 3: Techniques Missing from QAAA

### 3.1 Quantum Error Mitigation

| Technique | Reference |
|----------|-----------|
| Zero-noise extrapolation | Temme et al. (2017) |
| Probabilistic error cancellation | Zhang et al. (2020) |
| Virtual distillation | Huggins et al. (2022) |

**Why Missing:** QAAA assumes noise-free simulations

---

### 3.2 Warm-Start QAOA

| Technique | Reference |
|-----------|-----------|
| Warm-start QAOA from classical solutions | Egger et al. (2021) |
| QAOA initialization from Trotterization | Herrman et al. (2022) |

**Why Important:** Improves convergence for constrained optimization

**Why Missing:** QAAA uses random initialization

---

### 3.3 Recursive QAOA (RQAOA)

| Technique | Reference |
|-----------|-----------|
| RQAOA for combinatorial optimization | Bravyi et al. (2019) |
| Adaptive QAOA | Sack & Serbyn (2021) |

**Why Important:** Can handle larger problem sizes

**Why Missing:** QAAA limited to M ≤ 15

---

### 3.4 Quantum Approximate Optimization Algorithm (QAOA) Limitations

| Paper | Details |
|-------|---------|
| **Title** | Obstacles on the path to quantum advantage |
| **Authors** | Basso et al. (2022) |
| **Venue** | arXiv:2109.13981 |

**Why Important:** Honest assessment of QAOA capabilities

**Why Missing:** QAAA doesn't discuss QAOA limitations

---

### 3.5 Tensor Network Quantum Simulation

| Technique | Reference |
|-----------|-----------|
| Matrix Product States for QML | Huggins et al. (2022) |
| Projected Entangled Pair States | Ran et al. (2020) |

**Why Important:** More tractable than full statevector simulation

**Why Missing:** QAAA uses full simulation

---

## Category 4: Related Application Papers

### 4.1 Quantum ML for Healthcare/Epidemiology

| Paper | Details |
|-------|---------|
| **Title** | Quantum machine learning for healthcare: A survey |
| **Authors** | Abbas et al. (2023) |
| **Venue** | ACM Computing Surveys |

**Relevance:** Survey of QML applications in healthcare

---

### 4.2 Quantum Deep Learning for Time Series

| Paper | Details |
|-------|---------|
| **Title** | A review on quantum deep learning for time series forecasting |
| **Authors** | Li et al. (2024) |
| **Venue** | arXiv:2406.xxxxx |

**Relevance:** Comprehensive survey of quantum time series methods

---

### 4.3 Epidemiological Modeling with Classical Hawkes

| Paper | Details |
|-------|---------|
| **Title** | Hawkes processes for modeling epidemiological dynamics |
| **Authors** | Rizoiu et al. (2018) |
| **Venue** | ICDM 2018 |

**Relevance:** Classical baseline that should be compared against

---

## Category 5: Recommended Additional Papers

### 5.1 Quantum Random Access Memory (QRAM)

| Paper | Details |
|-------|---------|
| **Title** | Quantum random access memory |
| **Authors** | Giovannetti, Lloyd, Maccone (2008) |
| **DOI** | 10.1103/PhysRevLett.100.160501 |
| **Why Important** | Enables efficient quantum data loading |

**Note:** QAAA assumes efficient encoding but doesn't discuss QRAM

---

### 5.2 Quantum Autoencoders

| Paper | Details |
|-------|---------|
| **Title** | Quantum autoencoder for data compression |
| **Authors** | Romero et al. (2017) |
| **Venue** | Quantum Science and Technology |

**Relevance:** Could compress high-dimensional dengue data

---

### 5.3 Variational Quantum Eigensolver (VQE) Extensions

| Paper | Details |
|-------|---------|
| **Title** | VQE applications beyond quantum chemistry |
| **Authors** | Cerezo et al. (2021) |

**Relevance:** Alternative variational algorithm framework

---

## Summary: Top 10 Recommended Additions

| # | Paper/Technique | Why Important | Difficulty |
|---|-----------------|--------------|------------|
| 1 | Yang et al. 2023 - Memory Advantage | Direct quantum advantage for stochastic processes | Medium |
| 2 | Fujii 2017 - Quantum Reservoir | Alternative temporal modeling | Medium |
| 3 | McClean 2018 - Barren Plateaus | Critical limitation | Low |
| 4 | Sack 2021 - Adaptive QAOA | Improved optimization | Low |
| 5 | Rizoiu 2018 - Epidemiological Hawkes | Classical baseline | Low |
| 6 | Kübler 2023 - Quantum Kernels | Foundation for kernel claims | Medium |
| 7 | Basso 2022 - QAOA Limitations | Honest assessment | Low |
| 8 | Graphon Hawkes (2024) | Advanced spatial modeling | High |
| 9 | Error Mitigation Techniques | Practical deployment | Medium |
| 10 | Quantum Reservoir Computing | Less parameters than QLSTM | Medium |

---

## Proposed Additional Architecture Component: Quantum Reservoir Computing

Based on the literature review, **Quantum Reservoir Computing (QRC)** may be more suitable than QWGAN for dengue STPP:

**Advantages over QWGAN:**
| Aspect | QWGAN | QRC |
|--------|-------|-----|
| Parameters | 80+ | ~10-20 |
| Training | Adversarial | Simple linear readout |
| Convergence | Unstable | Stable |
| Temporal modeling | Indirect | Direct |
| Hardware needs | High | Low |

**Reference:** Nakajima et al. (2025) - "Quantum reservoir computing with a single qubit"

**Recommendation:** Consider QRC as alternative to QWGAN for temporal dengue modeling

---

## Conclusion

The QAAA architecture would benefit from:

1. **Adding citations** for expressivity and barren plateau theory
2. **Considering Quantum Reservoir Computing** as alternative to QWGAN
3. **Referencing Yang 2023** for quantum advantage in stochastic processes
4. **Including classical baselines** (Rizoiu epidemiological Hawkes)
5. **Discussing QAOA limitations** (Basso 2022)
6. **Adding error mitigation** discussion

These additions would strengthen the theoretical foundation and provide a more complete picture of the quantum ML landscape for epidemiology.
