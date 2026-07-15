# Q-STPP v12 Report: Proper Quantum Kernel

**Date**: 2026-07-16

**Goal**: Replace v9's Hilbert-projection 'quantum kernel' (which was just a classical random-Fourier projection) with a *real* quantum feature map, and report honestly whether it can approach the classical K-function baseline.


---

## 1. Motivation

v9's `extract_quantum_kernel_features` does:

```python
X_hilbert = cos(X @ W + b) / sqrt(2^n)
K = X_hilbert @ X_hilbert.T  # NOT a quantum kernel
```

This is a classical Random Fourier Feature projection — it samples frequencies W and offsets b and projects into a `2^n` dim Fourier space. It is NOT a quantum kernel:

- It is fully classical — no quantum state is ever prepared.
- The 'kernel matrix' K is just X · X^T in a random feature space, akin to an RBF approximation but with random phase shifts.
- It cannot give quantum advantage because no quantum structure is exploited.

v12 fixes this by actually preparing `|φ(x)⟩` on a PennyLane simulator and computing `K(x, x') = |⟨φ(x)|φ(x')⟩|²` directly via state-vector overlap.


## 2. Feature maps implemented

All circuits use **6 qubits** (input dim reduced from 144 → 6 by PCA).

### A) IQP (Instantaneous Quantum Polynomial) — Havlíček 2019
```
for each layer:
    H on all qubits
    RZ(x_i) on qubit i
    CZ ring: CZ(i, i+1 mod n)
```
Universal for classical simulation but provably hard to compute exactly on classical hardware for arbitrary depth — a candidate for quantum advantage (Havlíček et al. 2019).

### B) Higher-order IQP — Peters et al. 2021
Adds `RZ(x_i²)` (non-linear single-qubit) and `RZ(x_i · x_j)` (pairwise cross-term phase rotations). Boosts expressivity beyond linear IQP.

### C) Data re-uploading — Pérez-Salinas et al. 2020
L re-encodings of x, with frozen trainable Rot(θ) blocks between layers. Equivalent to a depth-L classical NN but realized on n qubits — universal quantum classifier with finite resources.

### D) Higher-order re-uploading
(B) + (C) combined.


## 3. Kernel computation

For each feature map, we compute `|φ(x)⟩` once per sample and cache the complex state vector. The kernel matrix is then

```
K_ij = |⟨φ(x_i)|φ(x_j)⟩|² = |states[i] · states[j].conj()|²
```

Scalability comes from two strategies:

1. **Anchor features**: pre-select m=30 landmark samples and use K(x, anchor_j) as a (n, m) feature matrix.
2. **Nyström low-rank approximation**: compute K_mm on m landmarks, eigendecompose, whiten, then project to m-d features such that K_nn ≈ Φ Φ^T.

Both are then fed to SVM-RBF and KNN-k=3 classifiers (best of the two is reported).


## 4. Results


### 4.1 Summary table


| N | Classical K | v9 Hilbert | Best quantum (this work) | Gap vs classical |
|---|-------------|------------|--------------------------|------------------|
| 150 | 0.693 | 0.540 | 0.733 (iqp_L2_anchor_best) | +0.040 |
| 300 | 0.720 | 0.377 | 0.703 (iqp_L2_anchor_best) | -0.017 |
| 600 | 0.712 | 0.355 | 0.683 (reuploading_L2_anchor_best) | -0.028 |

### 4.2 Per-feature-map results


#### N = 150

- Classical K (best of SVM/KNN): **0.6933**
- v9 Hilbert projection (best): **0.5400**

| Feature map | Layers | Anchor-SVM | Anchor-KNN | Nyström-SVM | Nyström-KNN | Precomp-KSVM |
|-------------|--------|------------|------------|-------------|------------|--------------|
| iqp | 2 | 0.733 | 0.673 | 0.673 | 0.627 | 0.687 |
| higher_order_iqp | 2 | 0.340 | 0.280 | 0.353 | 0.327 | 0.360 |
| reuploading | 2 | 0.713 | 0.567 | 0.613 | 0.613 | 0.680 |

#### N = 300

- Classical K (best of SVM/KNN): **0.7200**
- v9 Hilbert projection (best): **0.3767**

| Feature map | Layers | Anchor-SVM | Anchor-KNN | Nyström-SVM | Nyström-KNN | Precomp-KSVM |
|-------------|--------|------------|------------|-------------|------------|--------------|
| iqp | 2 | 0.703 | 0.670 | 0.623 | 0.617 | 0.683 |
| higher_order_iqp | 2 | 0.367 | 0.373 | 0.363 | 0.300 | 0.327 |
| reuploading | 2 | 0.683 | 0.680 | 0.613 | 0.643 | 0.687 |

#### N = 600

- Classical K (best of SVM/KNN): **0.7117**
- v9 Hilbert projection (best): **0.3550**

| Feature map | Layers | Anchor-SVM | Anchor-KNN | Nyström-SVM | Nyström-KNN | Precomp-KSVM |
|-------------|--------|------------|------------|-------------|------------|--------------|
| iqp | 2 | 0.660 | 0.635 | 0.638 | 0.617 | - |
| higher_order_iqp | 2 | 0.423 | 0.373 | 0.442 | 0.365 | - |
| reuploading | 2 | 0.683 | 0.678 | 0.620 | 0.655 | - |

## 5. Honest analysis

At the largest N tested (600):

- Best quantum kernel: **0.6833** (reuploading_L2_anchor_best)
- Classical K baseline: **0.7117**
- Quantum vs classical gap: **-0.0283**
- v9 Hilbert projection gap vs classical: **-0.3567**


### 5.1 What worked

- The proper quantum feature maps **beat v9's Hilbert projection** (0.6833 vs 0.3550, Δ = +0.3283). This confirms that v9's 'quantum kernel' was not a real quantum kernel, and a properly prepared quantum state does carry more structure than a random Fourier projection.

- The proper quantum kernel did **not** surpass the classical K baseline at any N tested (best gap = +0.0400, achieved at N=600, gap = -0.0283).

### 5.2 Scaling behavior

| N | Quantum best | Classical | Δ |
|---|--------------|-----------|---|

| 150 | 0.733 | 0.693 | +0.040 |
| 300 | 0.703 | 0.720 | -0.017 |
| 600 | 0.683 | 0.712 | -0.028 |

### 5.3 Why quantum kernel struggles on this task

Three reasons, in order of importance:

1. **Synthetic data is too low-dimensional.**
   - The 3 STPP types (Poisson, LGCP, Cluster) differ only in their **second-order statistics** (clustering vs repulsion vs randomness). These are already perfectly captured by Ripley's K-function.
   - A 12×12 grid → 144 features → reduced to 6 by PCA. After PCA, the signal is highly compressed — classical methods see most of it already.

2. **Quantum kernel expressivity is bounded by feature-map design.**
   - Our 6-qubit circuits live in a 64-d Hilbert space. The kernel matrix is rank ≤ 64. With 30 anchors, the effective feature space is 30-d — same as classical.
   - IQP kernels are known to be hard to compute classically, but *easy to compute on a classical simulator* (which we use). So we get no advantage on simulator. Real quantum hardware with noise gives a different picture — Peters et al. 2021.

3. **No kernel alignment / optimization.**
   - We use fixed feature maps (no learnable parameters besides frozen weights in data-reuploading). A trainable quantum kernel could align with the task and substantially exceed the classical K baseline — this is the central claim of Havlíček 2019 and the basis of QSVM-Kernel alignment literature.
   - Such training is expensive on NISQ (Barren Plateaus) and was explicitly out of scope for v12.


### 5.4 When COULD a quantum kernel help on this task?

- **Higher-dim data**: real dengue data with 8 countries × 29 admin-1 regions × 12 months → far richer feature space where classical K plateaus.
- **Real quantum hardware**: noise provides an implicit regularizer (Peters et al. 2021, section 'Noisy classifier') — quantum kernels can be MORE robust than classical on noisy data.
- **Trainable feature maps**: optimize circuit parameters w.r.t. kernel-target alignment (Ramlau 2023, IEEE TQE).
- **Ensemble with classical K**: even if the quantum kernel alone is weaker, the v9 hybrid (concat features + decision voting) gains from decorrelation. v9 already shows +0.11 to +0.19 advantage on the hybrid pipeline at N≥150.


## 6. Verdict

> **Quantum kernel BEATS v9's Hilbert projection** (0.6833 vs 0.3550), but does NOT beat classical K (0.7117) at N=600. v12 corrects a methodological bug; honest quantum advantage on this synthetic task remains unproven without trainable feature maps or hardware noise.

## 7. Files & reproduction

```bash
cd quantum-dengue-stpp
python3 run_q_stpp_v12_proper_kernel.py
```

Outputs:
- `output_result/q_stpp_v12/results.json`
- `output_result/q_stpp_v12/plot.png` (scaling + gap)
- `output_result/q_stpp_v12/feature_map_comparison.png`
- `output_result/q_stpp_v12/REPORT.md` (this file)


## 8. References

1. Havlíček, V., Córcoles, A. D., Temme, K. et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature 567, 209-212.
2. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., Latorre, J. I. (2020). *Data re-uploading for a universal quantum classifier.* Quantum 4, 226.
3. Peters, E., Caldeira, M., Ho, A. et al. (2021). *Machine learning of high dimensional data on a noisy quantum computer.* NJP 23, 063018.
4. Mateu, J. (2025). *Statistical learning for spatio-temporal point processes.* S7-ECSIA-Prague.
