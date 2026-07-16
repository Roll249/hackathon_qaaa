# Q-STPP v16: Theoretical Foundations

## 1. Spatial-Temporal Point Processes (STPP)

### 1.1 Definition

A **spatial-temporal point process** is a random mathematical model for events occurring in space and time. For dengue prediction:

- **Spatial coordinates**: x = (x, y) ∈ ℝ²
- **Temporal coordinate**: t ∈ ℝ⁺

The process is characterized by its **conditional intensity function**:

```
λ*(x, t | Hₜ) = lim_{dt→0, dx→0} Pr{event in [t, t+dt) × B(x, dx) | Hₜ} / (dt · dx)
```

Where:
- Hₜ = history of events up to time t
- B(x, dx) = ball of radius dx around x
- λ*(x, t) = expected number of events per unit area per unit time

### 1.2 Hawkes Process (Self-Exciting)

For dengue, we use **Hawkes processes** (self-exciting point processes):

```
λ(x, t) = μ(x) + Σ_{t_i < t} g(t - t_i) · h(x - x_i)
```

Where:
- μ(x) = background intensity (constant or spatially varying)
- g(·) = temporal kernel (triggering function)
- h(·) = spatial kernel (influence function)

**Exponential temporal kernel** (per Mateu 2025):
```
g(t - t_i) = β · exp(-β(t - t_i))
```

**Gaussian spatial kernel**:
```
h(x - x_i) = (1/2πσ²) · exp(-||x - x_i||² / 2σ²)
```

### 1.3 Intensity Estimation Goal

Given historical events {xᵢ, tᵢ}, predict future intensity:

```
λ_predicted(x, t) = λ̂(x, t | Hₜ)
```

**Objective**: Minimize L(r) - the error in L-function approximation.

---

## 2. Second-Order Statistics: K-function and L-function

### 2.1 Ripley's K-function

For spatial point processes, the **K-function** measures spatial clustering/regularity:

```
K(r) = (1/λ) · E[number of points within distance r of an arbitrary point]
```

**Estimator** (pair counting):
```
K̂(r) = (|W|/n(n-1)) · Σ_{i≠j} 1{||x_i - x_j|| < r}
```

Where:
- |W| = area of observation window
- n = number of points
- 1{·} = indicator function

### 2.2 L-function (Stabilized Transform)

The **L-function** is a stabilized transform of K:

```
L(r) = (K(r))^(1/3) · sign(K(r))
```

Or equivalently (for complete transform):
```
L(r) = sqrt(K(r) / π) - r
```

**Why L-function?**
- Stabilizes variance
- Under CSR (Complete Spatial Randomness), L(r) = 0
- L(r) > 0 → clustering
- L(r) < 0 → regularity

### 2.3 Space-Time L-function

For STPP, we combine spatial and temporal distance:

```
d²((x,t), (x',t')) = ||x - x'||² + α²|t - t'|²
```

Where α controls the space-time tradeoff.

**Space-time K-function**:
```
K(r) = (1/λ²) · E[number of space-time pairs within distance r]
```

### 2.4 L(r) Error Metric

For SOP augmentation, we measure how well a permutation preserves L-function:

```
L(r) error = ||L_perm(r) - L_target(r)||²
```

This is the **primary quality metric** for augmentation methods.

---

## 3. SOP (Second-Order Preserving) Permutations

### 3.1 Motivation

SOP permutations **shuffle event timestamps** while attempting to preserve second-order structure:

- **Original pattern**: events with real timestamps
- **Permutation**: same events, shuffled timestamps
- **Purpose**: Data augmentation for ML training

### 3.2 Problem Formulation

Given:
- Events: {(x₁, t₁), (x₂, t₂), ..., (xₙ, tₙ)}
- Target L-function: L_target(r) from original pattern
- Candidate permutation: π = permutation of indices

**Goal**: Find permutation that minimizes:

```
E(π) = ||L_π(r) - L_target(r)||²
```

Where L_π(r) uses times {t_{π(1)}, t_{π(2)}, ..., t_{π(n)}}.

### 3.3 Why SOP Matters for Dengue

- **Real timestamps**: Dengue cases have specific time patterns
- **Spatial clustering**: Hotspots are stable across time
- **Augmentation**: SOP permutations create realistic synthetic data
- **ML training**: More diverse training data improves generalization

### 3.4 Two Objectives

SOP optimization must balance:

1. **Quality**: Low L(r) error (preservation)
2. **Diversity**: High Hamming distance between permutations

```
Diversity(π_set) = (1/|Π|²) · Σ_{π,π'} Hamming(π, π')
```

---

## 4. Optimization Methods for SOP Search

### 4.1 Metropolis-Hastings (MH)

**Algorithm**:
1. Start with random permutation π₀
2. Propose swap: π' = swap(π, i, j)
3. Compute ΔE = E(π') - E(π)
4. Accept with probability: P(accept) = min(1, exp(-ΔE/T))
5. Repeat

**Key parameters**:
- T = temperature (controls exploration)
- Annealing schedule: T decreases over time

**Trade-offs**:
- ✅ High diversity (accepts worse moves)
- ❌ May not reach global optimum
- ✅ No tuning required (self-regulating)

### 4.2 Greedy Search

**Algorithm**:
1. Start with random permutation π₀
2. For each swap (i, j):
   - Compute E(π with swap)
   - Keep only if E decreases
3. Repeat until convergence

**Trade-offs**:
- ✅ Fast convergence to local optimum
- ✅ Lowest error when successful
- ❌ Mode collapse (low diversity)
- ❌ Sensitive to initial state

### 4.3 QAOA-Inspired Multi-Swap

**Inspired by Quantum Approximate Optimization**:

The QAOA ansatz applies p rounds of:
```
|ψ(β, γ)⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^⊗n
```

Our **classical approximation**:
1. Propose multiple swaps simultaneously
2. Accept if total improvement
3. Repeat

**Trade-offs**:
- ✅ Balances error and diversity
- ✅ Explores larger neighborhoods
- ❌ More hyperparameters

---

## 5. Quantum-Inspired Approaches (Research)

### 5.1 Quantum Amplitude Estimation

For rare event probability estimation:

```
P_success = |⟨ψ|target⟩|²
QAE provides: P̂ ≈ O(1/√N) vs classical O(1/N)
```

**Potential application**: Estimating tail probabilities in L-function distributions.

**Honest caveat**: Requires fault-tolerant quantum computer.

### 5.2 Quantum Kernel Methods

Define quantum feature map:
```
|x⟩ → |φ(x)⟩ = U_φ(x) |0⟩^⊗n
```

Kernel:
```
K(x, x') = |⟨φ(x)|φ(x')⟩|²
```

**Potential application**: Pattern classification with quantum-enhanced similarity.

**Honest caveat**: 
- Classical simulation: O(2ⁿ) - loses advantage
- Real hardware: NISQ noise dominates for n > 20

### 5.3 VQE for Kernel Parameter Optimization

Variational Quantum Eigensolver:
```
min_θ ⟨ψ(θ)|H|ψ(θ)⟩
```

**Potential application**: Optimizing Hawkes kernel parameters.

**Honest caveat**: Barren plateaus, parameter noise.

---

## 6. Fair Comparison Protocol

### 6.1 Identical Budget

For fair comparison, all methods use:
- **Same random seed**: Identical starting states
- **Same evaluation count**: N L-function evaluations per permutation
- **Same problem instance**: Same synthetic data

### 6.2 Metrics

**Quality metric**:
```
L(r) error = mean((L_perm - L_target)²)
```

**Diversity metric**:
```
Diversity = mean(Hamming(π_a, π_b)) / n
```

### 6.3 Reporting Standards

For honest claims:
1. Report BOTH quality AND diversity
2. Never optimize for one at expense of other
3. Include error bars (across seeds)
4. State assumptions explicitly

---

## 7. Mathematical References

### 7.1 Key Theorems

**Theorem 1 (Matern's Thinning Theorem)**:
Any point process can be generated by thinning a Poisson process with intensity-dependent probability.

**Theorem 2 (Campbell's Theorem)**:
For any function f:
```
E[Σ_{x∈PPP} f(x)] = ∫ f(x) λ(x) dx
```

**Theorem 3 (SLLN for K-function)**:
K̂(r) → K(r) almost surely as n → ∞.

### 7.2 Complexity Bounds

| Operation | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| K-function | Ω(N²) | O(N²) |
| L-function | Ω(N²) | O(N²) |
| MH acceptance | Ω(N log N) | O(N²) |
| Greedy search | Ω(N log N) | O(N³) |

### 7.3 Convergence Rates

| Method | Error Rate | Diversity Rate |
|--------|------------|---------------|
| MH | O(1/√T) | O(1) (asymptotic) |
| Greedy | O(exp(-cT)) | O(1/T) |
| QAOA-inspired | Problem-dependent | Problem-dependent |

---

## 8. Extensions (Future Work)

### 8.1 Non-Stationary Kernels

Per Mateu 2025, non-stationary kernels capture heterogeneous spread:

```
v(s, s') = ⟨φ_s, φ_s'⟩
```

Where φ_s is a location-dependent feature vector.

### 8.2 Marked Point Processes

For multi-serotype dengue:

```
λ_cl(t, s) = μ_cl + Σ α_cl,c'l' · g(t-t') · h(s-s')
```

Where c = crime/serotype category, l = landmark category.

### 8.3 Network-Aware Distances

Per Mateu 2025 for crime data:

```
d_net(s, s') = shortest_path_length_on_street_network(s, s')
```

Captures actual mobility patterns.

---

## 9. References

1. **Mateu, J. (2025)**. Statistical learning for spatio-temporal point processes: inference and testing. *ECSIA 2025, Prague*.

2. **Mohler, G. & Mateu, J. (2024)**. Second order preserving point process permutations. *Stat*.

3. **Daley, D.J. & Vere-Jones, D. (2003)**. An introduction to the theory of point processes. Springer.

4. **Diggle, P.J. (2013)**. Statistical analysis of spatial and spatio-temporal point patterns. CRC Press.

5. **Hawkes, A.G. (1971)**. Spectra of some self-exciting and mutually exciting point processes. *Biometrika*.

6. **Farhi, E., Goldstone, J. & Gutmann, S. (2014)**. A quantum approximate optimization algorithm. *arXiv:1411.4028*.

7. **Haviland, J. et al. (2024)**. A hybrid quantum-classical approach for optimization of spatial Coverage problems. *arXiv:2405.XXXXX*.

8. **Schuld, M. et al. (2015)**. An introduction to quantum machine learning. *Contemporary Physics*.

---

## Appendix: Notation Summary

| Symbol | Meaning |
|--------|---------|
| λ(x,t) | Conditional intensity at (x,t) |
| K(r) | Ripley's K-function |
| L(r) | Stabilized L-function |
| SOP | Second-Order Preserving |
| PPP | Poisson Point Process |
| MH | Metropolis-Hastings |
| QAOA | Quantum Approximate Optimization Algorithm |
| VQE | Variational Quantum Eigensolver |
| NISQ | Noisy Intermediate-Scale Quantum |
