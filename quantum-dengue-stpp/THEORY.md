# Q-STPP v15 (corrected) — Theoretical background

> **Honest scope note.** This document describes only the mathematics that the
> code (`run_q_stpp_v15_fair.py`) actually implements: spatio-temporal point
> patterns, Ripley's K/L second-order summaries, SOP permutations, and classical
> local search. Earlier drafts of this file described a "Sum-of-Squares (SOS)
> programming / Born-machine / PSD-certificate" pipeline — **none of that exists
> in the code** and it has been removed.

---

## 1. Spatio-temporal point processes

A spatio-temporal point process is a random collection of events
`{(xᵢ, yᵢ, tᵢ)}` in space × time. For dengue we treat case reports as such
events. The process is characterised by its conditional intensity

```
λ*(x, t | H_t) = expected event rate at (x, t) given history H_t.
```

We do **not** fit an intensity model in this repository. We work at the level of
**second-order summary statistics**, which is what the SOP method operates on.

---

## 2. Ripley's K- and L-functions (second-order structure)

For a stationary pattern with intensity λ, Ripley's K-function is

```
K(r) = (1/λ) · E[ number of further events within distance r of a typical event ].
```

Empirically, over N events with pairwise distances `d_ij`,

```
K̂(r) ≈ (1/N²) · #{ (i,j), i≠j : d_ij < r }.
```

A variance-stabilising transform gives an L-function. The classical planar
choice is `L(r) = √(K(r)/π)`. Because our events live in space × time, the code
uses a stabilised summary `L(r) = sign(K)·|K|^(1/3)` over a space-time distance
(with time rescaled to the spatial range). The exact transform is not important
for the experiment: it is applied **identically** to every method, so relative
comparisons of "how well is L(r) preserved" remain valid.

Implemented in `compute_L_summary(times, x, y, r_values)`.

---

## 3. Second-Order Preserving (SOP) permutations

**Goal (Mohler & Mateu 2024).** Augment a point pattern by permuting the event
time-stamps while keeping its second-order spatial structure — i.e. produce a
new labelling whose L(r) stays close to the original's `L_target(r)`.

Given the target, the SOP search minimises the discrepancy

```
E(π) = ‖ L(π) − L_target ‖²      (mean squared over the radii r)
```

over permutations π of the time index. This is `l_error(...)` in the code.

**Two objectives, not one.** A permutation set is useful for augmentation only
if it (a) has low `E(π)` **and** (b) is diverse — otherwise you have copies of a
single sample. We therefore also measure

```
diversity(Π) = mean over pairs (π_a, π_b) of  (fraction of positions that differ).
```

`diversity = 0` means total mode collapse; `diversity ≈ 1` means every
permutation differs everywhere. This is `set_diversity(...)`.

---

## 4. Search strategies compared

All three are classical local searches over permutations under an **identical
evaluation budget** (same number of `compute_L_summary` calls) and an
**identical RNG seed**.

### 4.1 Metropolis-Hastings (`mh`)

Standard Metropolis sampler with single-swap proposals:

```
propose π' by swapping two time indices
accept if  E(π') < E(π)   or with prob.  exp( −(E(π') − E(π)) / T_step ).
```

The temperature `T_step` is **scale-adaptive**: it starts at the initial error
magnitude of a random permutation and decays geometrically to 1% of it
(simulated annealing). This is the fix for the earlier bug where a fixed `T=1/10`
made the sampler accept ~90% of worsening moves and effectively random-walk.

MH is a *sampler*: by design it accepts some worsening moves, which keeps the
returned set diverse.

### 4.2 Grover-inspired greedy search (`grover`)

Same single-swap proposal, but **greedy** acceptance (accept iff the error
strictly decreases). The name is an analogy to amplitude amplification
(focusing effort on improving candidates); it is **not** a quantum circuit.
Greedy search reaches lower error but lower diversity.

### 4.3 QAOA-inspired multi-swap (`qaoa`)

**Multi-swap** proposals (several swaps before one evaluation), analogous to a
QAOA mixer perturbation, with greedy acceptance. Again purely classical.

---

## 5. Fair-comparison protocol

| Control | How it is enforced |
|---------|--------------------|
| Same randomness | every method re-instantiates `np.random.default_rng(seed)` with the same `seed`, so all start from the same random state (the stream then diverges only through the differing proposal/acceptance) |
| Same budget | each method spends exactly `evals_per_perm` L-summary evaluations per permutation |
| Same objective | all minimise `E(π) = ‖L(π) − L_target‖²` |
| Honest reporting | report L(r) error **and** diversity **and** time; ratios are clamped |

Complexity per L-summary evaluation is `O(N²)` (a pairwise distance sweep). With
`P` permutations and `B` evaluations each, a method costs `O(P·B·N²)`, identical
across the three strategies.

---

## 6. What is deliberately NOT here

- No quantum state, amplitude, Born rule, or measurement.
- No Sum-of-Squares programming, SDP, or PSD certificate.
- No intensity-model fitting, no R² regression, no neural network.

These appeared in earlier fabricated drafts and were removed to keep the theory
in one-to-one correspondence with the code.

---

## References

1. Ripley, B. D. (1977). Modelling spatial patterns. *JRSS-B* 39(2), 172–212.
2. Baddeley, A., Rubak, E. & Turner, R. (2015). *Spatial Point Patterns:
   Methodology and Applications with R*. CRC Press.
3. Mohler, G. & Mateu, J. (2024). Second-order preserving permutations. *Stat*.
4. Mateu, J. (2025). *Statistical learning for spatio-temporal point processes*
   (S7-ECSIA, Prague).
5. Metropolis, N. et al. (1953). Equation of state calculations by fast
   computing machines. *J. Chem. Phys.* 21, 1087.
