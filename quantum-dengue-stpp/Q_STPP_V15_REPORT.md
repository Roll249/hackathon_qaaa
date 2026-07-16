# Q-STPP v15 (corrected) — Report

**Date**: 2026-07-16
**Status**: methodology finalised; **result tables to be populated by a run** of
`run_q_stpp_v15_fair.py` (see §4).

> **Withdrawal notice.** A previous version of this report claimed a "32.7×
> better L(r) error", a "95.3% R²", and a "Sum-of-Squares / Born-machine"
> pipeline. Those claims were **invalid** and are withdrawn:
> - the "32.7×" came from an unfair comparison (unseeded baseline, a broken
>   acceptance temperature that crippled Metropolis-Hastings, unequal budgets,
>   and a ratio divided by a near-zero denominator);
> - the "R² 95.3%" was never computed by any code;
> - no SOS / Born-machine / SDP code ever existed.
>
> This report describes the corrected, fair experiment. It contains **no result
> numbers until the script is run** — none are hand-written.

---

## 1. Problem

Second-Order Preserving (SOP) permutations augment spatio-temporal point-process
data by permuting event time-stamps while preserving Ripley's L-function
(Mohler & Mateu 2024). A good augmenter must simultaneously:

1. **preserve L(r)** — low `E(π) = ‖L(π) − L_target‖²`, and
2. **stay diverse** — the returned set must not collapse to one permutation.

We compare three classical search strategies for producing such permutations.

---

## 2. Methods

All three are classical local searches over permutations (no quantum hardware,
no quantum simulator — see the note in §6):

| Key | Name | Proposal | Acceptance |
|-----|------|----------|------------|
| `mh` | Metropolis-Hastings | single swap | Metropolis, scale-adaptive annealed temperature |
| `grover` | Grover-inspired (greedy) | single swap | greedy (accept iff error drops) |
| `qaoa` | QAOA-inspired (multi-swap) | multi swap | greedy |

"Grover-/QAOA-inspired" are heuristic analogies (focused amplification / mixer
perturbations), **not** quantum circuits.

---

## 3. Fair-comparison protocol

The comparison is controlled on every axis that the earlier version got wrong:

- **Same randomness** — every method re-instantiates `np.random.default_rng(seed)`
  with the same `seed`, so all three start from the same random state; the stream
  diverges only through the differing proposal/acceptance, never through a
  different seed.
- **Same budget** — each method spends *exactly* `evals_per_perm` L-summary
  evaluations per permutation (the L-summary, an `O(N²)` distance sweep, is the
  dominant cost; we count evaluations, not swaps).
- **Same objective** — all minimise `E(π) = ‖L(π) − L_target‖²`.
- **Proper MH temperature** — geometric annealing derived from the initial error
  magnitude, replacing the fixed `1/10` that made MH accept ~90% of worsening
  moves.
- **Two metrics** — we report **L(r) error** (quality) *and* **set diversity**
  (mean pairwise normalised Hamming distance), so a low-error-but-collapsed
  method cannot look like a winner.
- **Bounded ratios** — any error ratio is computed with a clamped denominator; it
  is a secondary figure, not the headline.

Data: reproducible Hawkes-like space-time patterns from `simulate_hawkes(seed)`.

---

## 4. Results — to be generated

Run:

```bash
python3 run_q_stpp_v15_fair.py --seeds 1 2 3 4 5 --n_events 20 30 50
```

This writes:

- `output_result/q_stpp_v15_fair/fair_comparison_results.json`
- `output_result/q_stpp_v15_fair/fair_comparison_plot.png`

Then fill in the tables below from `fair_comparison_results.json["aggregate"]`.

### 4.1 Quality — mean L(r) error (lower is better)

| N (target) | MH | Grover-inspired | QAOA-inspired |
|-----------|----|-----------------|---------------|
| 20 | _tbd_ | _tbd_ | _tbd_ |
| 30 | _tbd_ | _tbd_ | _tbd_ |
| 50 | _tbd_ | _tbd_ | _tbd_ |

### 4.2 Diversity — mean pairwise Hamming distance (higher is better)

| N (target) | MH | Grover-inspired | QAOA-inspired |
|-----------|----|-----------------|---------------|
| 20 | _tbd_ | _tbd_ | _tbd_ |
| 30 | _tbd_ | _tbd_ | _tbd_ |
| 50 | _tbd_ | _tbd_ | _tbd_ |

---

## 5. How to interpret (expected shape, not a claim)

Because greedy search optimises error directly while the MH sampler keeps
diversity, we expect a **quality–diversity trade-off**:

- greedy (`grover`, `qaoa`): lower L(r) error, lower diversity;
- sampler (`mh`): slightly higher error, substantially higher diversity.

The honest takeaway is that method choice depends on the downstream need
(pure fidelity vs. augmentation diversity) — **not** that any method is "N×
better", and **not** that anything quantum is happening.

---

## 6. Honest limitations

- **Classical only** — no quantum hardware or simulator is used; the
  "quantum-inspired" label refers to the search heuristic, nothing more.
- **Synthetic data only** — Hawkes-like patterns, not real dengue surveillance.
- **Summary-level** — we compare L(r) preservation of permutation sets, not
  end-to-end outbreak-classification accuracy.

---

## 7. Files

- `run_q_stpp_v15_fair.py` — the experiment
- `output_result/q_stpp_v15_fair/` — results (created by a run)
- `ARCHITECTURE.md`, `THEORY.md` — matching design and background
- `DEVELOPMENT_HISTORY.md` — full history incl. withdrawn claims
