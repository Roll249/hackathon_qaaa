# Q-STPP v15 (corrected) — Architecture

**Last updated**: 2026-07-16
**Scope**: Fair comparison of classical / quantum-inspired strategies for
generating Second-Order Preserving (SOP) permutations, used to augment
spatio-temporal point-process (STPP) data for dengue outbreak modelling.

> **Honest scope note.** This version contains **no quantum hardware and no
> quantum simulator**. Everything here is classical NumPy. Two of the three
> search strategies are *quantum-inspired* heuristics (they borrow an idea from
> Grover amplification / QAOA mixers) but do not execute a quantum circuit. No
> quantum advantage is claimed. Earlier versions (v9–v12) that referenced a
> "quantum kernel" or reported large "quantum advantage" numbers have been
> withdrawn — see `DEVELOPMENT_HISTORY.md`.

---

## 1. What the system actually does

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Q-STPP v15 (corrected) pipeline                       │
└──────────────────────────────────────────────────────────────────────┘

  simulate_hawkes(seed)                     # reproducible space-time pattern
        │   → times, x, y
        ▼
  compute_L_summary(times, x, y, r)         # Ripley-K based second-order summary
        │   → L_target(r)
        ▼
  For each strategy ∈ {mh, grover, qaoa}:   # identical seed + identical budget
        │   generate n_perms permutations of the time-stamps
        │   each perm minimises  ‖L(perm) − L_target‖²  by local search
        ▼
  Report per strategy:
     • mean L(r) error   (quality — lower is better)
     • set diversity     (mean pairwise Hamming distance — higher is better)
     • wall-clock time
        │
        ▼
  fair_comparison_results.json  +  fair_comparison_plot.png
```

All of the above lives in a single file: `run_q_stpp_v15_fair.py`.

---

## 2. The three strategies

Every strategy is classical local search over permutations of the event
time-stamps. They differ **only** in how a candidate is proposed and accepted;
they share the same objective, the same RNG seed, and the same number of
L-summary evaluations.

| Key | Name | Proposal | Acceptance |
|-----|------|----------|------------|
| `mh` | Metropolis-Hastings | single swap | Metropolis, scale-adaptive annealed temperature |
| `grover` | Grover-inspired | single swap | greedy (accept iff error decreases) |
| `qaoa` | QAOA-inspired | multi-swap (mixer-like) | greedy |

"Grover-inspired" = focused hill-climbing (analogy to amplitude amplification).
"QAOA-inspired" = multi-swap perturbation (analogy to a mixer Hamiltonian).
Neither name implies any quantum computation.

---

## 3. The fairness protocol (why v15-initial was wrong)

The first v15 comparison was invalid for several reasons, all now fixed:

| Problem (old) | Fix (this version) |
|---------------|--------------------|
| MH used the unseeded global `np.random`; the others used `default_rng(42)` | Every method is seeded identically — each re-instantiates `np.random.default_rng(seed)` with the same `seed`, so all start from the same random state |
| Acceptance temperature was a fixed `10`, so MH accepted ~90% of worsening moves and barely optimised | Scale-adaptive geometric annealing derived from the initial error magnitude (no magic constant) |
| "Same budget" was false — step caps were 100 / 50 / 10 | Every method spends **exactly** `evals_per_perm` L-summary evaluations per permutation |
| Only mean error was reported, which rewards mode collapse | Report **both** L(r) error **and** set diversity |
| Improvement ratio divided by a near-zero error → 35–75× blow-ups | `ratio()` clamps the denominator; the headline is the two metrics, not a multiplier |

**The unit of budget** is one `compute_L_summary` call — the dominant O(N²)
cost. Counting evaluations (not swaps) makes the budget comparable across
single-swap and multi-swap proposals.

---

## 4. Why report diversity

SOP permutations exist to *augment* a dataset. A method that collapses to one
low-error permutation gives you the same sample ten times — useless for
augmentation. A method that keeps a spread of near-target permutations is what
you actually want. Greedy search (`grover`, `qaoa`) tends to reach lower error
but lower diversity; the MH sampler trades a little error for much more
diversity. The honest conclusion is a **quality–diversity trade-off**, not a
single winner.

---

## 5. Code organisation

```
quantum-dengue-stpp/
├── run_q_stpp_v15_fair.py      # the entire pipeline (data, methods, plots)
├── run.sh                      # convenience wrapper
├── requirements.txt            # numpy, scipy, matplotlib
│
├── ARCHITECTURE.md             # this file
├── THEORY.md                   # L-function + local-search background
├── Q_STPP_V15_REPORT.md        # methodology + results (populated by a run)
├── DEVELOPMENT_HISTORY.md      # version history incl. withdrawn claims
├── README.md                   # quick start
│
├── output_result/
│   ├── data/                   # legacy CSVs from earlier versions — NOT read by v15
│   └── q_stpp_v15_fair/        # results.json + plot.png (created by a run)
│
└── archive/                    # withdrawn / superseded versions (v4–v12)
```

---

## 6. Key functions (`run_q_stpp_v15_fair.py`)

| Function | Role |
|----------|------|
| `simulate_hawkes` | reproducible self-exciting space-time pattern |
| `compute_L_summary` | Ripley-K based second-order summary L(r) |
| `l_error` | mean squared deviation of a permutation's L(r) from target |
| `set_diversity` | mean pairwise normalised Hamming distance of a perm set |
| `_generate_perms` | fair local search shared by all three strategies |
| `evaluate_method` | run one strategy, return error / diversity / time |
| `run_single` | one (seed, N) cell across all methods |
| `aggregate` | average metrics across seeds, grouped by N |

---

## 7. Running

```bash
# defaults: seeds 1..5, N ∈ {20,30,50}, 10 perms, 200 evals/perm/method
python3 run_q_stpp_v15_fair.py

# custom sweep
python3 run_q_stpp_v15_fair.py --seeds 1 2 3 --n_events 20 40 60 --evals_per_perm 400
```

Dependencies: `numpy`, `scipy`, `matplotlib` only. Runtime is a few seconds per
(seed, N) cell on a laptop CPU; no GPU or quantum backend required.

---

## 8. What this version does and does not claim

**Does**: a controlled, reproducible, same-budget comparison of three classical
permutation-search heuristics on a synthetic space-time pattern, reporting both
L(r) preservation and augmentation diversity.

**Does not**: run any quantum circuit, demonstrate a quantum advantage, or
validate on real dengue surveillance data. Those remain future work.

---

## 9. References

1. Mateu, J. (2025). *Statistical learning for spatio-temporal point processes*
   (S7-ECSIA, Prague) — SOP permutations and the K/L-function baseline.
2. Mohler, G. & Mateu, J. (2024). Second-order preserving permutations. *Stat*.
3. Ripley, B. D. (1977). Modelling spatial patterns. *JRSS-B* — the K-function.
