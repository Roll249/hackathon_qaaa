# Quantum-Dengue-STPP — v15 (corrected)

**Hackathon project (QC4SG) — spatio-temporal point processes for dengue.**

Fair, reproducible comparison of three **classical** local-search strategies for
generating Second-Order Preserving (SOP) permutations that augment
spatio-temporal point-process data (Mohler & Mateu 2024).

> **This is a methods benchmark on *synthetic* space-time patterns, not a dengue
> forecast.** Despite the project name, the pipeline does **not** read any dengue
> surveillance data and has **not** been validated on real dengue outbreaks. The
> synthetic Hawkes patterns are a stand-in to compare permutation-search methods.
> Applying this to the OpenDengue-derived data in [`../dengue_dataset/`](../dengue_dataset/)
> is future work.

> **Honest scope.** This version contains **no quantum hardware and no quantum
> simulator** — everything is classical NumPy. Two of the three strategies are
> *quantum-inspired* heuristics (Grover-/QAOA-inspired) but run no quantum
> circuit, and **no quantum advantage is claimed**. Earlier versions (v9–v12)
> that advertised a "quantum kernel" or large "quantum advantage" numbers, and a
> v15 draft claiming "32.7× / 95.3% R²", have been **withdrawn** — see
> [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md).

---

## What it does

For a synthetic Hawkes-like space-time pattern, it generates SOP permutations
with three methods under a strictly fair protocol (same seed, same evaluation
budget) and reports **two** metrics:

- **L(r) error** — how well each permutation preserves Ripley's L-function
  (lower is better);
- **set diversity** — how different the generated permutations are from each
  other (higher is better; a collapsed set is useless for augmentation).

| Key | Method | Proposal | Acceptance |
|-----|--------|----------|------------|
| `mh` | Metropolis-Hastings | single swap | Metropolis, annealed temperature |
| `grover` | Grover-inspired | single swap | greedy |
| `qaoa` | QAOA-inspired | multi swap | greedy |

The expected result is a **quality–diversity trade-off**, not a single winner.

---

## Quick start

```bash
pip install numpy scipy matplotlib

# defaults: seeds 1..5, N ∈ {20,30,50}, 10 perms, 200 evals/perm/method
python3 run_q_stpp_v15_fair.py

# or use the wrapper
./run.sh
```

Outputs:

- `output_result/q_stpp_v15_fair/fair_comparison_results.json`
- `output_result/q_stpp_v15_fair/fair_comparison_plot.png`

Runtime: a few seconds per (seed, N) cell on a laptop CPU. No GPU, no quantum
backend.

---

## Project structure

```
quantum-dengue-stpp/
├── run_q_stpp_v15_fair.py   # the entire pipeline
├── run.sh                   # convenience wrapper
├── requirements.txt         # numpy, scipy, matplotlib
├── README.md                # this file
├── ARCHITECTURE.md          # system design (matches the code)
├── THEORY.md                # L-function + local-search background
├── Q_STPP_V15_REPORT.md     # methodology + results template
├── DEVELOPMENT_HISTORY.md   # history incl. withdrawn claims
├── output_result/
│   ├── data/                # legacy CSVs from earlier versions — NOT used by v15
│   └── q_stpp_v15_fair/     # results.json + plot.png (created by a run)
└── archive/                 # withdrawn / superseded versions (v4–v12)
```

---

## What this does and does NOT claim

**Does**: a controlled, same-budget comparison of three classical permutation
search heuristics, reporting both L(r) preservation and augmentation diversity.

**Does not**: run any quantum circuit, show a quantum advantage, or validate on
real dengue surveillance data.

---

## References

- Mateu, J. (2025). *Statistical learning for spatio-temporal point processes*
  (S7-ECSIA, Prague).
- Mohler, G. & Mateu, J. (2024). Second-order preserving permutations. *Stat*.
- Ripley, B. D. (1977). Modelling spatial patterns. *JRSS-B*.

## License

MIT (hackathon project).
