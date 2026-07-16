# Q-STPP — Quantum-*Inspired* Data Augmentation for Dengue Spatio-Temporal Point Processes

**Hackathon project (QC4SG).** This repository studies **Second-Order Preserving (SOP)
permutations** — a data-augmentation technique for spatio-temporal point-process (STPP)
data (Mohler & Mateu, 2024) — and compares three classical local-search strategies for
generating them, one of which is a plain Metropolis-Hastings sampler and two of which
borrow *ideas* from quantum algorithms (Grover, QAOA) without running any quantum
circuit. It also includes a real Southeast-Asia dengue surveillance dataset
(OpenDengue-derived) that the pipeline can run against.

> **Honest scope, read this first.** There is **no quantum hardware and no quantum
> simulator** anywhere in this repository. Every number in every result file was produced
> by classical NumPy/SciPy/pandas code. "Grover-inspired" and "QAOA-inspired" are names
> for classical heuristics that borrow an *idea* from those algorithms (focused
> amplification, mixer-style perturbation) — nothing here demonstrates a quantum
> advantage. Earlier versions of this project (v9–v12) made larger claims that were later
> found to be invalid and were formally withdrawn; see
> [`quantum-dengue-stpp/DEVELOPMENT_HISTORY.md`](quantum-dengue-stpp/DEVELOPMENT_HISTORY.md).
> This document, and the code it describes, intentionally stay inside what was actually
> run and measured.

---

## 1. Repository layout

```
hackathon_qaaa/
├── dengue_dataset/            # real OpenDengue-derived SEA dengue data + prep scripts
├── quantum-dengue-stpp/       # the Q-STPP pipeline (synthetic + real-data variants)
├── output_dataset/            # results of the 8-country real-data run (see §7)
├── message.txt                # research proposal (future-work vision, not implemented)
├── improve.md                 # exploratory notes on possible QML directions (not implemented)
└── S7-ECSIA-2025-Prague.pdf   # reference paper (Mateu, S7-ECSIA 2025)
```

### 1.1 `dengue_dataset/` — real data

Southeast-Asia dengue surveillance data derived from
[OpenDengue](https://opendengue.org), covering 11 SEA countries (spatial, yearly
resolution) and 8 countries / 233 admin1 regions (monthly resolution).

| File | What it is |
|---|---|
| `sea_dengue_spatial.csv` | Yearly case counts, various spatial resolutions, 11 countries |
| `sea_dengue_admin1_month.csv` | Monthly case counts at admin1 (province/state) resolution, 8 countries, 1993–2022 |
| `sea_dengue_admin1_month_pivot.csv` | The same data pivoted to date × region |
| `filter_sea.py` | Filters the raw OpenDengue extract down to the 11 SEA countries |
| `make_training_set.py` | Filters to `S_res=Admin1, T_res=Month`, sorts, cleans |
| `make_pivot.py` | Builds the pivot table above |
| `eda_analysis.py` / `eda_analysis_v2.py` | Exploratory data analysis (per-country stats, seasonality, epidemic-year detection, figures) |
| `check_training_set.py`, `inspect_data.py` | Quick data-sanity scripts |

None of these files require quantum anything — they are plain pandas data preparation.

### 1.2 `quantum-dengue-stpp/` — the pipeline

The actual experiment. Two entry points that share the same core code:

| File | Data source | Purpose |
|---|---|---|
| `run_q_stpp_v15_fair.py` | Synthetic Hawkes-process simulation | The original, fully controlled benchmark |
| `run_q_stpp_v15_real.py` | Real `dengue_dataset/sea_dengue_admin1_month.csv` | Same benchmark, real event dates + regions (§4) |

Supporting docs: `ARCHITECTURE.md` (system design), `THEORY.md` (L-function + local-search
background), `Q_STPP_V15_REPORT.md` (methodology template), `DEVELOPMENT_HISTORY.md`
(full history, including withdrawn claims from earlier versions), `archive/` (superseded
v4–v12 code, kept for the record, not used by anything current).

### 1.3 `output_dataset/` — the real-data run results

Output of running `run_q_stpp_v15_real.py` against all 8 countries in
`dengue_dataset/`. See §7 for the full analysis.

---

## 2. The problem: what is a Second-Order Preserving permutation, and why generate one?

A spatio-temporal point process is a set of events, each with a location `(x, y)` and a
time `t` — a dengue case report is a natural example. Statistical models of such
processes (and the deep-learning models increasingly used to forecast them) need more
training examples than a handful of years of surveillance data can provide.

**SOP augmentation** (Mohler & Mateu, 2024) creates *new* synthetic event sets from a
*real* one by **permuting which timestamp is attached to which location**, while trying
to preserve the pattern's second-order spatio-temporal structure — summarized by
**Ripley's K/L function**, which measures how clustered or dispersed the events are
across separation distances `r`. A permutation that keeps `L(r)` close to the original is
"second-order preserving": it looks statistically similar to the real data without being
a copy of it.

A useful augmenter must do **two** things at once:

1. **Preserve `L(r)`** — low error relative to the original pattern's `L(r)`.
2. **Stay diverse** — generate *different* permutations, not the same (or near-identical)
   one over and over. A method that always converges to the same low-error answer is
   useless for augmentation: it gives you one new sample, not many.

This repository's core question is: **which of three classical search strategies finds
the best trade-off between these two goals, under an identical computational budget?**

---

## 3. Core architecture — `run_q_stpp_v15_fair.py`, function by function

This is the file everything else imports from. Read top to bottom, it is the entire
pipeline.

### 3.1 `simulate_hawkes(n_events_target, mu, theta, omega, T, space_size, seed)`

Generates a synthetic self-exciting ("Hawkes-like") space-time point pattern via Ogata's
thinning algorithm: propose candidate event times from a homogeneous process at rate
`lam_max`, accept each one with probability `lam(t) / lam_max`, where `lam(t) = mu +
Σ θ·ω·exp(-ω·dt)` sums the decaying influence of past events within a 2-time-unit window
(`dt < 2.0`). `lam_max` is recomputed at every step from `n_active` — the number of
events still inside that decay window — so it stays a *tight* upper bound instead of
drifting upward with the total historical event count (a bug fixed in this version; see
§8). Returns `(times, x, y)` as three NumPy arrays. The seed makes the whole pattern
exactly reproducible.

### 3.2 `compute_L_summary(times, coords_x, coords_y, r_values, T=1.0, space_size=1.0)`

The second-order summary statistic everything else is built around.

1. Stack `(x, y, time·scale)` into one 3D point cloud (`time_scale = space_size / T`),
   treating time as a third spatial-like axis.
2. Compute the full pairwise Euclidean distance matrix (`scipy.spatial.distance.pdist` +
   `squareform`).
3. For each radius `r` in `r_values`, `K(r) = (count of pairs with distance < r, minus
   the N self-pairs at distance 0) / N²` — the fraction of point pairs that are neighbors
   within `r`, a direct space-time analogue of Ripley's K.
4. Return `L(r) = sign(K)·|K|^(1/3)` — a cube-root variance-stabilizing transform. This
   is **not** the classic 2D Ripley's `L(r) = √(K(r)/π) − r`; the exact transform doesn't
   matter for this comparison because it's applied identically to every method being
   compared, so *relative* differences stay valid even though the raw numbers aren't the
   textbook L-function.

`r_values` defaults to `np.linspace(0.05, 0.3, 8)` — 8 evenly spaced radii — in both
pipelines.

### 3.3 `l_error(L_perm, L_target)`

Mean squared error between a candidate permutation's `L(r)` curve and the original
pattern's `L(r)` curve, averaged over the 8 radii. This is the single number every search
method tries to minimize.

### 3.4 `set_diversity(perms)`

Given a *set* of `m` permutations, computes the mean pairwise **normalized Hamming
distance**: for every pair of permutations, the fraction of positions where they
disagree, averaged over all `m(m−1)/2` pairs. `0.0` means every permutation in the set is
identical (mode collapse — useless for augmentation); `1.0` means every pair disagrees
everywhere (maximal diversity). This is the metric that catches a search method that
"cheats" by always returning the same low-error answer.

### 3.5 `_generate_perms(strategy, ..., n_perms, evals_per_perm, rng)` — the three methods

For each of `n_perms` permutations to generate, the function starts from a random
permutation and takes `evals_per_perm − 1` local-search steps, each one spending
*exactly one* call to `compute_L_summary` (the dominant `O(N²)` cost) — so **every
method spends an identical number of L-summary evaluations**, which is what makes the
comparison "fair." The three strategies differ only in how a candidate is proposed and
whether it's accepted:

| Strategy | Proposal | Acceptance |
|---|---|---|
| `mh` | swap two random positions | **Metropolis**: always accept an improvement; accept a worse candidate with probability `exp(-(Δerror)/temperature)`. Temperature anneals geometrically from the initial error magnitude down to 1% of it over the run — this replaces a fixed, un-scaled temperature from an earlier version that made MH accept ~90% of worsening moves regardless of problem scale. |
| `grover` ("Grover-inspired") | swap two random positions | **Greedy**: accept only if the candidate's error is strictly lower. Loosely analogous to Grover's amplitude amplification always moving toward the marked (better) state — but implemented as ordinary hill-climbing, no amplitude, no circuit. |
| `qaoa` ("QAOA-inspired") | swap 1 to `~N/4` random position-pairs at once | **Greedy**, same rule as `grover`. The multi-swap proposal loosely mirrors a QAOA mixer perturbing several elements simultaneously — again, classical only. |

### 3.6 `evaluate_method(strategy, ..., seed)`

Runs `_generate_perms` for one method (with a fresh `np.random.default_rng(seed)` — the
*same* seed value is reused across the three methods so all three start from the same
initial random permutation; they diverge only because their proposal/acceptance logic
consumes randomness differently, never because of a different seed) and returns:

- `mean_error`, `std_error` — average and spread of `l_error` across the `n_perms`
  generated permutations (recomputed once more after generation, for reporting).
- `diversity` — `set_diversity` of the same `n_perms` permutations.
- `time` — wall-clock seconds for generation only (the reporting recomputation above is
  excluded from this).

### 3.7 `run_single`, `aggregate`, `print_summary`, `plot_summary`

- `run_single(seed, n_events, n_perms, evals_per_perm)`: builds one point pattern (from
  `simulate_hawkes` or, in the real-data script, from real events), computes its target
  `L(r)`, and runs all three methods against it. Returns `None` (and the caller prints a
  `SKIPPED` line) if the pattern has fewer than 10 events.
- `aggregate(rows)`: averages `mean_error`, `diversity`, and `time` across all seeds that
  share the same target `N`.
- `print_summary(agg)`: the console table (N × method → error, diversity, time).
- `plot_summary(agg, path)`: two-panel PNG — L(r) error vs. N, and diversity vs. N, one
  line per method.

### 3.8 The sweep (`main`)

Loops over every `(N, seed)` combination requested on the command line, calls
`run_single` for each, aggregates, prints, and writes `results.json` + a plot PNG.
Defaults: `seeds = [1..5]`, `n_events = [20, 30, 50]`, `n_perms = 10`,
`evals_per_perm = 200`.

---

## 4. The real-data variant — `run_q_stpp_v15_real.py`

Imports every function above unchanged (`compute_L_summary`, `evaluate_method`,
`aggregate`, `print_summary`, `plot_summary`) — only the **data loader** is new, so the
search algorithms and fairness protocol are identical to §3.

### 4.1 `load_real_events(df, country, year_start, year_end, seed, jitter_deg, max_events)`

Builds a `(times, x, y)` point pattern from `sea_dengue_admin1_month.csv`:

1. Filter to the requested `country` and year window, keeping only rows with
   `dengue_total > 0` — **one event per (admin1 region, month) record**, not one event
   per case. (Turning every individual case into its own point would mean millions of
   points with no per-case location anyway — the source data has no case-level
   coordinates. See §4.2 for exactly what's real vs. proxy.)
2. If more rows are available than `max_events`, randomly subsample down to
   `max_events` (seeded, so reproducible).
3. **Time**: each event's day-of-month is drawn uniformly at random (the source data has
   month resolution, not day resolution) via `_region_xy`'s sibling logic in
   `load_real_events` itself. Days elapsed since the earliest sampled event are then
   linearly rescaled into `[0, 2·jitter_deg]` — the same numeric range as the spatial
   jitter (§4.2) — so time and space contribute comparably to the distance metric inside
   `compute_L_summary`.
4. **Space**: `_region_xy(region, ...)` hashes the region name with SHA-256 into a seed,
   then draws a fixed `(±jitter_deg, ±jitter_deg)` offset from the country's centroid.
   Same region → same offset, every run, forever — so points from one region cluster
   together and points from different regions separate, without ever depending on the
   iteration order or `max_events` subsampling.

### 4.2 What's real and what's a proxy — read before interpreting any number

| Aspect | Status |
|---|---|
| Which (region, month) had cases | **Real** — from OpenDengue |
| Month of each event | **Real** |
| Day-of-month | Randomized (source data has no finer resolution) |
| Region-to-region spatial separation | **Proxy** — deterministic per-region jitter around a country centroid; this dataset has no admin1-level centroid or boundary data |
| Absolute lat/lon of any event | **Proxy** — do not treat as real geography |
| Case count (`dengue_total`) | Used only as a `> 0` filter; **not** used to weight or duplicate events |

Because spatial coordinates are a proxy, **absolute** `L(r)` error values are not
physically meaningful. What *is* meaningful: the **relative** comparison between `mh`,
`grover`, and `qaoa` on the *same* set of points — exactly the same caveat that already
applies to the synthetic pipeline (§3.2), just with a second, dataset-specific reason
layered on top.

### 4.3 CLI

```bash
python3 run_q_stpp_v15_real.py \
  --country CAMBODIA \
  --year_start 2005 --year_end 2006 \     # omit both to auto-use the country's full record
  --seeds 1 2 3 4 5 \
  --max_events 30 60 120 \
  --n_perms 10 --evals_per_perm 200 \
  --out_dir output_dataset/cambodia        # omit to use output_result/q_stpp_v15_real/
```

If `--year_start`/`--year_end` are omitted, the script auto-detects the first and last
year with `dengue_total > 0` for that country, so every country uses its full available
real record by default.

---

## 5. Running the pipeline

```bash
cd quantum-dengue-stpp
pip install -r requirements.txt          # numpy, scipy, matplotlib, pandas

# Synthetic Hawkes benchmark
python3 run_q_stpp_v15_fair.py
# or: ./run.sh          (./run.sh smoke for a fast sanity check)

# Real-data benchmark, one country
python3 run_q_stpp_v15_real.py --country "VIET NAM"
```

Both write a `*_results.json` (raw + aggregated numbers) and a `*_plot.png` (error and
diversity vs. N) to their output directory. Runtime is a few seconds to a couple of
minutes per full sweep on a laptop CPU — no GPU, no quantum backend, nothing to
provision.

---

## 6. Honest scope — what this project is and is not

**Is:**
- A controlled, same-budget comparison of three *classical* permutation-search
  heuristics for SOP data augmentation, reporting both quality (`L(r)` error) and
  diversity, on both synthetic and real event data.

**Is not:**
- A quantum computation of any kind — no circuit, no simulator, no quantum backend
  anywhere in the code that runs.
- A validated dengue forecasting or outbreak-prediction system — nothing here trains or
  evaluates a forecast model; the pipeline only compares how well permutation-search
  methods preserve a summary statistic.
- Applicable to real admin1 geography for spatial distance — see §4.2.

---

## 7. Results — the 8-country real-data run (`output_dataset/`)

`output_dataset/` was generated by running `run_q_stpp_v15_real.py` once per SEA country
present in the dataset, each with its **full available real record** (auto-detected year
range), `seeds = [1..5]`, `max_events = [30, 60, 120]`. Layout:

```
output_dataset/
├── SUMMARY.md                          # full per-country tables — read this for exact numbers
├── cambodia/      real_comparison_results.json, real_comparison_plot.png, run_log.txt
├── indonesia/     ...
├── laos/          ...
├── malaysia/      ...
├── singapore/     ...
├── thailand/      ...
├── timor_leste/   ...
└── vietnam/       ...
```

### 7.1 Data availability per country

| Country | Years used | Real (region, month) events available |
|---|---|---|
| Cambodia | 1998–2010 | 2,713 |
| Indonesia | 2004–2006 | 394 |
| Lao PDR | 1998–2010 | 1,335 |
| Malaysia | 1993–2010 | 1,903 |
| Singapore | 1993–2010 | 216 |
| Thailand | 1993–2022 | 31,685 |
| Timor-Leste | 2005 only | 30 |
| Viet Nam | 1994–2010 | 7,483 |

### 7.2 What each column in `SUMMARY.md` means

- **N (target)**: the `--max_events` cap requested for that row (30, 60, or 120).
- **N (actual)**: how many real events actually went into that run. Equal to N (target)
  whenever enough real data existed; capped at the country's total availability
  otherwise (see Timor-Leste below).
- **Method**: `MH` (Metropolis-Hastings), `Grover-inspired`, `QAOA-inspired` — see §3.5.
- **L(r) err**: mean squared error of the generated permutations' `L(r)` vs. the real
  pattern's `L(r)`, averaged over `n_perms=10` permutations and 5 seeds. Lower is better
  (closer preservation of the original space-time structure). Remember §4.2: the
  *absolute* value depends on proxy coordinates; compare *across methods*, not across
  countries.
- **diversity**: mean pairwise normalized Hamming distance across the 10 generated
  permutations, averaged over 5 seeds. Higher means the method produced more genuinely
  different augmented samples rather than near-copies of each other.

### 7.3 Observed patterns

Across the six countries with enough data to reach all three N tiers (Cambodia,
Indonesia, Lao PDR, Malaysia, Thailand, Viet Nam), three consistent patterns emerge —
each with a concrete, checkable explanation rather than a hand-wave:

**(a) `qaoa` and `grover` reach lower `L(r)` error than `mh`, almost everywhere.**
Both are pure greedy hill-climbers (§3.5: accept only strict improvements), while `mh`
spends part of its fixed 200-evaluation budget accepting worse candidates on purpose (to
avoid getting stuck). Under a *fixed* evaluation budget, greedy search reliably reaches a
lower final error than an annealed sampler — this is the expected, textbook trade-off,
not a surprise.

**(b) Diversity is high (0.96–0.99) for *all three* methods, including the two greedy
ones — a smaller quality/diversity trade-off than the "greedy = collapsed" intuition
would suggest.** The reason is budget, not method: with `evals_per_perm=200` fixed
regardless of `N`, a single swap only perturbs `2/N` of the permutation, so as `N` grows
past a few dozen, 200 steps of greedy hill-climbing cannot fully converge to *the* global
optimum — it converges to *a* local one, and which local optimum it lands on still
depends heavily on the random starting permutation and swap sequence. The result is a
diverse set even under greedy search, at this budget/N ratio. (A much larger
`evals_per_perm` would be expected to shrink this diversity for the greedy methods —
that's a testable follow-up, not run here.)

**(c) Diversity climbs toward its statistical ceiling as N grows: ≈0.967 at N=30,
≈0.983 at N=60, ≈0.992 at N=120 — matching `(N−1)/N` almost exactly** (29/30 = 0.9667,
59/60 = 0.9833, 119/120 = 0.9917). This is the expected fraction of positions at which
**two independent random permutations** of length N disagree — the mathematical ceiling
of the normalized-Hamming-distance metric itself (a fixed position holds the same value
in two independent uniform-random permutations with probability exactly `1/N`). The
close match across every country and every method says the generated permutation sets
are behaving close to *independent random draws* at this budget, for the reason in (b) —
it is a property of the **metric and the budget/N ratio**, not evidence that any method
is "more creative" as N grows.

**(d) `L(r)` error also shrinks as N grows** (e.g. Cambodia MH: 0.000187 → 0.000039 →
0.000005 across N=30→60→120). Part of this is the search converging further per (a)/(b);
part of it is that `K(r) = pairs/N²` is itself a lower-variance estimator at larger N
(more pairs average out sampling noise), so even a fairly random permutation's `L(r)`
tends to sit closer to the target's as N grows. This report does not decompose how much
of the drop is "better search" vs. "the metric got smoother" — flagging that as an open
question rather than claiming search quality improves with N.

### 7.4 Country-specific caveats — read before citing any single-country number

- **Singapore is a mathematical degeneracy, not a finding.** This dataset has only
  **one** admin1 region for Singapore, so every event shares the exact same proxy
  spatial coordinate. With zero spatial variation, *every* permutation of timestamps
  produces the exact same point *set* (only the time-to-index correspondence changes,
  and — because every point shares the same `(x,y)` — that correspondence is invisible
  to a summary statistic computed over the unordered point cloud). `L(r)` error is
  therefore exactly `0.000000` for all three methods at all three N: this is a direct
  consequence of Singapore having one region, not a claim that any method "solved" the
  problem.
- **Timor-Leste's N=60 and N=120 rows are not larger samples.** Only 30 real
  (region, month) events exist in the entire available record (2005 is the only year
  with data). The `max_events=60` and `max_events=120` runs therefore fall back to the
  same 30 events as `max_events=30` (§4.1, step 2 only subsamples *down*, never pads
  up) — the three N rows for Timor-Leste in `SUMMARY.md` are consequently identical.

### 7.5 What this run does *not* show

- **Not** a quantum result — every number above came from classical NumPy/pandas code
  (§6).
- **Not** a validated outbreak-forecasting benchmark — this measures permutation-search
  quality against a summary statistic, not forecast accuracy against held-out case
  counts.
- **Not** evidence about real admin1 spatial clustering of dengue — the spatial
  coordinates driving these numbers are a proxy (§4.2); a real geographic analysis would
  need true admin1 boundaries/centroids, which this dataset does not provide.

---

## 8. Notable fixes made during this review (for the record)

- `simulate_hawkes`'s `lam_max` upper bound now scales with `n_active` (events still
  inside the 2-time-unit decay window) instead of the *total* historical event count,
  which previously made the acceptance probability collapse as a run went on.
- Console output now forces UTF-8 (`sys.stdout.reconfigure`) so the box-drawing
  characters in the summary table don't crash on Windows' default `cp1252` console
  codepage.
- A silent failure mode was fixed: when a `(seed, N)` cell produced too few events to
  run (fewer than 10), it used to vanish from the output with no trace. It now prints an
  explicit `SKIPPED` line.
- An unused, never-called `ratio()` helper (and the report claim describing it) was
  removed as dead code.

---

## 9. Future work (not implemented — see `message.txt`)

`message.txt` (Vietnamese) sketches a longer-term research direction: replacing
classical SOP with genuine **quantum generative models** (QGAN / Quantum Born Machine /
VQC generator) trained on real dengue data, and comparing forecast accuracy (RMSE, MAE,
F1) with vs. without quantum augmentation. None of that is implemented in this
repository — the current code is exactly what's described in §3–§7, no more.

## References

- Mateu, J. (2025). *Statistical learning for spatio-temporal point processes*
  (S7-ECSIA, Prague) — `S7-ECSIA-2025-Prague.pdf`.
- Mohler, G. & Mateu, J. (2024). Second-order preserving permutations. *Stat*.
- Ripley, B. D. (1977). Modelling spatial patterns. *JRSS-B*.
- OpenDengue — https://opendengue.org (source of `dengue_dataset/`).

## License

MIT (hackathon project).
