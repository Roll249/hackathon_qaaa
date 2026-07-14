# Quantum-Dengue-STPP: Quantum-Enhanced Spatio-Temporal Point Processes for Dengue Zooning

**Hackathon Project — Quantum Computing × Public Health**
**Aligned with Mateu ECSIA 2025 (Prague)**

## Overview

This project implements and benchmarks **quantum-augmented Spatio-Temporal Point Process (STPP)** models for dengue fever surveillance in Southeast Asia. We compare three pipelines that share the same Siamese CNN backbone from Mateu's 2025 framework:

1. **K-function baseline** (classical statistical summary)
2. **Classical CNN Siamese** (the paper's recommended neural approach)
3. **Quantum CNN Siamese** (VQC replaces conv layer 2)

The end goal is **khoanh vùng bệnh** (point-pattern zoning / outbreak classification): given a new pattern of dengue cases, classify it as one of the known generative processes.

---

## Latest Version: v6 (Mateu-Aligned)

`run_q_stpp_v6.py` is the canonical entry point. It implements Mateu's exact framework:

| Module | Mateu Paper | v6 Implementation |
|--------|------------|-------------------|
| Discretization | Slide 14 (d1×d2 grid) | `discretize_to_grid` |
| CNN feature extractor | Slides 17-19, 43 | `CNNFeatureExtractor` |
| Siamese discriminant | Slide 30 | `SiameseDiscriminant` |
| Composite Bernoulli loss | Slide 36 | `composite_bernoulli_loss` |
| SOP permutations | Mohler-Mateu 2024 | `sop_permute_grid` |
| 1-NN classification | Slide 32 | `one_nn_accuracy` |
| K-function baseline | Slide 13 | `ripley_k` |

**Results (synthetic Poisson/LGCP/Cluster, 42 train + 18 test):**

| Method | 1-NN Accuracy | Params |
|--------|---------------|--------|
| K-function dissimilarity (baseline) | **0.8333** | - |
| Classical Siamese CNN | 0.7222 | 10,049 |
| Quantum Siamese CNN (hybrid) | 0.6111 | 1,931 |

Honest finding: on small synthetic data, the K-function summary (classical) is hard to beat. See `Q_STPP_V6_REPORT.md` for the full analysis.

---

## Research Question

> Can quantum circuits extract features from spatial point patterns that improve **outbreak zoning** (1-NN classification) compared to classical CNNs, when using Mateu's Siamese framework?

---

## Architecture (v6, aligned with Mateu 2025)

```
                 ┌────────────────────────────────────────┐
                 │  3 Process Types × 20 realizations     │
                 │  (Poisson, LGCP, Cluster)              │
                 └────────────────┬───────────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Discretize: W → d1×d2 grid    │  Paper slide 14
                  │  (8×8 count matrix per pattern) │
                  └───────────────┬────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
       ┌────────▼─────────┐                ┌────────▼─────────┐
       │ Classical CNN    │                │ Quantum CNN      │
       │ (10,049 params)  │                │ VQC 6q×2L        │
       │ Conv→Pool→FC→σ   │                │ Conv→Pool→VQC→FC │
       └────────┬─────────┘                └────────┬─────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Siamese Discriminant          │  Paper slide 30
                  │  p_θ(x,x') = σ(β_0 + Σ β|G-G'|)│
                  └───────────────┬────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  Composite Bernoulli Loss      │  Paper slide 36
                  │  l(θ) = Σ y log p + (1-y) log(1-p)│
                  └───────────────┬────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │  1-NN Classification Test      │  Paper slide 32
                  │  D(x,x') = 1 - p(x,x')         │
                  └────────────────────────────────┘
```

---

## Project Structure

```
quantum-dengue-stpp/
├── run_q_stpp_v6.py          # CANONICAL: Mateu-aligned v6 (current best)
├── run_q_stpp_v5.py          # v5: R² regression (intensity prediction)
├── run_q_stpp_v4.py          # v4: R² regression (preliminary)
├── main.py                   # Pre-consolidation entry point (legacy imports)
├── diagnose_correct.py       # Diagnostic harness for v5 bugs
├── run.sh                    # Quick-run wrapper
│
├── src/
│   ├── augmentation/         # Quantum augment modules (legacy)
│   │   ├── quantum_augment.py
│   │   ├── quantum_sop.py
│   │   ├── local_pqc.py
│   │   ├── data_reuploading_ansatz.py
│   │   ├── sop.py
│   │   ├── true_quantum.py
│   │   └── synthetic_events.py
│   ├── data/                 # data loading
│   ├── evaluation/           # metrics, spatial stats
│   ├── models/               # losses (ZINB, MSE), cnn_lstm, hawkes, etc
│   ├── optimization/         # quantum_natural_gradient
│   ├── pipeline/             # nisq_pipeline, grover_pipeline
│   └── utils/                # logging, serialization, etc
│
├── tests/                    # pytest tests
├── output_result/
│   ├── data/                 # raw dengue CSV data
│   ├── q_stpp_v6/            # v6 results
│   ├── q_stpp_v5/            # v5 results
│   └── quantum-dengue-presentation-v3.pdf, v4.pdf
│
├── Q_STPP_V6_REPORT.md       # Current Mateu-aligned analysis
├── Q_STPP_V5_REPORT.md       # v5 R² analysis
├── Q_STPP_V4_REPORT.md       # v4 R² analysis
├── PROJECT_ARCHITECTURE.md   # High-level project structure
└── README.md                 # this file
```

---

## Setup

```bash
# Clone
git clone https://github.com/Roll249/hackathon_qaaa.git
cd quantum-dengue-stpp

# Install
pip install -r requirements.txt

# Quick run (v6, ~25 seconds)
python run_q_stpp_v6.py

# v5 (R² benchmark)
python run_q_stpp_v5.py

# v4 (R² preliminary)
python run_q_stpp_v4.py
```

---

## Run Output (v6 sample)

```
╔══════════════════════════════════════════════════════════════════════╗
║  Q-STPP v6: ALIGNED WITH MATEU ECSIA 2025                         ║
║  Siamese CNN + Composite Bernoulli log-likelihood + 1-NN          ║
╚══════════════════════════════════════════════════════════════════════╝

  [1/6] Generating dataset of point patterns...
    X=(60, 8, 8), y=(60,), classes=['poisson', 'lgcp', 'cluster']
  [2/6] SOP augmentation...
  [3/6] Building Siamese models...
    Classical CNN: 10049 params
    Quantum hybrid: 1931 params
  [4/6] Training Siamese discriminants...
    Classical: 30 epochs, final loss=0.3575
    Quantum:   15 epochs, final loss=0.6558
  [5/6] Testing: 1-NN classification...
    Classical 1-NN accuracy: 0.7222
    Quantum 1-NN accuracy:   0.6111
  [6/6] K-function dissimilarity baseline...
    K-function 1-NN accuracy: 0.8333

  WINNER: K-function (acc=0.8333)
```

---

## Data

- **Source**: TYCHO (Treating Infectious Diseases) dataset
- **Coverage**: 8 Southeast Asian countries, admin1-month level
- **Period**: 1993-2022 (~20.7M cases)

In `output_result/data/`:
- `all_events.csv`, `train_events.csv`, `val_events.csv`, `test_events.csv`
- `country_summary.csv`
- `synthetic_events.csv`

For v6, we use **synthetic** Poisson/LGCP/Cluster patterns to benchmark the framework. Real-data application requires converting admin1-month to point events (legacy code).

---

## Key References

1. **Mateu, J. (2025)**. Statistical learning for spatio-temporal point processes: inference and testing. *ECSIA Prague*. Slides 14, 17-19, 30, 32, 36, 40, 43-47.
   - `S7-ECSIA-2025-Prague.pdf` (in repo root)
2. **Mohler & Mateu (2024)**. Second order preserving point process permutations. *Stat*. DOI: 10.1002/sta4.558.
3. **Dong, Mateu & Xie (2025)**. Spatio-temporal-network point processes for modeling crime incidents with landmarks. *Submitted*.
4. **Jalilian & Mateu (2023)**. Assessing similarities between spatial point patterns with a Siamese Neural Network discriminant model. *Advances in Data Analysis and Classification*, 17, 21-42.

---

## Version History

- **v6** (2026-07-15): Mateu ECSIA 2025 alignment. Siamese CNN + Bernoulli composite loss + 1-NN classification + K-function baseline. Quantum CNN hybrid. Honest: K-function wins on small synthetic data.
- **v5** (2026-07-15): R² regression benchmark with code-review fixes. Post-warm-start bias corrections. Quantum hybrid.
- **v4** (2026-07-14): R² regression preliminary. Initial QuantumIntensityGeneratorV4.
- **Pre-v4**: Multiple exploratory scripts (now removed).

See `Q_STPP_V5_REPORT.md` and `Q_STPP_V4_REPORT.md` for previous results.

---

## Limitations & Honest Findings

1. **K-function baseline wins on synthetic data** (Mateu's slide 47 confirms this for small N)
2. **Quantum CNN has 5× fewer params** than classical CNN (1,931 vs 10,049) — capacity gap
3. **Synthetic data is too small** (60 patterns) to test quantum advantage in feature extraction
4. **Real-data application requires point-event extraction** from admin1-month aggregates

For quantum advantage to manifest in practice:
- Larger training sets (1000+ per class)
- Real-world hierarchical spatial structure (street networks, hierarchical admin regions)
- Quantum kernel methods for K-function computation (paper slide 19)

---

## License

Hackathon project. MIT License.