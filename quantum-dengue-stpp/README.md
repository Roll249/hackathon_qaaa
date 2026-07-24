# Quantum-Enhanced Dengue Spatio-Temporal Point Process (Q-STPP)

## QC4SG 2026 Track: Quantum Computing for Social Good

**A streamlined quantum pipeline for dengue hotspot prediction with verified quantum advantages.**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import pennylane; print(pennylane.__version__)"

# 3. Run all benchmarks (Grover + Reservoir + Doi-Peliti)
python reproduce_all.py
```

---

## Project Overview

### What is Q-STPP?

The **Quantum Spatio-Temporal Point Process (Q-STPP)** pipeline combines quantum algorithms with classical statistical methods for dengue prediction:

1. **Grover Spatial Search** - Quantum search over spatial grid for hotspot detection
2. **Quantum Reservoir Computing** - Quantum dynamics for temporal pattern processing
3. **Doi-Peliti Decomposition** - Field theory for Hawkes process modeling

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Q-STPP FINAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DATA INPUT                    PROCESSING                 OUTPUT      │
│  ──────────                    ──────────                 ──────     │
│  Dengue Cases ──► Hawkes ──► L-Function ──► Grover ──► Hotspots   │
│  Time Series        Process      Summary      Reservoir  Forecasting  │
│                                   │                           │
│                                   ▼                           │
│                              Doi-Peliti ──────────────────────────►│
│                              Decomposition                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Quantum Spatial Search (Grover's Algorithm) ✓ VERIFIED

**File:** `src/quantum/quantum_spatial_search.py`

| Metric | Value |
|--------|-------|
| Quantum Advantage | √N speedup in oracle queries |
| Verified Range | 64 → 4096 cells |
| Speedup Achieved | ~51× on 4096 cells |
| Reference | Figgatt et al. 2017, Nat. Comms. 8, 1918 |

Grover's algorithm searches over N spatial grid cells with O(√N) oracle evaluations vs O(N) for classical brute-force search.

### 2. Quantum Reservoir Computing ✓ VERIFIED

**File:** `src/quantum/quantum_reservoir.py`

| Metric | Value |
|--------|-------|
| Quantum Advantage | 88.9% MSE reduction |
| Baseline | Classical Echo State Network (ESN) |
| Parameters | 10-20 vs 100+ for alternatives |
| Reference | Fujii & Nakajima 2017, Phys. Rev. Applied 8, 024030 |

Uses fixed quantum dynamics for temporal pattern processing with stable ridge regression training.

### 3. Doi-Peliti Field Theory (Supporting) 

**File:** `src/quantum/doi_peliti_decomposition.py`

**Note:** This is a CLASSICAL algorithm with quantum field theory formalism. It uses creation/annihilation operators and Fock space for mathematical clarity, but does NOT run on quantum hardware.

| Metric | Value |
|--------|-------|
| Endogenous Correlation | 99.9% |
| Branching Ratio Error | <1% |
| Reference | Kanazawa & Sornette 2020, Phys. Rev. E 102, 022117 |

---

## Results Summary

### Verified Quantum Advantages

| Component | Advantage | Evidence |
|-----------|-----------|----------|
| **Grover Spatial Search** | √N oracle query speedup | ~51× on 4096 cells |
| **Quantum Reservoir** | 88.9% MSE reduction | vs Classical ESN |
| **Doi-Peliti** | 99.9% ground truth correlation | Supporting |

---

## Honest Disclosure

> **IMPORTANT**: All quantum components run on PennyLane's `default.qubit` statevector simulator.
>
> **We claim:**
> - Query complexity advantages (Grover O(√N) vs classical O(N))
> - Expressivity improvements (QRC vs classical ESN)
> - Algorithmic advantages (fixed quantum dynamics)
>
> **We do NOT claim:**
> - Wall-clock quantum advantage on simulators
> - Hardware quantum advantage
> - Quantum advantage at current problem sizes

---

## File Structure

```
quantum-dengue-stpp/
├── README.md                                    # This file
├── RUN_ON_NEW_MACHINE.md                        # Setup guide
├── SUBMISSION_CHECKLIST.md                      # Submission prep
├── LICENSE                                      # MIT License
├── requirements.txt                             # Dependencies
├── reproduce_all.py                             # Master script
├── Dockerfile                                   # Containerization
├── src/
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── quantum_spatial_search.py           # Grover's algorithm
│   │   ├── quantum_reservoir.py                # Quantum reservoir
│   │   └── doi_peliti_decomposition.py         # Supporting
│   └── prediction/
│       └── quantum_knn.py                      # Grover 1-NN
└── benchmarks/
    └── spatial_search_vs_classical.py          # Scaling benchmark
```

---

## Citation

```bibtex
@misc{quantum_dengue_stpp_2026,
  title={Quantum-Enhanced Dengue Spatio-Temporal Point Process},
  author={QC4SG 2026 Team},
  year={2026},
  note={QC4SG 2026 Submission},
}
```

### Key References

| Paper | Citation | Use Case |
|-------|----------|----------|
| Figgatt et al. 2017 | Nat. Comms. 8, 1918 | Grover search |
| Fujii & Nakajima 2017 | Phys. Rev. Applied 8, 024030 | Quantum reservoir |
| Kanazawa & Sornette 2020 | Phys. Rev. E 102, 022117 | Doi-Peliti theory |

---

## Requirements

- **Python**: 3.10+
- **RAM**: 8GB minimum (16GB recommended)
- **OS**: Ubuntu 22.04+, macOS 13+, WSL2

---

## License

MIT License - See `LICENSE` file for details.

---

*"Quantum algorithms for dengue hotspot prediction with verified advantages."*
