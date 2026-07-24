# Running Quantum-Dengue-STPP on a New Machine

This guide provides step-by-step instructions for setting up the quantum-dengue-stpp project on a fresh machine.

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|------------|--------|-------------|
| Python | 3.10 | 3.11, 3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| GPU | Optional | NVIDIA with CUDA 11+ |

### Supported Operating Systems

- **Ubuntu**: 22.04 LTS or later
- **macOS**: 13 (Ventura) or later
- **Windows**: WSL2 with Ubuntu 22.04
- **Docker**: Containerized setup (see Dockerfile)

---

## Step-by-Step Installation

### Step 1: Check Python Version

```bash
python3 --version
# Should output: Python 3.10.x or higher
```

If Python is not installed or version is < 3.10:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**macOS:**
```bash
brew install python@3.11
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd quantum-dengue-stpp

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (WSL2):
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

**Option A: Standard Installation (CPU)**
```bash
pip install -r requirements.txt
```

**Option B: With GPU Support (NVIDIA)**
```bash
# First install CUDA if not already installed
# Then install PennyLane-Lightning GPU
pip install pennylane-lightning[gpu] --extra-index-url https://pennylaneai.github.io/Lightning-Wheels/pypi
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Test PennyLane installation
python -c "import pennylane; print(f'PennyLane version: {pennylane.__version__}')"

# Test NumPy
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test scikit-learn
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

Expected output:
```
PennyLane version: 0.x.x
NumPy version: 1.x.x
scikit-learn version: 1.x.x
```

---

## Quick Test Run

### Test 1: Import All Modules

```bash
python -c "
from src.quantum import (
    qaoa_solve_strict,
    quantum_knn_classify,
    run_sop_quantum,
    DoiPelitiDecomposer,
    StochasticDeclusterer,
)
print('All modules imported successfully!')
"
```

### Test 2: Run Main Pipeline

```bash
python scripts/run_q_stpp_v17.py --seeds 1 --n-events 20 --verbose
```

Expected output:
```
================================================================================
  Q-STPP v17 — QAOA-SOP-Augmented Pipeline
  seeds=[1]  n_events=[20]  M=10  k=4
================================================================================
[N=20 seed=1] data: {'source': 'synthetic'}
  N=20 seed=1: method=classical_greedy  ...
```

### Test 3: Run Quick Benchmark

```bash
python benchmarks/v18_quick_benchmark.py
```

---

## Expected Runtime

| Benchmark | Time | Notes |
|-----------|------|-------|
| `run_q_stpp_v17.py` (1 seed) | ~30s | N=20, M=10 |
| `v18_quick_benchmark.py` | ~2-5 min | All quick tests |
| `spatial_search_vs_classical.py` | ~5-10 min | Grid search |
| `v18_dp_vs_v17.py` | ~5-10 min | Pipeline comparison |
| `reproduce_all.py` | ~15-30 min | Full reproduction |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pennylane'`

**Solution:**
```bash
pip install pennylane
```

### Issue: `ImportError: cannot import name 'qml'`

**Solution:**
```bash
# Ensure PennyLane is installed with correct version
pip install pennylane>=0.38.0
```

### Issue: `MemoryError` on large N

**Solution:**
- Reduce N_events: `python run_q_stpp_v17.py --n-events 15`
- Close other applications
- Add more RAM

### Issue: PennyLane backend errors

**Solution:**
```bash
# Check available devices
python -c "import pennylane as qml; print(qml.device_info(qml.device('default.qubit')))"
```

### Issue: GPU not detected

**Solution:**
```bash
# Check CUDA installation
nvcc --version

# Verify PennyLane-Lightning GPU installation
pip show pennylane-lightning

# Test with CPU if GPU issues persist
export PENNYLANE_DEVICE="default.qubit"
```

---

## Docker Setup (Alternative)

### Build Docker Image

```bash
cd quantum-dengue-stpp
docker build -t quantum-dengue-stpp .
```

### Run in Docker

```bash
# Run quick test
docker run --rm quantum-dengue-stpp python -c "import pennylane; print(pennylane.__version__)"

# Run full reproduction
docker run --rm -v $(pwd)/output_result:/app/output_result quantum-dengue-stpp python reproduce_all.py
```

---

## GPU Setup (Optional)

### NVIDIA GPU Setup

1. **Install CUDA Toolkit 11.8+**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

2. **Install cuQuantum** (for larger simulations)
   ```bash
   pip install cuquantum
   ```

3. **Verify GPU access**
   ```bash
   python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=4); print('GPU ready!')"
   ```

### Apple Silicon (M1/M2/M3)

```bash
# Install via pip (works natively on Apple Silicon)
pip install pennylane
pip install pennylane-lightning
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PENNYLANE_DEVICE` | `default.qubit` | PennyLane backend device |
| `MPLCONFIGDIR` | (temp) | Matplotlib config directory |
| `OMP_NUM_THREADS` | (auto) | OpenMP threads |

---

## Next Steps

1. **Run `reproduce_all.py`** to verify all benchmarks:
   ```bash
   python reproduce_all.py
   ```

2. **Check output results** in `output_result/q_stpp_v18/`

3. **Review generated reports**:
   - `QUANTUM_OPTIMIZATION_REPORT.md`
   - `DOI_PELITI_SCIENTIFIC_NOTES.md`
   - `FINAL_SUBMISSION_REPORT.md`

---

## Need Help?

- **GitHub Issues**: https://github.com/your-org/quantum-dengue-stpp/issues
- **Documentation**: See README.md
- **QC4SG 2026**: https://qc4sg.org

---

*Last updated: July 2026*
