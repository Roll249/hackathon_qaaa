# Quantum Dengue STPP - Project Architecture & Pipeline

## Research Question

> **Can quantum generative models produce more diverse and realistic synthetic dengue fever data that better preserves spatio-temporal structure compared to classical augmentation methods, thereby improving outbreak prediction accuracy?**

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM DENGUE STPP SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   RAW DATA   │───▶│ DATA LAYER   │───▶│ AUGMENTATION │                  │
│  │   8 countries│    │ • loader.py  │    │   LAYER      │                  │
│  │ 1993-2022    │    │ • coordinates│    │ • SOP        │                  │
│  │ 29 regions   │    │ • climate    │    │ • Quantum    │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                   │                          │
│  ┌──────────────┐    ┌──────────────┐             │                          │
│  │    API       │◀───│  MODEL       │◀────────────┘                          │
│  │   LAYER      │    │   LAYER      │                                     │
│  │ • FastAPI    │    │ • CNN-LSTM   │    ┌──────────────┐                  │
│  │ • /predict   │    │ • NEST       │───▶│ EVALUATION   │                  │
│  │ • /batch      │    │ • Hawkes     │    │   LAYER      │                  │
│  └──────────────┘    └──────────────┘    │ • metrics.py │                  │
│                                          │ • spatial    │                  │
│  ┌──────────────┐    ┌──────────────┐    │   stats      │                  │
│  │ QUAPP        │    │ UTILS        │    └──────────────┘                  │
│  │ INTEGRATION  │    │ LAYER        │                                       │
│  │ • qiskit     │    │ • logging    │                                       │
│  │ • pennylane  │    │ • serialize  │                                       │
│  │ • simulator  │    │ • security   │                                       │
│  └──────────────┘    └──────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: DATA INGESTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Dengue Surveillance Data (8 countries, 1993-2022)                      │
│   ├── Thailand, Vietnam, Philippines, Indonesia                            │
│   ├── Malaysia, Singapore, Cambodia, Myanmar                                │
│   ├── Admin1-level monthly aggregated case counts                           │
│   └── Climate data (temperature, humidity, rainfall)                        │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ data/loader.py                                                   │      │
│   │ ├── load_raw_data() → spatial_df, long_df, pivot_df             │      │
│   │ ├── build_stpp_events() → (lat, lon, time, cases) tuples        │      │
│   │ └── temporal_split(0.7/0.15/0.15) → train/val/test              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                      │                                      │
│                                      ▼                                      │
│   STPP Events Dataset: ~50,000+ events                                      │
│   Format: (latitude, longitude, timestamp, case_count, country)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: SPATIAL GRIDDING (Preprocessing)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ create_adaptive_spatial_grid()                                   │      │
│   │                                                                  │      │
│   │ 1. Global [0,1]² normalization for quantum AngleEmbedding        │      │
│   │ 2. Country-adaptive grids for balanced representation             │      │
│   │ 3. 32×32×T tensor (H × W × Time)                                │      │
│   │                                                                  │      │
│   │ Output: 3D tensor ready for CNN-LSTM input                       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ validate_no_data_leakage()                                      │      │
│   │                                                                  │      │
│   │ CRITICAL: Prevents temporal contamination                        │      │
│   │ • Train: 1993-2008    Val: 2008-2015    Test: 2015-2022         │      │
│   │ • Strict temporal ordering enforced                              │      │
│   │ • Country coverage consistency check                             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: DATA AUGMENTATION LAYER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│   │ NO AUGMENT  │    │ SOP AUGMENT │    │ QUANTUM     │                      │
│   │ (Baseline)  │    │ (Classical) │    │ AUGMENT     │                      │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                      │
│          │                   │                   │                           │
│          │                   │                   ▼                           │
│          │                   │    ┌─────────────────────────────────┐        │
│          │                   │    │ augmentation/quantum_augment_v3 │        │
│          │                   │    │                                  │        │
│          │                   │    │ ├── Quantum Born Machine (QBM)   │        │
│          │                   │    │ ├── Variational Quantum Circuit │        │
│          │                   │    │ ├── Hybrid QGAN with CNN        │        │
│          │                   │    │ └── Local PQC (DBSCAN/K-Means) │        │
│          │                   │    └─────────────────────────────────┘        │
│          │                   │                   │                           │
│          │                   │                   ▼                           │
│          │                   │    ┌─────────────────────────────────┐        │
│          │                   │    │ ZINB Loss for Zero-Inflated    │        │
│          │                   │    │ Count Data                      │        │
│          │                   │    │ • models/zinb_loss.py           │        │
│          │                   │    └─────────────────────────────────┘        │
│          │                   │                                           │
│          └───────────────────┴───────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│   Augmented Events: +10-50% synthetic events preserving spatio-temporal      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 4: MODEL LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ CNN-LSTM (SpatioTemporalCNN)                                    │      │
│   │ ├── Conv2D layers for spatial feature extraction                │      │
│   │ ├── LSTM for temporal dynamics                                  │      │
│   │ ├── Spatial attention mechanism                                 │      │
│   │ └── Softplus output (non-negative predictions)                  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ NEST (Neural Spatio-Temporal)                                   │      │
│   │ ├── Encoder-Decoder architecture                                │      │
│   │ ├── Intensity function modeling                                  │      │
│   │ └── Softplus activation for count data                          │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ Hawkes Process                                                   │      │
│   │ ├── Multi-dimensional conditional intensity                     │      │
│   │ ├── Self-exciting temporal patterns                              │      │
│   │ └── Spatial kernel (Gaussian/RBF)                               │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: EVALUATION & METRICS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ Forecasting Metrics                                              │      │
│   │ ├── RMSE (Root Mean Square Error)                                │      │
│   │ ├── MAE (Mean Absolute Error)                                    │      │
│   │ ├── MAPE (Mean Absolute Percentage Error)                         │      │
│   │ └── R² Score                                                     │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ Point Process Quality Metrics                                    │      │
│   │ ├── K-function (spatial clustering)                              │      │
│   │ ├── L-function (standardized K)                                  │      │
│   │ ├── g(r) (pair correlation)                                      │      │
│   │ └── Quantum Fisher Information (QFI)                             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 6: DEPLOYMENT LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐    ┌─────────────────────┐                       │
│   │ LOCAL GPU TRAINING  │    │ QUAPP CLOUD        │                       │
│   │ run_gpu_pipeline.py │    │ (Quantum Backend)  │                       │
│   │ • NVIDIA CUDA       │    │ • Qiskit            │                       │
│   │ • Mixed precision   │    │ • PennyLane         │                       │
│   │ • 48×48 grid       │    │ • Simulators        │                       │
│   └─────────────────────┘    │ • Real Hardware     │                       │
│                              └──────────┬──────────┘                       │
│                                         │                                   │
│                                         ▼                                   │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ FastAPI Endpoints (src/api/endpoints.py)                        │      │
│   │ ├── GET  /health         → Health check                          │      │
│   │ ├── POST /predict        → Single prediction                     │      │
│   │ ├── POST /predict/batch  → Batch predictions                    │      │
│   │ └── GET  /metrics        → Model metrics                         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### Layer 1: Data Layer (`src/data/`)

| Module | Function | Purpose |
|--------|----------|---------|
| `loader.py` | `load_raw_data()` | Load raw dengue surveillance data |
| `loader.py` | `build_stpp_events()` | Convert to STPP events (lat, lon, time, cases) |
| `loader.py` | `create_adaptive_spatial_grid()` | Normalize to [0,1]² for quantum |
| `loader.py` | `validate_no_data_leakage()` | **CRITICAL**: Prevent temporal contamination |
| `loader.py` | `temporal_split()` | Time-based train/val/test (70/15/15) |
| `coordinates.py` | Geocoding | Convert admin1 regions to lat/lon centroids |
| `climate.py` | Weather data | Temperature, humidity, rainfall integration |

### Layer 2: Augmentation Layer (`src/augmentation/`)

| Module | Type | Description |
|--------|------|-------------|
| `sop.py` | Classical | Second-Order Preserving permutations |
| `sop_v2.py` | Classical | Enhanced SOP with validation |
| `quantum_augment.py` | Quantum | Basic Quantum Born Machine |
| `quantum_augment_v2.py` | Quantum | Variational Quantum Circuit |
| `quantum_augment_v3.py` | Quantum | Hybrid QGAN with CNN decoder |
| `local_pqc.py` | Quantum | Local PQC with spatial clustering |
| `true_quantum.py` | Quantum | Full quantum pipeline |
| `synthetic_events.py` | Synthetic | Gaussian noise-based augmentation |

### Layer 3: Model Layer (`src/models/`)

| Model | Architecture | Strength |
|-------|--------------|----------|
| `cnn_lstm.py` | CNN + LSTM + Attention | Spatial-temporal patterns |
| `cnn_lstm_v2.py` | Enhanced CNN-LSTM | Better feature extraction |
| `nest.py` | Neural Encoder-Decoder | Intensity function |
| `hawkes.py` | Multi-dimensional Hawkes | Self-exciting dynamics |
| `country_models.py` | Per-country models | Regional specialization |
| `zinb_loss.py` | ZINB Loss | Zero-inflated count data |

### Layer 4: Evaluation Layer (`src/evaluation/`)

| Module | Metrics | Purpose |
|--------|---------|---------|
| `metrics.py` | RMSE, MAE, MAPE, R² | Forecasting accuracy |
| `spatial_stats.py` | K, L, g(r) functions | Spatial pattern quality |
| `spatial_stats_fast.py` | Optimized versions | Faster computation |

### Layer 5: Utilities Layer (`src/utils/`)

| Module | Purpose |
|--------|---------|
| `logging.py` | TrainingLogger, DataLogger, QuantumLogger |
| `serialization.py` | Model save/load with metadata |
| `experiment_tracker.py` | MLflow-style tracking |
| `security.py` | API rate limiting, auth |

### Layer 6: API Layer (`src/api/`)

- `endpoints.py`: FastAPI REST API
  - `GET /health` - Health check
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions

---

## QuApp Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUAPP CLOUD PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│   │  Simulator  │    │   Qiskit     │    │  PennyLane  │                   │
│   │  (Default)  │    │  (IBM, etc)  │    │  (ManyUX)   │                   │
│   └─────────────┘    └─────────────┘    └─────────────┘                   │
│                                                                             │
│   Deployment: quapp deploy --handler quapp/handler.py                      │
│   Jobs:       quapp job run --function dengue-qbm                           │
│                                                                             │
│   quapp/                                                                    │
│   ├── handler.py    → Quantum function entry point                          │
│   ├── quapp_client.py → Client SDK usage                                   │
│   └── requirements.txt → quapp, qiskit, pennylane                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Features (v3)

| Feature | Status | Description |
|---------|--------|-------------|
| **Data Leakage Prevention** | ✅ | `validate_no_data_leakage()` prevents temporal contamination |
| **Adaptive Spatial Gridding** | ✅ | Country-specific [0,1]² normalization for quantum |
| **Softplus Output** | ✅ | Non-negative predictions for count data |
| **Unit Tests** | ✅ | 29 pytest tests, all passing |
| **Logging System** | ✅ | TrainingLogger, DataLogger, QuantumLogger |
| **Model Serialization** | ✅ | torch.save/load with metadata |
| **Experiment Tracking** | ✅ | Lightweight MLflow-style |
| **FastAPI Endpoints** | ✅ | REST API with /health, /predict |
| **CI/CD** | ✅ | GitHub Actions with 7 jobs |
| **Docker** | ✅ | Multi-stage Dockerfile + Compose |
| **Security** | ✅ | Rate limiting, API keys |

---

## Comparison: Classical vs Quantum Augmentation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUGMENTATION COMPARISON                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method          │ Diversity  │ Structure  │ Computational │ Quality     │
│                  │           │ Preservation│ Cost          │             │
│ ─────────────────┼───────────┼─────────────┼───────────────┼─────────────  │
│  No Augment      │    -      │     -       │      $0       │  Baseline    │
│  ────────────────┼───────────┼─────────────┼───────────────┼─────────────  │
│  SOP (Classical) │  Medium   │   Good      │    Low ($)    │  Good        │
│  ────────────────┼───────────┼─────────────┼───────────────┼─────────────  │
│  Quantum (QBM)   │  High     │   Very Good │   Medium ($$) │  Excellent   │
│  ────────────────┼────────────┼─────────────┼───────────────┼─────────────  │
│  Quantum (QGAN)  │  Highest  │   Excellent │   High ($$$)  │  Best        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Results (Expected)

| Metric | No Augment | SOP | Quantum Augment |
|--------|------------|-----|-----------------|
| RMSE   | Baseline   | -5% | **-10-15%**     |
| MAE    | Baseline   | -5% | **-10-15%**     |
| K-fn   | Target     | ~1.0| **~1.0**        |
| L-fn   | Target     | ~0  | **~0**          |

---

## File Structure

```
quantum-dengue-stpp/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── augmentation/   # Classical + Quantum augmentation
│   ├── models/         # CNN-LSTM, NEST, Hawkes
│   ├── evaluation/     # Metrics & spatial stats
│   ├── optimization/   # Hyperparameter tuning
│   ├── utils/          # Logging, serialization, security
│   └── api/            # FastAPI endpoints
├── quapp/              # QuApp cloud integration
├── tests/              # Pytest unit tests (29 tests)
├── outputs/            # Model outputs & plots
├── run_gpu_pipeline.py # Main GPU pipeline (NEWEST)
├── run_full_v2.py      # Full CPU pipeline
├── run_fast.py         # Quick test
└── Dockerfile          # Container deployment
```

---

## Research Hypothesis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HYPOTHESIS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  H₀: Quantum augmentation does NOT improve prediction accuracy            │
│      (No significant difference from classical methods)                     │
│                                                                             │
│  H₁: Quantum augmentation DOES improve prediction accuracy                 │
│      (Significant improvement in RMSE/MAE)                                 │
│                                                                             │
│  Evidence Required:                                                         │
│  ├── Lower RMSE/MAE on test set (p < 0.05)                                  │
│  ├── Better K/L-function preservation                                      │
│  ├── Higher Quantum Fisher Information                                      │
│  └── Generalization to unseen time periods                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
