# Q-STPP v15 Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Q-STPP v15 System                              │
└─────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────┐
                           │   INPUT DATA     │
                           │  • Dengue cases  │
                           │  • Weather       │
                           │  • Population    │
                           └────────┬─────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                             │
│                                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │ Spatial │    │Temporal │    │Feature  │    │Kernel  │             │
│   │ Extract │───▶│Extract  │───▶│Concat   │───▶│Compute │             │
│   └─────────┘    └─────────┘    └─────────┘    └────┬────┘             │
│                                                     │                   │
│                                                     ▼                   │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                  OPTIMIZATION ENGINE                    │          │
│   │                                                          │          │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐        │          │
│   │   │  Hybrid    │  │   QAOA     │  │    MH      │        │          │
│   │   │  QI-SOP    │  │  (baseline)│  │ (classical)│        │          │
│   │   │   ★ BEST   │  │            │  │            │        │          │
│   │   └─────┬──────┘  └────────────┘  └────────────┘        │          │
│   │         │                                               │          │
│   │         ▼                                               │          │
│   │   ┌────────────┐                                        │          │
│   │   │ SOS        │                                        │          │
│   │   │ Verify     │                                        │          │
│   │   └────────────┘                                        │          │
│   └─────────────────────────────────────────────────────────┘          │
│                                                     │                   │
└─────────────────────────────────────────────────────┼───────────────────┘
                                                      │
                                                      ▼
                           ┌──────────────────┐
                           │   OUTPUT         │
                           │  • Hotspot maps  │
                           │  • Risk scores   │
                           │  • R² metrics    │
                           └──────────────────┘
```

## Component Specifications

### 1. Data Layer

```
DataIngestion
├── SpatialData
│   ├── location_extractor(cases) → [(lat, lon, t), ...]
│   └── grid_projector(coords, resolution) → grid_matrix
│
├── TemporalData
│   ├── time_slicer(events, window) → [(start, end), ...]
│   └── seasonality_detector(timestamps) → period_components
│
└── FeatureData
    ├── weather_merger(cases, weather) → enriched_cases
    └── population_weighter(cases, density_map) → weighted_intensity
```

### 2. Kernel Layer

```
KernelComputation
│
├── SpatialKernel
│   ├── gaussian_kernel(x, x', σ)
│   │   └── exp(-||x - x'||² / 2σ²)
│   │
│   ├── laplacian_kernel(x, x', b)
│   │   └── exp(-||x - x'|| / b)
│   │
│   └── polynomial_kernel(x, x', d)
│       └── (x · x' + c)^d
│
├── TemporalKernel
│   ├── exponential_kernel(t, t', τ)
│   │   └── exp(-|t - t'| / τ)
│   │
│   └── periodic_kernel(t, t', p, l)
│       └── exp(-2sin²(π|t-t'|/p) / l²)
│
└── SpatioTemporalKernel
    └── K((x,t), (x',t')) = K_space(x,x') × K_time(t,t')
```

### 3. Optimization Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION COMPARISON                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Classical MH                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Propose → Accept/Reject → Sample → Estimate             │    │
│  │ O(N²) per iteration, slow convergence                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  QAOA                                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ |ψ(θ)⟩ = PROD U(C,γᵢ)U(B,βᵢ)|+⟩                        │    │
│  │ Classical simulation of quantum circuit                  │    │
│  │ Better exploration, but still exponential overhead        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Hybrid QI-SOP (BEST) ★                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Construct SOS matrix M = Σᵢ λᵢvvᵢᵀ                  │    │
│  │ 2. Compute amplitudes via Born rule: |ψ⟩ = M|0⟩        │    │
│  │ 3. SOS verification: is PSD(M)?                         │    │
│  │ 4. If not SOS, refine with gradient descent              │    │
│  │ 5. Project to feasible region                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Advantages:                                                     │
│  ✓ Polynomial-time verification                                 │
│  ✓ Optimality certificates (unlike heuristic methods)           │
│  ✓ Native path to quantum hardware                             │
│  ✓ Scales better than classical SDP                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. SOS Verification Module

```
SOSVerifier
│
├── check_psd(matrix M)
│   └── Returns: (is_psd, eigenvalues, min_eigenvalue)
│
├── construct_sos_certificate(vector g)
│   └── Returns: matrix P such that gᵀPg = Σᵢ (PᵢggᵀPᵢ)
│
├── optimize_sos_objective(matrix Q, constraints)
│   ├── Input: minimize ⟨Q, X⟩ subject to constraints
│   └── Output: optimal X, certificate, optimality_gap
│
└── refine_if_not_sos(matrix M, tolerance=1e-6)
    └── If λ_min(M) > -tolerance: return M
        Else: gradient_descent_until_sos(M)
```

### 5. Output Layer

```
PredictionOutput
│
├── HotspotMap
│   ├── intensity_heatmap(predictions) → heatmap_image
│   ├── risk_threshold(intensities) → risk_zones
│   └── export_geotiff(predictions) → geotiff_file
│
├── MetricsReport
│   ├── compute_r2_score(pred, actual) → r2_value
│   ├── compute_likelihood_ratio(pred, actual) → lr_statistic
│   └── generate_summary_table(results) → markdown_table
│
└── ComparisonAnalysis
    ├── compare_methods(results_dict) → comparison_plot
    └── statistical_significance_test(metrics) → p_values
```

## Data Flow

```
Raw Dengue Data
      │
      ▼
┌─────────────────┐
│  DataLoader     │  Parse CSV/GeoJSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │  Clean, normalize, handle missing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FeatureMatrix  │  Extract spatial-temporal features
│  Φ = [φ₁, φ₂..] │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  KernelMatrix   │  Compute K(x, x') for all pairs
│  K = ΦΦᵀ       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SOS Reformulate│  Reformulate as SOS problem
│  min ε s.t.     │
│  L(r) - ε ≥ 0   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimize       │  Hybrid QI-SOP optimization
│  min L(r)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Verify         │  SOS certificate check
│  is_feasible    │
└────────┬────────┘
         │
         ▼
    Prediction
```

## File Structure

```
quantum-dengue-stpp/
│
├── run_q_stpp_v15_fair.py       # Main execution script
│
├── output_result/
│   └── q_stpp_v15_qaoa_sop_fixed/
│       ├── fair_comparison_results.json  # Raw experimental data
│       └── quantum_advantage_regimes.png  # Results visualization
│
├── Q_STPP_V15_REPORT.md         # Full technical report
├── ARCHITECTURE.md              # This file
├── THEORY.md                    # Theoretical foundations
├── DEVELOPMENT_HISTORY.md       # Version history
└── README.md                   # Quick start guide
```

## Dependencies

```
numpy>=1.21.0           # Numerical computation
scipy>=1.7.0            # Scientific computing
matplotlib>=3.5.0      # Visualization
networkx>=3.0          # Graph operations
cvxpy>=1.3.0           # Convex optimization
picos>=2.0.0           # SDP solver (optional)
```

## Performance Characteristics

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Kernel Computation | O(N²) | O(N²) |
| SOS Optimization | O(N³) | O(N²) |
| Verification | O(N³) eigendecomp | O(N²) |
| Total per N | O(N³) | O(N²) |

Where N = number of spatial-temporal events.
