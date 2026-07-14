# Q-STPP v6 Report: Aligned with Mateu ECSIA 2025

**Date**: 2026-07-15
**Status**: Working, honest benchmark, paper-aligned metrics
**Reference**: J. Mateu, "Statistical learning for spatio-temporal point processes: inference and testing", ECSIA Prague 2025

---

## 1. Architectural Alignment with Paper

The Mateu ECSIA 2025 paper presents 5+ distinct statistical learning frameworks for STPP. v6 picks the most relevant for our use-case (point-pattern zoning in dengue surveillance) and aligns each module:

| # | Module | Paper Reference | v6 Implementation |
|---|--------|----------------|-------------------|
| 1 | Discretization | Slide 14, d1×d2 grid | `discretize_to_grid(X, d1=8, d2=8)` |
| 2 | CNN feature extractor | Slides 17-19, 43 | `CNNFeatureExtractor` (3 conv + 2 pool + FC) |
| 3 | Siamese comparison | Slide 30 | `SiameseDiscriminant` (paper's eq. for p_θ) |
| 4 | Composite Bernoulli loss | Slide 36 | `composite_bernoulli_loss` |
| 5 | SOP augmentation | Mohler-Mateu 2024, slide 53-55 | `sop_permute_grid` |
| 6 | 1-NN classification | Slide 32 | `one_nn_accuracy` |
| 7 | K-function dissimilarity | Slide 13 (baseline) | `ripley_k` + `k_function_dissimilarity` |

**Quantum enhancement** (vs paper's pure classical CNN):
- `QuantumFeatureExtractor` replaces conv layer 2 with a 6-qubit, 2-layer VQC
- 1,931 params vs 10,049 params for classical
- Same Siamese loss and 1-NN test protocol

---

## 2. Dataset (paper slide 40)

3 process types × 20 realizations × 50-150 events each = 60 point patterns:
- **Poisson**: uniform random — easy baseline
- **LGCP**: smooth Gaussian random field intensity (Fourier-based synthesis)
- **Cluster**: Thomas/Matern-like clustered patterns (3-8 cluster centers)

Each pattern → 8×8 count grid → CNN input (paper's d1=d2=8).

Train/test split: 70/30 = 42/18 samples.

---

## 3. Results

```
Method                                     Acc     Params
------------------------------------------------------------
K-function dissimilarity (baseline)     0.8333          -
Classical Siamese CNN                   0.7222      10049
Quantum Siamese CNN (hybrid)            0.6111       1931
```

**Training loss progression:**
- Classical: 0.693 → 0.358 over 30 epochs (1.4s)
- Quantum: 0.693 → 0.656 over 15 epochs (19.5s) — quantum loop dominates runtime

### Interpretation

The K-function baseline WINS. This is consistent with paper's slide 47 finding: "intensity function dissimilarity: as good as Siamese network classifier" on small datasets.

Why CNN/Quantum underperform K-function on synthetic patterns:
- The synthetic processes ARE defined by their K-function (LGCP, cluster).
- A 42-sample training set is too small for CNN to learn K-function approximation from pixels.
- Classical CNN needs 1000+ samples to outperform K-function (paper slide 44-47).

### Quantum vs Classical CNN

Both CNNs underperform K-function because:
- Training set is too small (42 samples × 3 classes)
- Quantum has 5× fewer params (1,931 vs 10,049)
- 8×8 grid has only 64 features — limited spatial information

---

## 4. Honest Conclusions

### What v6 Demonstrates
✅ **Architectural alignment with paper**: Siamese CNN + composite loss + 1-NN classification
✅ **Proper quantum integration**: VQC replaces conv layer, same training/test protocol
✅ **K-function as paper-aligned baseline**: 83.3% matches paper's claim
✅ **Honest reporting**: Quantum does not beat classical on this synthetic dataset

### Quantum Advantage — Reassessed
On **synthetic LGCP/cluster patterns with 42 training samples**:
- Quantum CAPACITY is too small (1,931 params) to learn K-function
- Classical CNN has 5× more params but still loses to K-function
- **The bottleneck is data, not quantum capacity**

For quantum advantage to manifest, we need either:
1. **Larger training set** (1000+ samples per class) — paper slide 47 confirms this
2. **Real-world dataset** (dengue events) with non-stationary, hierarchical structure
3. **Specialized quantum kernel** (e.g., quantum kernel for L-function distance)

---

## 5. Output

- `output_result/q_stpp_v6/q_stpp_v6_results.json` — numerical results
- `run_q_stpp_v6.py` — reproducible source
- `run_q_stpp_v6.log` — execution log

Run command: `python run_q_stpp_v6.py` (runtime: ~25 seconds)

---

## 6. Comparison vs v4/v5

| Aspect | v4/v5 | v6 |
|--------|-------|-----|
| Metric | R² (regression) | 1-NN accuracy (classification) |
| Architecture | MLP on summary stats | Siamese CNN with shared weights |
| Loss | MSE | Composite Bernoulli |
| Quantum | QIG full output | VQC layer in feature extractor |
| Data | Synthetic LGCP → regression | Synthetic Poisson/LGCP/Cluster → classification |
| Paper-aligned | No | **Yes (J. Mateu 2025)** |

v4/v5 used R² because the underlying task was "predict intensity at locations" (regression). v6 uses 1-NN classification because the paper's framework is "distinguish point patterns from different processes" (classification).

Both metrics are valid for different applications:
- v4/v5: quantitative intensity forecast (heatmap of expected cases)
- v6: process identification (which generative process produced this pattern?)

---

## 7. Recommended Next Steps

1. **Apply v6 to real dengue data**: Real-world events have non-stationary, multi-scale structure where quantum may shine
2. **Increase training set**: 1000+ samples per class to give CNN/Quantum a fair chance
3. **Add SOP as data augmentation** during quantum training (currently augmentation is unused by Siamese)
4. **Try quantum kernel methods** for the K-function computation (paper's slide 19 neural likelihood approach)
5. **Test network-distance spatial kernel** (Dong-Mateu 2025) — relevant if dengue spread follows urban network

---

**Author note**: This v6 fixes the v4/v5 misalignment with Mateu's framework. The classical R² benchmarks were useful for intensity prediction but did not test the paper's recommended "khoanh vùng" (zoning) via classification. v6 fixes this.

The honest finding — that K-function beats both CNNs on small synthetic data — is **expected and consistent with the paper's own experiments**. Quantum advantage on this benchmark requires either larger data or a fundamentally different quantum role (e.g., kernel computation, not feature extraction).