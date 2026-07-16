# RAPID-DENGUE: Real-time Alert and Prediction for Immediate Defense

## The Vision

**"Cứu người NGAY LẬP TỨC, không phải đợi 5-10 năm"**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAPID-DENGUE vs PHARMTOM LABS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pharmtom Labs (SEA Quantathon 2025 Winner)                      │
│  ────────────────────────────────────────────────               │
│  • Drug discovery via VQE                                        │
│  • Timeline: 5-10 năm cho thuốc mới                           │
│  • Impact: Long-term, không giúp được người hôm nay            │
│                                                                  │
│  RAPID-DENGUE (Our Project)                                      │
│  ────────────────────────────────────                            │
│  • Real-time hotspot prediction                                   │
│  • Timeline: Deploy được TRONG TUẦN NÀY                         │
│  • Impact: IMMEDIATE - cứu người ngay lập tức                 │
│                                                                  │
│  ✅ OUR ADVANTAGE: Speed to Impact                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## What We Do

**Real-time dengue hotspot prediction with actionable alerts**

```
Today's Data → ML Prediction → Alert to CDC → Action (Spray/Quarantine)
                  │
                  └── 24 hours before outbreak
                  └── Accuracy > 70%
                  └── Deploy in 5 minutes
```

---

## How It Works

### 1. Data Collection (Real-time)
```
├── WHO/TYCHO API          → Daily case counts
├── Hospital reports       → Hourly updates
├── Weather services       → Temperature, humidity, rainfall
└── Mobility data         → Google/Apple mobility trends
```

### 2. Feature Extraction (Classical ML)
```
├── K-function             → Spatial clustering detection
├── L-function             → Second-order statistics
├── CNN features           → Pattern recognition
└── GNN attention         → Influence propagation
```

### 3. Prediction Engine
```
├── 1-NN Classification   → Pattern matching
├── Risk Scoring           → Hotspot probability
└── Hotspot Map           → DỰ ĐOÁN locations
```

### 4. Alert System (Actionable)
```
├── Dashboard             → CDC/WHO monitoring
├── Push Notifications     → Local health departments
├── SMS Alerts            → Community awareness
└── Spray Optimization    → Resource allocation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAPID-DENGUE SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA INPUT                  PROCESSING              OUTPUT     │
│  ──────────                  ──────────              ──────     │
│  WHO/TYCHO ──────────────────────────┬──────────▶ Dashboard  │
│  Hospital  ────┬───▶ Feature ──────┤              │          │
│  Weather   ────┘     Extract ──────┤              ├──▶ Alerts │
│  Mobility  ────────────────────────▶│              │          │
│                                     │              ├──▶ Maps   │
│                                     │              │          │
│                                     ▼              │          │
│                               PREDICTION ◀─────────┘          │
│                                     │                          │
│                                     ▼                          │
│                             HOTSPOT MAP                       │
│                             (Actionable)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quantum-Enhanced Components

### What's Classical (Production-Ready)
```
✅ K-function computation
✅ L-function computation
✅ 1-NN classification
✅ Risk scoring
✅ Dashboard
✅ Alert system
```

### What's Quantum-Research (Future)
```
🔬 QAOA for SOP (N > 200)
🔬 Quantum kernels (N ≤ 15)
🔬 VQE optimization
🔬 Grover for nearest neighbor
```

### Our Philosophy
```
Classical-first → Quantum-where-useful → Honest claims
```

---

## Impact Metrics

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Prediction Accuracy | > 70% | Validate against actual outbreaks |
| Alert Speed | < 5 min | Data → Prediction pipeline |
| Coverage | District-level | Grid resolution |
| Lives Saved | Measurable | Before/after intervention |
| Cost Reduction | > 50% | Optimized spray routes |

---

## Comparison with Other Approaches

| Approach | Timeline | Impact | Quantum |
|----------|----------|--------|---------|
| **RAPID-DENGUE** | **This week** | **Immediate** | Classical-first |
| Pharmtom Labs | 5-10 years | Long-term | VQE |
| Traditional CDC | 2-4 weeks | Delayed | None |
| Google Flu Trends | Historical | Retrospective | ML |

---

## Team Tasks

See `TEAM_ASSIGNMENTS.md` for detailed task assignments.

### Priority P1 Tasks (Week 1-2)
```
1. MODULE-3.3: Hotspot Prediction - MAIN OUTPUT
2. MODULE-6.1: Dashboard Visualization
3. MODULE-1.1: Real Data Integration (TYCHO)
4. MODULE-6.3: Alert Generation
```

### Research P2 Tasks (Week 3-4)
```
1. MODULE-2.1: K-function Optimization
2. MODULE-3.1: 1-NN Classification
3. MODULE-4.3: QAOA Benchmark
```

---

## Quick Start

```bash
# Run real-time prediction
python run_rapid_dengue.py --mode predict

# Start dashboard
python run_rapid_dengue.py --mode dashboard

# Run with real TYCHO data
python run_rapid_dengue.py --source tycho --location vietnam
```

---

## Hackathon Story

**Wrong:**
> "Quantum-powered dengue drug discovery using VQE for molecular binding"

**Right:**
> "While Pharmtom Labs waits 5-10 years for quantum drug discovery, 
> we save lives TODAY with real-time dengue hotspot prediction.
> 
> Our quantum-inspired ML predicts outbreaks 24 hours before transmission,
> allowing CDC to spray and quarantine IMMEDIATELY.
> 
> Classical AI deployed this week. Quantum-ready tomorrow."

---

## References

- **SEA Quantathon 2025 Winner**: Pharmtom Labs (Drug Discovery)
- **Our Approach**: Real-time prediction (Immediate impact)
- **Philosophy**: Speed to impact > Long-term research

---

## Contact

**Project**: RAPID-DENGUE
**Goal**: Save lives THIS WEEK
**Strategy**: Classical-first, Quantum-where-useful
**Timeline**: Deploy NOW

---

*"The best time to predict dengue was 10 years ago.*
*The second best time is TODAY."*
