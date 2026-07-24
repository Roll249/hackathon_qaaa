# Dengue Hotspot Forecast Pipeline

## Overview

This pipeline generates `hotspots.json` — the real-time dengue risk map data consumed by the Next.js demo at `../q_dengue_epidemiology-main/web/`.

## Data Source

**OpenDengue V1.1** (Imperial College London)
- URL: https://opendengue.org/ | GitHub: https://github.com/OpenDengue/master-repo
- License: Creative Commons Attribution 4.0
- Vietnam subset: 12,273 monthly records, 64 provinces, 1994–2010
- Resolution: Admin1 (province) × Month
- Source: TYCHO/Imperial College London

**Citation:**
> Clarke J, Lim A, Gupte P, Pigott DM, van Panhuis WG, Brady OJ. *A global dataset of publicly available dengue case count data.* Sci Data. 2024;11(1):296.

### Data Quality Notes

- Data ends at December 2010 (last available release in the repo)
- Monthly resolution converted to weekly using 4-way equal distribution per month
- Some provinces have incomplete records for earlier years (pre-2000 gaps)
- 0-values may represent either true zero cases OR missing reports — not distinguishable in aggregated data

## Pipeline Architecture

```
OpenDengue CSV (monthly, VN Admin1)
        │
        ▼  monthly_to_weekly()
Continuous weekly time series (epi weeks 1-52)
        │
        ▼  fit_sarima() per province
SARIMA(1,1,1)(1,1,1,52) — 12-week-ahead forecast
        │
        ▼  compute_risk_score()
Risk score (z-score vs same-season historical baseline)
        │
        ▼  build_hotspots()
hotspots.json (matching web app schema)
```

### Models

| Stage | Model | Details |
|-------|-------|---------|
| Weekly conversion | Deterministic | Each month → 4 consecutive epi weeks, equal distribution |
| Forecasting | SARIMA(1,1,1)(1,1,1,52) | Seasonal ARIMA with 52-week period |
| Risk scoring | Z-score | Avg z-score of last 12 weeks vs same-epi-weeks in prior years |
| Trend | Ratio comparison | Last 4 weeks vs previous 4 weeks |

### Risk Score Interpretation

- `riskScore ∈ [0, 1]`
- Computed as `(z + 2) / 4` where `z` is the average z-score of the last 12 weeks compared to the same epidemic weeks across all prior years
- ≥ 0.70 = **high** (top hotspot)
- 0.50–0.69 = **medium**
- < 0.50 = **low**

## How to Run

### One-command regeneration

```bash
cd quantum-dengue-stpp
python scripts/generate_hotspots.py
```

### With options

```bash
# Change number of provinces
python scripts/generate_hotspots.py --n-provinces 30

# Change forecast horizon
python scripts/generate_hotspots.py --forecast-weeks 8

# Set random seed for reproducibility
python scripts/generate_hotspots.py --seed 123
```

### Output

```
../q_dengue_epidemiology-main/web/public/data/hotspots.json
```

### Web app

```bash
cd ../q_dengue_epidemiology-main/web
npm run dev
# → http://localhost:3000
```

## Output Schema

```json
{
  "metadata": {
    "generated_at": "2026-07-24T09:45:38Z",
    "data_source": "OpenDengue V1.1 — Vietnam Admin1 Monthly (TYCHO/Imperial College London)",
    "time_range": "12 weeks observed + 12 weeks forecast using SARIMA(1,1,1)(1,1,1,52)",
    "last_week": 49
  },
  "provinces": [{
    "id": "HCM",
    "name": "TP. Hồ Chí Minh",
    "code": "79",
    "coordinates": { "lat": 10.8231, "lng": 106.6297 },
    "riskScore": 0.44,
    "cases": 2072,
    "trend": "down",
    "weeklyData": [
      { "week": 38, "cases": 234, "risk": 0.45 },
      ...12 historical weeks...,
      { "week": 50, "cases": 312, "risk": 0.52, "predicted": 1 },
      ...12 forecast weeks...
    ]
  }],
  "hotspots": [
    { "provinceId": "CMA", "provinceName": "Cà Mau", "coordinates": [9.1875, 104.9874],
      "riskScore": 1.0, "intensity": "high" },
    ...
  ]
}
```

## Limitations

1. **Data recency**: OpenDengue V1.1 ends December 2010. The pipeline generates
   forecasts relative to that reference point. For a real deployment, update to the
   latest OpenDengue release (V1.3 as of 2025).
2. **Weekly approximation**: Monthly → weekly conversion is 4-way equal split, not
   actual daily data. Real daily/weekly MOH data would improve accuracy.
3. **Province matching**: Province IDs are mapped from OpenDengue names to web app
   IDs. Some provinces may use fallback IDs (e.g., `TGI` for Tiền Giang) if not
   in the metadata table.
4. **Synthetic forecasts**: SARIMA forecasts from 2010 data cannot predict actual
   2026 outbreak levels. This is a demonstration of the pipeline architecture.
5. **Missing recent provinces**: Some current provinces in the web app (e.g., Hải
   Phòng) have no OpenDengue Admin1 data and are not included.

## Files

| File | Purpose |
|------|---------|
| `scripts/generate_hotspots.py` | Main pipeline script |
| `dengue_dataset/sea_dengue_admin1_month.csv` | Source data (OpenDengue Vietnam) |
| `dengue_dataset/README.md` | Data provenance and processing notes |
| `../q_dengue_epidemiology-main/web/public/data/hotspots.json` | Generated output |
