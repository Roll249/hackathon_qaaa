# Exploratory Data Analysis Report: Southeast Asia Dengue Surveillance Dataset

**Analysis Date:** May 30, 2026  
**Dataset Location:** `/home/khang/Work/hackathon/dengue_dataset/`

---

## Executive Summary

This report presents a comprehensive exploratory data analysis of the Southeast Asia Dengue Surveillance Dataset, covering dengue case data across 8 countries from 1955 to 2022. The dataset contains over **20.7 million cumulative dengue cases** across the region, with significant temporal and geographic variation.

**Key Findings:**
- Total of 69,595 spatial records and 55,030 monthly administrative records
- 8 countries with 233 unique administrative regions
- Peak epidemic year: 2019 with 2.18 million cases
- Strong seasonal patterns with peak transmission typically during rainy season (May-October)
- Thailand and Vietnam account for the majority of cases in the monthly dataset

---

## 1. Dataset Overview

### 1.1 Files Analyzed

| File | Records | Columns | Description |
|------|---------|---------|-------------|
| `sea_dengue_spatial.csv` | 69,595 | 14 | Yearly aggregated data with geographic hierarchy |
| `sea_dengue_admin1_month.csv` | 55,030 | 14 | Long format: admin1-level monthly data |
| `sea_dengue_admin1_month_pivot.csv` | 360 | 234 | Pivot table: dates × 233 regions |

### 1.2 Column Structure

**Core columns (all files):**
- `adm_0_name`: Country name
- `adm_1_name`: First-level administrative region
- `adm_2_name`: Second-level administrative region
- `full_name`: Combined geographic identifier
- `ISO_A0`: ISO 3166-1 alpha-3 country code
- `FAO_GAUL_code`: FAO geographic code
- `calendar_start_date`: Start of reporting period
- `calendar_end_date`: End of reporting period
- `Year`: Year of observation
- `dengue_total`: Total dengue cases
- `S_res`: Spatial resolution (Admin0/Admin1/Admin2)
- `T_res`: Temporal resolution (Year/Month)
- `sourceID`: Data source identifier

### 1.3 Temporal Coverage

| Dataset | Start Date | End Date | Duration |
|---------|------------|----------|----------|
| Spatial | 1955-01-01 | 2022-12-25 | 68 years |
| Monthly (long) | 1993-01-01 | 2022-12-01 | 30 years |
| Pivot | 1993-01-01 | 2022-12-01 | 30 years |

---

## 2. Geographic Coverage

### 2.1 Countries Included

The dataset covers 8 countries in Southeast Asia (plus 3 additional countries in the spatial data):

| Country | Spatial Records | Monthly Records | Regions |
|---------|-----------------|----------------|---------|
| BRUNEI DARUSSALAM | 118 | - | - |
| CAMBODIA | 4,295 | 3,680 | 24 |
| INDONESIA | 1,277 | 421 | 30 |
| LAO PEOPLE'S DEMOCRATIC REPUBLIC | 3,362 | 2,731 | 18 |
| MALAYSIA | 2,788 | 1,944 | 10 |
| MYANMAR | 128 | - | - |
| PHILIPPINES | 9,875 | - | - |
| SINGAPORE | 1,307 | 216 | 1 |
| THAILAND | 33,763 | 33,720 | 77 |
| TIMOR-LESTE | 118 | 45 | 9 |
| VIET NAM | 12,564 | 12,273 | 64 |

### 2.2 Administrative Regions

Total unique administrative regions: **233**

**Breakdown by country:**
- **Thailand**: 77 regions (largest coverage)
- **Vietnam**: 64 regions
- **Indonesia**: 30 regions
- **Cambodia**: 24 regions
- **Laos**: 18 regions
- **Malaysia**: 10 regions
- **Timor-Leste**: 9 regions
- **Singapore**: 1 region (city-state)

---

## 3. Summary Statistics

### 3.1 Overall Case Statistics

| Statistic | Value |
|-----------|-------|
| **Grand Total Cases** | **20,770,132** |
| Mean cases per record | 298.4 |
| Median cases per record | 19.0 |
| Standard Deviation | 3,139.2 |
| Minimum | 0 |
| Maximum | 354,517 |
| 25th Percentile | 4.0 |
| 75th Percentile | 74.0 |
| 99th Percentile | 4,905.1 |

### 3.2 Total Cases by Country (All Years)

| Rank | Country | Total Cases | Percentage |
|------|---------|-------------|------------|
| 1 | Philippines | 5,072,797 | 24.4% |
| 2 | Vietnam | 4,542,778 | 21.9% |
| 3 | Indonesia | 3,710,844 | 17.9% |
| 4 | Thailand | 3,212,348 | 15.5% |
| 5 | Malaysia | 2,062,270 | 9.9% |
| 6 | Myanmar | 782,170 | 3.8% |
| 7 | Cambodia | 590,160 | 2.8% |
| 8 | Singapore | 409,713 | 2.0% |
| 9 | Laos | 373,290 | 1.8% |
| 10 | Timor-Leste | 9,972 | 0.05% |
| 11 | Brunei | 3,790 | 0.02% |

---

## 4. Temporal Patterns

### 4.1 Yearly Trends

**Peak Epidemic Years:**
| Year | Total Cases | Notable Countries |
|------|-------------|-------------------|
| 2019 | 2,185,806 | Philippines, Vietnam, Malaysia |
| 2015 | 1,208,971 | Philippines, Malaysia, Vietnam |
| 2013 | 1,206,363 | Philippines, Vietnam, Malaysia |
| 2016 | 1,176,031 | Philippines, Malaysia, Vietnam |
| 2017 | 971,587 | Vietnam, Philippines, Malaysia |

**Notable Historical Outbreaks:**
- **1987**: 789,095 cases (major regional epidemic)
- **1998**: Significant outbreak coinciding with El Niño
- **2010**: 645,760 cases (pre-2013 surge)
- **2019**: Record-breaking 2.18 million cases

### 4.2 Yearly Totals by Country (1990-2022)

```
Year    Cambodia  Indonesia  Laos   Malaysia  Singapore  Thailand  Timor-Leste  Vietnam
1990        0       1,400     0       0         1,045     35,406       0          6,026
1991        0       2,167     0       0         2,228     36,685       0         12,116
1992        0       1,500     0       0         2,482     36,876       0         15,117
1993     1,600      3,267     0      2,167      2,764     81,696       0         28,000
...
2019    136,871   138,127  77,986   260,383     32,006    87,866       0        579,772
2020     11,792   108,303   7,692    87,171     34,959    50,911    1,451           0
```

*Note: 2020 shows reduced cases for some countries, likely due to COVID-19 pandemic effects and reporting disruptions*

---

## 5. Monthly Seasonality Patterns

### 5.1 Average Monthly Cases by Country

| Month | Cambodia | Indonesia | Laos | Malaysia | Singapore | Thailand | Timor-Leste | Vietnam |
|-------|----------|-----------|------|----------|-----------|----------|-------------|---------|
| Jan | 9.9 | 436.2 | 6.1 | 191.2 | 303.3 | 21.7 | 12.6 | 55.0 |
| Feb | 7.0 | 588.7 | 5.4 | 154.9 | 262.2 | 18.8 | 27.7 | 34.0 |
| Mar | 10.9 | 708.2 | 6.8 | 141.7 | 224.4 | 23.7 | 46.1 | 36.3 |
| Apr | 18.8 | 313.4 | 8.9 | 129.1 | 220.5 | 26.8 | 29.0 | 45.5 |
| May | 48.9 | 191.0 | 24.0 | 141.2 | 280.7 | 51.2 | 0.3 | 68.6 |
| Jun | 99.5 | 175.7 | 44.4 | 161.4 | 430.6 | 91.2 | - | 120.6 |
| Jul | 119.7 | 222.0 | 93.6 | 184.9 | 507.7 | 104.5 | - | 173.9 |
| Aug | 94.7 | 378.5 | 86.7 | 171.1 | 534.1 | 92.9 | - | 190.1 |
| Sep | 59.4 | 410.0 | 77.1 | 163.5 | 564.3 | 66.3 | - | 190.6 |
| Oct | 34.7 | 480.8 | 46.2 | 165.6 | 487.4 | 48.3 | - | 190.1 |
| Nov | 18.0 | 656.7 | 23.2 | 150.9 | 347.6 | 39.6 | - | 146.4 |
| Dec | 10.6 | 804.3 | 16.4 | 166.0 | 383.5 | 23.2 | - | 95.2 |

### 5.2 Peak Transmission Months

| Country | Peak Month | Pattern |
|---------|------------|---------|
| Cambodia | **July** | Strong monsoon-related seasonality |
| Indonesia | **December** | Year-round with end-year peak |
| Laos | **July** | Strong monsoon-related seasonality |
| Malaysia | **January** | Unusual pattern (year-round moderate) |
| Singapore | **September** | Moderate seasonality |
| Thailand | **July** | Strong monsoon-related seasonality |
| Timor-Leste | **March** | Limited data |
| Vietnam | **September** | Strong late-year seasonality |

**Seasonal Interpretation:**
- Countries with monsoon climates (Thailand, Cambodia, Laos, Vietnam) show peak transmission during/wet season (May-October)
- Indonesia shows less pronounced seasonality with higher cases toward year-end
- Singapore's city-state transmission is more constant throughout the year
- Malaysia's January peak is atypical and may reflect year-end reporting patterns

---

## 6. Per-Country Detailed Analysis

### 6.1 Country Summaries

#### CAMBODIA
- **Total Cases:** 164,181
- **Period:** 1998-2010 (13 years)
- **Peak Year:** 2007 (38,418 cases)
- **Regions:** 24
- **Mean Monthly Cases:** 44.6
- **Zero-Inflation:** 26.3%
- **Pattern:** Strong July peak, typical of monsoon climate

#### INDONESIA
- **Total Cases:** 187,447
- **Period:** 2004-2006 (2.3 years, limited coverage)
- **Peak Year:** 2004 (69,105 cases)
- **Regions:** 30
- **Mean Monthly Cases:** 445.2 (highest)
- **Zero-Inflation:** 6.4%
- **Pattern:** December peak, more distributed throughout year

#### LAO PEOPLE'S DEMOCRATIC REPUBLIC
- **Total Cases:** 99,934
- **Period:** 1998-2010 (13 years)
- **Peak Year:** 2010 (22,903 cases)
- **Regions:** 18
- **Mean Monthly Cases:** 36.6
- **Zero-Inflation:** 51.1% (highest)
- **Pattern:** Strong July peak, high zero-inflation suggests incomplete reporting

#### MALAYSIA
- **Total Cases:** 311,271
- **Period:** 1993-2010 (18 years)
- **Peak Year:** 2010 (42,413 cases)
- **Regions:** 10
- **Mean Monthly Cases:** 160.1
- **Zero-Inflation:** 2.1% (lowest)
- **Pattern:** January peak, consistent year-round transmission

#### SINGAPORE
- **Total Cases:** 81,831
- **Period:** 1993-2010 (18 years)
- **Peak Year:** 2005 (14,173 cases)
- **Regions:** 1
- **Mean Monthly Cases:** 378.8
- **Zero-Inflation:** 0.0% (complete reporting)
- **Pattern:** September peak, urban transmission pattern

#### THAILAND
- **Total Cases:** 1,708,632
- **Period:** 1993-2022 (30 years, longest coverage)
- **Peak Year:** 1998 (124,062 cases)
- **Regions:** 77 (most comprehensive)
- **Mean Monthly Cases:** 50.7
- **Zero-Inflation:** 6.0%
- **Pattern:** Strong July peak, excellent regional coverage

#### TIMOR-LESTE
- **Total Cases:** 1,041
- **Period:** 2005 (6 months, very limited)
- **Peak Year:** 2005 (1,041 cases)
- **Regions:** 9
- **Mean Monthly Cases:** 23.1
- **Zero-Inflation:** 33.3%
- **Pattern:** Limited data, March peak

#### VIET NAM
- **Total Cases:** 1,379,503
- **Period:** 1994-2010 (17 years)
- **Peak Year:** 1998 (234,920 cases)
- **Regions:** 64
- **Mean Monthly Cases:** 112.4
- **Zero-Inflation:** 39.0%
- **Pattern:** September peak, high variability

---

## 7. Data Quality Assessment

### 7.1 Missing Values

| Column | Spatial Data | Long Format |
|--------|--------------|-------------|
| adm_1_name | 4,173 (6.0%) | 0 |
| adm_2_name | 60,535 (87.0%) | 55,030 (100%) |
| Other columns | <1% | <1% |

**Notes:**
- `adm_2_name` is only populated for Admin2 resolution data
- `adm_1_name` missing in some Admin0 aggregated records
- Core case data (`dengue_total`) has no missing values

### 7.2 Zero-Inflation

| Dataset | Zero Records | Percentage |
|---------|--------------|------------|
| Spatial | 9,340 | 13.42% |
| Long Format | 9,271 | 16.85% |

**High Zero-Inflation Countries:**
- Laos: 51.1% (suggests underreporting or surveillance gaps)
- Vietnam: 39.0%
- Timor-Leste: 33.3%
- Cambodia: 26.3%

**Low Zero-Inflation Countries:**
- Singapore: 0.0% (complete surveillance)
- Malaysia: 2.1% (excellent surveillance)

### 7.3 Outlier Detection

Using the IQR method (values > Q3 + 1.5×IQR):

- **Q1:** 4.0
- **Q3:** 74.0
- **IQR:** 70.0
- **Upper Bound:** 179.0
- **Outliers:** 9,475 records (13.61%)

**Top 10 Highest Case Records:**
| Country | Region | Year | Cases |
|---------|--------|------|-------|
| Vietnam | N/A | 1987 | 354,517 |
| Vietnam | N/A | 2019 | 320,702 |
| Thailand | N/A | 1987 | 174,285 |
| Vietnam | N/A | 2017 | 172,232 |
| Vietnam | N/A | 1983 | 143,380 |
| Vietnam | N/A | 2018 | 131,447 |
| Malaysia | N/A | 2019 | 130,101 |
| Vietnam | N/A | 2015 | 97,484 |
| Vietnam | N/A | 1991 | 94,630 |
| Philippines | N/A | 2015 | 86,916 |

---

## 8. Epidemic Detection

### 8.1 Methodology

Epidemic years were identified using a rolling mean + 2 standard deviations threshold:
- Years with cases exceeding (3-year rolling mean + 2×rolling std) were flagged

### 8.2 Notable Epidemic Patterns

**Historical Major Epidemics:**
1. **1987**: 789,095 cases - Major regional outbreak
2. **1998**: 613,597 cases - El Niño-associated outbreak
3. **2013**: 1,206,363 cases - Pre-2015 surge
4. **2015**: 1,208,971 cases - Near-record outbreak
5. **2019**: 2,185,806 cases - Record-breaking outbreak

**Country-Specific Patterns:**
- **Thailand (1998)**: 124,062 cases - Major epidemic
- **Vietnam (1998)**: 234,920 cases - Peak historical year
- **Philippines (2019)**: 872,795 cases - Largest single-country outbreak
- **Malaysia (2019)**: 260,383 cases - Peak year
- **Cambodia (2007)**: 38,418 cases - Regional peak

---

## 9. Point Process Simulation

### 9.1 Methodology

To demonstrate feasibility of converting aggregate counts to point process events:

1. Sample 500 events from the admin1-month data
2. Assign approximate geographic coordinates (country centroids with random jitter)
3. Preserve temporal and case count information

### 9.2 Output Format

Generated events include:
- `latitude`: Simulated latitude (±3° from country centroid)
- `longitude`: Simulated longitude
- `timestamp`: Original calendar date
- `cases`: Original case count
- `country`: Country name
- `region`: Original admin1 region name
- `month`/`year`: Temporal components

### 9.3 Caveats

**Important Notes:**
- Coordinates are simulated approximations (not actual region centroids)
- Real implementation would require a geographic lookup table with admin1 boundary polygons
- Could integrate with datasets like GADM or Natural Earth for accurate coordinates

---

## 10. Visualizations

Generated visualization files (saved to `/home/khang/Work/hackathon/dengue_dataset/figures/`):

| File | Description |
|------|-------------|
| `overview.png` | 4-panel overview: (1) Total cases by country, (2) Yearly trends, (3) Monthly seasonality heatmap, (4) Top outbreak years |
| `country_trends.png` | 8-panel yearly trends by country with epidemic markers |
| `monthly_patterns.png` | 8-panel monthly seasonality bar charts |
| `point_locations.png` | Scatter plot of simulated point events |

---

## 11. Key Findings and Observations

### 11.1 Data Strengths
- **Longitudinal coverage**: 30+ years of monthly data for several countries
- **Regional granularity**: Admin1-level data for most countries
- **Low missing data**: Core case variables have minimal missing values
- **Consistent structure**: Standardized column names across all files

### 11.2 Data Limitations
- **Incomplete country coverage**: Not all 11 countries have monthly data
- **Variable time spans**: Some countries have decades of data, others only months
- **Zero-inflation**: Some countries show high proportions of zero cases
- **Limited 2020+ data**: COVID-19 pandemic may have affected reporting

### 11.3 Epidemiological Patterns
1. **Strong interannual variability**: Large swings between epidemic and non-epidemic years
2. **Seasonal transmission**: Clear monsoon-related peaks in most countries
3. **Urban-rural differences**: Singapore shows distinct pattern from mainland countries
4. **El Niño correlation**: 1997-98 and 2015-16 outbreaks coincide with El Niño events

### 11.4 Surveillance Quality
- **Best surveillance**: Singapore (0% zeros), Malaysia (2.1% zeros)
- **Moderate surveillance**: Indonesia, Thailand, Philippines
- **Gaps in surveillance**: Laos, Vietnam, Cambodia (high zero-inflation)

---

## 12. Recommendations

### 12.1 For Further Analysis
1. **Time series modeling**: ARIMA/SARIMA for forecasting
2. **Spatiotemporal analysis**: Join with climate/demographic data
3. **Zero-inflated models**: Account for excess zeros in statistical models
4. **Outbreak detection**: Implement EWMA or CUSUM charts for real-time alerting

### 12.2 For Data Enhancement
1. **Geographic coordinates**: Add lat/lon for all admin1 regions
2. **Climate covariates**: Integrate temperature, rainfall, humidity
3. **Population data**: Enable per-capita rate calculations
4. **Vector data**: Add mosquito surveillance indices if available

### 12.3 For Visualization
1. **Interactive dashboards**: Build Plotly/Dash applications
2. **Choropleth maps**: Create animated yearly maps
3. **Heatmaps**: Explore region×month patterns
4. **Network graphs**: Show cross-border transmission patterns

---

## 13. Output Files Generated

| File | Description |
|------|-------------|
| `country_summary.csv` | Summary statistics by country |
| `simulated_point_events.csv` | 500 simulated point process events |
| `figures/overview.png` | Multi-panel overview visualization |
| `figures/country_trends.png` | Country-specific yearly trends |
| `figures/monthly_patterns.png` | Monthly seasonality charts |
| `figures/point_locations.png` | Geographic distribution plot |

---

## Appendix: Technical Notes

- **Python Environment**: pandas, numpy, matplotlib
- **Coordinate Reference**: WGS84 (simulated)
- **Statistical Methods**: IQR-based outlier detection, rolling mean epidemic threshold
- **Visualization Style**: seaborn whitegrid theme

---

*Report generated by automated EDA pipeline*
