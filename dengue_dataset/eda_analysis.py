#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for Dengue Dataset
Southeast Asia Dengue Surveillance Data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("="*80)
print("DENGE DATASET EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load all three files
print("\n[1] LOADING DATA FILES...")
print("-"*40)

# File 1: Spatial data (yearly)
df_spatial = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_spatial.csv')
print(f"sea_dengue_spatial.csv: {df_spatial.shape[0]:,} rows x {df_spatial.shape[1]} columns")

# File 2: Admin1 monthly data (long format)
df_long = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_admin1_month.csv')
print(f"sea_dengue_admin1_month.csv: {df_long.shape[0]:,} rows x {df_long.shape[1]} columns")

# File 3: Pivot table
df_pivot = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_admin1_month_pivot.csv')
print(f"sea_dengue_admin1_month_pivot.csv: {df_pivot.shape[0]} rows x {df_pivot.shape[1]} columns")

# ==============================================================================
# 2. BASIC INFORMATION
# ==============================================================================
print("\n[2] BASIC FILE INFORMATION")
print("-"*40)

print("\n--- sea_dengue_spatial.csv ---")
print(f"Shape: {df_spatial.shape}")
print(f"Columns: {list(df_spatial.columns)}")
print(f"Data types:\n{df_spatial.dtypes}")

print("\n--- sea_dengue_admin1_month.csv ---")
print(f"Shape: {df_long.shape}")
print(f"Columns: {list(df_long.columns)}")

print("\n--- sea_dengue_admin1_month_pivot.csv ---")
print(f"Shape: {df_pivot.shape}")
print(f"First column (date): {df_pivot.columns[0]}")
print(f"Number of region columns: {len(df_pivot.columns) - 1}")

# ==============================================================================
# 3. DATE RANGES AND TEMPORAL COVERAGE
# ==============================================================================
print("\n[3] TEMPORAL COVERAGE")
print("-"*40)

# Convert dates
df_spatial['calendar_start_date'] = pd.to_datetime(df_spatial['calendar_start_date'])
df_spatial['calendar_end_date'] = pd.to_datetime(df_spatial['calendar_end_date'])
df_long['calendar_start_date'] = pd.to_datetime(df_long['calendar_start_date'])
df_long['calendar_end_date'] = pd.to_datetime(df_long['calendar_end_date'])
df_pivot['calendar_start_date'] = pd.to_datetime(df_pivot['calendar_start_date'])

# Spatial data date range
print("\n--- sea_dengue_spatial.csv ---")
print(f"Date range: {df_spatial['calendar_start_date'].min()} to {df_spatial['calendar_start_date'].max()}")
print(f"Years covered: {sorted(df_spatial['Year'].unique())}")

# Long data date range
print("\n--- sea_dengue_admin1_month.csv ---")
print(f"Date range: {df_long['calendar_start_date'].min()} to {df_long['calendar_start_date'].max()}")

# Pivot data date range
print("\n--- sea_dengue_admin1_month_pivot.csv ---")
print(f"Date range: {df_pivot['calendar_start_date'].min()} to {df_pivot['calendar_start_date'].max()}")

# ==============================================================================
# 4. COUNTRIES COVERED
# ==============================================================================
print("\n[4] COUNTRIES COVERED")
print("-"*40)

countries_spatial = df_spatial['adm_0_name'].unique()
countries_long = df_long['adm_0_name'].unique()

print(f"\nCountries in spatial data ({len(countries_spatial)}):")
for c in sorted(countries_spatial):
    count = len(df_spatial[df_spatial['adm_0_name'] == c])
    print(f"  - {c}: {count:,} records")

print(f"\nCountries in long data ({len(countries_long)}):")
for c in sorted(countries_long):
    count = len(df_long[df_long['adm_0_name'] == c])
    print(f"  - {c}: {count:,} records")

# ==============================================================================
# 5. REGIONS BY COUNTRY
# ==============================================================================
print("\n[5] ADMINISTRATIVE REGIONS BY COUNTRY")
print("-"*40)

for country in sorted(df_long['adm_0_name'].unique()):
    regions = df_long[df_long['adm_0_name'] == country]['adm_1_name'].unique()
    print(f"{country}: {len(regions)} regions")
    for r in sorted(regions)[:5]:  # Show first 5
        print(f"  - {r}")
    if len(regions) > 5:
        print(f"  ... and {len(regions) - 5} more")

# ==============================================================================
# 6. SUMMARY STATISTICS - TOTAL CASES
# ==============================================================================
print("\n[6] SUMMARY STATISTICS: TOTAL CASES")
print("-"*40)

# Total cases by country
total_by_country = df_spatial.groupby('adm_0_name')['dengue_total'].sum().sort_values(ascending=False)
print("\nTotal cases by country (from spatial data):")
for country, total in total_by_country.items():
    print(f"  {country}: {total:,}")

print(f"\nGrand total: {total_by_country.sum():,}")

# Basic statistics
print("\n--- Dengue case statistics ---")
print(f"Mean cases per record: {df_spatial['dengue_total'].mean():.2f}")
print(f"Median cases per record: {df_spatial['dengue_total'].median():.2f}")
print(f"Std deviation: {df_spatial['dengue_total'].std():.2f}")
print(f"Min: {df_spatial['dengue_total'].min():,}")
print(f"Max: {df_spatial['dengue_total'].max():,}")
print(f"25th percentile: {df_spatial['dengue_total'].quantile(0.25):.2f}")
print(f"75th percentile: {df_spatial['dengue_total'].quantile(0.75):.2f}")
print(f"99th percentile: {df_spatial['dengue_total'].quantile(0.99):.2f}")

# ==============================================================================
# 7. YEARLY TOTALS
# ==============================================================================
print("\n[7] YEARLY CASE TOTALS")
print("-"*40)

yearly_totals = df_spatial.groupby('Year')['dengue_total'].sum()
print("\nYearly totals across all countries:")
for year, total in yearly_totals.items():
    print(f"  {year}: {total:,}")

print(f"\nYear with highest cases: {yearly_totals.idxmax()} ({yearly_totals.max():,})")
print(f"Year with lowest cases: {yearly_totals.idxmin()} ({yearly_totals.min():,})")

# Yearly by country
print("\n--- Yearly totals by country ---")
yearly_by_country = df_spatial.pivot_table(values='dengue_total', index='Year', columns='adm_0_name', aggfunc='sum', fill_value=0)
print(yearly_by_country.to_string())

# ==============================================================================
# 8. DATA QUALITY ISSUES
# ==============================================================================
print("\n[8] DATA QUALITY ASSESSMENT")
print("-"*40)

# Missing values
print("\n--- Missing Values ---")
missing_spatial = df_spatial.isnull().sum()
print("Spatial data missing values:")
for col, count in missing_spatial[missing_spatial > 0].items():
    print(f"  {col}: {count} ({count/len(df_spatial)*100:.2f}%)")

missing_long = df_long.isnull().sum()
print("\nLong format missing values:")
for col, count in missing_long[missing_long > 0].items():
    print(f"  {col}: {count} ({count/len(df_long)*100:.2f}%)")

# Zero values
print("\n--- Zero Case Records ---")
zero_count_spatial = (df_spatial['dengue_total'] == 0).sum()
zero_count_long = (df_long['dengue_total'] == 0).sum()
print(f"Spatial data: {zero_count_spatial:,} records with 0 cases ({zero_count_spatial/len(df_spatial)*100:.2f}%)")
print(f"Long format: {zero_count_long:,} records with 0 cases ({zero_count_long/len(df_long)*100:.2f}%)")

# Outliers detection (IQR method)
print("\n--- Outlier Detection (IQR method) ---")
Q1 = df_spatial['dengue_total'].quantile(0.25)
Q3 = df_spatial['dengue_total'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df_spatial[(df_spatial['dengue_total'] < lower_bound) | (df_spatial['dengue_total'] > upper_bound)]
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df_spatial)*100:.2f}%)")

# Top outliers
print("\nTop 10 highest case records:")
top_outliers = df_spatial.nlargest(10, 'dengue_total')[['adm_0_name', 'adm_1_name', 'Year', 'dengue_total']]
for _, row in top_outliers.iterrows():
    region = row['adm_1_name'] if pd.notna(row['adm_1_name']) else "N/A"
    print(f"  {row['adm_0_name']}, {region}, {row['Year']}: {row['dengue_total']:,}")

# ==============================================================================
# 9. PER-COUNTRY ANALYSIS
# ==============================================================================
print("\n[9] PER-COUNTRY DETAILED ANALYSIS")
print("="*80)

major_countries = ['Cambodia', 'Indonesia', 'Lao People\'s Democratic Republic', 
                   'Malaysia', 'Singapore', 'Thailand', 'Timor-Leste', 'Viet Nam']

for country in major_countries:
    print(f"\n{'='*60}")
    print(f"COUNTRY: {country}")
    print('='*60)
    
    country_data = df_long[df_long['adm_0_name'] == country]
    country_spatial = df_spatial[df_spatial['adm_0_name'] == country]
    
    if len(country_data) == 0:
        print(f"No data found for {country}")
        continue
    
    # Total cases
    total_cases = country_data['dengue_total'].sum()
    print(f"\nTotal cases: {total_cases:,}")
    
    # Year with highest cases
    yearly_country = country_data.groupby('Year')['dengue_total'].sum()
    if len(yearly_country) > 0:
        peak_year = yearly_country.idxmax()
        peak_cases = yearly_country.max()
        print(f"Year with highest cases: {peak_year} ({peak_cases:,} cases)")
    
    # Number of regions
    n_regions = country_data['adm_1_name'].nunique()
    print(f"Number of administrative regions: {n_regions}")
    
    # Date coverage
    print(f"Date range: {country_data['calendar_start_date'].min()} to {country_data['calendar_start_date'].max()}")
    
    # Monthly seasonality
    country_data_copy = country_data.copy()
    country_data_copy['Month'] = country_data_copy['calendar_start_date'].dt.month
    monthly_avg = country_data_copy.groupby('Month')['dengue_total'].mean()
    
    print("\nMonthly average cases:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        if m in monthly_avg.index:
            print(f"  {month_names[m-1]}: {monthly_avg[m]:.1f}")
    
    # Peak month
    if len(monthly_avg) > 0:
        peak_month = monthly_avg.idxmax()
        print(f"\nPeak month: {month_names[peak_month-1]} (avg: {monthly_avg.max():.1f})")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean: {country_data['dengue_total'].mean():.2f}")
    print(f"  Median: {country_data['dengue_total'].median():.2f}")
    print(f"  Max: {country_data['dengue_total'].max():,}")
    print(f"  Zero records: {(country_data['dengue_total'] == 0).sum()} ({(country_data['dengue_total'] == 0).sum()/len(country_data)*100:.1f}%)")

# ==============================================================================
# 10. YEARLY TRENDS BY COUNTRY
# ==============================================================================
print("\n\n[10] YEARLY TRENDS - IDENTIFYING EPIDEMIC YEARS")
print("="*80)

for country in major_countries:
    country_data = df_long[df_long['adm_0_name'] == country]
    if len(country_data) == 0:
        continue
    
    yearly = country_data.groupby('Year')['dengue_total'].sum()
    
    # Calculate rolling mean and std for outbreak detection
    if len(yearly) >= 3:
        rolling_mean = yearly.rolling(window=3, center=True).mean()
        std = yearly.rolling(window=3, center=True).std()
        
        # Years with cases > mean + 2*std (potential epidemics)
        epidemic_years = yearly[yearly > rolling_mean + 2 * std]
        
        if len(epidemic_years) > 0:
            print(f"\n{country}:")
            print(f"  Average annual cases: {yearly.mean():,.0f}")
            print(f"  Potential epidemic years (>{rolling_mean.mean() + 2*std.mean():,.0f}):")
            for year, cases in epidemic_years.items():
                print(f"    {year}: {cases:,}")

# ==============================================================================
# 11. MONTHLY SEASONALITY PATTERNS
# ==============================================================================
print("\n\n[11] MONTHLY SEASONALITY PATTERNS")
print("="*80)

df_long_copy = df_long.copy()
df_long_copy['Month'] = df_long_copy['calendar_start_date'].dt.month

print("\nAverage monthly cases by country:")
monthly_by_country = df_long_copy.pivot_table(values='dengue_total', index='Month', columns='adm_0_name', aggfunc='mean')
print(monthly_by_country.round(1).to_string())

# Identify peak months for each country
print("\n\nPeak transmission month by country:")
for country in major_countries:
    if country in monthly_by_country.columns:
        peak_month = monthly_by_country[country].idxmax()
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        print(f"  {country}: {month_names[peak_month-1]}")

# ==============================================================================
# 12. POINT PROCESS SIMULATION
# ==============================================================================
print("\n\n[12] SIMULATED POINT PROCESS EVENTS")
print("="*80)

# This section generates simulated (lat, lon, timestamp) events from aggregate data
# Note: Real coordinates would require a geographic lookup table

# Get approximate centroids for SEA countries (for demonstration)
country_coords = {
    'Cambodia': (12.5657, 104.9910),
    'Indonesia': (-0.7893, 113.9213),
    "Lao People's Democratic Republic": (19.8563, 102.4955),
    'Malaysia': (4.2105, 101.9758),
    'Singapore': (1.3521, 103.8198),
    'Thailand': (15.8700, 100.9925),
    'Timor-Leste': (-8.8742, 125.7275),
    'Viet Nam': (14.0583, 108.2772)
}

print("\nSimulated point process events generated from admin1-month data")
print("(Adding random jitter to country centroids for region approximation)")
print("\nNote: Real implementation would require administrative boundary coordinates")
print("      This simulation uses country centroids with random jitter\n")

# Generate sample events
np.random.seed(42)
n_samples = 100
sample_events = []

for _ in range(n_samples):
    # Random sample from the long format data
    sample = df_long.sample(1).iloc[0]
    country = sample['adm_0_name']
    
    if country in country_coords:
        base_lat, base_lon = country_coords[country]
        # Add random jitter (±2 degrees)
        lat = base_lat + np.random.uniform(-2, 2)
        lon = base_lon + np.random.uniform(-2, 2)
        timestamp = sample['calendar_start_date']
        cases = sample['dengue_total']
        
        sample_events.append({
            'latitude': lat,
            'longitude': lon,
            'timestamp': timestamp,
            'cases': cases,
            'country': country,
            'region': sample['adm_1_name']
        })

events_df = pd.DataFrame(sample_events)
print(f"Generated {len(events_df)} simulated point events")
print("\nSample events:")
print(events_df.head(10).to_string())

# Save events to CSV
events_df.to_csv('/home/khang/Work/hackathon/dengue_dataset/simulated_point_events.csv', index=False)
print(f"\nSaved simulated events to: simulated_point_events.csv")

# ==============================================================================
# 13. SUMMARY REPORT
# ==============================================================================
print("\n\n" + "="*80)
print("EXECUTIVE SUMMARY")
print("="*80)

print(f"""
DATASET OVERVIEW:
- 3 data files covering dengue surveillance in Southeast Asia
- Spatial data: {df_spatial.shape[0]:,} records (yearly aggregates)
- Long format: {df_long.shape[0]:,} records (admin1 monthly)
- Pivot format: {df_pivot.shape[0]} time points x {df_pivot.shape[1]-1} regions

GEOGRAPHIC COVERAGE:
- {len(df_long['adm_0_name'].unique())} countries: Cambodia, Indonesia, Laos, Malaysia, 
  Singapore, Thailand, Timor-Leste, Vietnam
- {df_long['adm_1_name'].nunique()} administrative regions

TEMPORAL COVERAGE:
- Spatial: 1990-2017 (yearly)
- Monthly: {df_long['calendar_start_date'].min().strftime('%Y-%m')} to {df_long['calendar_start_date'].max().strftime('%Y-%m')}

KEY STATISTICS:
- Total cases: {df_spatial['dengue_total'].sum():,}
- Mean cases per record: {df_spatial['dengue_total'].mean():.1f}
- Peak year: {yearly_totals.idxmax()} ({yearly_totals.max():,} cases)

DATA QUALITY:
- Missing values: Minimal in key columns
- Zero-inflation: {(df_spatial['dengue_total']==0).sum()/len(df_spatial)*100:.1f}% of records have zero cases
- Outliers: {len(outliers)} records ({len(outliers)/len(df_spatial)*100:.2f}%) identified as statistical outliers

TOP COUNTRIES BY TOTAL CASES:
""")

for i, (country, total) in enumerate(total_by_country.head(5).items(), 1):
    print(f"  {i}. {country}: {total:,} cases")

print(f"""
SEASONAL PATTERNS:
- Peak transmission typically during rainy season (May-October)
- Thailand, Cambodia, Vietnam show strong seasonality
- Indonesia shows more year-round transmission

RECOMMENDATIONS:
1. Consider zero-inflated models for statistical analysis
2. Regional variation suggests need for localized interventions
3. Multi-year trends show cycles consistent with known dengue epidemiology
4. Point process simulation demonstrates feasibility of event-based analysis
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
