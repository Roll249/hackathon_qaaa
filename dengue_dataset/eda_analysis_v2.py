#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for Dengue Dataset - Enhanced Version
Southeast Asia Dengue Surveillance Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("="*80)
print("DENGE DATASET EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load all three files
print("\n[1] LOADING DATA FILES...")

df_spatial = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_spatial.csv')
df_long = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_admin1_month.csv')
df_pivot = pd.read_csv('/home/khang/Work/hackathon/dengue_dataset/sea_dengue_admin1_month_pivot.csv')

print(f"sea_dengue_spatial.csv: {df_spatial.shape[0]:,} rows x {df_spatial.shape[1]} columns")
print(f"sea_dengue_admin1_month.csv: {df_long.shape[0]:,} rows x {df_long.shape[1]} columns")
print(f"sea_dengue_admin1_month_pivot.csv: {df_pivot.shape[0]} rows x {df_pivot.shape[1]} columns")

# Convert dates
df_spatial['calendar_start_date'] = pd.to_datetime(df_spatial['calendar_start_date'])
df_spatial['calendar_end_date'] = pd.to_datetime(df_spatial['calendar_end_date'])
df_long['calendar_start_date'] = pd.to_datetime(df_long['calendar_start_date'])
df_long['calendar_end_date'] = pd.to_datetime(df_long['calendar_end_date'])
df_pivot['calendar_start_date'] = pd.to_datetime(df_pivot['calendar_start_date'])

# Create normalized country names for matching
df_long['country_upper'] = df_long['adm_0_name'].str.upper()
df_spatial['country_upper'] = df_spatial['adm_0_name'].str.upper()

# ==============================================================================
# PER-COUNTRY DETAILED ANALYSIS
# ==============================================================================
print("\n\n[9] PER-COUNTRY DETAILED ANALYSIS")
print("="*80)

major_countries_upper = ['CAMBODIA', 'INDONESIA', "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 
                         'MALAYSIA', 'SINGAPORE', 'THAILAND', 'TIMOR-LESTE', 'VIET NAM']

country_analysis = {}

for country_upper in major_countries_upper:
    country_data = df_long[df_long['country_upper'] == country_upper]
    country_spatial = df_spatial[df_spatial['country_upper'] == country_upper]
    
    if len(country_data) == 0:
        print(f"\n{country_upper}: No data found")
        continue
    
    # Total cases
    total_cases = country_data['dengue_total'].sum()
    
    # Year with highest cases
    yearly_country = country_data.groupby('Year')['dengue_total'].sum()
    peak_year = yearly_country.idxmax() if len(yearly_country) > 0 else 'N/A'
    peak_cases = yearly_country.max() if len(yearly_country) > 0 else 0
    
    # Number of regions
    n_regions = country_data['adm_1_name'].nunique()
    
    # Monthly seasonality
    country_data_copy = country_data.copy()
    country_data_copy['Month'] = country_data_copy['calendar_start_date'].dt.month
    monthly_avg = country_data_copy.groupby('Month')['dengue_total'].mean()
    peak_month = monthly_avg.idxmax() if len(monthly_avg) > 0 else 0
    
    # Epidemic years detection (using 2 std above rolling mean)
    if len(yearly_country) >= 5:
        rolling_mean = yearly_country.rolling(window=3, center=True, min_periods=1).mean()
        rolling_std = yearly_country.rolling(window=3, center=True, min_periods=1).std().fillna(0)
        threshold = rolling_mean + 2 * rolling_std
        epidemic_years = yearly_country[yearly_country > threshold].index.tolist()
    else:
        epidemic_years = []
    
    country_analysis[country_upper] = {
        'total_cases': total_cases,
        'peak_year': peak_year,
        'peak_cases': peak_cases,
        'n_regions': n_regions,
        'peak_month': peak_month,
        'epidemic_years': epidemic_years,
        'yearly_data': yearly_country,
        'monthly_avg': monthly_avg,
        'mean': country_data['dengue_total'].mean(),
        'median': country_data['dengue_total'].median(),
        'max': country_data['dengue_total'].max(),
        'zero_pct': (country_data['dengue_total'] == 0).sum() / len(country_data) * 100,
        'date_start': country_data['calendar_start_date'].min(),
        'date_end': country_data['calendar_start_date'].max()
    }
    
    print(f"\n{'='*60}")
    print(f"COUNTRY: {country_upper}")
    print('='*60)
    print(f"Total cases: {total_cases:,}")
    print(f"Peak year: {peak_year} ({peak_cases:,} cases)")
    print(f"Regions: {n_regions}")
    print(f"Date range: {country_analysis[country_upper]['date_start'].strftime('%Y-%m')} to {country_analysis[country_upper]['date_end'].strftime('%Y-%m')}")
    print(f"Peak month: {peak_month}")
    print(f"Epidemic years: {epidemic_years}")
    print(f"\nStatistics:")
    print(f"  Mean: {country_analysis[country_upper]['mean']:.1f}")
    print(f"  Median: {country_analysis[country_upper]['median']:.1f}")
    print(f"  Max: {country_analysis[country_upper]['max']:,}")
    print(f"  Zero records: {country_analysis[country_upper]['zero_pct']:.1f}%")

# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================
print("\n\n[10] CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Total cases by country
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 Total cases by country (bar chart)
ax1 = axes[0, 0]
total_by_country = df_spatial.groupby('adm_0_name')['dengue_total'].sum().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(total_by_country)))
ax1.barh(total_by_country.index, total_by_country.values / 1e6, color=colors)
ax1.set_xlabel('Total Cases (Millions)')
ax1.set_title('Total Dengue Cases by Country (All Years)')
ax1.grid(axis='x', alpha=0.3)

# 1.2 Yearly totals across all countries
ax2 = axes[0, 1]
yearly_totals = df_spatial.groupby('Year')['dengue_total'].sum()
ax2.plot(yearly_totals.index, yearly_totals.values / 1e6, 'b-', linewidth=2, marker='o', markersize=3)
ax2.fill_between(yearly_totals.index, yearly_totals.values / 1e6, alpha=0.3)
ax2.set_xlabel('Year')
ax2.set_ylabel('Total Cases (Millions)')
ax2.set_title('Yearly Dengue Cases (All Countries)')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
ax2.tick_params(axis='x', rotation=45)

# 1.3 Monthly seasonality heatmap
ax3 = axes[1, 0]
monthly_pivot = df_long.copy()
monthly_pivot['Month'] = monthly_pivot['calendar_start_date'].dt.month
monthly_pivot['country'] = monthly_pivot['country_upper']
monthly_avg_pivot = monthly_pivot.pivot_table(values='dengue_total', index='Month', columns='country', aggfunc='mean')
# Normalize by country max for comparison
monthly_norm = monthly_avg_pivot.div(monthly_avg_pivot.max())
im = ax3.imshow(monthly_norm.T.values, aspect='auto', cmap='YlOrRd')
ax3.set_yticks(range(len(monthly_norm.columns)))
ax3.set_yticklabels([c[:15] for c in monthly_norm.columns], fontsize=8)
ax3.set_xticks(range(12))
ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax3.set_xlabel('Month')
ax3.set_title('Monthly Seasonality (Normalized by Peak)')
plt.colorbar(im, ax=ax3, label='Relative Intensity')

# 1.4 Top 10 outbreak years by country
ax4 = axes[1, 1]
yearly_country = df_spatial.pivot_table(values='dengue_total', index='Year', columns='adm_0_name', aggfunc='sum', fill_value=0)
recent = yearly_country.loc[2010:]
top_years = recent.sum(axis=1).nlargest(10)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_years)))
ax4.bar(range(len(top_years)), top_years.values / 1e6, color=colors)
ax4.set_xticks(range(len(top_years)))
ax4.set_xticklabels(top_years.index, rotation=45)
ax4.set_xlabel('Year')
ax4.set_ylabel('Total Cases (Millions)')
ax4.set_title('Top 10 Outbreak Years (2010-2022)')

plt.tight_layout()
plt.savefig('/home/khang/Work/hackathon/dengue_dataset/figures/overview.png', dpi=150, bbox_inches='tight')
print("Saved: figures/overview.png")

# Figure 2: Per-country analysis
fig2, axes2 = plt.subplots(4, 2, figsize=(16, 20))
axes2 = axes2.flatten()

for idx, (country, data) in enumerate(sorted(country_analysis.items())):
    if idx >= 8:
        break
    ax = axes2[idx]
    
    # Yearly trend
    yearly = data['yearly_data']
    if len(yearly) > 0:
        ax.bar(yearly.index, yearly.values / 1000, color='steelblue', alpha=0.7)
        ax.plot(yearly.index, yearly.values / 1000, 'b-', linewidth=1, alpha=0.5)
        
        # Mark epidemic years
        for ey in data['epidemic_years']:
            if ey in yearly.index:
                ax.axvline(x=ey, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f"{country}\nTotal: {data['total_cases']:,} | Peak: {data['peak_year']} ({data['peak_cases']:,})", fontsize=10)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cases (thousands)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/home/khang/Work/hackathon/dengue_dataset/figures/country_trends.png', dpi=150, bbox_inches='tight')
print("Saved: figures/country_trends.png")

# Figure 3: Monthly patterns
fig3, axes3 = plt.subplots(4, 2, figsize=(14, 16))
axes3 = axes3.flatten()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx, (country, data) in enumerate(sorted(country_analysis.items())):
    if idx >= 8:
        break
    ax = axes3[idx]
    
    monthly = data['monthly_avg']
    if len(monthly) > 0:
        bars = ax.bar(monthly.index, monthly.values, color='coral', alpha=0.7)
        # Highlight peak month
        peak = monthly.idxmax()
        bars[peak-1].set_color('darkred')
        
        ax.set_title(f"{country} - Peak Month: {month_names[peak-1]}", fontsize=10)
        ax.set_xlabel('Month')
        ax.set_ylabel('Avg Cases')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)

plt.tight_layout()
plt.savefig('/home/khang/Work/hackathon/dengue_dataset/figures/monthly_patterns.png', dpi=150, bbox_inches='tight')
print("Saved: figures/monthly_patterns.png")

# ==============================================================================
# POINT PROCESS SIMULATION
# ==============================================================================
print("\n\n[11] SIMULATED POINT PROCESS EVENTS")
print("="*80)

# Approximate country centroids (lat, lon)
country_coords = {
    'CAMBODIA': (12.5657, 104.9910),
    'INDONESIA': (-0.7893, 113.9213),
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": (19.8563, 102.4955),
    'MALAYSIA': (4.2105, 101.9758),
    'SINGAPORE': (1.3521, 103.8198),
    'THAILAND': (15.8700, 100.9925),
    'TIMOR-LESTE': (-8.8742, 125.7275),
    'VIET NAM': (14.0583, 108.2772)
}

np.random.seed(42)
n_samples = 500
sample_events = []

for i in range(n_samples):
    sample = df_long.sample(1).iloc[0]
    country = sample['country_upper']
    
    if country in country_coords:
        base_lat, base_lon = country_coords[country]
        lat = base_lat + np.random.uniform(-3, 3)
        lon = base_lon + np.random.uniform(-3, 3)
        timestamp = sample['calendar_start_date']
        cases = sample['dengue_total']
        region = sample['adm_1_name']
        
        sample_events.append({
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'timestamp': timestamp,
            'cases': cases,
            'country': country.title(),
            'region': region,
            'month': timestamp.month,
            'year': timestamp.year
        })

events_df = pd.DataFrame(sample_events)
print(f"Generated {len(events_df)} simulated point events")

# Save events
events_df.to_csv('/home/khang/Work/hackathon/dengue_dataset/simulated_point_events.csv', index=False)
print("Saved: simulated_point_events.csv")

# Figure 4: Geographic distribution
fig4, ax4 = plt.subplots(figsize=(12, 8))

for country in events_df['country'].unique():
    country_events = events_df[events_df['country'] == country]
    ax4.scatter(country_events['longitude'], country_events['latitude'], 
                s=np.sqrt(country_events['cases'] + 1) * 2, 
                alpha=0.5, label=country, edgecolors='white', linewidths=0.5)

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.set_title('Simulated Dengue Event Locations (Point Process)')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(90, 145)
ax4.set_ylim(-15, 30)

plt.tight_layout()
plt.savefig('/home/khang/Work/hackathon/dengue_dataset/figures/point_locations.png', dpi=150, bbox_inches='tight')
print("Saved: figures/point_locations.png")

# ==============================================================================
# FINAL SUMMARY TABLE
# ==============================================================================
print("\n\n[12] SUMMARY TABLE")
print("="*80)

summary_data = []
for country, data in sorted(country_analysis.items()):
    summary_data.append({
        'Country': country.title(),
        'Total Cases': f"{data['total_cases']:,}",
        'Peak Year': data['peak_year'],
        'Peak Cases': f"{data['peak_cases']:,}",
        'Regions': data['n_regions'],
        'Peak Month': month_names[data['peak_month']-1] if data['peak_month'] > 0 else 'N/A',
        'Epidemic Years': ', '.join(map(str, data['epidemic_years'][:5])) if data['epidemic_years'] else 'None',
        'Mean Monthly': f"{data['mean']:.1f}",
        'Zero %': f"{data['zero_pct']:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('/home/khang/Work/hackathon/dengue_dataset/country_summary.csv', index=False)
print("\nSaved: country_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
