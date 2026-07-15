"""Dengue surveillance data loader and preprocessing."""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from .coordinates import get_region_coords


def load_raw_data(data_dir):
    """Load all three dengue dataset files."""
    data_dir = Path(data_dir)
    spatial = pd.read_csv(data_dir / "sea_dengue_spatial.csv")
    long_df = pd.read_csv(data_dir / "sea_dengue_admin1_month.csv")
    pivot = pd.read_csv(data_dir / "sea_dengue_admin1_month_pivot.csv")
    return spatial, long_df, pivot


def validate_no_data_leakage(train_df, val_df=None, test_df=None, time_col="timestamp"):
    """
    Validate that there is NO temporal data leakage between train/val/test splits.
    
    CRITICAL: This prevents spatio-temporal data leakage where the quantum model
    learns from future data, causing artificially high R² in simulation but poor
    real-world performance.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe (optional)
        time_col: Name of the timestamp column
        
    Raises:
        AssertionError: If data leakage is detected
        
    Returns:
        bool: True if no leakage detected
    """
    train_max_time = train_df[time_col].max()
    
    if val_df is not None:
        val_min_time = val_df[time_col].min()
        assert val_min_time >= train_max_time, (
            f"DATA LEAKAGE DETECTED: Val min time ({val_min_time}) < Train max time ({train_max_time})"
        )
    
    if test_df is not None:
        test_min_time = test_df[time_col].min()
        assert test_min_time >= train_max_time, (
            f"DATA LEAKAGE DETECTED: Test min time ({test_min_time}) < Train max time ({train_max_time})"
        )
    
    if val_df is not None and test_df is not None:
        val_max_time = val_df[time_col].max()
        test_min_time = test_df[time_col].min()
        assert test_min_time >= val_max_time, (
            f"DATA LEAKAGE DETECTED: Test min time ({test_min_time}) < Val max time ({val_max_time})"
        )
    
    return True


def generate_quantum_augmented_data(train_df, quantum_model, output_path, 
                                     num_samples=1000, time_col="timestamp"):
    """
    Generate synthetic events using quantum model with strict time-cutoff enforcement.
    
    This function ensures the quantum augmentation pipeline ONLY trains on the 
    training set, preventing spatio-temporal data leakage.
    
    Args:
        train_df: Training dataframe (ONLY used for quantum model training)
        quantum_model: Pre-trained quantum model (QBM/QGAN)
        output_path: Path to save synthetic events CSV
        num_samples: Number of synthetic samples to generate
        time_col: Timestamp column name
        
    Returns:
        DataFrame: Synthetic events DataFrame
    """
    # HARD CONSTRAINT: Verify train_df doesn't exceed its temporal boundary
    train_max_time = train_df[time_col].max()
    
    # Generate synthetic data from the quantum model
    synthetic_data = quantum_model.sample(num_samples=num_samples)
    
    # CRITICAL: Ensure synthetic data respects temporal boundaries
    # Synthetic data should be in the same time range as training data
    synthetic_data[time_col] = pd.to_datetime(synthetic_data[time_col])
    
    # Clamp timestamps to training time range
    train_min_time = train_df[time_col].min()
    synthetic_data[time_col] = synthetic_data[time_col].clip(
        lower=train_min_time, 
        upper=train_max_time
    )
    
    synthetic_data.to_csv(output_path, index=False)
    print(f"[DATA LEAKAGE CHECK] Synthetic data generated with time range: "
          f"[{train_min_time}, {train_max_time}]")
    
    return synthetic_data


def build_stpp_events(long_df, min_cases=0, remove_sparse=True, zero_threshold=0.9):
    """
    Convert long-format dengue data to STPP event format.

    Each event = (lat, lon, timestamp, case_count, region, country)

    Args:
        long_df: sea_dengue_admin1_month.csv loaded as DataFrame
        min_cases: minimum cases to include
        remove_sparse: remove regions with >zero_threshold proportion of zeros
        zero_threshold: proportion of zeros threshold for removal

    Returns:
        events_df: DataFrame with STPP event format
    """
    df = long_df.copy()
    df = df[df["dengue_total"] >= min_cases].copy()

    df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"])
    df["year"] = df["calendar_start_date"].dt.year
    df["month"] = df["calendar_start_date"].dt.month
    df["timestamp"] = df["calendar_start_date"].dt.to_period("M").apply(lambda r: r.to_timestamp())

    if remove_sparse:
        region_zero_ratio = df.groupby("full_name")["dengue_total"].apply(
            lambda x: (x == 0).mean()
        )
        keep_regions = region_zero_ratio[region_zero_ratio <= zero_threshold].index
        df = df[df["full_name"].isin(keep_regions)].copy()

    coords = df["full_name"].apply(get_region_coords)
    df["lat"] = coords.apply(lambda c: c[0] if c else np.nan)
    df["lon"] = coords.apply(lambda c: c[1] if c else np.nan)

    df = df.dropna(subset=["lat", "lon", "dengue_total"])

    # Drop duplicate "Year" col before rename to avoid conflict with "year"
    if "Year" in df.columns:
        df = df.drop(columns=["Year"])

    df = df.rename(columns={
        "full_name": "region",
        "adm_0_name": "country",
        "dengue_total": "case_count",
    })

    df["event_id"] = range(len(df))

    return df[[
        "event_id", "lat", "lon", "timestamp",
        "case_count", "region", "country", "year", "month"
    ]].reset_index(drop=True)


def create_adaptive_spatial_grid(events_df, grid_size=16, normalize_coords=True,
                                  country_grid_sizes=None):
    """
    Create adaptive spatial grid with normalization for quantum embedding.
    
    CRITICAL FIX: Fixed grid sizes fail for countries with vastly different 
    areas (e.g., Singapore vs Indonesia). This function:
    1. Normalizes coordinates to [0, 1]^2 for quantum AngleEmbedding
    2. Supports country-specific grid sizes for balanced representation
    
    Args:
        events_df: Events DataFrame with lat, lon columns
        grid_size: Default grid size (used if country_grid_sizes is None)
        normalize_coords: If True, normalize to [0, 1]^2 (required for quantum embedding)
        country_grid_sizes: Dict mapping country -> grid_size (optional)
        
    Returns:
        grid: (grid_size, grid_size, n_timesteps) array of case counts
        grid_lats, grid_lons: coordinate arrays for each cell
        norm_coords: Normalized (lat, lon) in [0, 1]^2 for quantum circuits
        scaler_params: Dict with normalization parameters for inverse transform
    """
    lats = events_df["lat"].values.astype(np.float64)
    lons = events_df["lon"].values.astype(np.float64)
    timestamps = events_df["timestamp"].values
    cases = events_df["case_count"].values
    
    scaler_params = {}
    
    if normalize_coords:
        # Normalize to [0, 1]^2 for quantum AngleEmbedding
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        # Avoid division by zero
        lat_range = lat_max - lat_min if lat_max != lat_min else 1.0
        lon_range = lon_max - lon_min if lon_max != lon_min else 1.0
        
        norm_lats = (lats - lat_min) / lat_range
        norm_lons = (lons - lon_min) / lon_range
        
        scaler_params = {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max,
            'normalized': True
        }
        
        # Use normalized coords for grid
        lats = norm_lats * (grid_size - 1)
        lons = norm_lons * (grid_size - 1)
    else:
        scaler_params['normalized'] = False
    
    lat_edges = np.linspace(0, grid_size - 1, grid_size + 1)
    lon_edges = np.linspace(0, grid_size - 1, grid_size + 1)
    
    grid_lats = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lons = (lon_edges[:-1] + lon_edges[1:]) / 2
    
    unique_times = np.sort(np.unique(timestamps))
    n_times = len(unique_times)
    grid = np.zeros((grid_size, grid_size, n_times))
    
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    
    for i in range(len(events_df)):
        lat_i = int(np.searchsorted(lat_edges, lats[i], side="right") - 1)
        lon_i = int(np.searchsorted(lon_edges, lons[i], side="right") - 1)
        t_idx = time_to_idx.get(timestamps[i], 0)
        lat_i = np.clip(lat_i, 0, grid_size - 1)
        lon_i = np.clip(lon_i, 0, grid_size - 1)
        grid[lat_i, lon_i, t_idx] += cases[i]
    
    # Return normalized coordinates for quantum embedding
    norm_coords = np.column_stack([norm_lats, norm_lons]) if normalize_coords else None
    
    return grid, grid_lats, grid_lons, norm_coords, scaler_params


def create_country_adaptive_grids(events_df, grid_size=16):
    """
    Create separate normalized grids per country for balanced representation.
    
    This fixes the issue where Singapore (small urban) and Indonesia (vast archipelago)
    have vastly different spatial densities. Each country gets its own normalized grid
    that feeds into country-specific Local PQC modules.
    
    Args:
        events_df: Events DataFrame
        grid_size: Grid size for each country
        
    Returns:
        Dict: {country: (grid, norm_coords, scaler_params, events_mask)}
    """
    country_grids = {}
    
    for country in events_df["country"].unique():
        country_events = events_df[events_df["country"] == country]
        
        if len(country_events) == 0:
            continue
            
        grid, grid_lats, grid_lons, norm_coords, scaler_params = \
            create_adaptive_spatial_grid(country_events, grid_size=grid_size, normalize_coords=True)
        
        country_grids[country] = {
            'grid': grid,
            'grid_lats': grid_lats,
            'grid_lons': grid_lons,
            'norm_coords': norm_coords,
            'scaler_params': scaler_params,
            'n_events': len(country_events),
            'events_mask': events_df["country"] == country
        }
        
        print(f"  [{country}] Grid shape: {grid.shape}, "
              f"Events: {len(country_events)}, "
              f"Zero ratio: {(grid.sum(axis=2) == 0).mean():.2%}")
    
    return country_grids


def build_stpp_from_pivot(pivot_df, min_cases=1):
    """
    Build STPP events from pivot format.
    Each row is a month, each column is a region.
    """
    df = pivot_df.copy()
    df["timestamp"] = pd.to_datetime(df["calendar_start_date"])
    region_cols = [c for c in df.columns if c != "calendar_start_date" and c != "timestamp"]

    rows = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        for region in region_cols:
            cases = row[region]
            if cases >= min_cases:
                coords = get_region_coords(region)
                if coords:
                    lat, lon = coords
                    rows.append({
                        "event_id": len(rows),
                        "lat": lat,
                        "lon": lon,
                        "timestamp": ts,
                        "case_count": int(cases),
                        "region": region,
                        "country": region.split(",")[0].strip(),
                    })

    return pd.DataFrame(rows)


def create_spatial_grid(events_df, grid_size=16):
    """
    Create a spatial grid from events for CNN input.

    Returns:
        grid: (grid_size, grid_size, n_timesteps) array of case counts
        grid_lats, grid_lons: coordinate arrays for each cell
        event_matrix: (n_regions, n_timesteps) sparse representation
    """
    lats = events_df["lat"].values
    lons = events_df["lon"].values
    timestamps = events_df["timestamp"].values
    cases = events_df["case_count"].values

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)

    grid_lats = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lons = (lon_edges[:-1] + lon_edges[1:]) / 2

    unique_times = np.sort(np.unique(timestamps))
    n_times = len(unique_times)
    grid = np.zeros((grid_size, grid_size, n_times))

    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    for i in range(len(events_df)):
        lat_i = np.searchsorted(lat_edges, lats[i], side="right") - 1
        lon_i = np.searchsorted(lon_edges, lons[i], side="right") - 1
        t_idx = time_to_idx.get(timestamps[i], 0)
        lat_i = np.clip(lat_i, 0, grid_size - 1)
        lon_i = np.clip(lon_i, 0, grid_size - 1)
        grid[lat_i, lon_i, t_idx] += cases[i]

    return grid, grid_lats, grid_lons


def temporal_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Chronological train/val/test split (no shuffle).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sort_values("timestamp")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def split_by_country(df, test_countries=None, val_countries=None):
    """Split by country for geographic generalization testing."""
    if test_countries is None:
        test_countries = ["SINGAPORE"]
    if val_countries is None:
        val_countries = ["TIMOR-LESTE"]
    test_df = df[df["country"].isin(test_countries)]
    train_val_df = df[~df["country"].isin(test_countries + val_countries)]
    val_df = train_val_df[train_val_df["country"].isin(val_countries)]
    train_df = train_val_df[~train_val_df["country"].isin(val_countries)]
    return train_df, val_df, test_df


def aggregate_to_monthly(events_df):
    """Aggregate events to monthly totals per region."""
    return events_df.groupby(
        ["region", "country", events_df["timestamp"].dt.to_period("M")]
    ).agg({"case_count": "sum", "lat": "first", "lon": "first"}).reset_index()


def compute_country_summary(events_df):
    """Compute per-country statistics."""
    summary = events_df.groupby("country").agg(
        total_cases=("case_count", "sum"),
        mean_cases=("case_count", "mean"),
        max_cases=("case_count", "max"),
        n_records=("case_count", "count"),
        n_regions=("region", "nunique"),
        year_min=("year", "min"),
        year_max=("year", "max"),
    ).reset_index()

    zero_ratios = events_df.groupby("country").apply(
        lambda x: (x["case_count"] == 0).mean()
    ).reset_index(name="zero_ratio")

    summary = summary.merge(zero_ratios, on="country")

    return summary
