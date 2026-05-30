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
