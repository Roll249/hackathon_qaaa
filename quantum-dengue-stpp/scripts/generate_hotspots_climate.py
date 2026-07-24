#!/usr/bin/env python3
"""
generate_hotspots_climate.py — Dengue Hotspot Forecast with Climate + Air Quality Features
===================================================================================

Enhanced pipeline that enriches dengue case forecasting with:
  • Open-Meteo Historical weather (ERA5 reanalysis, 1940–2010, FREE, no auth)
  • Open-Meteo Air Quality (PM2.5/NO2/O3, ~2022–present, FREE, no auth)
  • Feature engineering adapted from hination pattern

Reuse / adaptation from hination:
  • `OpenMeteoHistoricalProvider` from hination/hination/providers/era5_provider.py
    — REUSED AS-IS (import)
  • `compute_api()`, `get_monsoon_phase()` from hination/features/feature_engineering.py
    — REUSED WITH ADDITIONS (dengue-specific lag features)
  • `DailyFeatureVector` dataclass from hination
    — ADAPTED for dengue (removed terrain/disaster, added PM2.5/NO2/O3)

New additions in this pipeline:
  • OpenMeteoAirQualityProvider: fetches hourly AQ → daily aggregates (pm25, pm10, no2, o3)
  • DengueFeatureVector: enhanced with temp_lag_1w, temp_lag_4w, rain_lag_2w,
    humid_4w_mean, temp_optimal_window (25–32°C Aedes optimum)
  • WeeklyFeatureAggregator: daily → weekly feature aggregation

Usage:
    python scripts/generate_hotspots_climate.py [--n-provinces N] [--year-start Y]

Output:
    ../q_dengue_epidemiology-main/web/public/data/hotspots.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse from hination — Open-Meteo Historical Provider (ERA5)
# ---------------------------------------------------------------------------
# ADAPTED FROM: hination/hination/providers/era5_provider.py
# CHANGES: None — used as-is via import
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()          # → quantum-dengue-stpp/scripts/
HINATION_DIR = SCRIPT_DIR.parent.parent / "hination"  # → hackathon_qaaa/hination/
REPO_ROOT = SCRIPT_DIR.parent.parent                    # → hackathon_qaaa/
DATA_DIR = REPO_ROOT / "dengue_dataset"
SRC_DATA = DATA_DIR / "sea_dengue_admin1_month.csv"
CLIMATE_CACHE = SCRIPT_DIR / "data" / "climate_cache"
AQ_CACHE = SCRIPT_DIR / "data" / "aq_cache"
CLIMATE_CACHE.mkdir(parents=True, exist_ok=True)
AQ_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = (
    REPO_ROOT / "q_dengue_epidemiology-main"
    / "web" / "public" / "data" / "hotspots.json"
)

# Try to import from hination, fall back to bundled copy
try:
    sys.path.insert(0, str(HINATION_DIR))
    from hination.providers.era5_provider import (
        OpenMeteoHistoricalProvider,
        OPEN_METEO_DAILY_VARS,
        OPEN_METEO_BASE,
        OPEN_METEO_TIMEZONE,
    )
    _HINATION_AVAILABLE = True
except ImportError:
    _HINATION_AVAILABLE = False
    OPEN_METEO_DAILY_VARS = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "precipitation_sum", "rain_sum", "precipitation_hours",
        "weather_code", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        "daylight_duration", "sunshine_duration", "uv_index_max",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "relative_humidity_2m_mean", "surface_pressure_mean",
    ]
    OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_TIMEZONE = "Asia/Ho_Chi_Minh"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

FORECAST_WEEKS = 12
HISTORICAL_WEEKS = 12
DEFAULT_N_PROVINCES = 25
YEAR_START = 1994   # earliest year in OpenDengue data
YEAR_END = 2010     # latest year in OpenDengue data
RISK_HIGH = 0.70
RISK_MEDIUM = 0.50

# ---------------------------------------------------------------------------
# Province metadata (keyed by dataset name → (id, display_name, code, lat, lng))
# ---------------------------------------------------------------------------

PROVINCE_META: dict[str, tuple[str, str, str, float, float]] = {
    "HO CHI MINH CITY":  ("HCM",  "TP. Hồ Chí Minh", "79", 10.8231, 106.6297),
    "HA NOI CITY":        ("HN",   "Hà Nội",           "01", 21.0285, 105.8542),
    "DA NANG CITY":      ("DN",   "Đà Nẵng",         "48", 16.0544, 108.2022),
    "HAI PHONG CITY":    ("HP",   "Hải Phòng",        "31", 20.8449, 106.6881),
    "CAN THO CITY":      ("CT",   "Cần Thơ",          "92", 10.0452, 105.7469),
    "KHANH HOA":         ("KH",   "Khánh Hòa",        "37", 12.2388, 109.1967),
    "THANH HOA":         ("TH",   "Thanh Hóa",       "38", 19.8075, 105.7850),
    "NGHE AN":           ("ND",   "Nghệ An",           "40", 18.6745, 105.7934),
    "BINH DINH":         ("BD",   "Bình Định",         "52", 13.7833, 109.2167),
    "LONG AN":           ("LA",   "Long An",            "62", 10.6929, 106.3686),
    "THAI BINH":         ("TB",   "Thái Bình",        "34", 20.4453, 106.3139),
    "BAC NINH":          ("BN",   "Bắc Ninh",        "27", 21.1214, 106.1131),
    "QUANG NINH":        ("QNH",  "Quảng Ninh",        "32", 21.3901, 107.0467),
    "TAY NINH":          ("TNH",  "Tây Ninh",         "72", 11.3350, 106.0981),
    "BA RIA-VUNG TAU":   ("VT",   "Bà Rịa - Vũng Tàu","77", 10.4806, 107.1791),
    "DAK LAK":           ("DL",   "Đắk Lắk",         "66", 12.7100, 108.2378),
    "LAM DONG":          ("LDD",  "Lâm Đồng",         "68", 11.9388, 108.4419),
    "SOC TRANG":          ("SG",   "Sóc Trăng",         "94",  9.6029, 105.9739),
    "BAC LIEU":          ("BL",   "Bạc Liêu",         "95",  9.2941, 105.7268),
    "KIEN GIANG":        ("KG",   "Kiên Giang",         "91", 10.0440, 105.0800),
    "THUA THIEN - HUE": ("TTH",  "Thừa Thiên Huế",    "46", 16.4677, 107.5795),
    "AN GIANG":          ("AG",   "An Giang",          "80", 10.5211, 105.1219),
    "DONG THAP":         ("DDT",  "Đồng Tháp",        "68", 10.7326, 105.6937),
    "BINH THUAN":        ("BTH",  "Bình Thuận",         "52", 10.9294, 109.0284),
    "NINH THUAN":        ("NT",   "Ninh Thuận",        "58", 11.5763, 108.8864),
    "GIA LAI":           ("GLA",  "Gia Lai",          "52", 13.8070, 108.1094),
    "DAK NONG":          ("DNG2", "Đắk Nông",        "67", 12.2675, 107.6099),
    "BAC GIANG":          ("BGG",  "Bắc Giang",        "24", 21.2744, 106.2141),
    "HUNG YEN":          ("HYN",  "Hưng Yên",         "33", 20.6464, 105.5251),
    "SON LA":             ("SLA",  "Sơn La",           "52", 21.1151, 103.9223),
    "LAI CHAU":          ("LCI",  "Lai Châu",          "02", 22.3878, 103.3616),
    "HA GIANG":          ("HGG",  "Hà Giang",           "02", 22.8233, 104.0234),
    "TUYEN QUANG":       ("TNG",  "Tuyên Quang",       "34", 21.8190, 105.2140),
    "YEN BAI":           ("YB",   "Yên Bái",           "15", 21.7168, 105.0670),
    "THAI NGUYEN":       ("TNH2", "Thái Nguyên",       "34", 21.5941, 105.8480),
    "LANG SON":           ("LS",   "Lạng Sơn",         "20", 21.8537, 106.0177),
    "HAI DUONG":         ("HD",   "Hải Dương",        "30", 20.9371, 106.3007),
    "NINH BINH":         ("NB",   "Ninh Bình",          "18", 20.1091, 105.4311),
    "HA TINH":           ("HT",   "Hà Tĩnh",         "42", 18.3595, 105.6423),
    "QUANG BINH":         ("QBI",  "Quảng Bình",         "44", 17.4685, 106.6207),
    "QUANG NGAI":        ("QNG",  "Quảng Ngãi",       "48", 15.1213, 108.7925),
    "QUANG NAM":         ("QNM",  "Quảng Nam",         "46", 15.5730, 108.4742),
    "QUANG TRI":         ("QT",   "Quảng Trị",         "46", 16.7406, 106.4123),
    "PHU YEN":           ("PY",   "Phú Yên",          "52", 13.0887, 108.4251),
    "KON TUM":           ("KTM",  "Kon Tum",            "58", 14.3548, 108.0305),
    "BINH DUONG":        ("BDG",  "Bình Dương",        "70", 11.0695, 106.6508),
    "BINH PHUOC":        ("BP",   "Bình Phước",        "70", 11.5333, 106.8825),
    "DONG NAI":          ("DND",  "Đồng Nai",         "75", 11.0686, 107.0071),
    "BEN TRE":           ("BTR",  "Bến Tre",           "83", 10.2477, 105.7877),
    "TRA VINH":          ("TVN",  "Trà Vinh",          "84",  9.8043, 105.9792),
    "VINH LONG":         ("VLO",  "Vĩnh Long",          "86", 10.2545, 105.9729),
    "HAU GIANG":         ("HAG",  "Hậu Giang",         "93",  9.6058, 105.8204),
    "CA MAU":            ("CMA",  "Cà Mau",            "96",  9.1875, 104.9874),
    "TIEN GIANG":        ("TGI",  "Tiền Giang",       "82", 10.3747, 106.3647),
    "DIEN BIEN":         ("DBI",  "Điện Biên",        "02", 21.0125, 103.0208),
    "BAC KAN":           ("BKN",  "Bắc Kạn",           "24", 22.1476, 105.8388),
    "LAO CAI":          ("LCI2", "Lào Cai",            "02", 22.4866, 103.8667),
}


# ---------------------------------------------------------------------------
# REUSE FROM hination/features/feature_engineering.py
# compute_api() — REUSED AS-IS
# get_monsoon_phase() — REUSED WITH ADAPTATION (broadened for all VN regions)
# ---------------------------------------------------------------------------

MONSOON_ONSET_DAY = 120   # ~Apr 30 (earlier for southern VN)
MONSOON_END_DAY = 280     # ~Oct 7


def compute_api(precip_history: list[float], decay: float = 0.85) -> float:
    """REUSED FROM hination/features/feature_engineering.py
    Antecedent Precipitation Index: Σ P_t * decay^(t)
    """
    api = 0.0
    for i, p in enumerate(precip_history):
        api += p * (decay ** i)
    return api


def get_monsoon_phase(day_of_year: int) -> str:
    """ADAPTED FROM hination (generalized for full Vietnam).
    Vietnam spans ~8°N–23°N; southern VN monsoon onset ~May, northern ~Jun.
    Uses midpoint values.
    """
    if day_of_year < MONSOON_ONSET_DAY - 30:
        return "dry"
    elif day_of_year < MONSOON_ONSET_DAY:
        return "pre_monsoon"
    elif day_of_year <= MONSOON_END_DAY:
        return "monsoon"
    elif day_of_year <= MONSOON_END_DAY + 30:
        return "post_monsoon"
    else:
        return "dry"


# ---------------------------------------------------------------------------
# NEW: Open-Meteo Air Quality Provider
# Endpoint: https://air-quality-api.open-meteo.com/v1/air-quality
# Coverage: ~2022–present (historical), FREE, no auth
# ---------------------------------------------------------------------------

AQ_BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQ_HOURLY_VARS = ["pm10", "pm2_5", "carbon_monoxide",
                   "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
AQ_DAILY_VARS = ["pm10", "pm2_5", "nitrogen_dioxide", "ozone"]  # available as daily agg


class OpenMeteoAirQualityProvider:
    """
    NEW PROVIDER — not in hination.
    Fetches historical air quality data from Open-Meteo Air Quality API.
    Data available from ~2022 onward.
    
    Cache: data/aq_cache/{lat}_{lon}_{date}.json (1 file/day/location)
    """

    def __init__(self, cache_dir: Path, rate_limit_delay: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dengue-q/1.0"})

    def _day_cache_path(self, lat: float, lon: float, dt: date) -> Path:
        return self.cache_dir / f"{lat:.4f}_{lon:.4f}_{dt.isoformat()}.json"

    def _is_cached(self, lat: float, lon: float, dt: date) -> bool:
        return self._day_cache_path(lat, lon, dt).exists()

    def fetch_daily(self, lat: float, lon: float, dt: date) -> dict | None:
        """Fetch 1 day of AQ data (daily aggregates)."""
        cache_path = self._day_cache_path(lat, lon, dt)
        if cache_path.exists():
            with cache_path.open() as f:
                return json.load(f)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": dt.isoformat(),
            "end_date": dt.isoformat(),
            "daily": ",".join(AQ_DAILY_VARS),
            "timezone": OPEN_METEO_TIMEZONE,
        }
        try:
            resp = self.session.get(AQ_BASE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                if daily.get("time"):
                    result = {
                        "date": dt.isoformat(),
                        "lat": lat, "lon": lon,
                        "pm10": daily.get("pm10", [None])[0],
                        "pm2_5": daily.get("pm2_5", [None])[0],
                        "no2": daily.get("nitrogen_dioxide", [None])[0],
                        "o3": daily.get("ozone", [None])[0],
                    }
                    with cache_path.open("w") as f:
                        json.dump(result, f)
                    time.sleep(self.rate_limit_delay)
                    return result
        except Exception:
            pass
        return None

    def fetch_range(
        self, lat: float, lon: float,
        start_date: date, end_date: date,
    ) -> list[dict]:
        """Fetch daily AQ for a date range, skipping already-cached days."""
        results = []
        cur = start_date
        while cur <= end_date:
            if not self._is_cached(lat, lon, cur):
                rec = self.fetch_daily(lat, lon, cur)
                if rec:
                    results.append(rec)
            cur += timedelta(days=1)
        return results


# ---------------------------------------------------------------------------
# NEW: Climate provider wrapper
# ---------------------------------------------------------------------------

def _get_climate_provider() -> Any:
    """Get or create the Open-Meteo Historical provider."""
    if _HINATION_AVAILABLE:
        return OpenMeteoHistoricalProvider(cache_dir=CLIMATE_CACHE, rate_limit_delay=1.0)
    # Bundled fallback — minimal reimplementation
    return _BundledClimateProvider(cache_dir=CLIMATE_CACHE)


class _BundledClimateProvider:
    """
    Fallback climate provider — implements batch fetching pattern from hination.
    Makes 1 API call per year per location (not 1 per day).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "dengue-q/1.0"})

    def _cache_path(self, lat: float, lon: float, dt: date) -> Path:
        return self.cache_dir / f"{lat:.4f}_{lon:.4f}_{dt.isoformat()}.json"

    def _is_cached(self, lat: float, lon: float, dt: date) -> bool:
        return self._cache_path(lat, lon, dt).exists()

    def fetch_range(self, lat: float, lon: float,
                    start: date, end: date) -> list[dict]:
        """Batch-fetch by year — 1 call per year, cache each day."""
        records = []
        for year in range(start.year, end.year + 1):
            yr_start = date(year, 1, 1) if year == start.year else date(year, 1, 1)
            yr_end = date(year, 12, 31) if year == end.year else date(year, 12, 31)

            # Check how many days are already cached
            missing = 0
            cur = yr_start
            while cur <= yr_end:
                if not self._is_cached(lat, lon, cur):
                    missing += 1
                cur += timedelta(days=1)

            if missing == 0:
                # All cached — load from disk
                cur = yr_start
                while cur <= yr_end:
                    cp = self._cache_path(lat, lon, cur)
                    if cp.exists():
                        with cp.open() as f:
                            recs = json.load(f).get("records", [])
                            records.extend(recs)
                    cur += timedelta(days=1)
                continue

            # Fetch batch for the year
            batch_recs = self._fetch_year_batch(lat, lon, yr_start, yr_end)
            for rec in batch_recs:
                dt_str = rec["date"]
                cp = self._cache_path(lat, lon, date.fromisoformat(dt_str))
                if not cp.exists():
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    with cp.open("w") as f:
                        json.dump({"records": [rec]}, f)
                records.append(rec)
            time.sleep(0.5)  # rate limit
        return records

    def _fetch_year_batch(self, lat: float, lon: float,
                           start: date, end: date) -> list[dict]:
        """Fetch 1 full year in 1 API call, split into daily records."""
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": ",".join(OPEN_METEO_DAILY_VARS),
            "timezone": OPEN_METEO_TIMEZONE,
        }
        try:
            resp = self.session.get(OPEN_METEO_BASE, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                if not daily.get("time"):
                    return []
                n = len(daily["time"])
                recs = []
                for i in range(n):
                    dt_str = daily["time"][i]
                    rec = {
                        "date": dt_str,
                        "year": int(dt_str[:4]),
                        "month": int(dt_str[5:7]),
                        "day": int(dt_str[8:10]),
                    }
                    for v in OPEN_METEO_DAILY_VARS:
                        vals = daily.get(v, [None])
                        rec[v] = float(vals[i]) if i < len(vals) and vals[i] is not None else None
                    recs.append(rec)
                return recs
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Step 1: Load dengue data
# ---------------------------------------------------------------------------

def load_dengue() -> pd.DataFrame:
    print("[1/7] Loading OpenDengue Vietnam data...")
    df = pd.read_csv(SRC_DATA)
    vn = df[df["adm_0_name"] == "VIET NAM"].copy()
    print(f"  → {len(vn):,} rows, {vn['adm_1_name'].nunique()} provinces, "
          f"{int(vn['Year'].min())}–{int(vn['Year'].max())}")
    return vn


def rank_provinces(vn: pd.DataFrame, n: int) -> list[str]:
    totals = vn.groupby("adm_1_name")["dengue_total"].sum().sort_values(ascending=False)
    top = totals.head(n)
    print(f"  → Top {n} provinces by total cases:")
    for name, total in top.items():
        print(f"      {name}: {int(total):,} cases")
    return list(top.index)


# ---------------------------------------------------------------------------
# Step 2: Fetch climate data for each province
# ---------------------------------------------------------------------------

import requests

# Epi week of month (deterministic)
_EPI_WEEK_OF_MONTH = {
    1:  1,  2:  5,  3:  9,  4: 14,  5: 18,  6: 23,
    7: 27,  8: 32,  9: 36, 10: 40, 11: 45, 12: 49,
}


def _epi_week_of_month(month: int) -> int:
    return _EPI_WEEK_OF_MONTH.get(month, 1)


def fetch_climate_for_province(
    prov_name: str, lat: float, lon: float,
    start_year: int, end_year: int,
    climate_provider,
    aq_provider,
) -> pd.DataFrame:
    """Fetch daily climate (and AQ where available) for a province."""
    print(f"  Fetching climate for {prov_name} ({lat:.4f}, {lon:.4f}) "
          f"{start_year}–{end_year}...")

    records = []
    for year in range(start_year, end_year + 1):
        start_d = date(year, 1, 1)
        end_d = date(year, 12, 31)
        daily = climate_provider.fetch_range(lat, lon, start_d, end_d)
        records.extend(daily)

    if not records:
        print(f"  WARNING: No climate data fetched for {prov_name}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="first")

    # Fill missing values via linear interpolation
    numeric_cols = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "relative_humidity_2m_mean",
        "shortwave_radiation_sum", "surface_pressure_mean",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "et0_fao_evapotranspiration", "uv_index_max",
        "daylight_duration",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear")

    print(f"  → {len(df)} daily records, years {df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Feature engineering (adapted from hination, dengue-enhanced)
# ---------------------------------------------------------------------------

def compute_dengue_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add dengue-specific climate features to daily dataframe.
    
    ADAPTED FROM: hination/features/feature_engineering.py
    - REUSED: compute_api(), get_monsoon_phase()
    - ADDED: dengue-specific lags, temp_optimal_window, PM2.5 features
    """
    if daily_df.empty:
        return daily_df

    df = daily_df.copy().sort_values("date").reset_index(drop=True)

    # --- Basic weather features ---
    df["temp_avg"] = pd.to_numeric(df.get("temperature_2m_mean", 28.0), errors="coerce").fillna(28.0)
    df["temp_max"] = pd.to_numeric(df.get("temperature_2m_max", 32.0), errors="coerce").fillna(32.0)
    df["temp_min"] = pd.to_numeric(df.get("temperature_2m_min", 24.0), errors="coerce").fillna(24.0)
    df["rainfall_mm"] = pd.to_numeric(df.get("precipitation_sum", 0.0), errors="coerce").fillna(0.0)
    df["humidity_pct"] = pd.to_numeric(df.get("relative_humidity_2m_mean", 75.0), errors="coerce").fillna(75.0)
    df["wind_speed_max"] = pd.to_numeric(df.get("wind_speed_10m_max", 10.0), errors="coerce").fillna(10.0)
    df["pressure_mean"] = pd.to_numeric(df.get("surface_pressure_mean", 1013.0), errors="coerce").fillna(1013.0)
    df["solar_radiation"] = pd.to_numeric(df.get("shortwave_radiation_sum", 15.0), errors="coerce").fillna(15.0)
    df["uv_index"] = pd.to_numeric(df.get("uv_index_max", 10.0), errors="coerce").fillna(10.0)
    df["daylight_hours"] = pd.to_numeric(df.get("daylight_duration", 12.0), errors="coerce").fillna(12.0)

    # --- Accumulated rainfall (REUSED from hination pattern) ---
    precip = df["rainfall_mm"].values
    df["precip_1d"] = precip
    df["precip_3d"] = pd.Series(precip).rolling(3, min_periods=1).sum().values
    df["precip_7d"] = pd.Series(precip).rolling(7, min_periods=1).sum().values
    df["precip_14d"] = pd.Series(precip).rolling(14, min_periods=1).sum().values
    df["precip_30d"] = pd.Series(precip).rolling(30, min_periods=1).sum().values

    # --- API (REUSED from hination compute_api) ---
    def rolling_api(precip_arr: np.ndarray, days: int, decay: float = 0.85) -> np.ndarray:
        """Compute rolling API using exponential decay."""
        result = np.full(len(precip_arr), np.nan)
        for i in range(len(precip_arr)):
            window = precip_arr[max(0, i - days + 1):i + 1]
            result[i] = compute_api(list(window[::-1]), decay)
        return result

    df["api_7d"] = rolling_api(precip, 7)
    df["api_14d"] = rolling_api(precip, 14)

    # --- Dengue-specific lags (NEW) ---
    # Aedes aegypti egg→adult ~10-14 days; virus incubation ~8-12 days
    temp = df["temp_avg"].values
    rain = df["rainfall_mm"].values
    humid = df["humidity_pct"].values

    def rolling_mean(arr: np.ndarray, days: int) -> np.ndarray:
        return pd.Series(arr).rolling(days, min_periods=1).mean().values

    df["temp_lag_1w"] = pd.Series(temp).shift(7).bfill().values
    df["temp_lag_2w"] = pd.Series(temp).shift(14).bfill().values
    df["temp_lag_4w"] = pd.Series(temp).shift(28).bfill().values

    df["rain_lag_2w"] = pd.Series(rain).shift(14).fillna(0).rolling(14, min_periods=1).sum().values
    df["rain_lag_4w"] = pd.Series(rain).shift(28).fillna(0).rolling(28, min_periods=1).sum().values

    df["humid_4w_mean"] = rolling_mean(humid, 28)

    # --- Temperature optimal window (NEW) ---
    # Aedes aegypti thrives at 25-32°C; <18°C or >35°C = reduced vector competence
    df["temp_optimal_window"] = (
        (df["temp_avg"] >= 25.0) & (df["temp_avg"] <= 32.0)
    ).astype(int)

    # --- Anomaly z-scores ---
    # Compare current week to historical mean for same week-of-year
    df["day_of_year"] = df["date"].dt.dayofyear
    by_woy = df.groupby("day_of_year")["temp_avg"].transform("mean")
    std_woy = df.groupby("day_of_year")["temp_avg"].transform("std").fillna(1.0) + 1e-6
    df["temp_anomaly_zscore"] = (df["temp_avg"] - by_woy) / std_woy

    by_woy_p = df.groupby("day_of_year")["rainfall_mm"].transform("mean")
    std_woy_p = df.groupby("day_of_year")["rainfall_mm"].transform("std").fillna(1.0) + 1e-6
    df["precip_anomaly_zscore"] = (df["rainfall_mm"] - by_woy_p) / std_woy_p

    # --- Seasonality ---
    df["sin_week"] = np.sin(2 * math.pi * df["day_of_year"] / 365.0)
    df["cos_week"] = np.cos(2 * math.pi * df["day_of_year"] / 365.0)
    df["monsoon_phase"] = df["day_of_year"].apply(get_monsoon_phase)

    # --- Air quality (where available — typically 2022+) ---
    if "pm2_5" in df.columns:
        df["pm25_avg"] = df["pm2_5"]
        df["pm10_avg"] = df["pm10"]
        df["no2_avg"] = df["no2"]
        df["o3_avg"] = df["o3"]
    else:
        df["pm25_avg"] = np.nan
        df["pm10_avg"] = np.nan
        df["no2_avg"] = np.nan
        df["o3_avg"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Step 4: Daily → Weekly aggregation
# ---------------------------------------------------------------------------

def daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily features to weekly, aligned to WHO epi weeks."""
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy().sort_values("date")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Epidemic week approximation: map month to epi week
    # epi_year: Jan-Jun = calendar year, Jul-Dec = calendar year + 1
    df["epi_year"] = df.apply(
        lambda r: r["year"] if r["month"] <= 6 else r["year"] + 1, axis=1
    )
    df["epi_week"] = df["month"].map(_EPI_WEEK_OF_MONTH)

    # Group features by (epi_year, epi_week) — take mean for continuous vars, sum for precip
    agg_dict = {
        "temp_avg": "mean", "temp_max": "mean", "temp_min": "mean",
        "rainfall_mm": "sum", "humidity_pct": "mean",
        "precip_1d": "sum", "precip_3d": "sum", "precip_7d": "sum",
        "precip_14d": "sum", "precip_30d": "sum",
        "api_7d": "last", "api_14d": "last",
        "temp_lag_1w": "last", "temp_lag_2w": "last", "temp_lag_4w": "last",
        "rain_lag_2w": "last", "rain_lag_4w": "last",
        "humid_4w_mean": "last",
        "temp_optimal_window": "mean",
        "temp_anomaly_zscore": "mean", "precip_anomaly_zscore": "mean",
        "sin_week": "mean", "cos_week": "mean",
        "wind_speed_max": "mean", "pressure_mean": "mean",
        "solar_radiation": "mean", "uv_index": "mean",
        "pm25_avg": "mean", "pm10_avg": "mean", "no2_avg": "mean", "o3_avg": "mean",
        "daylight_hours": "mean",
    }

    weekly = df.groupby(["epi_year", "epi_week"]).agg(agg_dict).reset_index()
    weekly["epi_year_wk"] = weekly["epi_year"].astype(str) + "-W" + weekly["epi_week"].astype(str).str.zfill(2)
    return weekly


# ---------------------------------------------------------------------------
# Step 5: Monthly → Weekly dengue (as before)
# ---------------------------------------------------------------------------

def monthly_dengue_to_weekly(
    monthly_cases: np.ndarray,
    dates: list[datetime],
) -> tuple[list[int], list[float]]:
    """Convert monthly dengue cases to weekly (4 weeks per month)."""
    buckets: dict[tuple[int, int], float] = {}
    for val, dt in zip(monthly_cases, dates):
        month = dt.month
        year = dt.year
        ewk = _EPI_WEEK_OF_MONTH.get(month, 1)
        for offset in range(4):
            wk = ((ewk + offset - 1) % 52) + 1
            key = (year, month, offset)
            buckets[key] = (buckets.get(key, 0.0) + val / 4.0)

    sorted_keys = sorted(buckets.keys())
    weeks = [buckets[k][0] if isinstance(buckets[k], tuple) else _EPI_WEEK_OF_MONTH.get(sorted_keys[0][1], 1) for _ in sorted_keys]
    # Rebuild correctly
    week_list = []
    case_list = []
    prev_year_month = None
    for key in sorted_keys:
        year, month, offset = key
        ewk = ((_EPI_WEEK_OF_MONTH.get(month, 1) + offset - 1) % 52) + 1
        week_list.append(ewk)
        if isinstance(buckets[key], tuple):
            case_list.append(buckets[key][1])
        else:
            case_list.append(buckets[key])
    return week_list, case_list


def _fix_monthly_to_weekly(monthly_cases, dates):
    """Proper monthly → weekly conversion."""
    buckets = {}
    for val, dt in zip(monthly_cases, dates):
        month = dt.month
        year = dt.year
        ewk = _EPI_WEEK_OF_MONTH.get(month, 1)
        for offset in range(4):
            wk = ((ewk + offset - 1) % 52) + 1
            key = (year, month, offset)
            if key not in buckets:
                buckets[key] = 0.0
            buckets[key] += val / 4.0

    sorted_keys = sorted(buckets.keys())
    week_nums = []
    case_counts = []
    for key in sorted_keys:
        _, month, offset = key
        ewk = ((_EPI_WEEK_OF_MONTH.get(month, 1) + offset - 1) % 52) + 1
        week_nums.append(ewk)
        case_counts.append(buckets[key])
    return week_nums, case_counts


# ---------------------------------------------------------------------------
# Step 6: SARIMA forecasting with climate features
# ---------------------------------------------------------------------------

def fit_sarima(series: np.ndarray, steps: int, seed: int = 42) -> np.ndarray:
    """SARIMA(1,1,1)(1,1,1,52) forecast. Fallback to seasonal naive."""
    try:
        import statsmodels.api as sm

        n = len(series)
        if n < 24:
            return _seasonal_naive(series, steps, seed)

        period = 52
        padded = np.tile(series[-period:], (n // period + 2))[:n]
        model = sm.tsa.SARIMAX(
            padded,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),   # 12-week seasonal period
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=True,
        )
        fit = model.fit(disp=False, maxiter=200)
        return np.maximum(fit.forecast(steps=steps), 0.0)
    except Exception:
        return _seasonal_naive(series, steps, seed)


def _seasonal_naive(series: np.ndarray, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period = min(len(series), 52)
    pattern = series[-period:]
    return np.array([
        max(pattern[i % period] + rng.normal(0, max(pattern[i % period] * 0.05, 1)), 0)
        for i in range(steps)
    ])


# ---------------------------------------------------------------------------
# Step 7: Risk scoring with climate-enhanced features
# ---------------------------------------------------------------------------

def compute_risk_score_climate(
    all_cases: np.ndarray,
    all_weeks: list[int],
    weekly_df: pd.DataFrame,    # climate features per week
    forecast: np.ndarray,
    last_week: int,
) -> tuple[float, str, list[dict]]:
    """Risk score using seasonal baseline + climate features."""
    cases = np.array(all_cases)
    n_hist = min(HISTORICAL_WEEKS, len(cases))

    # --- Seasonal baseline from dengue cases ---
    wk_to_cases: dict[int, list[float]] = {}
    for wk, c in zip(all_weeks, cases):
        wk_to_cases.setdefault(wk, []).append(c)

    season_mean: dict[int, float] = {wk: np.mean(v) for wk, v in wk_to_cases.items()}
    season_std: dict[int, float] = {wk: np.std(v) + 1e-6 for wk, v in wk_to_cases.items()}

    recent = cases[-n_hist:]
    recent_wks = all_weeks[-n_hist:]

    z_scores = []
    for wk, c in zip(recent_wks, recent):
        m = season_mean.get(wk, float(np.mean(recent)))
        s = season_std.get(wk, float(np.std(recent)) + 1e-6)
        z_scores.append((c - m) / s)

    avg_z = float(np.mean(z_scores))
    risk = float(np.clip((avg_z + 2) / 4, 0, 1))

    # Climate modifier: if recent humidity + temp + rain are elevated, boost risk
    if not weekly_df.empty and len(weekly_df) >= n_hist:
        clim_recent = weekly_df.tail(n_hist)
        humid_mean = float(clim_recent["humidity_pct"].mean()) if "humidity_pct" in clim_recent.columns else 75.0
        rain_mean = float(clim_recent["precip_7d"].mean()) if "precip_7d" in clim_recent.columns else 10.0
        opt_frac = float(clim_recent["temp_optimal_window"].mean()) if "temp_optimal_window" in clim_recent.columns else 0.5

        # Climate boost: elevated humidity (70%+) + rain + optimal temp → higher risk
        climate_boost = 0.0
        if humid_mean > 70:
            climate_boost += 0.05
        if rain_mean > 20:
            climate_boost += 0.05
        if opt_frac > 0.7:
            climate_boost += 0.03

        risk = float(np.clip(risk + climate_boost, 0, 1))

    # Trend
    if len(recent) >= 8:
        r4 = float(np.mean(recent[-4:]))
        p4 = float(np.mean(recent[-8:-4]))
        if r4 > p4 * 1.1:
            trend = "up"
        elif r4 < p4 * 0.9:
            trend = "down"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Build weeklyData
    weekly_data: list[dict] = []
    for wk, c in zip(recent_wks, recent):
        m = season_mean.get(wk, float(np.mean(recent)))
        s = season_std.get(wk, float(np.std(recent)) + 1e-6)
        w_risk = float(np.clip((c - m) / (4 * s) + 0.5, 0, 1))
        weekly_data.append({"week": wk, "cases": int(round(c)), "risk": round(w_risk, 2)})

    for j, fc in enumerate(forecast):
        # Forecast weeks increment sequentially from last_week without wrap.
        # Front-end displays week numbers as-is (no modulo), so W50, W51, W52, W53...
        # are valid epi-week identifiers (ISO allows 1-53, standard 1-52).
        wk = last_week + j + 1
        m = season_mean.get(wk, float(np.mean(cases)))
        s = season_std.get(wk, float(np.std(cases)) + 1e-6)
        w_z = (fc - m) / s
        w_risk = float(np.clip((w_z + 2) / 4, 0, 1))
        weekly_data.append({
            "week": wk, "cases": int(round(fc)),
            "risk": round(w_risk, 2), "predicted": 1,
        })

    return risk, trend, weekly_data


# ---------------------------------------------------------------------------
# Province metadata
# ---------------------------------------------------------------------------

def find_meta(province_name: str) -> tuple[str, str, str, float, float]:
    if province_name in PROVINCE_META:
        return (*PROVINCE_META[province_name],)
    stripped = province_name.replace(" CITY", "")
    for key, val in PROVINCE_META.items():
        if key.replace(" CITY", "") == stripped:
            return (*val,)
    for key, val in PROVINCE_META.items():
        kn = key.replace(" CITY", "").replace("-", " ")
        pn = province_name.replace(" CITY", "").replace("-", " ")
        if kn in pn or pn in kn:
            return (*val,)
    suffix = province_name[:4].upper().replace(" ", "_")
    return suffix, province_name.title(), "??", 16.0, 108.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    global SEED, FORECAST_WEEKS, HISTORICAL_WEEKS
    t0 = time.time()

    _def_seed = SEED
    _def_fw = FORECAST_WEEKS
    _def_hw = HISTORICAL_WEEKS

    parser = argparse.ArgumentParser(
        description="Generate dengue hotspots with climate + AQ features"
    )
    parser.add_argument("--n-provinces", type=int, default=DEFAULT_N_PROVINCES)
    parser.add_argument("--forecast-weeks", type=int, default=_def_fw)
    parser.add_argument("--seed", type=int, default=_def_seed)
    parser.add_argument("--year-start", type=int, default=YEAR_START,
                        help="First year for climate data fetch")
    args = parser.parse_args()

    SEED = args.seed
    FORECAST_WEEKS = args.forecast_weeks
    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 70)
    print("Dengue Hotspot Forecast — Climate + Air Quality Enhanced Pipeline")
    print(f"  n_provinces={args.n_provinces}, forecast_weeks={FORECAST_WEEKS}, "
          f"climate_years={args.year_start}–{YEAR_END}")
    print("=" * 70)

    # Step 1: Dengue data
    vn = load_dengue()
    top_provinces = rank_provinces(vn, args.n_provinces)

    # Step 2: Climate provider
    climate_provider = _get_climate_provider()
    aq_provider = OpenMeteoAirQualityProvider(cache_dir=AQ_CACHE, rate_limit_delay=1.5)
    print(f"\n[2/7] Climate providers initialized")
    print(f"  Climate cache: {CLIMATE_CACHE}")
    print(f"  AQ cache: {AQ_CACHE}")
    print(f"  NOTE: Air Quality API available from ~2022 onward; "
          f"dengue data ends {YEAR_END} — AQ will be unavailable for this analysis.")

    # Step 3-5: Process each province
    print(f"\n[3-5/7] Fetching climate + dengue data for {len(top_provinces)} provinces...")
    province_records = []
    hotspots_list = []
    LAST_WEEK = 49  # default; will be overridden per-province from actual data

    for i, prov_name in enumerate(top_provinces):
        pid, pname, code, lat, lng = find_meta(prov_name)
        print(f"\n  [{i+1}/{len(top_provinces)}] {pname} (ID:{pid})")

        # Dengue time series
        pdata = vn[vn["adm_1_name"] == prov_name].copy()
        pdata = pdata.sort_values("calendar_start_date")
        pdata = pdata.drop_duplicates(subset="calendar_start_date", keep="first")
        monthly_cases = pdata["dengue_total"].fillna(0).values
        raw_dates = pd.to_datetime(pdata["calendar_start_date"].values)
        dates = [datetime(d.year, d.month, d.day) for d in raw_dates]
        dengue_weeks, dengue_cases = _fix_monthly_to_weekly(monthly_cases, dates)

        # --- Compute dynamic last_week from actual observed weeks ---
        # Group observed data: epidemic year ends around week 49 (Dec),
        # resumes week 1 (Jan). Use the maximum week in the last epidemic cycle.
        if dengue_weeks:
            # Sort and deduplicate
            unique_sorted = sorted(set(dengue_weeks))
            # Last observed week = max week number (accounting for year boundary)
            # If there are weeks near 1-9 (Jan-Mar), they're from next epi year
            # The "last week" for forecast anchor = the week before W1 appears
            last_wk = max(unique_sorted) if unique_sorted else LAST_WEEK
            # Guard: if max week is near 49 (Dec) or 52, that's the boundary
            # If we have weeks 1-9 from the "next" epi year at the tail,
            # the real last historical week should be from the tail of actual data
            prov_last_week = unique_sorted[-1] if unique_sorted else LAST_WEEK
        else:
            prov_last_week = LAST_WEEK

        # Fetch climate
        climate_df = fetch_climate_for_province(
            prov_name, lat, lng,
            args.year_start, YEAR_END,
            climate_provider, aq_provider,
        )

        # Feature engineering
        if not climate_df.empty:
            climate_df = compute_dengue_features(climate_df)
            weekly_df = daily_to_weekly(climate_df)
        else:
            weekly_df = pd.DataFrame()

        # Forecasting
        recent_cases = np.array(dengue_cases[-104:]) if len(dengue_cases) >= 104 else np.array(dengue_cases)
        forecast = fit_sarima(recent_cases, FORECAST_WEEKS, seed=SEED + i)

        # Risk scoring — pass prov_last_week for correct forecast wrap
        risk, trend, weekly_data = compute_risk_score_climate(
            all_cases=np.array(dengue_cases),
            all_weeks=dengue_weeks,
            weekly_df=weekly_df,
            forecast=forecast,
            last_week=prov_last_week,
        )

        # Deduplicate weekly_data by week, keep first occurrence (historical has priority)
        seen_weeks: set[int] = set()
        deduped: list[dict] = []
        for w in weekly_data:
            if w["week"] not in seen_weeks:
                seen_weeks.add(w["week"])
                deduped.append(w)
        weekly_data = deduped

        total_cases = sum(w["cases"] for w in weekly_data[:HISTORICAL_WEEKS])

        record = {
            "id": pid, "name": pname, "code": code,
            "coordinates": {"lat": round(lat, 4), "lng": round(lng, 4)},
            "riskScore": round(risk, 2),
            "cases": total_cases,
            "trend": trend,
            "weeklyData": weekly_data,
        }
        province_records.append(record)

        if risk >= RISK_MEDIUM:
            intensity = "high" if risk >= RISK_HIGH else "medium"
            hotspots_list.append({
                "provinceId": pid, "provinceName": pname,
                "coordinates": [round(lat, 4), round(lng, 4)],
                "riskScore": round(risk, 2), "intensity": intensity,
            })

        print(f"    risk={risk:.2f}, trend={trend}, cases={total_cases}, "
              f"climate_records={len(climate_df)}")

    hotspots_list.sort(key=lambda x: x["riskScore"], reverse=True)

    # Step 6: Validation
    print(f"\n[6/7] Validation:")
    print(f"  Provinces: {len(province_records)}")
    print(f"  Hotspots (risk≥{RISK_MEDIUM}): {len(hotspots_list)}")
    for p in province_records:
        assert "id" in p and "name" in p and "coordinates" in p
        assert "riskScore" in p and "weeklyData" in p
        assert len(p["weeklyData"]) == HISTORICAL_WEEKS + FORECAST_WEEKS
    print("  Schema checks: PASSED")

    # Step 7: Save
    # Derive global last_week from the province with most complete data
    prov_last_weeks = []
    for p in province_records:
        for w in p["weeklyData"]:
            prov_last_weeks.append((w.get("predicted", 0), w["week"]))
    # Max forecast week = global last_week
    forecast_weeks = [w for pred, w in prov_last_weeks if pred == 1]
    observed_weeks = [w for pred, w in prov_last_weeks if pred == 0]
    global_last_week = max(forecast_weeks) if forecast_weeks else LAST_WEEK
    global_first_week = min(observed_weeks) if observed_weeks else 1

    data = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_source": (
                "OpenDengue V1.1 (TYCHO/Imperial College London) + "
                "Open-Meteo Historical ERA5 + Open-Meteo Air Quality API"
            ),
            "time_range": f"{HISTORICAL_WEEKS}W observed + {FORECAST_WEEKS}W SARIMA forecast",
            "last_week": global_last_week,
            "climate_vars": [
                "temp_avg", "temp_max", "temp_min", "rainfall_mm", "humidity_pct",
                "precip_7d", "precip_14d", "api_7d", "api_14d",
                "temp_lag_1w", "temp_lag_2w", "temp_lag_4w",
                "rain_lag_2w", "rain_lag_4w", "humid_4w_mean",
                "temp_optimal_window (25-32°C Aedes optimum)",
                "temp_anomaly_zscore", "precip_anomaly_zscore",
                "sin_week", "cos_week", "monsoon_phase",
                "pm25_avg (2022+ only)", "pm10_avg (2022+ only)",
                "no2_avg (2022+ only)", "o3_avg (2022+ only)",
            ],
        },
        "provinces": province_records,
        "hotspots": hotspots_list[:10],
    }

    print(f"\n[7/7] Saving to {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → {OUTPUT_FILE.stat().st_size:,} bytes")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"  Provinces: {len(province_records)}")
    print(f"  Hotspots: {len(hotspots_list)}")
    return data


if __name__ == "__main__":
    main()
