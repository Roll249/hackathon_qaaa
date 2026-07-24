#!/usr/bin/env python3
"""
generate_hotspots.py — Dengue Hotspot Forecast Pipeline
======================================================

End-to-end pipeline that:
  1. Loads real dengue case data (OpenDengue V1.1, Vietnam provinces)
  2. Converts monthly → weekly (epidemic-week aligned)
  3. Fits SARIMA per-province for 12-week-ahead forecasts
  4. Computes risk scores relative to same-epidemic-season historical baseline
  5. Outputs hotspots.json matching the web app schema exactly

Usage:
    python scripts/generate_hotspots.py [--n-provinces N] [--forecast-weeks W]

Output:
    ../q_dengue_epidemiology-main/web/public/data/hotspots.json

Dependencies: pandas, numpy, statsmodels
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DATA_DIR = Path(__file__).parent.parent.parent / "dengue_dataset"
SRC_DATA = DATA_DIR / "sea_dengue_admin1_month.csv"
OUTPUT_FILE = (
    Path(__file__).parent.parent.parent
    / "q_dengue_epidemiology-main"
    / "web"
    / "public"
    / "data"
    / "hotspots.json"
)

FORECAST_WEEKS = 12      # weeks to forecast
HISTORICAL_WEEKS = 12    # historical weeks in output weeklyData
DEFAULT_N_PROVINCES = 25
RISK_HIGH = 0.70
RISK_MEDIUM = 0.50

# ---- Province metadata keyed by dataset name → (id, display_name, code, lat, lng) ----
# id: matches the web app's existing Province.id values for known provinces
PROVINCE_META: dict[str, tuple[str, str, str, float, float]] = {
    # KEY = exact name as it appears in the OpenDengue CSV (uppercase, CITY suffix)
    "HO CHI MINH CITY":  ("HCM",  "TP. Hồ Chí Minh", "79", 10.8231, 106.6297),
    "HA NOI CITY":        ("HN",   "Hà Nội",           "01", 21.0285, 105.8542),
    "DA NANG CITY":      ("DN",   "Đà Nẵng",         "48", 16.0544, 108.2022),
    "HAI PHONG CITY":    ("HP",   "Hải Phòng",        "31", 20.8449, 106.6881),
    "CAN THO CITY":      ("CT",   "Cần Thơ",          "92", 10.0452, 105.7469),
    "KHANH HOA":         ("KH",   "Khánh Hòa",        "37", 12.2388, 109.1967),
    "THANH HOA":         ("TH",   "Thanh Hóa",        "38", 19.8075, 105.7850),
    "NGHE AN":           ("ND",   "Nghệ An",           "40", 18.6745, 105.7934),
    "BINH DINH":        ("BD",   "Bình Định",         "52", 13.7833, 109.2167),
    "LONG AN":           ("LA",   "Long An",            "62", 10.6929, 106.3686),
    "THAI BINH":        ("TB",   "Thái Bình",         "34", 20.4453, 106.3139),
    "BAC NINH":         ("BN",   "Bắc Ninh",         "27", 21.1214, 106.1131),
    "QUANG NINH":        ("QNH",  "Quảng Ninh",        "32", 21.3901, 107.0467),
    "TAY NINH":         ("TNH",  "Tây Ninh",          "72", 11.3350, 106.0981),
    "BA RIA-VUNG TAU":   ("VT",   "Bà Rịa - Vũng Tàu","77", 10.4806, 107.1791),
    "DAK LAK":          ("DL",   "Đắk Lắk",          "66", 12.7100, 108.2378),
    "LAM DONG":         ("LDD",  "Lâm Đồng",          "68", 11.9388, 108.4419),
    "SOC TRANG":         ("SG",   "Sóc Trăng",         "94",  9.6029, 105.9739),
    "BAC LIEU":          ("BL",   "Bạc Liêu",         "95",  9.2941, 105.7268),
    "KIEN GIANG":        ("KG",   "Kiên Giang",         "91", 10.0440, 105.0800),
    "THUA THIEN - HUE":  ("TTH",  "Thừa Thiên Huế",    "46", 16.4677, 107.5795),
    "AN GIANG":          ("AG",   "An Giang",          "80", 10.5211, 105.1219),
    "DONG THAP":         ("DDT",  "Đồng Tháp",        "68", 10.7326, 105.6937),
    "BINH THUAN":        ("BTH",  "Bình Thuận",        "52", 10.9294, 109.0284),
    "NINH THUAN":        ("NT",   "Ninh Thuận",        "58", 11.5763, 108.8864),
    "GIA LAI":          ("GLA",  "Gia Lai",           "52", 13.8070, 108.1094),
    "DAK NONG":         ("DNG2", "Đắk Nông",         "67", 12.2675, 107.6099),
    "BAC GIANG":         ("BGG",  "Bắc Giang",        "24", 21.2744, 106.2141),
    "HUNG YEN":         ("HYN",  "Hưng Yên",          "33", 20.6464, 105.5251),
    "SON LA":           ("SLA",  "Sơn La",            "52", 21.1151, 103.9223),
    "LAI CHAU":         ("LCI",  "Lai Châu",           "02", 22.3878, 103.3616),
    "HA GIANG":         ("HGG",  "Hà Giang",           "02", 22.8233, 104.0234),
    "TUYEN QUANG":      ("TNG",  "Tuyên Quang",        "34", 21.8190, 105.2144),
    "YEN BAI":          ("YB",   "Yên Bái",            "15", 21.7168, 105.0670),
    "THAI NGUYEN":      ("TNH2", "Thái Nguyên",        "34", 21.5941, 105.8480),
    "LANG SON":         ("LS",   "Lạng Sơn",          "20", 21.8537, 106.0177),
    "HAI DUONG":        ("HD",   "Hải Dương",         "30", 20.9371, 106.3007),
    "NINH BINH":        ("NB",   "Ninh Bình",           "18", 20.1091, 105.4311),
    "HA TINH":          ("HT",   "Hà Tĩnh",           "42", 18.3595, 105.6423),
    "QUANG BINH":       ("QBI",  "Quảng Bình",         "44", 17.4685, 106.6207),
    "QUANG NGAI":       ("QNG",  "Quảng Ngãi",        "48", 15.1213, 108.7925),
    "QUANG NAM":        ("QNM",  "Quảng Nam",          "46", 15.5730, 108.4742),
    "QUANG TRI":        ("QT",   "Quảng Trị",          "46", 16.7406, 106.4123),
    "PHU YEN":          ("PY",   "Phú Yên",           "52", 13.0887, 108.4251),
    "KON TUM":          ("KTM",  "Kon Tum",             "58", 14.3548, 108.0305),
    "BINH DUONG":       ("BDG",  "Bình Dương",        "70", 11.0695, 106.6508),
    "BINH PHUOC":       ("BP",   "Bình Phước",         "70", 11.5333, 106.8825),
    "DONG NAI":         ("DND",  "Đồng Nai",           "75", 11.0686, 107.0071),
    "BEN TRE":          ("BTR",  "Bến Tre",            "83", 10.2477, 105.7877),
    "TRA VINH":         ("TVN",  "Trà Vinh",           "84",  9.8043, 105.9792),
    "VINH LONG":        ("VLO",  "Vĩnh Long",          "86", 10.2545, 105.9729),
    "HAU GIANG":        ("HAG",  "Hậu Giang",          "93",  9.6058, 105.8204),
    "CA MAU":           ("CMA",  "Cà Mau",             "96",  9.1875, 104.9874),
    "TIEN GIANG":       ("TGI",  "Tiền Giang",         "82", 10.3747, 106.3647),
    "DIEN BIEN":        ("DBI",  "Điện Biên",          "02", 21.0125, 103.0208),
    "BAC KAN":          ("BKN",  "Bắc Kạn",            "24", 22.1476, 105.8388),
    "LAO CAI":         ("LCI2", "Lào Cai",             "02", 22.4866, 103.8667),
}


# ---------------------------------------------------------------------------
# Step 1 — Load real OpenDengue Vietnam data
# ---------------------------------------------------------------------------

def load_vietnam_data() -> pd.DataFrame:
    print("[1/6] Loading OpenDengue Vietnam data...")
    df = pd.read_csv(SRC_DATA)
    vn = df[df["adm_0_name"] == "VIET NAM"].copy()
    print(f"  → {len(vn):,} rows, {vn['adm_1_name'].nunique()} provinces, "
          f"{int(vn['Year'].min())}-{int(vn['Year'].max())}")
    return vn


def rank_provinces(vn: pd.DataFrame, n: int) -> list[str]:
    totals = vn.groupby("adm_1_name")["dengue_total"].sum().sort_values(ascending=False)
    top = totals.head(n)
    print(f"  → Top {n} provinces by total cases:")
    for name, total in top.items():
        print(f"      {name}: {int(total):,} cases")
    return list(top.index)


# ---------------------------------------------------------------------------
# Step 2 — Monthly → Weekly conversion
# ---------------------------------------------------------------------------

# Approximate WHO Epidemic Week start per month (Jan=W1)
_EPI_WEEK_OF_MONTH = {
    1:  1,  2:  5,  3:  9,  4: 14,  5: 18,  6: 23,
    7: 27,  8: 32,  9: 36, 10: 40, 11: 45, 12: 49,
}


def _epi_week(date: datetime) -> int:
    """Epi week ~month mapping (deterministic)."""
    return _EPI_WEEK_OF_MONTH.get(date.month, 1)


def monthly_to_weekly(cases: np.ndarray, dates: list[datetime]) -> tuple[list[int], list[float]]:
    """Convert monthly case series to weekly, distributed across 4 consecutive epi weeks.

    Each month's cases are spread evenly across the 4 epi weeks starting from that
    month's epi week (wrapped modulo 52). This produces a continuous weekly series.

    Returns:
        weeks:   list of epi week numbers (1-52), monotonically increasing
        counts:  list of weekly case counts (floats)
    """
    # Build (year, month) → (epi_week_start, cases/4)
    buckets: dict[tuple[int, int], tuple[int, float]] = {}
    for val, date in zip(cases, dates):
        month = date.month
        year = date.year
        ewk = _EPI_WEEK_OF_MONTH.get(month, 1)
        per_week = val / 4.0
        # Store 4 entries per month: (wk, wk+1, wk+2, wk+3) mod 52
        for offset in range(4):
            wk = ((ewk + offset - 1) % 52) + 1
            key = (year, month, offset)
            buckets[key] = (wk, per_week)

    # Sort by (year, month, offset) → produce continuous weekly series
    sorted_keys = sorted(buckets.keys())
    weeks = [buckets[k][0] for k in sorted_keys]
    counts = [buckets[k][1] for k in sorted_keys]
    return weeks, counts


# ---------------------------------------------------------------------------
# Step 3 — SARIMA forecasting
# ---------------------------------------------------------------------------

def fit_sarima(series: np.ndarray, steps: int, seed: int = 42) -> np.ndarray:
    """Fit SARIMA(1,1,1)(1,1,1,12) and forecast `steps` ahead.

    Falls back to seasonal naive if data is too short or fitting fails.
    """
    try:
        import statsmodels.api as sm

        n = len(series)
        if n < 24:
            return _seasonal_naive(series, steps, seed)

        # Pad to multiple of 52 for seasonal differencing
        period = 52
        reps = (n + period - 1) // period
        padded = np.tile(series[-period:], reps)[:n]

        model = sm.tsa.SARIMAX(
            padded,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=True,
        )
        fit = model.fit(disp=False, maxiter=200)
        return np.maximum(fit.forecast(steps=steps), 0.0)

    except Exception:
        return _seasonal_naive(series, steps, seed)


def _seasonal_naive(series: np.ndarray, steps: int, seed: int) -> np.ndarray:
    """Repeat the last 52-week seasonal pattern with small noise."""
    rng = np.random.default_rng(seed)
    period = min(len(series), 52)
    pattern = series[-period:]
    forecast = []
    for i in range(steps):
        base = pattern[i % period]
        noise = rng.normal(0, max(base * 0.05, 1))
        forecast.append(max(base + noise, 0))
    return np.array(forecast)


# ---------------------------------------------------------------------------
# Step 4 — Risk scoring (season-aware)
# ---------------------------------------------------------------------------

def compute_risk_score(
    all_cases: np.ndarray,
    all_weeks: list[int],
    forecast: np.ndarray,
    last_week: int,
) -> tuple[float, str, list[dict]]:
    """Compute risk score relative to same-epidemic-season historical baseline.

    Risk = z-score of current/recent cases vs same-epi-weeks in previous years.
    Risk is clipped to [0, 1].

    Args:
        all_cases:   full weekly case count history (for seasonal baseline)
        all_weeks:   epi week numbers for all_cases
        forecast:    FORECAST_WEEKS-ahead forecast
        last_week:   last observed epi week

    Returns:
        (risk_score, trend, weekly_data_list)
    """
    rng = np.random.default_rng(SEED)

    # --- Seasonal baseline: avg cases per epi week across all years ---
    week_to_cases: dict[int, list[float]] = {}
    for wk, c in zip(all_weeks, all_cases):
        week_to_cases.setdefault(wk, []).append(c)

    season_mean: dict[int, float] = {}
    season_std: dict[int, float] = {}
    for wk, vals in week_to_cases.items():
        season_mean[wk] = np.mean(vals)
        season_std[wk] = np.std(vals) + 1e-6

    # --- Recent window: last HISTORICAL_WEEKS of observed data ---
    recent = all_cases[-HISTORICAL_WEEKS:]
    recent_wks = all_weeks[-HISTORICAL_WEEKS:]

    # Compute z-score for each recent week vs its seasonal mean
    z_scores = []
    for wk, c in zip(recent_wks, recent):
        mean_s = season_mean.get(wk, float(np.mean(recent) + 1e-6))
        std_s = season_std.get(wk, float(np.std(recent) + 1e-6))
        z = (c - mean_s) / std_s
        z_scores.append(z)

    avg_z = float(np.mean(z_scores))
    risk = float(np.clip((avg_z + 2) / 4, 0, 1))

    # --- Trend: compare last 4 weeks vs previous 4 weeks ---
    recent4 = float(np.mean(recent[-4:])) if len(recent) >= 4 else float(np.mean(recent[-2:]))
    prev4 = float(np.mean(recent[-8:-4])) if len(recent) >= 8 else float(np.mean(recent[:4]))
    if recent4 > prev4 * 1.1:
        trend = "up"
    elif recent4 < prev4 * 0.9:
        trend = "down"
    else:
        trend = "stable"

    # --- Build weeklyData array ---
    weekly_data: list[dict] = []

    for wk, c in zip(recent_wks, recent):
        mean_s = season_mean.get(wk, float(np.mean(recent) + 1e-6))
        std_s = season_std.get(wk, float(np.std(recent) + 1e-6))
        w_risk = float(np.clip((c - mean_s) / (4 * std_s) + 0.5, 0, 1))
        weekly_data.append({"week": wk, "cases": int(round(c)), "risk": round(w_risk, 2)})

    # Append forecast weeks
    for j, fc in enumerate(forecast):
        wk = ((last_week + j + 1 - 1) % 52) + 1
        mean_s = season_mean.get(wk, float(np.mean(all_cases) + 1e-6))
        std_s = season_std.get(wk, float(np.std(all_cases) + 1e-6))
        w_z = (fc - mean_s) / std_s
        w_risk = float(np.clip((w_z + 2) / 4, 0, 1))
        weekly_data.append({
            "week": wk,
            "cases": int(round(fc)),
            "risk": round(w_risk, 2),
            "predicted": 1,
        })

    return risk, trend, weekly_data


# ---------------------------------------------------------------------------
# Step 5 — Province metadata lookup
# ---------------------------------------------------------------------------

def find_meta(province_name: str) -> tuple[str, str, str, float, float]:
    """Look up province metadata by dataset name."""
    # Try exact match (dataset uses uppercase, no CITY → strip it)
    norm = province_name.replace(" CITY", "").strip().upper()

    if province_name in PROVINCE_META:
        pid, pname, code, lat, lng = PROVINCE_META[province_name]
        return pid, pname, code, lat, lng

    # Try without CITY suffix
    stripped = province_name.replace(" CITY", "")
    for key, val in PROVINCE_META.items():
        if key.replace(" CITY", "") == stripped:
            return (*val,)

    # Partial match
    for key, val in PROVINCE_META.items():
        key_norm = key.replace(" CITY", "").replace("-", " ")
        prov_norm = province_name.replace(" CITY", "").replace("-", " ")
        if key_norm in prov_norm or prov_norm in key_norm:
            return (*val,)

    # Fallback
    suffix = province_name[:4].upper().replace(" ", "_")
    return suffix, province_name.title(), "??", 16.0, 108.0


# ---------------------------------------------------------------------------
# Step 6 — Build hotspots JSON
# ---------------------------------------------------------------------------

def build_hotspots(vn: pd.DataFrame, province_names: list[str],
                   n_provinces: int) -> dict:
    print(f"\n[3/6] Building hotspots for {n_provinces} provinces...")

    province_records: list[dict] = []
    hotspots_list: list[dict] = []

    # Reference last_week — use last epi week in dataset as "current"
    LAST_WEEK = 49  # Week 49 ≈ December (last month in dataset)

    for i, prov_name in enumerate(province_names):
        pdata = vn[vn["adm_1_name"] == prov_name].copy()
        pdata = pdata.sort_values("calendar_start_date")
        pdata = pdata.drop_duplicates(subset="calendar_start_date", keep="first")

        monthly_cases = pdata["dengue_total"].fillna(0).values
        raw_dates = pd.to_datetime(pdata["calendar_start_date"].values)
        dates = [datetime(d.year, d.month, d.day) for d in raw_dates]

        weekly_weeks, weekly_cases = monthly_to_weekly(monthly_cases, dates)

        if len(weekly_cases) < 26:
            print(f"  [{i+1}/{n_provinces}] SKIP {prov_name}: only {len(weekly_cases)} weeks")
            continue

        # Last 2 years of data (≈104 weeks)
        recent_cases = np.array(weekly_cases[-104:]) if len(weekly_cases) >= 104 else np.array(weekly_cases)
        recent_weeks = weekly_weeks[-len(recent_cases):]

        # Fit SARIMA
        forecast = fit_sarima(recent_cases, FORECAST_WEEKS, seed=SEED + i)

        # Compute risk (pass full history for seasonal baseline)
        risk, trend, weekly_data = compute_risk_score(
            all_cases=np.array(weekly_cases),
            all_weeks=weekly_weeks,
            forecast=forecast,
            last_week=LAST_WEEK,
        )

        total_cases = sum(w["cases"] for w in weekly_data[:HISTORICAL_WEEKS])
        pid, pname, code, lat, lng = find_meta(prov_name)

        record = {
            "id": pid,
            "name": pname,
            "code": code,
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
                "provinceId": pid,
                "provinceName": pname,
                "coordinates": [round(lat, 4), round(lng, 4)],
                "riskScore": round(risk, 2),
                "intensity": intensity,
            })

        print(f"  [{i+1}/{n_provinces}] {pname} (ID:{pid}): risk={risk:.2f}, "
              f"trend={trend}, cases={total_cases}")

    hotspots_list.sort(key=lambda x: x["riskScore"], reverse=True)

    return {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_source": (
                "OpenDengue V1.1 — Vietnam Admin1 Monthly "
                "(TYCHO/Imperial College London) via quantum-dengue-stpp pipeline"
            ),
            "time_range": (
                f"{HISTORICAL_WEEKS} weeks observed + {FORECAST_WEEKS} weeks forecast "
                "using SARIMA(1,1,1)(1,1,1,52)"
            ),
            "last_week": LAST_WEEK,
        },
        "provinces": province_records,
        "hotspots": hotspots_list[:10],
    }


# ---------------------------------------------------------------------------
# Step 7 — Save & validate
# ---------------------------------------------------------------------------

def save_output(data: dict, path: Path):
    print(f"\n[5/6] Saving to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size = path.stat().st_size
    print(f"  → {size:,} bytes")


def validate(data: dict):
    print(f"\n[4/6] Validation:")
    print(f"  Provinces: {len(data['provinces'])}")
    print(f"  Hotspots (risk≥{RISK_MEDIUM}): {len(data['hotspots'])}")
    for p in data["provinces"]:
        assert "id" in p,          f"Missing id in {p['name']}"
        assert "name" in p,        f"Missing name in {p['name']}"
        assert "coordinates" in p, f"Missing coordinates in {p['name']}"
        assert "riskScore" in p,  f"Missing riskScore in {p['name']}"
        assert "weeklyData" in p,  f"Missing weeklyData in {p['name']}"
        assert len(p["weeklyData"]) == HISTORICAL_WEEKS + FORECAST_WEEKS, \
            f"{p['name']}: weeklyData len={len(p['weeklyData'])} != {HISTORICAL_WEEKS + FORECAST_WEEKS}"
        for wd in p["weeklyData"][:HISTORICAL_WEEKS]:
            assert "predicted" not in wd, f"Historical week should not have 'predicted': {wd}"
        for wd in p["weeklyData"][HISTORICAL_WEEKS:]:
            assert "predicted" in wd, f"Forecast week should have 'predicted': {wd}"
    print("  Schema checks: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global SEED, FORECAST_WEEKS, HISTORICAL_WEEKS
    t0 = time.time()

    _default_seed = SEED
    _default_fw = FORECAST_WEEKS
    _default_hw = HISTORICAL_WEEKS

    parser = argparse.ArgumentParser(description="Generate dengue hotspots forecast JSON")
    parser.add_argument("--n-provinces", type=int, default=DEFAULT_N_PROVINCES,
                        help=f"Number of provinces (default: {DEFAULT_N_PROVINCES})")
    parser.add_argument("--forecast-weeks", type=int, default=_default_fw,
                        help=f"Forecast horizon weeks (default: {_default_fw})")
    parser.add_argument("--seed", type=int, default=_default_seed, help="Random seed")
    args = parser.parse_args()

    SEED = args.seed
    FORECAST_WEEKS = args.forecast_weeks

    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 60)
    print("Dengue Hotspot Forecast Pipeline")
    print(f"  n_provinces={args.n_provinces}, forecast_weeks={FORECAST_WEEKS}, seed={SEED}")
    print("=" * 60)

    vn = load_vietnam_data()
    top_provinces = rank_provinces(vn, args.n_provinces)
    data = build_hotspots(vn, top_provinces, args.n_provinces)
    validate(data)
    save_output(data, OUTPUT_FILE)

    elapsed = time.time() - t0
    print(f"\n[6/6] Done in {elapsed:.1f}s")
    print(f"  Provinces: {len(data['provinces'])}")
    print(f"  Hotspots: {len(data['hotspots'])}")
    print(f"  Output:   {OUTPUT_FILE}")
    print(f"  Source:   {data['metadata']['data_source']}")
    print(f"  Generated:{data['metadata']['generated_at']}")


if __name__ == "__main__":
    main()
