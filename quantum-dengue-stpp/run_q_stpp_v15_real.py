#!/usr/bin/env python3
"""
Q-STPP v15 — REAL-DATA variant: fair SOP comparison on OpenDengue-derived events.

WHAT THIS DOES
--------------
Same fair 3-method SOP comparison as run_q_stpp_v15_fair.py (mh / grover /
qaoa; identical seed, identical evaluation budget), but the input point
pattern is built from the real dengue_dataset/sea_dengue_admin1_month.csv
(OpenDengue-derived) instead of a synthetic Hawkes simulation.

HONEST SCOPE — READ BEFORE INTERPRETING NUMBERS
-------------------------------------------------
* Real: report dates (month) and the admin1 region with a positive case
  count -- one point EVENT is created per (region, month) record that has
  dengue_total > 0. The actual case COUNT is used only to filter out
  zero-case months; it does not weight or duplicate events. Turning every
  individual case into its own point (millions of them, no per-case
  location in the source data anyway) is out of scope.
* Real, jittered within the month: the event's day-of-month is drawn
  uniformly at random, because the source data has month resolution, not
  day resolution.
* PROXY, not real: spatial coordinates. This dataset has no admin1-level
  centroid/boundary, only a country name. Each admin1 region is assigned a
  FIXED (deterministic, seeded by region name) jitter offset around the
  country's centroid, so points from the same region cluster together and
  points from different regions separate -- but the offsets are NOT real
  admin1 geography. Do not read spatial distances in the output as
  physically meaningful; only the relative comparison between the three
  SOP methods (computed on the exact same points) is meaningful, exactly
  as documented in run_q_stpp_v15_fair.py.

This script imports its core (L-function, local search, aggregation,
plotting) from run_q_stpp_v15_fair.py unchanged -- only the data loader is
new -- so it inherits the same fairness guarantees and the same "no
quantum hardware, no quantum advantage" scope.
"""

import os
import sys
import hashlib
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_q_stpp_v15_fair import (  # noqa: E402
    METHODS, METHOD_LABELS, compute_L_summary, evaluate_method,
    aggregate, print_summary, plot_summary,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, '..', 'dengue_dataset',
                                  'sea_dengue_admin1_month.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_result', 'q_stpp_v15_real')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country centroids (lat, lon) -- same proxy values used in
# dengue_dataset/eda_analysis_v2.py. Real admin1 centroids are not in the
# source data, so this is the finest anchor available.
COUNTRY_CENTROIDS = {
    'CAMBODIA': (12.5657, 104.9910),
    'INDONESIA': (-0.7893, 113.9213),
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": (19.8563, 102.4955),
    'MALAYSIA': (4.2105, 101.9758),
    'SINGAPORE': (1.3521, 103.8198),
    'THAILAND': (15.8700, 100.9925),
    'TIMOR-LESTE': (-8.8742, 125.7275),
    'VIET NAM': (14.0583, 108.2772),
}


def _region_xy(region, base_lat, base_lon, jitter_deg):
    """Deterministic per-region jitter: same region -> same offset every run."""
    h = int(hashlib.sha256(region.encode('utf-8')).hexdigest(), 16) % (2**32)
    r_rng = np.random.default_rng(h)
    x = base_lon + r_rng.uniform(-jitter_deg, jitter_deg)
    y = base_lat + r_rng.uniform(-jitter_deg, jitter_deg)
    return x, y


def load_real_events(df, country, year_start, year_end, seed, jitter_deg=1.5,
                     max_events=None):
    """Build a (times, x, y) point pattern from real admin1-month dengue records.

    One event per (region, month) row with dengue_total > 0 in the window
    [year_start, year_end]. Day-of-month and (if max_events caps the pool)
    the subsample are drawn from `seed`. See module docstring for what is
    real vs. proxy in the resulting coordinates.
    """
    sub = df[(df['adm_0_name'] == country) &
             (df['Year'] >= year_start) & (df['Year'] <= year_end) &
             (df['dengue_total'] > 0)]
    if len(sub) < 10:
        return None

    base_lat, base_lon = COUNTRY_CENTROIDS[country]
    rng = np.random.default_rng(seed)

    month_starts = sub['calendar_start_date'].values
    days_in_month = sub['calendar_start_date'].dt.days_in_month.values
    day_offsets = rng.integers(0, days_in_month)
    event_dates = pd.to_datetime(month_starts) + pd.to_timedelta(day_offsets, unit='D')

    regions = sub['adm_1_name'].tolist()
    xy = [_region_xy(r, base_lat, base_lon, jitter_deg) for r in regions]
    xs = np.array([p[0] for p in xy])
    ys = np.array([p[1] for p in xy])

    n_available = len(sub)
    if max_events and n_available > max_events:
        idx = rng.choice(n_available, size=max_events, replace=False)
        event_dates = event_dates[idx]
        xs, ys = xs[idx], ys[idx]

    order = np.argsort(event_dates.values)
    event_dates = event_dates[order]
    xs, ys = xs[order], ys[order]

    days_elapsed = (event_dates - event_dates.min()).days.values.astype(float)
    span_days = max(days_elapsed.max(), 1.0)
    space_extent = 2 * jitter_deg
    times = days_elapsed / span_days * space_extent  # same numeric scale as the spatial jitter

    return times, xs, ys, n_available


def run_single_real(df, country, year_start, year_end, seed, n_perms,
                    evals_per_perm, max_events, jitter_deg):
    loaded = load_real_events(df, country, year_start, year_end, seed,
                              jitter_deg, max_events)
    if loaded is None:
        return None
    times, cx, cy, n_available = loaded

    r_values = np.linspace(0.05, 0.3, 8)
    L_target = compute_L_summary(times, cx, cy, r_values)

    out = {'seed': seed, 'n_events': int(len(times)), 'n_events_target': max_events,
          'n_events_available': n_available, 'evals_per_perm': evals_per_perm,
          'n_perms': n_perms}
    for m in METHODS:
        out[m] = evaluate_method(m, times, cx, cy, r_values, L_target,
                                 n_perms, evals_per_perm, seed)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Q-STPP v15 (real data): fair SOP comparison on OpenDengue-derived events')
    parser.add_argument('--data_path', default=DEFAULT_DATA_PATH)
    parser.add_argument('--country', default='CAMBODIA', choices=sorted(COUNTRY_CENTROIDS))
    parser.add_argument('--year_start', type=int, default=None,
                        help='Defaults to the first year with dengue_total>0 for this country')
    parser.add_argument('--year_end', type=int, default=None,
                        help='Defaults to the last year with dengue_total>0 for this country')
    parser.add_argument('--jitter_deg', type=float, default=1.5,
                        help='Half-width of the proxy spatial jitter around the country centroid, in degrees')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--max_events', type=int, nargs='+', default=[30, 60, 120])
    parser.add_argument('--n_perms', type=int, default=10)
    parser.add_argument('--evals_per_perm', type=int, default=200)
    parser.add_argument('--out_dir', default=None,
                        help='Defaults to output_result/q_stpp_v15_real/')
    args = parser.parse_args()

    out_dir = args.out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.data_path, low_memory=False)
    df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
    df['Year'] = df['Year'].astype(int)

    country_years = df.loc[(df['adm_0_name'] == args.country) & (df['dengue_total'] > 0), 'Year']
    if args.year_start is None:
        args.year_start = int(country_years.min())
    if args.year_end is None:
        args.year_end = int(country_years.max())

    print("=" * 78)
    print("  Q-STPP v15 (REAL DATA): FAIR SOP COMPARISON")
    print(f"  Country: {args.country}  |  Years: {args.year_start}-{args.year_end}")
    print("  Event = one (region, month) record with dengue_total > 0.")
    print("  Dates are real (day-of-month sampled within the report month).")
    print("  Coordinates are a PROXY (country centroid + per-region jitter) --")
    print("  this dataset has no true admin1 centroids. See module docstring.")
    print("  All methods classical -- no quantum hardware, no quantum advantage claimed")
    print("=" * 78)

    n_available = len(df[(df['adm_0_name'] == args.country) &
                        (df['Year'] >= args.year_start) & (df['Year'] <= args.year_end) &
                        (df['dengue_total'] > 0)])
    print(f"\n  {n_available} real (region, month) events available for "
         f"{args.country} {args.year_start}-{args.year_end}\n")

    rows = []
    for n in args.max_events:
        for seed in args.seeds:
            r = run_single_real(df, args.country, args.year_start, args.year_end,
                               seed, args.n_perms, args.evals_per_perm, n, args.jitter_deg)
            if r is not None:
                rows.append(r)
                line = "  ".join(
                    f"{m}: err={r[m]['mean_error']:.5f} div={r[m]['diversity']:.2f}"
                    for m in METHODS)
                print(f"  N={n:>3} seed={seed}: {line}")
            else:
                print(f"  N={n:>3} seed={seed}: SKIPPED (fewer than 10 real events available)")

    if not rows:
        print("\n  No (N, seed) cell produced enough events -- nothing to aggregate.")
        return

    agg = aggregate(rows)
    print_summary(agg)

    import json
    results_path = os.path.join(out_dir, 'real_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump({'runs': rows, 'aggregate': agg, 'config': vars(args)},
                  f, indent=2, default=str)
    plot_path = os.path.join(out_dir, 'real_comparison_plot.png')
    plot_summary(agg, plot_path)

    print(f"\n  Results: {results_path}")
    print(f"  Plot:    {plot_path}")


if __name__ == '__main__':
    main()
