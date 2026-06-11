"""
Realistic synthetic event generation for quantum augmentation.

Key improvements:
1. Resamples from real distributions instead of random lat/lon
2. Preserves spatial structure (country-specific regions)
3. Maintains temporal patterns (seasonality)
4. Respects case count distributions
"""
import numpy as np
import pandas as pd
from typing import Optional, List


def generate_realistic_synthetic_events(
    train_events: pd.DataFrame,
    n_synthetic: int,
    preserve_country: bool = True,
    preserve_temporal: bool = True,
    case_noise_scale: float = 0.15,
    spatial_noise_scale: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic events by resampling from real distributions.

    Unlike the original implementation which generates random lat/lon,
    this function samples from realistic distributions that preserve:
    - Country-specific spatial patterns
    - Seasonal temporal patterns
    - Case count distributions

    Args:
        train_events: training events DataFrame
        n_synthetic: number of synthetic events to generate
        preserve_country: sample from country-specific distributions
        preserve_temporal: preserve monthly seasonality
        case_noise_scale: scale factor for case count noise
        spatial_noise_scale: scale factor for spatial noise (in degrees)
        seed: random seed

    Returns:
        DataFrame of synthetic events
    """
    np.random.seed(seed)

    if len(train_events) == 0:
        return pd.DataFrame()

    base_event_id = train_events['event_id'].max() + 1 if 'event_id' in train_events.columns else 0
    base_timestamp = train_events['timestamp'].min()

    synthetic_events = []

    countries = train_events['country'].unique()

    for i in range(n_synthetic):
        if preserve_country and len(countries) > 0:
            country = np.random.choice(countries)
            country_events = train_events[train_events['country'] == country]
        else:
            country_events = train_events

        if len(country_events) == 0:
            country_events = train_events

        idx = np.random.randint(len(country_events))
        sample = country_events.iloc[idx]

        lat = sample['lat'] + np.random.normal(0, spatial_noise_scale)
        lon = sample['lon'] + np.random.normal(0, spatial_noise_scale)

        lat = np.clip(lat, train_events['lat'].min() - 1, train_events['lat'].max() + 1)
        lon = np.clip(lon, train_events['lon'].min() - 1, train_events['lon'].max() + 1)

        region = sample.get('region', 'SYNTHETIC')
        if preserve_country:
            region = f"SYNTHETIC_{country}"

        if preserve_temporal:
            base_year = sample.get('year', base_timestamp.year)
            base_month = sample.get('month', base_timestamp.month)

            month_shift = np.random.randint(-1, 2)
            year_shift = np.random.randint(-1, 2) if np.random.random() < 0.1 else 0

            new_month = np.clip(base_month + month_shift, 1, 12)
            new_year = base_year + year_shift
        else:
            month_offset = np.random.randint(0, 12)
            new_month = month_offset + 1
            new_year = base_timestamp.year + np.random.randint(0, 5)

        timestamp = pd.Timestamp(year=new_year, month=new_month, day=15)

        base_cases = sample['case_count']
        case_noise = np.random.uniform(1 - case_noise_scale, 1 + case_noise_scale)
        case_count = max(0, int(base_cases * case_noise))

        synthetic_events.append({
            'event_id': base_event_id + i,
            'lat': float(lat),
            'lon': float(lon),
            'timestamp': timestamp,
            'case_count': case_count,
            'region': region,
            'country': country if preserve_country else 'SYNTHETIC',
            'year': new_year,
            'month': new_month,
            'augmented': True,
            'aug_method': 'quantum_resampling',
        })

    return pd.DataFrame(synthetic_events)


def generate_quantum_style_events(
    qgan_samples: np.ndarray,
    train_events: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Convert QGAN samples to realistic synthetic events.

    Instead of generating random lat/lon from samples (which loses spatial structure),
    this function uses QGAN samples to perturb real events while preserving structure.

    Args:
        qgan_samples: QGAN-generated samples (n_samples, sample_dim)
        train_events: training events for resampling reference
        seed: random seed

    Returns:
        DataFrame of synthetic events
    """
    np.random.seed(seed)

    if len(qgan_samples) == 0 or len(train_events) == 0:
        return pd.DataFrame()

    n_samples = len(qgan_samples)
    base_event_id = train_events['event_id'].max() + 1 if 'event_id' in train_events.columns else 0
    base_timestamp = train_events['timestamp'].min()

    indices = np.random.choice(len(train_events), size=n_samples, replace=True)
    reference_events = train_events.iloc[indices].copy()

    for i, s in enumerate(qgan_samples):
        ref = reference_events.iloc[i]

        case_scale = max(0.5, min(2.0, abs(s[0]) if len(s) > 0 else 1.0))
        case_count = max(1, int(ref['case_count'] * case_scale))

        spatial_perturb = abs(s[1:3]) if len(s) > 2 else np.array([0.5, 0.5])
        lat = ref['lat'] + (spatial_perturb[0] - 0.5) * 2
        lon = ref['lon'] + (spatial_perturb[1] - 0.5) * 2

        lat = np.clip(lat, train_events['lat'].min() - 1, train_events['lat'].max() + 1)
        lon = np.clip(lon, train_events['lon'].min() - 1, train_events['lon'].max() + 1)

        temporal_shift = int(abs(s[-1]) * 3) if len(s) > 0 else 0
        timestamp = ref['timestamp'] + pd.Timedelta(days=temporal_shift * 30)

        yield_event = {
            'event_id': base_event_id + i,
            'lat': float(lat),
            'lon': float(lon),
            'timestamp': timestamp,
            'case_count': case_count,
            'region': ref.get('region', 'SYNTHETIC_QGAN'),
            'country': ref.get('country', 'SYNTHETIC'),
            'year': timestamp.year,
            'month': timestamp.month,
            'augmented': True,
            'aug_method': 'qgan_style',
        }

    return pd.DataFrame(list(yield_event for _ in range(n_samples)))


def batch_generate_synthetic(
    train_events: pd.DataFrame,
    n_batches: int = 3,
    events_per_batch: int = None,
    total_ratio: float = 3.0,
    **kwargs
) -> pd.DataFrame:
    """
    Generate synthetic events in batches for better diversity.

    Args:
        train_events: training events
        n_batches: number of batches to generate
        events_per_batch: events per batch (auto-calculated if None)
        total_ratio: total augmentation ratio
        **kwargs: passed to generate_realistic_synthetic_events

    Returns:
        Combined DataFrame of all synthetic events
    """
    if events_per_batch is None:
        events_per_batch = int(len(train_events) * (total_ratio - 1)) // n_batches

    all_synthetic = []

    for i in range(n_batches):
        batch_events = generate_realistic_synthetic_events(
            train_events,
            n_synthetic=events_per_batch,
            seed=kwargs.get('seed', 42) + i,
            **kwargs
        )
        if len(batch_events) > 0:
            all_synthetic.append(batch_events)

    if all_synthetic:
        return pd.concat(all_synthetic, ignore_index=True)
    return pd.DataFrame()
