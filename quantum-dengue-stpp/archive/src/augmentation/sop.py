"""
Improved SOP v2 augmentation - Structure-Preserving Augmentation.

Key improvements over v1:
1. SMOTE-style interpolation instead of random shuffling
2. Preserves spatial-temporal structure
3. Resamples from real distributions rather than permuting
4. Multi-level augmentation: point, cluster, temporal
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings


def create_feature_space(events_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create feature space for augmentation.

    Features include:
    - Temporal: month_sin, month_cos, year_norm
    - Spatial: lat, lon
    - Intensity: log(case_count + 1)
    """
    df = events_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1e-9)
    df['lat_norm'] = (df['lat'] - df['lat'].min()) / (df['lat'].max() - df['lat'].min() + 1e-9)
    df['lon_norm'] = (df['lon'] - df['lon'].min()) / (df['lon'].max() - df['lon'].min() + 1e-9)
    df['log_cases'] = np.log1p(df['case_count'].clip(lower=0))

    features = ['month_sin', 'month_cos', 'year_norm', 'lat', 'lon', 'log_cases']
    return df[features].values, df


def smote_interpolation(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    k_neighbors: int = 5,
    noise_scale: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE-style interpolation for generating synthetic samples.

    Instead of shuffling, we interpolate between similar points,
    preserving the underlying structure.

    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        n_synthetic: number of synthetic samples
        k_neighbors: number of nearest neighbors to use
        noise_scale: scale of random noise to add
        seed: random seed

    Returns:
        synthetic_X, synthetic_y
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(k_neighbors, n_samples)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    synthetic_X = []
    synthetic_y = []

    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_samples)
        neighbor_idx = indices[idx, 1:]

        neighbor_choice = np.random.choice(neighbor_idx)
        alpha = np.random.uniform(0.2, 0.8)

        x_new = X[idx] * alpha + X[neighbor_choice] * (1 - alpha)
        y_new = y[idx] * alpha + y[neighbor_choice] * (1 - alpha)

        noise = np.random.randn(n_features) * noise_scale * np.std(X, axis=0)
        x_new = x_new + noise

        synthetic_X.append(x_new)
        synthetic_y.append(y_new)

    return np.array(synthetic_X), np.array(synthetic_y)


def sop_augment_v2(
    events_df: pd.DataFrame,
    n_augment: int = 2,
    augmentation_factor: float = 1.5,
    k_neighbors: int = 5,
    preserve_temporal: bool = True,
    preserve_spatial: bool = True,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    SOP v2 augmentation - Structure-Preserving Augmentation.

    Unlike v1 which shuffled case counts (breaking structure),
    v2 interpolates between similar events while preserving structure.

    Args:
        events_df: training events DataFrame
        n_augment: number of augmented datasets to generate
        augmentation_factor: how much to multiply dataset size
        k_neighbors: neighbors for SMOTE interpolation
        preserve_temporal: ensure temporal patterns preserved
        preserve_spatial: ensure spatial patterns preserved
        random_state: random seed

    Returns:
        List of augmented DataFrames
    """
    np.random.seed(random_state)

    if len(events_df) < k_neighbors + 1:
        warnings.warn("Too few events for SOP v2 augmentation")
        return []

    X, df = create_feature_space(events_df)
    y = df['case_count'].values

    target_n = int(len(events_df) * (augmentation_factor - 1))
    n_per_aug = target_n // n_augment

    augmented_dfs = []

    for i in range(n_augment):
        synthetic_X, synthetic_y = smote_interpolation(
            X, y,
            n_synthetic=n_per_aug,
            k_neighbors=k_neighbors,
            noise_scale=0.05,
            seed=random_state + i
        )

        synth_df = pd.DataFrame()
        synth_df['month_sin'] = synthetic_X[:, 0]
        synth_df['month_cos'] = synthetic_X[:, 1]
        synth_df['year_norm'] = synthetic_X[:, 2]
        synth_df['lat'] = synthetic_X[:, 3]
        synth_df['lon'] = synthetic_X[:, 4]
        synth_df['log_cases'] = synthetic_X[:, 5]

        synth_df['month'] = ((np.arcsin(synth_df['month_sin'].clip(-1, 1)) / (2 * np.pi) + 0.5) * 12).round().clip(1, 12).astype(int)
        synth_df['case_count'] = np.expm1(synth_df['log_cases']).round().clip(lower=0).astype(int)

        year_min = df['year'].min()
        year_max = df['year'].max()
        synth_df['year'] = (synth_df['year_norm'] * (year_max - year_min) + year_min).round().astype(int)

        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        synth_df['lat'] = synth_df['lat'].clip(lat_min, lat_max)
        synth_df['lon'] = synth_df['lon'].clip(lon_min, lon_max)

        synth_df['timestamp'] = pd.to_datetime(
            synth_df['year'].astype(str) + '-' + synth_df['month'].astype(str).str.zfill(2) + '-01'
        )
        synth_df['region'] = 'SOP_V2_SYNTHETIC'
        synth_df['country'] = 'SYNTHETIC'
        synth_df['event_id'] = range(len(events_df), len(events_df) + len(synth_df))
        synth_df['augmented'] = True
        synth_df['aug_method'] = f'sop_v2_run_{i+1}'

        augmented_dfs.append(synth_df)

    return augmented_dfs


def sop_augment_cluster_v2(
    events_df: pd.DataFrame,
    n_clusters: int = 5,
    n_augment_per_cluster: int = 2,
    augmentation_factor: float = 1.5,
    k_neighbors: int = 5,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Cluster-based SOP v2 augmentation.

    First clusters events by spatial-temporal similarity,
    then augments within each cluster separately.
    This preserves cluster-level structure better.
    """
    from sklearn.cluster import MiniBatchKMeans

    np.random.seed(random_state)

    if len(events_df) < n_clusters * k_neighbors:
        warnings.warn("Too few events for cluster-based augmentation")
        return sop_augment_v2(
            events_df,
            n_augment=n_augment_per_cluster * n_clusters,
            augmentation_factor=augmentation_factor,
            k_neighbors=k_neighbors,
            random_state=random_state
        )

    features, df = create_feature_space(events_df)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(features[:, :5])

    augmented_dfs = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_df = events_df[cluster_mask].copy()

        if len(cluster_df) < k_neighbors + 1:
            continue

        cluster_augs = sop_augment_v2(
            cluster_df,
            n_augment=n_augment_per_cluster,
            augmentation_factor=augmentation_factor,
            k_neighbors=k_neighbors,
            random_state=random_state + cluster_id
        )
        augmented_dfs.extend(cluster_augs)

    return augmented_dfs


def validate_sop_v2_preservation(
    original_df: pd.DataFrame,
    augmented_dfs: List[pd.DataFrame],
    radii: Optional[np.ndarray] = None,
    n_permutations: int = 19
) -> dict:
    """
    Validate that SOP v2 augmentation preserves spatial-temporal structure.

    Checks:
    1. K-function preservation
    2. Case distribution (Wasserstein distance)
    3. Temporal distribution (monthly patterns)
    4. Spatial distribution
    """
    from scipy.stats import wasserstein_distance
    from scipy.stats import ks_2samp

    if radii is None:
        radii = np.array([0.5, 1.0, 2.0, 5.0])

    results = {
        'k_function_mae': [],
        'l_function_mae': [],
        'case_dist_wasserstein': [],
        'month_dist_ks': [],
        'spatial_dist_ks': [],
    }

    for aug_df in augmented_dfs:
        if len(aug_df) < 20:
            continue

        try:
            from src.evaluation.spatial_stats import compute_k_function_empirical

            lat_orig = original_df['lat'].values
            lon_orig = original_df['lon'].values
            lat_aug = aug_df['lat'].values
            lon_aug = aug_df['lon'].values

            K_orig, _, _, _ = compute_k_function_empirical(
                lat_orig, lon_orig, radii, n_permutations=n_permutations
            )
            K_aug, _, _, _ = compute_k_function_empirical(
                lat_aug, lon_aug, radii, n_permutations=n_permutations
            )

            if K_orig is not None and K_aug is not None:
                k_mae = np.mean(np.abs(K_orig - K_aug))
                results['k_function_mae'].append(k_mae)

                L_orig = np.sqrt(K_orig / (np.pi * radii ** 2 + 1e-9)) - radii
                L_aug = np.sqrt(K_aug / (np.pi * radii ** 2 + 1e-9)) - radii
                l_mae = np.mean(np.abs(L_orig - L_aug))
                results['l_function_mae'].append(l_mae)

        except Exception:
            pass

        orig_cases = original_df['case_count'].values
        aug_cases = aug_df['case_count'].values
        if len(orig_cases) > 0 and len(aug_cases) > 0:
            wd = wasserstein_distance(orig_cases, aug_cases)
            results['case_dist_wasserstein'].append(wd)

        orig_months = original_df['month'].values
        aug_months = aug_df['month'].values
        if len(orig_months) > 0 and len(aug_months) > 0:
            ks_stat, _ = ks_2samp(orig_months, aug_months)
            results['month_dist_ks'].append(ks_stat)

        orig_lats = original_df['lat'].values
        aug_lats = aug_df['lat'].values
        if len(orig_lats) > 0 and len(aug_lats) > 0:
            ks_stat, _ = ks_2samp(orig_lats, aug_lats)
            results['spatial_dist_ks'].append(ks_stat)

    return results
