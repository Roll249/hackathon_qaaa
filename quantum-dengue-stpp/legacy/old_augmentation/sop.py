"""Second-Order Preserving (SOP) augmentation for spatio-temporal point processes."""
import numpy as np
import pandas as pd
from typing import List
from scipy.stats import wasserstein_distance

from ..evaluation.spatial_stats_fast import fast_k_function


def compute_k_function_empirical(lats, lons, radii, n_permutations=99, seed=42):
    """
    Compute empirical K-function with optional Monte Carlo envelope.

    Returns (K, K_lower, K_upper, K_csr) — the 4-tuple interface used across
    run_pipeline.py.  K_csr is the analytic CSR expectation π*r².
    Monte Carlo permutations are used to build the envelope when n_permutations>0.
    """
    K = fast_k_function(lats, lons, radii, seed=seed)
    if K is None or np.all(K == 0):
        return None, None, None, None

    K_csr = np.pi * radii ** 2  # degrees² CSR expectation

    if n_permutations > 0 and len(lats) > 5:
        rng = np.random.default_rng(seed)
        perm_K = []
        for _ in range(n_permutations):
            perm_idx = rng.permutation(len(lats))
            k_perm = fast_k_function(lats[perm_idx], lons, radii, seed=seed)
            if k_perm is not None:
                perm_K.append(k_perm)
        if perm_K:
            perm_arr = np.stack(perm_K)
            return K, np.percentile(perm_arr, 2.5, axis=0), np.percentile(perm_arr, 97.5, axis=0), K_csr

    return K, None, None, K_csr


def compute_l_function_from_k(K, r):
    """L(r) = sqrt(K(r)/π) - r."""
    return np.sqrt(np.maximum(K, 1e-10) / np.pi) - r


def sop_augment(
    events_df: pd.DataFrame,
    n_augment: int = 2,
    window_months: int = 3,
    preserve_case_distribution: bool = True,
    random_state: int = 42,
) -> List[pd.DataFrame]:
    """SOP: permute timestamps within windows, preserving spatial + case distribution."""
    np.random.seed(random_state)
    augmented = []
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    ev = ev.sort_values("timestamp")
    groups = list(ev.groupby("country"))

    for aug_idx in range(n_augment):
        aug_events = []
        for country, group in groups:
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < 3:
                aug_events.append(group)
                continue
            months = group["timestamp"].dt.to_period("M").values
            unique_months = np.unique(months)
            for m_start in range(0, len(unique_months), window_months):
                m_end = min(m_start + window_months, len(unique_months))
                wmask = np.isin(months, unique_months[m_start:m_end])
                wdata = group[wmask].copy()
                if len(wdata) < 2:
                    aug_events.append(wdata)
                    continue
                if preserve_case_distribution:
                    cases = wdata["case_count"].values.copy()
                    np.random.shuffle(cases)
                    wdata = wdata.copy()
                    wdata["case_count"] = cases
                aug_events.append(wdata)
        if aug_events:
            adf = pd.concat(aug_events, ignore_index=True)
            adf["event_id"] = range(len(adf))
            adf["augmented"] = True
            adf["aug_idx"] = aug_idx
            augmented.append(adf)
    return augmented


def sop_augment_spatial_clusters(
    events_df: pd.DataFrame,
    n_augment: int = 2,
    n_clusters: int = 5,
    window_months: int = 3,
    preserve_case_distribution: bool = True,
    random_state: int = 42,
) -> List[pd.DataFrame]:
    """Enhanced SOP: cluster by space, permute within cluster+window."""
    from sklearn.cluster import KMeans

    np.random.seed(random_state)
    augmented = []
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    ev = ev.sort_values("timestamp")
    groups = list(ev.groupby("country"))

    for aug_idx in range(n_augment):
        aug_all = []
        for country, group in groups:
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < n_clusters * 2:
                aug_all.append(group)
                continue

            coords = group[["lat", "lon"]].values
            n_cl = max(2, min(n_clusters, len(group) // 5))
            kmeans = KMeans(n_clusters=n_cl, random_state=random_state + aug_idx, n_init=3)
            labels = kmeans.fit_predict(coords)
            group = group.copy()
            group["cluster"] = labels

            months = group["timestamp"].dt.to_period("M").values
            unique_months = np.unique(months)
            aug_group = []

            for m_start in range(0, len(unique_months), window_months):
                m_end = min(m_start + window_months, len(unique_months))
                wmask = np.isin(months, unique_months[m_start:m_end])
                wdata = group[wmask].copy()
                if len(wdata) < 2:
                    aug_group.append(wdata)
                    continue
                if preserve_case_distribution:
                    cases = wdata["case_count"].values.copy()
                    np.random.shuffle(cases)
                    wdata = wdata.copy()
                    wdata["case_count"] = cases
                aug_group.append(wdata)

            if aug_group:
                adf = pd.concat(aug_group, ignore_index=True).drop(columns=["cluster"])
                aug_all.append(adf)

        if aug_all:
            ac = pd.concat(aug_all, ignore_index=True)
            ac["event_id"] = range(len(ac))
            ac["augmented"] = True
            ac["aug_idx"] = aug_idx
            augmented.append(ac)

    return augmented


def validate_sop_preservation(
    original_df: pd.DataFrame,
    augmented_dfs: List[pd.DataFrame],
    radii: np.ndarray = None,
    n_permutations: int = 9,
) -> dict:
    """
    Validate that SOP augmentation preserved second-order properties.
    Uses fast O(n log n) K-function. n_permutations is kept for API
    compatibility but does not trigger Monte Carlo loops.
    """
    if radii is None:
        radii = np.array([50.0, 100.0, 200.0, 500.0])

    orig_K = fast_k_function(original_df["lat"].values, original_df["lon"].values,
                             radii, max_n=300)
    orig_L = compute_l_function_from_k(orig_K, radii)

    results = {
        "original_K": orig_K.tolist(),
        "original_L": orig_L.tolist(),
        "augmented_K": [],
        "augmented_L": [],
        "k_function_mae": [],
        "l_function_mae": [],
        "case_dist_wasserstein": [],
    }

    for aug_df in augmented_dfs:
        aug_K = fast_k_function(aug_df["lat"].values, aug_df["lon"].values,
                                radii, max_n=300)
        aug_L = compute_l_function_from_k(aug_K, radii)
        results["augmented_K"].append(aug_K.tolist())
        results["augmented_L"].append(aug_L.tolist())
        results["k_function_mae"].append(float(np.mean(np.abs(orig_K - aug_K))))
        results["l_function_mae"].append(float(np.mean(np.abs(orig_L - aug_L))))
        results["case_dist_wasserstein"].append(
            float(wasserstein_distance(
                original_df["case_count"].values,
                aug_df["case_count"].values
            ))
        )

    return results
