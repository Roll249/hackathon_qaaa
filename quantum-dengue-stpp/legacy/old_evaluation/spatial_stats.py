"""Spatial and temporal statistics for dengue point process analysis."""
import numpy as np
from scipy import stats


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def _vectorized_haversine(coords1, coords2):
    """Vectorized haversine for two (N,2) arrays of (lat, lon). Returns (N,M) distance matrix."""
    lat1 = np.asarray(coords1[:, 0])
    lon1 = np.asarray(coords1[:, 1])
    lat2 = np.asarray(coords2[:, 0])
    lon2 = np.asarray(coords2[:, 1])
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2[:, None] - lat1[None, :])
    dlam = np.radians(lon2[:, None] - lon1[None, :])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1)[None, :] * np.cos(phi2)[:, None] * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def compute_k_function(lats, lons, radii=None, area=None, method="ripley"):
    r"""
    Compute Ripley's K-function for a spatial point pattern.

    K(r) = (1/λ) × E[number of events within distance r of a randomly chosen event]
    where λ = n / A (intensity = number of events / study area).

    Args:
        lats: array of latitudes
        lons: array of longitudes
        radii: array of radii (in degrees); if None, default [0.5,1,2,5] degrees
        area: study region area in km^2; if None, estimated from bounding box
        method: "ripley" or "translational"

    Returns:
        K: array of K(r) values at each radius
        r: the radii used
        lamb: intensity λ
    """
    n = len(lats)
    if n < 2:
        return np.array([0.0]), np.array([0.0]), 0.0

    if radii is None:
        radii = np.array([0.5, 1.0, 2.0, 5.0])
    radii = np.asarray(radii)

    coords = np.column_stack([lats, lons])
    dists = _vectorized_haversine(coords, coords)
    np.fill_diagonal(dists, np.inf)

    if area is None:
        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        bbox_area = lat_range * lon_range * (111.0 * 111.0 * np.cos(np.radians(lats.mean())))
        area = max(bbox_area, 1.0)

    lamb = n / area

    K = np.zeros(len(radii))
    for i, r in enumerate(radii):
        count = np.sum(dists <= r, axis=1)
        K[i] = (count.sum() / n) / lamb

    return K, radii, lamb


def compute_l_function(K, r):
    r"""L(r) = sqrt(K(r)/π) - r. For CSR: L(r) ≈ 0. L>0=clustering, L<0=regularity."""
    return np.sqrt(np.maximum(K, 0) / np.pi) - r


def compute_pc_function(lats, lons, r_max=5.0, n_bins=20):
    r"""Compute pair correlation function g(r)."""
    n = len(lats)
    if n < 3:
        return np.zeros(n_bins), np.zeros(n_bins)

    coords = np.column_stack([lats, lons])
    dists = _vectorized_haversine(coords, coords)
    np.fill_diagonal(dists, np.inf)
    dists[dists == 0] = np.nan
    dists_flat = dists.flatten()
    dists_flat = dists_flat[~np.isnan(dists_flat)]
    dists_flat = dists_flat[dists_flat <= r_max]

    lat_range = lats.max() - lats.min()
    lon_range = lons.max() - lons.min()
    bbox_area = lat_range * lon_range * (111.0 * 111.0 * np.cos(np.radians(lats.mean())))
    lamb = n / max(bbox_area, 1.0)

    hist, bin_edges = np.histogram(dists_flat, bins=n_bins, range=(0, r_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    g = np.zeros(n_bins)
    mask = hist > 0
    r_valid = bin_centers[mask]
    hist_valid = hist[mask]
    g[mask] = hist_valid / (2 * np.pi * r_valid * bin_width * lamb * n * n)

    return bin_centers, g


def spatial_autocorrelation(lats, lons, values, nbins=5):
    """Compute Global Moran's I using k-nearest-neighbor spatial weights."""
    n = len(values)
    if n < 3:
        return np.nan, 1.0

    coords = np.column_stack([lats, lons])
    dists = _vectorized_haversine(coords, coords)

    W = np.zeros_like(dists)
    for i in range(n):
        nearest = np.argsort(dists[i])[1:min(nbins + 1, n)]
        W[i, nearest] = 1.0

    np.fill_diagonal(W, 0)
    W_row_sum = W.sum(axis=1, keepdims=True)
    W_row_sum[W_row_sum == 0] = 1
    W = W / W_row_sum

    y = values - values.mean()

    I = (n / float(y @ y)) * (y.T @ W @ y) / (y @ y)

    V = (n / float(y @ y)) * (y.T @ W @ y - (1 / float(n)) * (y @ np.ones(n)) ** 2)

    if V > 0:
        se_I = np.sqrt(V)
        z_stat = I / se_I
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1.0

    return float(I), float(p_value)


def temporal_autocorrelation(cases, max_lag=12):
    """Compute ACF and PACF for a time series."""
    acf = np.zeros(max_lag + 1)
    pacf = np.zeros(max_lag + 1)
    mean = np.mean(cases)
    var = np.var(cases)
    n = len(cases)

    if var == 0:
        return acf, pacf

    for lag in range(max_lag + 1):
        if lag == 0:
            cov = var
        else:
            cov = np.mean((cases[:-lag] - mean) * (cases[lag:] - mean))
        acf[lag] = cov / var

    pacf[0] = 1.0
    if max_lag > 0:
        pacf[1] = acf[1]
        for k in range(2, max_lag + 1):
            numerator = acf[k] - np.sum(pacf[1:k] * acf[1:k][::-1])
            denominator = 1 - np.sum(pacf[1:k] * acf[1:k])
            pacf[k] = numerator / denominator if denominator != 0 else 0.0

    return acf, pacf


def seasonal_decomposition(cases, period=12):
    """Simple seasonal decomposition."""
    n = len(cases)
    if n < 2 * period:
        return np.full(n, cases.mean()), np.zeros(n), np.zeros(n)

    trend = np.zeros(n)
    half = period // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        trend[i] = np.mean(cases[start:end])

    detrended = cases - trend
    seasonal = np.zeros(period)
    for p in range(period):
        indices = np.arange(p, n, period)
        seasonal[p] = np.mean(detrended[indices])

    seasonal_full = np.tile(seasonal, n // period + 1)[:n]
    residual = cases - trend - seasonal_full

    return trend, seasonal_full, residual


def detect_outliers_iqr(data, multiplier=3.0):
    """Detect outliers using IQR method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (data < lower) | (data > upper), lower, upper


def zero_inflation_ratio(data):
    """Compute proportion of zeros."""
    return np.mean(np.array(data) == 0)


def compute_overdispersion(data):
    """Compute dispersion index: Var/Mean. >1 indicates overdispersion."""
    mean = np.mean(data)
    var = np.var(data)
    return float(var / mean) if mean > 0 else 0.0
