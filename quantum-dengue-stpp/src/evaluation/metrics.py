"""Evaluation metrics for forecasting and point process quality."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List


def compute_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard forecasting metrics.

    Args:
        y_true: ground truth values
        y_pred: predicted values

    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {m: np.nan for m in ["RMSE", "MAE", "MAPE", "R2", "Pearson_r", "Spearman_r"]}

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    if len(y_true) > 2:
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
    else:
        pearson_r = spearman_r = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Pearson_r": float(pearson_r),
        "Spearman_r": float(spearman_r),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None):
    """
    Compute binary classification metrics.

    Args:
        y_true: ground truth
        y_pred: predictions or probabilities
        threshold: threshold for binary classification; if None, use y_pred > 0 as threshold
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if threshold is None:
        threshold = np.median(y_true[y_true > 0])

    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)

    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
    }


def compute_point_process_metrics(
    lats_true, lons_true, cases_true,
    lats_gen, lons_gen, cases_gen,
    radii=None
) -> Dict[str, float]:
    """
    Compute point process quality metrics.

    Compares spatial statistics between true and generated events.
    """
    if radii is None:
        radii = np.array([0.5, 1.0, 2.0, 5.0])

    from ..evaluation.spatial_stats import (
        compute_k_function, compute_l_function,
        compute_pc_function, haversine_distance
    )

    K_true, _, _ = compute_k_function(lats_true, lons_true, radii=radii)
    K_gen, _, _ = compute_k_function(lats_gen, lons_gen, radii=radii)

    L_true = compute_l_function(K_true, radii)
    L_gen = compute_l_function(K_gen, radii)

    r_pc, g_true = compute_pc_function(lats_true, lons_true, r_max=5.0)
    _, g_gen = compute_pc_function(lats_gen, lons_gen, r_max=5.0)

    k_mae = np.mean(np.abs(K_true - K_gen)) if K_true is not None else np.nan
    l_mae = np.mean(np.abs(L_true - L_gen)) if L_true is not None else np.nan
    g_corr = np.corrcoef(g_true, g_gen)[0, 1] if len(g_true) > 1 and len(g_gen) > 1 else np.nan

    dist_true = []
    for i in range(min(len(lats_true), 100)):
        for j in range(i + 1, min(len(lats_true), 100)):
            d = haversine_distance(lats_true[i], lons_true[i], lats_true[j], lons_true[j])
            dist_true.append(d)
    mean_dist_true = np.mean(dist_true) if dist_true else np.nan

    dist_gen = []
    for i in range(min(len(lats_gen), 100)):
        for j in range(i + 1, min(len(lats_gen), 100)):
            d = haversine_distance(lats_gen[i], lons_gen[i], lats_gen[j], lons_gen[j])
            dist_gen.append(d)
    mean_dist_gen = np.mean(dist_gen) if dist_gen else np.nan

    return {
        "K_function_MAE": float(k_mae) if not np.isnan(k_mae) else 0.0,
        "L_function_MAE": float(l_mae) if not np.isnan(l_mae) else 0.0,
        "g_function_correlation": float(g_corr) if not np.isnan(g_corr) else 0.0,
        "mean_pair_distance_true": float(mean_dist_true),
        "mean_pair_distance_gen": float(mean_dist_gen),
        "mean_pair_distance_error": float(abs(mean_dist_true - mean_dist_gen)),
        "case_count_wass_dist": float(
            abs(np.mean(cases_true) - np.mean(cases_gen))
        ),
    }


def k_function_comparison_plot(K_true, K_gen, radii, label_true="Original", label_gen="Generated"):
    """Return data for K-function comparison visualization."""
    return {
        "radii": radii.tolist(),
        "K_true": K_true.tolist() if K_true is not None else [],
        "K_gen": K_gen.tolist() if K_gen is not None else [],
        "label_true": label_true,
        "label_gen": label_gen,
    }


def comprehensive_evaluation(
    results_dict: Dict,
    augmentation_methods: List[str],
    metrics: List[str] = None
) -> Dict:
    """
    Generate comprehensive comparison table across all methods.

    Args:
        results_dict: {method: {metric: value}}
        augmentation_methods: list of method names
        metrics: list of metric names to compare

    Returns:
        Summary comparison dict
    """
    if metrics is None:
        all_metrics = set()
        for m in results_dict.values():
            all_metrics.update(m.keys())
        metrics = sorted(all_metrics)

    summary = {"metric": metrics}
    for method in augmentation_methods:
        if method in results_dict:
            summary[method] = [
                results_dict[method].get(m, np.nan) for m in metrics
            ]

    best_per_metric = {}
    for m in metrics:
        vals = {method: summary[method][i]
                for method in augmentation_methods
                if method in summary}
        non_nan_vals = {k: v for k, v in vals.items() if not np.isnan(v)}
        if non_nan_vals:
            best_per_metric[m] = min(non_nan_vals, key=non_nan_vals.get)

    summary["best_method"] = best_per_metric
    return summary
