"""Evaluation metrics for geochemical anomaly detection.

Implements AUC, Average Precision, PR-AUC, and Distance to Deposit metrics
following the GeoChemAD benchmark protocol.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from scipy.spatial import cKDTree


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth labels (1 = anomaly/positive, 0 = background).
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).

    Returns
    -------
    float
        AUC score.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Average Precision.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth labels.
    scores : np.ndarray
        Anomaly scores.

    Returns
    -------
    float
        Average Precision score.
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return average_precision_score(labels, scores)


def compute_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth labels.
    scores : np.ndarray
        Anomaly scores.

    Returns
    -------
    float
        PR-AUC score.
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def compute_dtd(
    sample_coordinates: np.ndarray,
    anomaly_scores: np.ndarray,
    site_coordinates: np.ndarray,
    top_k_percent: float = 10.0,
) -> float:
    """Compute average Distance to Deposit (DTD) metric.

    Measures how close the top-scored anomalous samples are to known
    mineralization sites. Lower values indicate spatially meaningful predictions.

    Parameters
    ----------
    sample_coordinates : np.ndarray
        All sample coordinates, shape (n_samples, 2).
    anomaly_scores : np.ndarray
        Anomaly scores for all samples, shape (n_samples,).
    site_coordinates : np.ndarray
        Known deposit site coordinates, shape (n_sites, 2).
    top_k_percent : float
        Percentage of top-scored samples to evaluate.

    Returns
    -------
    float
        Average minimum distance from top anomalous samples to nearest deposit.
    """
    if len(site_coordinates) == 0:
        return float("inf")

    n_top = max(1, int(len(anomaly_scores) * top_k_percent / 100))
    top_indices = np.argsort(anomaly_scores)[-n_top:]
    top_coords = sample_coordinates[top_indices]

    site_tree = cKDTree(site_coordinates)
    distances, _ = site_tree.query(top_coords, k=1)

    return float(distances.mean())


def evaluate_model(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_coordinates: np.ndarray = None,
    site_coordinates: np.ndarray = None,
) -> dict:
    """Compute all evaluation metrics.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground truth labels.
    scores : np.ndarray
        Anomaly scores.
    sample_coordinates : np.ndarray, optional
        Sample coordinates for spatial metrics.
    site_coordinates : np.ndarray, optional
        Deposit coordinates for spatial metrics.

    Returns
    -------
    dict
        Dictionary with keys: auc, ap, pr_auc, dtd.
    """
    results = {
        "auc": compute_auc(labels, scores),
        "ap": compute_ap(labels, scores),
        "pr_auc": compute_pr_auc(labels, scores),
    }

    if sample_coordinates is not None and site_coordinates is not None:
        # Map scores back to full sample set for DTD
        results["dtd"] = compute_dtd(
            sample_coordinates, scores, site_coordinates
        )

    return results
