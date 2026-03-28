import numpy as np

from geochemad.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_dtd,
    compute_pr_auc,
    evaluate_model,
)


def test_classification_metrics_handle_constant_labels():
    labels = np.ones(4)
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    assert compute_auc(labels, scores) == 0.5
    assert compute_ap(labels, scores) == 0.0
    assert compute_pr_auc(labels, scores) == 0.0


def test_classification_metrics_return_perfect_scores_for_ranked_example():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert compute_auc(labels, scores) == 1.0
    assert compute_ap(labels, scores) == 1.0
    assert compute_pr_auc(labels, scores) == 1.0


def test_compute_dtd_returns_mean_distance_for_top_scored_samples():
    sample_coordinates = np.array([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0]])
    scores = np.array([0.9, 0.1, 0.8])
    site_coordinates = np.array([[0.0, 0.0]])

    dtd = compute_dtd(sample_coordinates, scores, site_coordinates, top_k_percent=50.0)

    assert dtd == np.sqrt(2) / 2


def test_evaluate_model_includes_spatial_metric_when_coordinates_are_provided():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    sample_coordinates = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [2.0, 2.0]])
    site_coordinates = np.array([[1.0, 1.0]])

    metrics = evaluate_model(
        labels=labels,
        scores=scores,
        sample_coordinates=sample_coordinates,
        site_coordinates=site_coordinates,
    )

    assert set(metrics) == {"auc", "ap", "pr_auc", "dtd"}
    assert metrics["auc"] == 1.0
