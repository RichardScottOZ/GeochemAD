import numpy as np

from geochemad.models.statistical import KNNAnomaly, MahalanobisDistance, ZScore


def test_zscore_returns_max_absolute_zscore_per_row():
    features = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])

    model = ZScore().fit(features)
    scores = model.score(features)

    np.testing.assert_allclose(scores, np.array([1.22474487, 0.0, 1.22474487]))


def test_mahalanobis_distance_scores_all_samples():
    features = np.array([[1.0, 1.0], [2.0, 2.5], [3.0, 2.0], [4.0, 4.5]])

    model = MahalanobisDistance().fit(features)
    scores = model.score(features)

    assert scores.shape == (4,)
    assert np.all(scores >= 0)


def test_knn_anomaly_supports_training_and_unseen_samples():
    train_features = np.array([[0.0, 0.0], [0.0, 1.0], [3.0, 3.0], [3.0, 4.0]])
    test_features = np.array([[0.1, 0.2], [10.0, 10.0]])

    model = KNNAnomaly(k=1).fit(train_features)

    train_scores = model.score(train_features)
    test_scores = model.score(test_features)

    assert train_scores.shape == (4,)
    assert test_scores.shape == (2,)
    assert test_scores[1] > test_scores[0]
