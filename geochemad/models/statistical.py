"""Statistical anomaly detection methods.

Implements Z-score, Mahalanobis Distance, and k-Nearest Neighbors approaches
as described in the GeoChemAD benchmark.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.spatial import cKDTree


class BaseAnomalyDetector:
    """Base class for anomaly detection models."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "BaseAnomalyDetector":
        """Fit the model on training data."""
        raise NotImplementedError

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        """Compute anomaly scores for input samples.

        Returns
        -------
        np.ndarray
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        raise NotImplementedError

    def fit_score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        """Fit the model and compute anomaly scores."""
        self.fit(features, coordinates)
        return self.score(features, coordinates)


class ZScore(BaseAnomalyDetector):
    """Z-score based anomaly detection.

    Detects anomalies based on deviations from global data distributions.
    The anomaly score is the maximum absolute z-score across all elements.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "ZScore":
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        z_scores = np.abs((features - self.mean) / self.std)
        # Use max z-score across elements as anomaly score
        return z_scores.max(axis=1)


class MahalanobisDistance(BaseAnomalyDetector):
    """Mahalanobis Distance based anomaly detection.

    Measures the distance of each sample from the centroid of the data
    distribution, accounting for correlations between elements.
    """

    def __init__(self, regularization: float = 1e-6):
        super().__init__()
        self.mean = None
        self.cov_inv = None
        self.regularization = regularization

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "MahalanobisDistance":
        self.mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        # Regularize to ensure invertibility
        cov += np.eye(cov.shape[0]) * self.regularization
        self.cov_inv = np.linalg.inv(cov)
        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        scores = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            scores[i] = mahalanobis(features[i], self.mean, self.cov_inv)
        return scores


class KNNAnomaly(BaseAnomalyDetector):
    """k-Nearest Neighbors based anomaly detection.

    Measures local neighbourhood sparsity. The anomaly score is the average
    distance to the k nearest neighbours.
    """

    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self.tree = None
        self.train_features = None

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "KNNAnomaly":
        self.train_features = features.copy()
        self.tree = cKDTree(features)
        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        k_query = min(self.k + 1, self.tree.n)
        distances, _ = self.tree.query(features, k=k_query)

        # If scoring training data, skip self (distance ~0)
        if distances.shape[1] > self.k:
            distances = distances[:, 1:]

        # Average distance to k nearest neighbours
        return distances.mean(axis=1)
