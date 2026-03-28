"""Classical machine learning anomaly detection methods.

Implements Isolation Forest and One-Class SVM as described in the
GeoChemAD benchmark.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from geochemad.models.statistical import BaseAnomalyDetector


class IsolationForestAD(BaseAnomalyDetector):
    """Isolation Forest anomaly detection.

    Identifies anomalies through isolation-based partitioning.
    Anomaly score is the negative of sklearn's decision function
    (so higher = more anomalous).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: str = "auto",
        random_state: int = 42,
    ):
        super().__init__()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "IsolationForestAD":
        self.model.fit(features)
        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        # Negate so higher = more anomalous
        return -self.model.decision_function(features)


class OneClassSVMAD(BaseAnomalyDetector):
    """One-Class SVM anomaly detection.

    Learns a boundary around normal data; samples outside are anomalous.
    Uses RBF kernel by default.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
    ):
        super().__init__()
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "OneClassSVMAD":
        self.model.fit(features)
        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        # Negate so higher = more anomalous
        return -self.model.decision_function(features)
