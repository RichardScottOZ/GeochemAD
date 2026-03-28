"""Preprocessing utilities for geochemical data.

Implements:
- Abnormal value handling (missing, negative, zero)
- Centered Log-Ratio (CLR) transformation
- Isometric Log-Ratio (ILR) transformation
- Feature selection methods
- Normalization
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def handle_abnormal_values(
    features: np.ndarray,
    method: str = "half_detection_limit",
    abnormal_values: tuple = (-9999, -0.5),
    detection_limit_fraction: float = 0.5,
) -> np.ndarray:
    """Handle abnormal values in geochemical data.

    Geochemical surveys frequently contain abnormal entries including missing,
    negative, or zero values.

    Parameters
    ----------
    features : np.ndarray
        Concentration matrix of shape (n_samples, n_elements).
    method : str
        "half_detection_limit" replaces abnormal values with half the column
        minimum positive value. "remove" returns only rows with no abnormal values.
    abnormal_values : tuple
        Values treated as abnormal indicators.
    detection_limit_fraction : float
        Fraction of column minimum to use as replacement.

    Returns
    -------
    np.ndarray
        Cleaned feature matrix.
    """
    features = features.copy()

    # Create mask for abnormal values
    abnormal_mask = np.isnan(features) | np.isinf(features)
    for val in abnormal_values:
        abnormal_mask |= features == val

    # Also treat zero and negative values as abnormal for log-ratio transforms
    abnormal_mask |= features <= 0

    if method == "remove":
        row_mask = ~abnormal_mask.any(axis=1)
        return features[row_mask]

    elif method == "half_detection_limit":
        for col in range(features.shape[1]):
            col_data = features[:, col]
            col_abnormal = abnormal_mask[:, col]
            positive_vals = col_data[~col_abnormal & (col_data > 0)]
            if len(positive_vals) > 0:
                replacement = positive_vals.min() * detection_limit_fraction
            else:
                replacement = 1e-6
            features[col_abnormal, col] = replacement
        return features

    else:
        raise ValueError(f"Unknown method: {method}. Use 'remove' or 'half_detection_limit'.")


def clr_transform(features: np.ndarray) -> np.ndarray:
    """Apply Centered Log-Ratio (CLR) transformation.

    CLR addresses the compositional closure issue by transforming constrained
    compositional data into unconstrained real space.

    CLR(x_i) = ln(x_i / g(x))  where g(x) is the geometric mean.

    Parameters
    ----------
    features : np.ndarray
        Positive concentration matrix of shape (n_samples, n_elements).

    Returns
    -------
    np.ndarray
        CLR-transformed features.
    """
    # Ensure all values are positive
    features = np.maximum(features, 1e-10)

    log_features = np.log(features)
    geometric_mean = log_features.mean(axis=1, keepdims=True)
    return log_features - geometric_mean


def ilr_transform(features: np.ndarray) -> np.ndarray:
    """Apply Isometric Log-Ratio (ILR) transformation.

    ILR maps D-part compositional data to (D-1)-dimensional real space
    using an orthonormal basis, preserving distances.

    Parameters
    ----------
    features : np.ndarray
        Positive concentration matrix of shape (n_samples, n_elements).

    Returns
    -------
    np.ndarray
        ILR-transformed features with shape (n_samples, n_elements - 1).
    """
    features = np.maximum(features, 1e-10)
    D = features.shape[1]
    log_features = np.log(features)

    # Construct Helmert sub-matrix as orthonormal basis
    ilr_coords = np.zeros((features.shape[0], D - 1))
    for i in range(D - 1):
        # ILR coordinate i:
        # sqrt(i+1 / (i+2)) * ln( g(x_1,...,x_{i+1}) / x_{i+2} )
        coeff = np.sqrt((i + 1) / (i + 2))
        log_geo_mean = log_features[:, : i + 1].mean(axis=1)
        ilr_coords[:, i] = coeff * (log_geo_mean - log_features[:, i + 1])

    return ilr_coords


def select_features(
    features: np.ndarray,
    element_columns: list,
    method: str = "all",
    target_element: str = "Au",
    n_components: int = 20,
    variance_ratio: float = 0.95,
    manual_elements: list = None,
) -> tuple:
    """Select relevant features for anomaly detection.

    Parameters
    ----------
    features : np.ndarray
        Concentration matrix.
    element_columns : list
        Column names.
    method : str
        "all": use all elements.
        "manual": use domain-knowledge selected elements.
        "pca": use PCA components.
    target_element : str
        Target element name.
    n_components : int
        Number of PCA components if method is "pca".
    variance_ratio : float
        Cumulative variance ratio threshold for PCA.
    manual_elements : list, optional
        List of element names for manual selection.

    Returns
    -------
    tuple of (selected_features, selected_columns_or_components)
    """
    if method == "all":
        return features, element_columns

    elif method == "manual":
        if manual_elements is None:
            # Default pathfinder elements for common target elements
            pathfinder_map = {
                "Au": ["Au_ppb", "As_ppm", "Sb_ppm", "Ag_ppm", "Cu_ppm", "Pb_ppm",
                        "Zn_ppm", "Bi_ppm", "Te_ppm", "Se_ppm", "W_ppm", "Mo_ppm"],
                "Cu": ["Cu_ppm", "Au_ppb", "Ag_ppm", "Mo_ppm", "Pb_ppm", "Zn_ppm",
                        "As_ppm", "Bi_ppm", "Co_ppm", "Ni_ppm", "Fe_pct"],
                "W": ["W_ppm", "Sn_ppm", "Mo_ppm", "Bi_ppm", "Cu_ppm", "As_ppm",
                       "F_ppm", "Li_ppm", "Be_ppm", "Cs_ppm", "Ta_ppm"],
                "Ni": ["Ni_ppm", "Co_ppm", "Cr_ppm", "Cu_ppm", "MgO_pct",
                        "Fe_pct", "Mn_ppm", "Pt_ppb", "Pd_ppb", "S_pct"],
            }
            manual_elements = pathfinder_map.get(target_element, [])

        # Find matching columns
        selected_idx = []
        selected_cols = []
        for elem in manual_elements:
            for i, col in enumerate(element_columns):
                if col == elem or elem.split("_")[0] in col:
                    if i not in selected_idx:
                        selected_idx.append(i)
                        selected_cols.append(col)
                        break

        if len(selected_idx) == 0:
            return features, element_columns

        return features[:, selected_idx], selected_cols

    elif method == "pca":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        pca = PCA(n_components=min(n_components, features.shape[1]))
        transformed = pca.fit_transform(scaled)

        # Select components to reach variance threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = np.searchsorted(cumvar, variance_ratio) + 1
        n_keep = max(n_keep, 2)  # At least 2 components

        component_names = [f"PC{i+1}" for i in range(n_keep)]
        return transformed[:, :n_keep], component_names

    else:
        raise ValueError(f"Unknown feature selection method: {method}")


def normalize_features(
    features: np.ndarray,
    method: str = "standard",
    fit_params: dict = None,
) -> tuple:
    """Normalize features.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    method : str
        "standard" for zero-mean unit-variance; "minmax" for [0, 1] scaling.
    fit_params : dict, optional
        Pre-computed normalization parameters for inference.

    Returns
    -------
    tuple of (normalized_features, fit_params)
    """
    if method == "standard":
        if fit_params is None:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std[std == 0] = 1.0
            fit_params = {"mean": mean, "std": std}
        normalized = (features - fit_params["mean"]) / fit_params["std"]

    elif method == "minmax":
        if fit_params is None:
            fmin = features.min(axis=0)
            fmax = features.max(axis=0)
            frange = fmax - fmin
            frange[frange == 0] = 1.0
            fit_params = {"min": fmin, "range": frange}
        normalized = (features - fit_params["min"]) / fit_params["range"]

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, fit_params


def preprocess_pipeline(
    features: np.ndarray,
    element_columns: list,
    transform: str = "clr",
    feature_selection: str = "all",
    target_element: str = "Au",
    normalize: bool = True,
    handle_abnormal: str = "half_detection_limit",
) -> tuple:
    """Full preprocessing pipeline.

    Parameters
    ----------
    features : np.ndarray
        Raw concentration matrix.
    element_columns : list
        Column names.
    transform : str
        "raw", "clr", or "ilr".
    feature_selection : str
        Feature selection method.
    target_element : str
        Target element for feature selection.
    normalize : bool
        Whether to normalize after transformation.
    handle_abnormal : str
        How to handle abnormal values.

    Returns
    -------
    tuple of (processed_features, metadata_dict)
    """
    metadata = {}

    # Step 1: Handle abnormal values
    cleaned = handle_abnormal_values(features, method=handle_abnormal)
    if cleaned.shape[0] != features.shape[0]:
        metadata["removed_samples"] = features.shape[0] - cleaned.shape[0]
    metadata["n_samples_after_cleaning"] = cleaned.shape[0]

    # Step 2: Feature selection (before transform for manual/all)
    if feature_selection != "pca":
        selected, sel_cols = select_features(
            cleaned, element_columns, method=feature_selection,
            target_element=target_element
        )
    else:
        selected, sel_cols = cleaned, element_columns

    # Step 3: Log-ratio transformation
    if transform == "clr":
        transformed = clr_transform(selected)
    elif transform == "ilr":
        transformed = ilr_transform(selected)
    elif transform == "raw":
        transformed = selected
    else:
        raise ValueError(f"Unknown transform: {transform}")

    metadata["transform"] = transform
    metadata["n_features"] = transformed.shape[1]

    # Step 3b: PCA after transform if requested
    if feature_selection == "pca":
        transformed, sel_cols = select_features(
            transformed, sel_cols, method="pca", target_element=target_element
        )
        metadata["n_features_after_pca"] = transformed.shape[1]

    # Step 4: Normalize
    if normalize:
        transformed, norm_params = normalize_features(transformed)
        metadata["norm_params"] = norm_params

    metadata["columns"] = sel_cols

    return transformed, metadata
