import numpy as np
import pytest

from geochemad.data.preprocessing import (
    clr_transform,
    handle_abnormal_values,
    ilr_transform,
    normalize_features,
    preprocess_pipeline,
)


def test_handle_abnormal_values_replaces_with_half_detection_limit():
    features = np.array([[10.0, -9999.0], [0.0, 8.0], [4.0, -0.5]])

    cleaned = handle_abnormal_values(features)

    expected = np.array([[10.0, 4.0], [2.0, 8.0], [4.0, 4.0]])
    np.testing.assert_allclose(cleaned, expected)


def test_handle_abnormal_values_remove_drops_rows_with_any_abnormal_value():
    features = np.array([[1.0, 2.0], [0.0, 5.0], [3.0, 4.0]])

    cleaned = handle_abnormal_values(features, method="remove")

    np.testing.assert_allclose(cleaned, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_clr_transform_rows_sum_to_zero():
    transformed = clr_transform(np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]]))

    np.testing.assert_allclose(transformed.sum(axis=1), np.zeros(2))


def test_ilr_transform_reduces_one_dimension():
    transformed = ilr_transform(np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 6.0]]))

    assert transformed.shape == (2, 2)


def test_normalize_features_supports_reusing_fit_params():
    features = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

    normalized, fit_params = normalize_features(features)
    reused, _ = normalize_features(features, fit_params=fit_params)

    np.testing.assert_allclose(normalized, reused)
    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(2))


def test_preprocess_pipeline_tracks_removed_samples_and_selected_columns():
    features = np.array([[10.0, 100.0, 5.0], [0.0, 120.0, 6.0], [20.0, 80.0, 7.0]])
    element_columns = ["Au_ppb", "Cu_ppm", "As_ppm"]

    processed, metadata = preprocess_pipeline(
        features=features,
        element_columns=element_columns,
        transform="raw",
        feature_selection="manual",
        target_element="Au",
        normalize=False,
        handle_abnormal="remove",
    )

    assert processed.shape == (2, 3)
    assert metadata["removed_samples"] == 1
    assert metadata["n_samples_after_cleaning"] == 2
    assert metadata["columns"] == element_columns


def test_preprocess_pipeline_rejects_unknown_transform():
    with pytest.raises(ValueError, match="Unknown transform"):
        preprocess_pipeline(
            features=np.array([[1.0, 2.0]]),
            element_columns=["Au_ppb", "Cu_ppm"],
            transform="bad-transform",
        )
