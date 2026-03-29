from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geochemad.data.dataset import GeoChemADDataset, load_dataset


def test_load_from_dataframes_parses_coordinates_and_features(sample_frames):
    samples_df, sites_df = sample_frames

    dataset = GeoChemADDataset("sed1").load_from_dataframes(samples_df, sites_df)

    assert dataset.coordinates.shape == (4, 2)
    assert dataset.features.shape == (4, 3)
    assert dataset.element_columns == ["Au_ppb", "Cu_ppm", "As_ppm"]
    assert dataset.n_samples == 4
    assert dataset.n_elements == 3
    assert dataset.n_sites == 2
    assert dataset.get_target_element_index() == 0
    np.testing.assert_allclose(dataset.get_target_element_values(), [10.0, 12.0, 14.0, 18.0])


def test_create_evaluation_set_returns_positive_and_background_samples(sample_frames):
    samples_df, sites_df = sample_frames
    dataset = GeoChemADDataset("sed1").load_from_dataframes(samples_df, sites_df)

    evaluation = dataset.create_evaluation_set(n_background=2, random_state=7)

    assert len(evaluation["positive_indices"]) >= 1
    assert len(evaluation["negative_indices"]) == 2
    assert len(evaluation["labels"]) == len(evaluation["indices"])
    assert set(np.unique(evaluation["labels"])) == {0.0, 1.0}
    assert set(evaluation["positive_indices"]).isdisjoint(set(evaluation["negative_indices"]))


def test_load_reads_csv_files_and_defaults_missing_sites_to_empty(tmp_path: Path):
    subset_dir = tmp_path / "rock2"
    subset_dir.mkdir()
    pd.DataFrame(
        {
            "SAMPLEID": ["S001", "S002"],
            "SAMPLETYPE": ["rockchip", "rockchip"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "Au_ppb": [1.0, 2.0],
            "Cu_ppm": [3.0, 4.0],
        }
    ).to_csv(subset_dir / "samples.csv", index=False)

    dataset = load_dataset("rock2", data_dir=str(tmp_path))

    assert dataset.n_samples == 2
    assert dataset.n_sites == 0
    assert dataset.site_coordinates.shape == (0, 2)


def test_load_raises_when_samples_file_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GeoChemADDataset("soil1", data_dir=str(tmp_path)).load()


def test_load_from_dataframes_rejects_missing_required_sample_columns(sample_frames):
    samples_df, sites_df = sample_frames

    with pytest.raises(ValueError, match="missing required columns: SAMPLETYPE"):
        GeoChemADDataset("sed1").load_from_dataframes(
            samples_df.drop(columns=["SAMPLETYPE"]),
            sites_df,
        )


def test_load_from_dataframes_rejects_missing_required_site_columns(sample_frames):
    samples_df, sites_df = sample_frames

    with pytest.raises(ValueError, match="Sites data is missing required columns: y"):
        GeoChemADDataset("sed1").load_from_dataframes(
            samples_df,
            sites_df.drop(columns=["y"]),
        )


def test_load_from_dataframes_rejects_non_numeric_or_non_finite_values(sample_frames):
    samples_df, sites_df = sample_frames
    bad_samples = samples_df.copy()
    bad_samples.loc[0, "x"] = np.inf

    with pytest.raises(ValueError, match="finite numeric values"):
        GeoChemADDataset("sed1").load_from_dataframes(bad_samples, sites_df)

    bad_features = samples_df.copy()
    bad_features.loc[0, "Au_ppb"] = np.nan

    with pytest.raises(ValueError, match="Element concentration columns must contain only finite numeric values"):
        GeoChemADDataset("sed1").load_from_dataframes(bad_features, sites_df)


def test_load_from_dataframes_requires_at_least_one_element_column(sample_frames):
    samples_df, sites_df = sample_frames
    samples_only_metadata = samples_df[["SAMPLEID", "SAMPLETYPE", "x", "y"]]

    with pytest.raises(ValueError, match="at least one element concentration column"):
        GeoChemADDataset("sed1").load_from_dataframes(samples_only_metadata, sites_df)


def test_create_evaluation_set_validates_arguments(sample_frames):
    samples_df, sites_df = sample_frames
    dataset = GeoChemADDataset("sed1").load_from_dataframes(samples_df, sites_df)

    with pytest.raises(ValueError, match="n_background must be non-negative"):
        dataset.create_evaluation_set(n_background=-1)

    with pytest.raises(ValueError, match="buffer_distance must be positive"):
        dataset.create_evaluation_set(buffer_distance=0)
