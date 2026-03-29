from pathlib import Path

import pandas as pd
import pytest

from scripts.download_data import generate_sample_data


@pytest.fixture
def sample_frames():
    samples_df = pd.DataFrame(
        {
            "SAMPLEID": ["S001", "S002", "S003", "S004"],
            "SAMPLETYPE": ["sediment"] * 4,
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "Au_ppb": [10.0, 12.0, 14.0, 18.0],
            "Cu_ppm": [100.0, 120.0, 90.0, 80.0],
            "As_ppm": [5.0, 6.0, 8.0, 7.0],
        }
    )
    sites_df = pd.DataFrame(
        {
            "SiteID": ["D001", "D002"],
            "x": [0.1, 2.8],
            "y": [0.1, 0.9],
        }
    )
    return samples_df, sites_df


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "synthetic-data"
    generate_sample_data(str(data_dir), n_samples=24, n_elements=8)
    return data_dir
