"""Dataset loading and management for GeoChemAD benchmark.

The GeoChemAD dataset comprises eight subsets from the Geological Survey of
Western Australia (GSWA), covering sediment, rock chip, and soil samples with
target elements Au, Cu, W, and Ni.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from geochemad.config import DATASET_INFO, TARGET_ELEMENT_COLUMNS


class GeoChemADDataset:
    """Dataset class for a single GeoChemAD subset.

    Each subset contains:
    - Geochemical samples with spatial coordinates and multi-element concentrations
    - Known mineralization sites with spatial coordinates

    Parameters
    ----------
    subset_id : str
        One of: sed1, sed2, rock1, rock2, rock3, soil1, soil2, soil3
    data_dir : str
        Root directory containing the dataset CSV files.
    """

    def __init__(self, subset_id: str, data_dir: str = "data"):
        if subset_id not in DATASET_INFO:
            raise ValueError(
                f"Unknown subset '{subset_id}'. Choose from: {list(DATASET_INFO.keys())}"
            )

        self.subset_id = subset_id
        self.data_dir = data_dir
        self.info = DATASET_INFO[subset_id]
        self.target_element = self.info["target_element"]

        self.samples_df: Optional[pd.DataFrame] = None
        self.sites_df: Optional[pd.DataFrame] = None

        self._coordinates: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None
        self._element_columns: Optional[list] = None

    def load(self) -> "GeoChemADDataset":
        """Load CSV files for this subset.

        Expects files at:
          {data_dir}/{subset_id}/samples.csv
          {data_dir}/{subset_id}/sites.csv
        """
        subset_dir = os.path.join(self.data_dir, self.subset_id)

        samples_path = os.path.join(subset_dir, "samples.csv")
        sites_path = os.path.join(subset_dir, "sites.csv")

        if not os.path.exists(samples_path):
            raise FileNotFoundError(
                f"Samples file not found: {samples_path}. "
                f"Download the dataset first using scripts/download_data.py"
            )

        self.samples_df = pd.read_csv(samples_path)
        if os.path.exists(sites_path):
            self.sites_df = pd.read_csv(sites_path)
        else:
            self.sites_df = pd.DataFrame(columns=["SiteID", "x", "y"])

        self._parse_data()
        return self

    def load_from_dataframes(
        self, samples_df: pd.DataFrame, sites_df: pd.DataFrame
    ) -> "GeoChemADDataset":
        """Load from pre-existing DataFrames (useful for testing)."""
        self.samples_df = samples_df.copy()
        self.sites_df = sites_df.copy()
        self._parse_data()
        return self

    def _parse_data(self) -> None:
        """Parse loaded DataFrames into arrays."""
        # Extract coordinates
        self._coordinates = self.samples_df[["x", "y"]].values.astype(np.float64)

        # Identify element concentration columns (exclude metadata)
        metadata_cols = {"SAMPLEID", "SAMPLETYPE", "x", "y", "SiteID", "ProjectID"}
        self._element_columns = [
            col for col in self.samples_df.columns if col not in metadata_cols
        ]
        self._features = self.samples_df[self._element_columns].values.astype(
            np.float64
        )

    @property
    def coordinates(self) -> np.ndarray:
        """Sample spatial coordinates, shape (n_samples, 2)."""
        if self._coordinates is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._coordinates

    @property
    def features(self) -> np.ndarray:
        """Geochemical concentration matrix, shape (n_samples, n_elements)."""
        if self._features is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._features

    @property
    def element_columns(self) -> list:
        """List of element column names."""
        if self._element_columns is None:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._element_columns

    @property
    def n_samples(self) -> int:
        return self.coordinates.shape[0]

    @property
    def n_elements(self) -> int:
        return self.features.shape[1]

    @property
    def site_coordinates(self) -> np.ndarray:
        """Known mineralization site coordinates, shape (n_sites, 2)."""
        if self.sites_df is None or self.sites_df.empty:
            return np.empty((0, 2))
        return self.sites_df[["x", "y"]].values.astype(np.float64)

    @property
    def n_sites(self) -> int:
        return self.site_coordinates.shape[0]

    def get_target_element_index(self) -> int:
        """Get the column index for the target element."""
        target_col = TARGET_ELEMENT_COLUMNS.get(self.target_element)
        if target_col and target_col in self._element_columns:
            return self._element_columns.index(target_col)
        # Fallback: search for partial match
        for i, col in enumerate(self._element_columns):
            if self.target_element in col:
                return i
        raise ValueError(
            f"Target element '{self.target_element}' not found in columns"
        )

    def get_target_element_values(self) -> np.ndarray:
        """Get target element concentration values for all samples."""
        idx = self.get_target_element_index()
        return self.features[:, idx]

    def create_evaluation_set(
        self,
        n_background: Optional[int] = None,
        buffer_distance: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> dict:
        """Create an evaluation set with positive (near deposits) and negative (background) samples.

        Following the paper, background samples are randomly selected from the
        research area. Positive samples are those geochemical sample points
        nearest to known deposit sites.

        Parameters
        ----------
        n_background : int, optional
            Number of background samples. Defaults to same as number of sites.
        buffer_distance : float, optional
            Max distance to assign a sample as positive. If None, uses nearest sample.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            - positive_indices: indices of positive samples
            - negative_indices: indices of background samples
            - labels: binary labels (1=positive, 0=negative)
            - indices: combined indices
        """
        rng = np.random.RandomState(random_state)
        site_coords = self.site_coordinates

        if len(site_coords) == 0:
            raise ValueError("No mineralization sites available for evaluation")

        # Find nearest sample to each deposit site
        tree = cKDTree(self.coordinates)
        distances, nearest_indices = tree.query(site_coords, k=1)
        positive_indices = np.unique(nearest_indices)

        if buffer_distance is not None:
            # Also include samples within buffer distance of any site
            site_tree = cKDTree(site_coords)
            nearby = site_tree.query_ball_point(self.coordinates, r=buffer_distance)
            extra_positives = np.where([len(n) > 0 for n in nearby])[0]
            positive_indices = np.unique(
                np.concatenate([positive_indices, extra_positives])
            )

        # Select background samples (excluding positives)
        all_indices = np.arange(self.n_samples)
        candidate_bg = np.setdiff1d(all_indices, positive_indices)

        if n_background is None:
            n_background = len(positive_indices)

        n_background = min(n_background, len(candidate_bg))
        negative_indices = rng.choice(candidate_bg, size=n_background, replace=False)

        # Combine
        indices = np.concatenate([positive_indices, negative_indices])
        labels = np.concatenate(
            [np.ones(len(positive_indices)), np.zeros(len(negative_indices))]
        )

        return {
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "labels": labels,
            "indices": indices,
        }


def load_dataset(subset_id: str, data_dir: str = "data") -> GeoChemADDataset:
    """Convenience function to load a GeoChemAD subset.

    Parameters
    ----------
    subset_id : str
        One of: sed1, sed2, rock1, rock2, rock3, soil1, soil2, soil3
    data_dir : str
        Root directory containing the dataset files.

    Returns
    -------
    GeoChemADDataset
        Loaded dataset object.
    """
    return GeoChemADDataset(subset_id, data_dir).load()
