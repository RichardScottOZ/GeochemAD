"""Download GeoChemAD dataset from GSWA data portal.

The data is sourced from the Geological Survey of Western Australia's
DMPE Data and Software Center (https://dasc.dmirs.wa.gov.au/).

This script provides instructions and utilities for downloading and
organizing the dataset files.
"""

import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geochemad.config import DATASET_INFO, ALL_DATASET_IDS


DATA_SOURCE_INFO = """
GeoChemAD Dataset Download Instructions
========================================

The GeoChemAD dataset is compiled from government-led geological surveys
from the Geological Survey of Western Australia (GSWA).

Data Source:
  DMPE Data and Software Center
  https://dasc.dmirs.wa.gov.au/

Datasets Used:
  - CM02: Near-Surface Geochemistry (regional geochemical assays)
  - CM01: Mineralization Sites (verified deposit locations)

All data uses the GDA2020 coordinate system.

Expected Directory Structure:
  data/
  ├── sed1/
  │   ├── samples.csv    # Geochemical samples
  │   └── sites.csv      # Known mineralization sites
  ├── sed2/
  │   ├── samples.csv
  │   └── sites.csv
  ├── rock1/
  │   ├── samples.csv
  │   └── sites.csv
  ├── rock2/
  │   ├── samples.csv
  │   └── sites.csv
  ├── rock3/
  │   ├── samples.csv
  │   └── sites.csv
  ├── soil1/
  │   ├── samples.csv
  │   └── sites.csv
  ├── soil2/
  │   ├── samples.csv
  │   └── sites.csv
  └── soil3/
      ├── samples.csv
      └── sites.csv

CSV Format - samples.csv:
  Required columns: SAMPLEID, SAMPLETYPE, x (longitude), y (latitude)
  Plus geochemical concentration columns (e.g., Ag_ppm, Au_ppb, Cu_ppm, ...)

CSV Format - sites.csv:
  Required columns: SiteID, x (longitude), y (latitude)
  Optional: ProjectID, and other metadata
"""


def create_directory_structure(data_dir: str):
    """Create the expected directory structure."""
    for subset_id in ALL_DATASET_IDS:
        subset_dir = os.path.join(data_dir, subset_id)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"Created: {subset_dir}/")


def generate_sample_data(data_dir: str, n_samples: int = 100, n_elements: int = 20):
    """Generate synthetic sample data for testing the pipeline.

    This creates small synthetic datasets that mimic the structure of the
    real GeoChemAD data, useful for testing the code pipeline.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)

    element_names = [
        "Ag_ppm", "Al_pct", "As_ppm", "Au_ppb", "Ba_ppm",
        "Bi_ppm", "Ca_pct", "Cd_ppm", "Ce_ppm", "Co_ppm",
        "Cr_ppm", "Cu_ppm", "Fe_pct", "Ga_ppm", "Hg_ppb",
        "K_pct", "La_ppm", "Li_ppm", "Mg_pct", "Mn_ppm",
        "Mo_ppm", "Na_pct", "Ni_ppm", "P_ppm", "Pb_ppm",
        "Rb_ppm", "S_pct", "Sb_ppm", "Sc_ppm", "Se_ppm",
        "Si_pct", "Sn_ppm", "Sr_ppm", "Te_ppm", "Th_ppm",
        "Ti_pct", "Tl_ppm", "U_ppm", "V_ppm", "W_ppm",
        "Y_ppm", "Zn_ppm", "Zr_ppm",
    ]

    for subset_id, info in DATASET_INFO.items():
        subset_dir = os.path.join(data_dir, subset_id)
        os.makedirs(subset_dir, exist_ok=True)

        n = min(n_samples, info["num_samples"])
        n_elem = min(n_elements, len(element_names))
        n_sites = min(5, info["num_anomaly"])

        # Generate sample data
        x_center = 115.0 + rng.uniform(-2, 2)
        y_center = -30.0 + rng.uniform(-2, 2)

        samples = {
            "SAMPLEID": [f"S{i:05d}" for i in range(n)],
            "SAMPLETYPE": [info["source"]] * n,
            "x": x_center + rng.uniform(-0.5, 0.5, n),
            "y": y_center + rng.uniform(-0.5, 0.5, n),
        }

        for elem in element_names[:n_elem]:
            # Log-normal distribution for concentrations
            samples[elem] = np.abs(rng.lognormal(mean=1.0, sigma=1.5, size=n))
            # Add some abnormal values
            n_abnormal = max(1, n // 20)
            abnormal_idx = rng.choice(n, n_abnormal, replace=False)
            samples[elem][abnormal_idx] = rng.choice([-9999, -0.5, 0], n_abnormal)

        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(os.path.join(subset_dir, "samples.csv"), index=False)

        # Generate site data
        sites = {
            "SiteID": [f"D{i:03d}" for i in range(n_sites)],
            "x": x_center + rng.uniform(-0.3, 0.3, n_sites),
            "y": y_center + rng.uniform(-0.3, 0.3, n_sites),
        }
        sites_df = pd.DataFrame(sites)
        sites_df.to_csv(os.path.join(subset_dir, "sites.csv"), index=False)

        print(f"Generated synthetic data for {subset_id}: {n} samples, {n_sites} sites")

    print(f"\nSynthetic data saved to {data_dir}/")
    print("NOTE: This is synthetic data for testing only. For real experiments,")
    print("download the actual data from GSWA (see --info flag).")


def main():
    parser = argparse.ArgumentParser(
        description="Download or prepare GeoChemAD dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory for dataset files"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print download instructions"
    )
    parser.add_argument(
        "--create-dirs", action="store_true",
        help="Create expected directory structure"
    )
    parser.add_argument(
        "--generate-synthetic", action="store_true",
        help="Generate synthetic test data"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples per subset for synthetic data"
    )
    args = parser.parse_args()

    if args.info:
        print(DATA_SOURCE_INFO)
        print("\nDataset Summary:")
        print(f"{'ID':<8} {'Source':<12} {'Target':<6} {'#Samples':<10} {'#Sites':<8} {'#Elem':<6}")
        print("-" * 60)
        for sid, info in DATASET_INFO.items():
            print(f"{sid:<8} {info['source']:<12} {info['target_element']:<6} "
                  f"{info['num_samples']:<10} {info['num_anomaly']:<8} {info['num_elem']:<6}")
        return

    if args.create_dirs:
        create_directory_structure(args.data_dir)
        return

    if args.generate_synthetic:
        generate_sample_data(args.data_dir, n_samples=args.n_samples)
        return

    print("Use --info for download instructions, --create-dirs to set up directories,")
    print("or --generate-synthetic to create test data.")


if __name__ == "__main__":
    main()
