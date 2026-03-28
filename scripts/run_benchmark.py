"""Main entry point for running the GeoChemAD benchmark."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geochemad.config import BenchmarkConfig, PreprocessConfig, ALL_DATASET_IDS
from geochemad.benchmark import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root directory containing dataset CSV files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=ALL_DATASET_IDS,
        help="Dataset subsets to evaluate (default: all)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=["ZS", "MD", "KNN", "IF", "OSVM", "AE", "VAE",
                 "VAE-G", "VAE-CG", "VAE-D", "T1", "T2"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=20,
        help="Number of evaluation runs with random background sampling"
    )
    parser.add_argument(
        "--transform", type=str, default="clr",
        choices=["raw", "clr", "ilr"],
        help="Data transformation method"
    )
    parser.add_argument(
        "--feature-selection", type=str, default="all",
        choices=["all", "manual", "pca"],
        help="Feature selection method"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    preprocess_config = PreprocessConfig(
        transform=args.transform,
        feature_selection=args.feature_selection,
    )

    config = BenchmarkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datasets=args.datasets or ALL_DATASET_IDS,
        num_runs=args.num_runs,
        random_seed=args.seed,
        device=args.device,
        preprocess=preprocess_config,
    )

    results = run_benchmark(
        config=config,
        model_names=args.models,
        verbose=not args.quiet,
    )

    print(f"\nResults saved to {args.output_dir}/")
    return results


if __name__ == "__main__":
    main()
