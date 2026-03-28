"""Benchmarking pipeline for GeoChemAD.

Runs all anomaly detection methods on all dataset subsets and collects
performance metrics as described in the paper.
"""

import os
import time
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch

from geochemad.config import (
    BenchmarkConfig,
    DATASET_INFO,
)
from geochemad.data.dataset import GeoChemADDataset
from geochemad.data.preprocessing import preprocess_pipeline
from geochemad.evaluation.metrics import evaluate_model
from geochemad.models.statistical import ZScore, MahalanobisDistance, KNNAnomaly
from geochemad.models.classical_ml import IsolationForestAD, OneClassSVMAD
from geochemad.models.autoencoder import AutoEncoderAD, VAEAD
from geochemad.models.generative import VAEGANAD, VAECascadeGANAD, VAEDiffusionAD
from geochemad.models.transformer import VanillaTransformerAD
from geochemad.models.geochemformer import GeoChemFormer


def get_all_models(config: BenchmarkConfig, device: str) -> dict:
    """Initialize all anomaly detection models.

    Returns
    -------
    dict
        Model name -> model instance.
    """
    return {
        "ZS": ZScore(),
        "MD": MahalanobisDistance(),
        "KNN": KNNAnomaly(k=10),
        "IF": IsolationForestAD(random_state=config.random_seed),
        "OSVM": OneClassSVMAD(),
        "AE": AutoEncoderAD(
            hidden_dims=config.ae_config.hidden_dims,
            latent_dim=config.ae_config.latent_dim,
            learning_rate=config.ae_config.learning_rate,
            batch_size=config.ae_config.batch_size,
            num_epochs=config.ae_config.num_epochs,
            dropout=config.ae_config.dropout,
            device=device,
        ),
        "VAE": VAEAD(
            hidden_dims=config.ae_config.hidden_dims,
            latent_dim=config.ae_config.latent_dim,
            learning_rate=config.ae_config.learning_rate,
            batch_size=config.ae_config.batch_size,
            num_epochs=config.ae_config.num_epochs,
            dropout=config.ae_config.dropout,
            device=device,
        ),
        "VAE-G": VAEGANAD(
            hidden_dims=config.vaegan_config.hidden_dims,
            latent_dim=config.vaegan_config.latent_dim,
            learning_rate=config.vaegan_config.learning_rate,
            batch_size=config.vaegan_config.batch_size,
            num_epochs=config.vaegan_config.num_epochs,
            dropout=config.vaegan_config.dropout,
            lambda_recon=config.vaegan_config.lambda_recon,
            lambda_kl=config.vaegan_config.lambda_kl,
            lambda_adv=config.vaegan_config.lambda_adv,
            device=device,
        ),
        "VAE-CG": VAECascadeGANAD(
            hidden_dims=config.vaegan_config.hidden_dims,
            latent_dim=config.vaegan_config.latent_dim,
            learning_rate=config.vaegan_config.learning_rate,
            batch_size=config.vaegan_config.batch_size,
            num_epochs=config.vaegan_config.num_epochs,
            dropout=config.vaegan_config.dropout,
            device=device,
        ),
        "VAE-D": VAEDiffusionAD(
            hidden_dims=config.ae_config.hidden_dims,
            latent_dim=config.ae_config.latent_dim,
            learning_rate=config.ae_config.learning_rate,
            batch_size=config.ae_config.batch_size,
            num_epochs=config.ae_config.num_epochs,
            device=device,
        ),
        "T1": VanillaTransformerAD(
            d_model=config.transformer_config.d_model,
            n_heads=config.transformer_config.n_heads,
            n_layers=config.transformer_config.n_layers,
            d_ff=config.transformer_config.d_ff,
            dropout=config.transformer_config.dropout,
            learning_rate=config.transformer_config.learning_rate,
            batch_size=config.transformer_config.batch_size,
            num_epochs=config.transformer_config.num_epochs,
            mask_ratio=config.transformer_config.mask_ratio,
            device=device,
        ),
        "T2": GeoChemFormer(
            k_neighbours=config.geochemformer_config.k_neighbours,
            scl_d_model=config.geochemformer_config.scl_d_model,
            scl_n_heads=config.geochemformer_config.scl_n_heads,
            scl_n_layers=config.geochemformer_config.scl_n_layers,
            scl_d_ff=config.geochemformer_config.scl_d_ff,
            scl_dropout=config.geochemformer_config.scl_dropout,
            scl_epochs=config.geochemformer_config.scl_epochs,
            scl_learning_rate=config.geochemformer_config.scl_learning_rate,
            scl_batch_size=config.geochemformer_config.scl_batch_size,
            edm_d_model=config.geochemformer_config.edm_d_model,
            edm_n_heads=config.geochemformer_config.edm_n_heads,
            edm_n_layers=config.geochemformer_config.edm_n_layers,
            edm_d_ff=config.geochemformer_config.edm_d_ff,
            edm_dropout=config.geochemformer_config.edm_dropout,
            edm_epochs=config.geochemformer_config.edm_epochs,
            edm_learning_rate=config.geochemformer_config.edm_learning_rate,
            edm_batch_size=config.geochemformer_config.edm_batch_size,
            elem_embed_dim=config.geochemformer_config.elem_embed_dim,
            device=device,
        ),
    }


def run_single_evaluation(
    model,
    model_name: str,
    features: np.ndarray,
    coordinates: np.ndarray,
    dataset: GeoChemADDataset,
    random_state: int,
) -> dict:
    """Run a single evaluation for one model on one dataset.

    Parameters
    ----------
    model : BaseAnomalyDetector
        Anomaly detection model.
    model_name : str
        Model identifier.
    features : np.ndarray
        Preprocessed features.
    coordinates : np.ndarray
        Spatial coordinates.
    dataset : GeoChemADDataset
        Dataset object with sites info.
    random_state : int
        Random seed for background sampling.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    # Create evaluation set
    eval_set = dataset.create_evaluation_set(random_state=random_state)

    # Fit model on all samples (unsupervised)
    needs_coords = model_name in ("T2",)
    if needs_coords:
        model.fit(features, coordinates)
    else:
        model.fit(features)

    # Score all samples
    if needs_coords:
        all_scores = model.score(features, coordinates)
    else:
        all_scores = model.score(features)

    # Get scores for evaluation subset
    eval_scores = all_scores[eval_set["indices"]]

    # Compute metrics
    metrics = evaluate_model(
        labels=eval_set["labels"],
        scores=eval_scores,
        sample_coordinates=coordinates[eval_set["indices"]],
        site_coordinates=dataset.site_coordinates,
    )

    return metrics


def run_benchmark(
    config: BenchmarkConfig,
    datasets: dict = None,
    model_names: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full GeoChemAD benchmark.

    Evaluates all models on all datasets, repeating evaluation with random
    background sampling as specified in the paper.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    datasets : dict, optional
        Pre-loaded datasets. If None, loads from config.data_dir.
    model_names : list, optional
        Subset of model names to evaluate.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results table with AUC scores indexed by dataset and model.
    """
    # Determine device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if verbose:
        print(f"Using device: {device}")

    # Initialize models
    all_models = get_all_models(config, device)
    if model_names:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    # Results storage
    results = {}

    for subset_id in config.datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {subset_id}")
            print(f"{'='*60}")

        # Load dataset
        if datasets and subset_id in datasets:
            dataset = datasets[subset_id]
        else:
            try:
                dataset = GeoChemADDataset(subset_id, config.data_dir).load()
            except FileNotFoundError as e:
                if verbose:
                    print(f"  Skipping {subset_id}: {e}")
                continue

        # Preprocess
        processed, metadata = preprocess_pipeline(
            features=dataset.features,
            element_columns=dataset.element_columns,
            transform=config.preprocess.transform,
            feature_selection=config.preprocess.feature_selection,
            target_element=dataset.target_element,
            normalize=config.preprocess.normalize,
            handle_abnormal=config.preprocess.handle_abnormal,
        )

        if verbose:
            print(f"  Samples: {processed.shape[0]}, Features: {processed.shape[1]}")
            print(f"  Sites: {dataset.n_sites}")

        results[subset_id] = {}

        for model_name, model in all_models.items():
            if verbose:
                print(f"  Running {model_name}...", end="", flush=True)

            run_metrics = []
            start_time = time.time()

            for run_idx in range(config.num_runs):
                seed = config.random_seed + run_idx

                try:
                    # Re-initialize model for each run
                    fresh_model = get_all_models(config, device)[model_name]

                    metrics = run_single_evaluation(
                        model=fresh_model,
                        model_name=model_name,
                        features=processed,
                        coordinates=dataset.coordinates,
                        dataset=dataset,
                        random_state=seed,
                    )
                    run_metrics.append(metrics)
                except Exception as e:
                    if verbose:
                        print(f" Error in run {run_idx}: {e}")
                    continue

            elapsed = time.time() - start_time

            if run_metrics:
                avg_auc = np.mean([m["auc"] for m in run_metrics])
                std_auc = np.std([m["auc"] for m in run_metrics])
                results[subset_id][model_name] = {
                    "auc_mean": float(avg_auc),
                    "auc_std": float(std_auc),
                    "ap_mean": float(np.mean([m["ap"] for m in run_metrics])),
                    "pr_auc_mean": float(np.mean([m["pr_auc"] for m in run_metrics])),
                    "n_runs": len(run_metrics),
                    "time_seconds": elapsed,
                }
                if verbose:
                    print(f" AUC: {avg_auc:.4f} ± {std_auc:.4f} ({elapsed:.1f}s)")
            else:
                results[subset_id][model_name] = {
                    "auc_mean": float("nan"),
                    "auc_std": float("nan"),
                    "n_runs": 0,
                }
                if verbose:
                    print(" FAILED")

    # Build results DataFrame
    auc_table = {}
    for subset_id in results:
        auc_table[subset_id] = {}
        for model_name in results[subset_id]:
            auc_table[subset_id][model_name] = results[subset_id][model_name]["auc_mean"]

    df = pd.DataFrame(auc_table).T
    if len(df) > 0:
        df.loc["Avg."] = df.mean()
        df.loc["Var."] = df.var()

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    df.to_csv(os.path.join(config.output_dir, "benchmark_results.csv"))

    with open(os.path.join(config.output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("Results Summary (AUC)")
        print(f"{'='*60}")
        print(df.to_string(float_format="%.4f"))

    return df
