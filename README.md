# GeochemAD

**Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration**

Implementation of the GeoChemAD benchmark and GeoChemFormer model from [Ding et al. (2026)](https://arxiv.org/abs/2603.13068v1).

## Overview

Geochemical anomaly detection plays a critical role in mineral exploration. GeoChemAD is an open-source benchmark dataset compiled from government-led geological surveys (GSWA), covering multiple regions, sampling sources, and target elements.

This repository provides:
- **8 geochemical dataset subsets** (sediment, rock chip, soil) with diverse target elements (Au, Cu, W, Ni)
- **12 unsupervised anomaly detection methods** spanning statistical, ML, deep generative, and transformer approaches
- **GeoChemFormer**: a transformer-based framework with spatial context learning and element dependency modelling
- **Standardized evaluation** using AUC, AP, PR-AUC, and Distance-to-Deposit metrics

### Implemented Models

| Category | Models |
|----------|--------|
| Statistical | Z-score (ZS), Mahalanobis Distance (MD), k-Nearest Neighbors (KNN) |
| Classical ML | Isolation Forest (IF), One-Class SVM (OSVM) |
| Deep Generative | AutoEncoder (AE), VAE, VAE-GAN, VAE-Cascade-GAN, VAE-Diffusion |
| Transformer | Vanilla Transformer (T1), **GeoChemFormer** (T2) |

### Dataset Subsets

| ID | Source | Target Element | #Samples | Area (km²) | #Sites |
|----|--------|---------------|----------|------------|--------|
| sed1 | Sediment | Au | 1,392 | ~8,523 | 32 |
| sed2 | Sediment | Cu | 2,994 | ~6,671 | 21 |
| rock1 | Rock Chip | W | 3,790 | ~3,177 | 7 |
| rock2 | Rock Chip | Au | 224 | ~2 | 12 |
| rock3 | Rock Chip | Cu | 9,624 | ~7,423 | 21 |
| soil1 | Soil | Au | 2,734 | ~5.7 | 14 |
| soil2 | Soil | Au | 5,163 | ~57 | 17 |
| soil3 | Soil | Ni | 21,040 | ~2,018 | 13 |

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

The dataset is sourced from the [Geological Survey of Western Australia (GSWA)](https://dasc.dmirs.wa.gov.au/). For information on downloading and formatting the data:

```bash
python scripts/download_data.py --info
```

To generate synthetic test data for pipeline validation:

```bash
python scripts/download_data.py --generate-synthetic --data-dir data
```

### 2. Run the Benchmark

Run all models on all datasets:

```bash
python scripts/run_benchmark.py --data-dir data --output-dir results
```

Run specific models and datasets:

```bash
python scripts/run_benchmark.py \
    --data-dir data \
    --datasets sed1 sed2 \
    --models ZS MD KNN IF AE T2 \
    --transform clr \
    --num-runs 20
```

### 3. Python API

```python
from geochemad.data import GeoChemADDataset
from geochemad.data.preprocessing import preprocess_pipeline
from geochemad.models import GeoChemFormer, AutoEncoderAD, ZScore
from geochemad.evaluation import evaluate_model

# Load dataset
dataset = GeoChemADDataset("sed1", data_dir="data").load()

# Preprocess
features, metadata = preprocess_pipeline(
    features=dataset.features,
    element_columns=dataset.element_columns,
    transform="clr",
    target_element=dataset.target_element,
)

# Train GeoChemFormer
model = GeoChemFormer(k_neighbours=128)
model.fit(features, dataset.coordinates)

# Score samples
scores = model.score(features, dataset.coordinates)

# Evaluate
eval_set = dataset.create_evaluation_set(random_state=42)
metrics = evaluate_model(
    labels=eval_set["labels"],
    scores=scores[eval_set["indices"]],
)
print(f"AUC: {metrics['auc']:.4f}")
```

## Data Preprocessing

The pipeline supports:
- **Abnormal value handling**: Replace -9999, negative, zero values with half detection limit
- **Compositional transforms**: Centered Log-Ratio (CLR) and Isometric Log-Ratio (ILR)
- **Feature selection**: All elements, manual (pathfinder elements), PCA-based
- **Normalization**: Standard scaling or min-max

```bash
# Compare preprocessing strategies
python scripts/run_benchmark.py --transform raw --models AE T2
python scripts/run_benchmark.py --transform clr --models AE T2
python scripts/run_benchmark.py --transform ilr --models AE T2
```

## GeoChemFormer Architecture

GeoChemFormer is a two-stage transformer framework:

**Stage 1 - Spatial Context Learning (SCL):**
- Builds KD-tree to find K nearest neighbours for each sample
- Processes token sequence: `[target_element, query_location, neighbour_1, ..., neighbour_K]`
- Learns to predict target element concentration from neighbourhood context
- Produces spatially informed representations

**Stage 2 - Element Dependency Modelling (EDM):**
- Processes token sequence: `[geo_context, element_1, ..., element_C]`
- Models inter-element dependencies conditioned on spatial context
- Anomaly score = mean squared reconstruction error across all elements

## Project Structure

```
geochemad/
├── __init__.py
├── config.py                # Configuration and hyperparameters
├── benchmark.py             # Main benchmarking pipeline
├── data/
│   ├── dataset.py           # Dataset loading and management
│   └── preprocessing.py     # CLR, ILR, feature selection, normalization
├── models/
│   ├── statistical.py       # Z-score, Mahalanobis, kNN
│   ├── classical_ml.py      # Isolation Forest, One-Class SVM
│   ├── autoencoder.py       # AE, VAE
│   ├── generative.py        # VAE-GAN, VAE-Cascade-GAN, VAE-Diffusion
│   ├── transformer.py       # Vanilla Transformer (T1)
│   └── geochemformer.py     # GeoChemFormer (T2)
├── evaluation/
│   └── metrics.py           # AUC, AP, PR-AUC, DTD metrics
└── utils/
    └── spatial.py           # KD-tree, IDW, Kriging interpolation
scripts/
├── run_benchmark.py         # CLI entry point
└── download_data.py         # Data download/preparation utilities
```

## Citation

```bibtex
@article{ding2026geochemad,
  title={GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration},
  author={Ding, Yihao and Zhang, Yiran and Gonzalez, Chris and Holden, Eun-Jung and Liu, Wei},
  journal={arXiv preprint arXiv:2603.13068},
  year={2026}
}
```

## License

This project implements the methods described in the GeoChemAD paper. The dataset is sourced from publicly available GSWA geological surveys.
