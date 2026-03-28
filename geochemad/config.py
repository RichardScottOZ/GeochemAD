"""Configuration and hyperparameters for GeoChemAD benchmark."""

from dataclasses import dataclass, field
from typing import Optional


# Dataset metadata: source type, target element, number of elements
DATASET_INFO = {
    "sed1": {"source": "sediment", "target_element": "Au", "num_elem": 124, "num_samples": 1392, "num_anomaly": 32},
    "sed2": {"source": "sediment", "target_element": "Cu", "num_elem": 124, "num_samples": 2994, "num_anomaly": 21},
    "rock1": {"source": "rockchip", "target_element": "W", "num_elem": 124, "num_samples": 3790, "num_anomaly": 7},
    "rock2": {"source": "rockchip", "target_element": "Au", "num_elem": 124, "num_samples": 224, "num_anomaly": 12},
    "rock3": {"source": "rockchip", "target_element": "Cu", "num_elem": 124, "num_samples": 9624, "num_anomaly": 21},
    "soil1": {"source": "soil", "target_element": "Au", "num_elem": 126, "num_samples": 2734, "num_anomaly": 14},
    "soil2": {"source": "soil", "target_element": "Au", "num_elem": 126, "num_samples": 5163, "num_anomaly": 17},
    "soil3": {"source": "soil", "target_element": "Ni", "num_elem": 126, "num_samples": 21040, "num_anomaly": 13},
}

ALL_DATASET_IDS = list(DATASET_INFO.keys())

# Target element column name mapping
TARGET_ELEMENT_COLUMNS = {
    "Au": "Au_ppb",
    "Cu": "Cu_ppm",
    "W": "W_ppm",
    "Ni": "Ni_ppm",
}


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""

    transform: str = "clr"  # "raw", "clr", "ilr"
    feature_selection: str = "all"  # "all", "manual", "pca"
    pca_variance_ratio: float = 0.95
    handle_abnormal: str = "half_detection_limit"  # "remove", "half_detection_limit"
    abnormal_values: tuple = (-9999, -0.5)
    normalize: bool = True


@dataclass
class AutoEncoderConfig:
    """Configuration for AutoEncoder and VAE models."""

    hidden_dims: list = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    dropout: float = 0.1
    weight_decay: float = 1e-5


@dataclass
class VAEGANConfig:
    """Configuration for VAE-GAN variants."""

    hidden_dims: list = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    dropout: float = 0.1
    lambda_recon: float = 1.0
    lambda_kl: float = 0.5
    lambda_adv: float = 0.1


@dataclass
class TransformerConfig:
    """Configuration for vanilla Transformer model."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    mask_ratio: float = 0.15


@dataclass
class GeoChemFormerConfig:
    """Configuration for GeoChemFormer model."""

    # Spatial Context Learning
    k_neighbours: int = 128
    scl_d_model: int = 128
    scl_n_heads: int = 4
    scl_n_layers: int = 4
    scl_d_ff: int = 256
    scl_dropout: float = 0.1
    scl_epochs: int = 50
    scl_learning_rate: float = 1e-3
    scl_batch_size: int = 64

    # Element Dependency Modelling
    edm_d_model: int = 128
    edm_n_heads: int = 4
    edm_n_layers: int = 4
    edm_d_ff: int = 256
    edm_dropout: float = 0.1
    edm_epochs: int = 100
    edm_learning_rate: float = 1e-3
    edm_batch_size: int = 128

    # Element embedding
    elem_embed_dim: int = 64


@dataclass
class DiffusionConfig:
    """Configuration for VAE-Diffusion model."""

    hidden_dims: list = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 8
    num_timesteps: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class BenchmarkConfig:
    """Configuration for the full benchmark."""

    data_dir: str = "data"
    output_dir: str = "results"
    datasets: list = field(default_factory=lambda: ALL_DATASET_IDS)
    num_runs: int = 20  # Number of runs for averaging (with random background sampling)
    random_seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ae_config: AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    vaegan_config: VAEGANConfig = field(default_factory=VAEGANConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    geochemformer_config: GeoChemFormerConfig = field(default_factory=GeoChemFormerConfig)
    diffusion_config: DiffusionConfig = field(default_factory=DiffusionConfig)
