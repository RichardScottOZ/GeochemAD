"""AutoEncoder and Variational AutoEncoder models for anomaly detection.

Reconstruction-based anomaly detection: samples with high reconstruction
error are flagged as anomalous.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from geochemad.models.statistical import BaseAnomalyDetector


class _AutoEncoder(nn.Module):
    """Standard AutoEncoder network."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class _VAE(nn.Module):
    """Variational AutoEncoder network."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class AutoEncoderAD(BaseAnomalyDetector):
    """AutoEncoder-based anomaly detection.

    Anomaly score is the mean squared reconstruction error.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        device: str = "auto",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = self._resolve_device(device)
        self.model = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "AutoEncoderAD":
        input_dim = features.shape[1]
        self.model = _AutoEncoder(
            input_dim, self.hidden_dims, self.latent_dim, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                x_hat, _ = self.model(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            x_hat, _ = self.model(x)
            errors = ((x - x_hat) ** 2).mean(dim=1)
        return errors.cpu().numpy()


class VAEAD(BaseAnomalyDetector):
    """Variational AutoEncoder-based anomaly detection.

    Anomaly score is the reconstruction error plus KL divergence.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        kl_weight: float = 0.5,
        device: str = "auto",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.device = self._resolve_device(device)
        self.model = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "VAEAD":
        input_dim = features.shape[1]
        self.model = _VAE(
            input_dim, self.hidden_dims, self.latent_dim, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                x_hat, mu, logvar, _ = self.model(batch)

                recon_loss = nn.functional.mse_loss(x_hat, batch, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            x_hat, mu, logvar, _ = self.model(x)
            recon_error = ((x - x_hat) ** 2).mean(dim=1)
        return recon_error.cpu().numpy()
