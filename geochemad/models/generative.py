"""Generative models for anomaly detection.

Implements VAE-GAN, VAE-Cascade-GAN, and VAE-Diffusion variants
as described in the GeoChemAD benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from geochemad.models.statistical import BaseAnomalyDetector


class _Discriminator(nn.Module):
    """Discriminator network for GAN variants."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _VAEEncoder(nn.Module):
    """VAE encoder for generative models."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Decoder network for generative models."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class VAEGANAD(BaseAnomalyDetector):
    """VAE-GAN anomaly detection.

    Combines VAE reconstruction with adversarial training. The anomaly
    score is the reconstruction error from the VAE component.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        dropout: float = 0.1,
        lambda_recon: float = 1.0,
        lambda_kl: float = 0.5,
        lambda_adv: float = 0.1,
        device: str = "auto",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_adv = lambda_adv
        self.device = self._resolve_device(device)
        self.encoder = None
        self.decoder = None
        self.discriminator = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "VAEGANAD":
        input_dim = features.shape[1]
        self.encoder = _VAEEncoder(
            input_dim, self.hidden_dims, self.latent_dim, self.dropout
        ).to(self.device)
        self.decoder = _Decoder(
            self.latent_dim, self.hidden_dims, input_dim, self.dropout
        ).to(self.device)
        self.discriminator = _Discriminator(
            input_dim, [64, 32], self.dropout
        ).to(self.device)

        opt_vae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        bce = nn.BCELoss()

        for epoch in range(self.num_epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                bs = batch.size(0)

                # Encode and decode
                mu, logvar = self.encoder(batch)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_hat = self.decoder(z)

                # Train discriminator
                real_labels = torch.ones(bs, 1, device=self.device)
                fake_labels = torch.zeros(bs, 1, device=self.device)

                d_real = self.discriminator(batch)
                d_fake = self.discriminator(x_hat.detach())
                d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)

                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

                # Train VAE + generator
                d_fake_for_g = self.discriminator(x_hat)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                adv_loss = bce(d_fake_for_g, real_labels)

                vae_loss = (
                    self.lambda_recon * recon_loss
                    + self.lambda_kl * kl_loss
                    + self.lambda_adv * adv_loss
                )

                opt_vae.zero_grad()
                vae_loss.backward()
                opt_vae.step()

        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            mu, logvar = self.encoder(x)
            x_hat = self.decoder(mu)
            errors = ((x - x_hat) ** 2).mean(dim=1)
        return errors.cpu().numpy()


class VAECascadeGANAD(BaseAnomalyDetector):
    """VAE-Cascade-GAN anomaly detection.

    A cascaded architecture with two stage VAE-GAN where the second
    stage refines the reconstruction from the first stage.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        dropout: float = 0.1,
        lambda_recon: float = 1.0,
        lambda_kl: float = 0.5,
        lambda_adv: float = 0.1,
        device: str = "auto",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_adv = lambda_adv
        self.device = self._resolve_device(device)

        # Stage 1 and Stage 2 components
        self.encoder1 = None
        self.decoder1 = None
        self.disc1 = None
        self.encoder2 = None
        self.decoder2 = None
        self.disc2 = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _train_stage(self, loader, encoder, decoder, disc, input_dim):
        opt_vae = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr
        )
        opt_disc = torch.optim.Adam(disc.parameters(), lr=self.lr)
        bce = nn.BCELoss()

        for epoch in range(self.num_epochs // 2):
            for (batch,) in loader:
                batch = batch.to(self.device)
                bs = batch.size(0)

                mu, logvar = encoder(batch)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_hat = decoder(z)

                real_labels = torch.ones(bs, 1, device=self.device)
                fake_labels = torch.zeros(bs, 1, device=self.device)

                d_real = disc(batch)
                d_fake = disc(x_hat.detach())
                d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)
                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

                d_fake_g = disc(x_hat)
                recon_loss = nn.functional.mse_loss(x_hat, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                adv_loss = bce(d_fake_g, real_labels)
                vae_loss = (
                    self.lambda_recon * recon_loss
                    + self.lambda_kl * kl_loss
                    + self.lambda_adv * adv_loss
                )
                opt_vae.zero_grad()
                vae_loss.backward()
                opt_vae.step()

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "VAECascadeGANAD":
        input_dim = features.shape[1]

        # Stage 1
        self.encoder1 = _VAEEncoder(input_dim, self.hidden_dims, self.latent_dim, self.dropout).to(self.device)
        self.decoder1 = _Decoder(self.latent_dim, self.hidden_dims, input_dim, self.dropout).to(self.device)
        self.disc1 = _Discriminator(input_dim, [64, 32], self.dropout).to(self.device)

        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._train_stage(loader, self.encoder1, self.decoder1, self.disc1, input_dim)

        # Get stage 1 residuals
        self.encoder1.eval()
        self.decoder1.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            mu, _ = self.encoder1(x)
            x_hat1 = self.decoder1(mu)
            residuals = (x - x_hat1).cpu().numpy()

        # Stage 2: refine residuals
        self.encoder2 = _VAEEncoder(input_dim, self.hidden_dims, self.latent_dim, self.dropout).to(self.device)
        self.decoder2 = _Decoder(self.latent_dim, self.hidden_dims, input_dim, self.dropout).to(self.device)
        self.disc2 = _Discriminator(input_dim, [64, 32], self.dropout).to(self.device)

        residual_dataset = TensorDataset(torch.FloatTensor(residuals))
        residual_loader = DataLoader(residual_dataset, batch_size=self.batch_size, shuffle=True)

        self._train_stage(residual_loader, self.encoder2, self.decoder2, self.disc2, input_dim)

        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.encoder1.eval()
        self.decoder1.eval()
        self.encoder2.eval()
        self.decoder2.eval()

        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            mu1, _ = self.encoder1(x)
            x_hat1 = self.decoder1(mu1)

            residual = x - x_hat1
            mu2, _ = self.encoder2(residual)
            residual_hat = self.decoder2(mu2)

            # Final reconstruction
            x_hat = x_hat1 + residual_hat
            errors = ((x - x_hat) ** 2).mean(dim=1)

        return errors.cpu().numpy()


class VAEDiffusionAD(BaseAnomalyDetector):
    """VAE-Diffusion anomaly detection.

    Combines VAE encoding with a denoising diffusion model in the latent
    space. Anomaly score is the reconstruction error.
    """

    def __init__(
        self,
        hidden_dims: list = None,
        latent_dim: int = 8,
        num_timesteps: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.1,
        device: str = "auto",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.dropout = dropout
        self.device = self._resolve_device(device)

        self.encoder = None
        self.decoder = None
        self.denoiser = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_denoiser(self):
        """Build the denoising network for the diffusion process."""
        return nn.Sequential(
            nn.Linear(self.latent_dim + 1, 128),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        ).to(self.device)

    def _get_noise_schedule(self):
        """Get beta schedule and derived quantities."""
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_cumprod

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "VAEDiffusionAD":
        input_dim = features.shape[1]

        self.encoder = _VAEEncoder(
            input_dim, self.hidden_dims, self.latent_dim, self.dropout
        ).to(self.device)
        self.decoder = _Decoder(
            self.latent_dim, self.hidden_dims, input_dim, self.dropout
        ).to(self.device)
        self.denoiser = self._build_denoiser()

        betas, alphas, alphas_cumprod = self._get_noise_schedule()

        # Phase 1: Train VAE
        opt_vae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs // 2):
            for (batch,) in loader:
                batch = batch.to(self.device)
                mu, logvar = self.encoder(batch)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_hat = self.decoder(z)

                recon_loss = nn.functional.mse_loss(x_hat, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.5 * kl_loss

                opt_vae.zero_grad()
                loss.backward()
                opt_vae.step()

        # Phase 2: Train diffusion denoiser in latent space
        self.encoder.eval()
        opt_diff = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs // 2):
            for (batch,) in loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    mu, _ = self.encoder(batch)
                    z_clean = mu

                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (z_clean.size(0),), device=self.device)
                alpha_t = alphas_cumprod[t].unsqueeze(1)

                # Add noise
                noise = torch.randn_like(z_clean)
                z_noisy = torch.sqrt(alpha_t) * z_clean + torch.sqrt(1 - alpha_t) * noise

                # Predict noise
                t_norm = t.float().unsqueeze(1) / self.num_timesteps
                denoiser_input = torch.cat([z_noisy, t_norm], dim=1)
                noise_pred = self.denoiser(denoiser_input)

                loss = nn.functional.mse_loss(noise_pred, noise)
                opt_diff.zero_grad()
                loss.backward()
                opt_diff.step()

        self.is_fitted = True
        return self

    def _denoise(self, z_noisy):
        """Run the reverse diffusion process."""
        betas, alphas, alphas_cumprod = self._get_noise_schedule()
        z = z_noisy.clone()

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((z.size(0), 1), t / self.num_timesteps, device=self.device)
            denoiser_input = torch.cat([z, t_tensor], dim=1)
            noise_pred = self.denoiser(denoiser_input)

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]

            z = (1 / torch.sqrt(alpha_t)) * (
                z - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )

            if t > 0:
                z += torch.sqrt(beta_t) * torch.randn_like(z)

        return z

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.encoder.eval()
        self.decoder.eval()
        self.denoiser.eval()

        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            mu, _ = self.encoder(x)

            # Add noise and denoise
            noise = torch.randn_like(mu) * 0.1
            z_noisy = mu + noise
            z_denoised = self._denoise(z_noisy)

            x_hat = self.decoder(z_denoised)
            errors = ((x - x_hat) ** 2).mean(dim=1)

        return errors.cpu().numpy()
