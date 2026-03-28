"""Vanilla Transformer (T1) for anomaly detection.

Implements a standard Transformer-based anomaly detection model that treats
geochemical elements as a token sequence and uses masked reconstruction
for training.
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset

from geochemad.models.statistical import BaseAnomalyDetector


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerAD(nn.Module):
    """Vanilla Transformer for anomaly detection via masked reconstruction.

    Each element concentration is treated as a token. A [CLS] token is
    prepended. The model is trained with random masking (like BERT) to
    reconstruct masked element values.
    """

    def __init__(
        self,
        n_elements: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.d_model = d_model

        # Element embedding: identity + value projection
        self.element_embed = nn.Embedding(n_elements, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.token_combine = nn.Linear(2 * d_model, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = _PositionalEncoding(d_model, max_len=n_elements + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction head
        self.recon_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, values, mask=None):
        """
        Parameters
        ----------
        values : torch.Tensor
            Element concentrations, shape (batch, n_elements).
        mask : torch.Tensor, optional
            Boolean mask indicating which elements to mask, shape (batch, n_elements).

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed values, shape (batch, n_elements).
        """
        batch_size = values.size(0)

        # Create element identity indices
        elem_ids = torch.arange(self.n_elements, device=values.device)
        elem_ids = elem_ids.unsqueeze(0).expand(batch_size, -1)

        # Embed element identity and value
        id_embed = self.element_embed(elem_ids)  # (B, C, d_model)
        val_embed = self.value_proj(values.unsqueeze(-1))  # (B, C, d_model)

        # Combine
        tokens = self.token_combine(torch.cat([id_embed, val_embed], dim=-1))

        # Apply mask by zeroing out masked positions' value contribution
        if mask is not None:
            # Replace masked tokens with just identity embedding
            mask_expand = mask.unsqueeze(-1).float()
            tokens = tokens * (1 - mask_expand) + id_embed * mask_expand

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional encoding
        tokens = self.pos_encoding(tokens)
        tokens = self.norm(tokens)

        # Transformer encoding
        encoded = self.transformer(tokens)

        # Reconstruct from element tokens (skip CLS)
        elem_encoded = encoded[:, 1:, :]
        reconstructed = self.recon_head(elem_encoded).squeeze(-1)

        return reconstructed


class VanillaTransformerAD(BaseAnomalyDetector):
    """Vanilla Transformer-based anomaly detection (T1 in the paper).

    Uses masked reconstruction training. Anomaly score is the mean squared
    reconstruction error across all elements.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        mask_ratio: float = 0.15,
        device: str = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.mask_ratio = mask_ratio
        self.device = self._resolve_device(device)
        self.model = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "VanillaTransformerAD":
        n_elements = features.shape[1]
        self.model = _TransformerAD(
            n_elements=n_elements,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)

                # Random masking
                mask = (
                    torch.rand(batch.shape, device=self.device) < self.mask_ratio
                )

                reconstructed = self.model(batch, mask=mask)

                # Loss on masked positions only
                loss = nn.functional.mse_loss(
                    reconstructed[mask], batch[mask]
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        self.is_fitted = True
        return self

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            reconstructed = self.model(x, mask=None)
            errors = ((x - reconstructed) ** 2).mean(dim=1)
        return errors.cpu().numpy()
