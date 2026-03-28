"""GeoChemFormer: Transformer-based framework for geochemical anomaly detection.

Implements the two-stage GeoChemFormer model from the paper:
  Stage 1: Spatial Context Learning (SCL) - learns spatially informed
           representations from neighbourhood context.
  Stage 2: Element Dependency Modelling (EDM) - detects anomalies by
           modelling dependencies among elemental concentrations
           conditioned on spatial context.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.spatial import cKDTree

from geochemad.models.statistical import BaseAnomalyDetector


class _SpatialPositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(2, d_model)

    def forward(self, coords):
        """
        Parameters
        ----------
        coords : torch.Tensor
            Spatial coordinates, shape (..., 2).
        """
        return self.proj(coords)


class _SpatialContextEncoder(nn.Module):
    """Spatial Context Encoder for Stage 1.

    Processes a token sequence [target_element_token, query_token, neighbour_1, ..., neighbour_K]
    and outputs a spatially informed query representation.

    Parameters
    ----------
    n_features : int
        Number of geochemical elements (c).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    d_ff : int
        Feed-forward dimension.
    dropout : float
        Dropout rate.
    n_target_elements : int
        Number of distinct target elements for the element token embedding.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_target_elements: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Target element token embedding
        self.target_element_embed = nn.Embedding(n_target_elements, d_model)

        # Query token: project (x, y) + positional encoding
        self.query_proj = nn.Linear(2, d_model)
        self.spatial_pe = _SpatialPositionalEncoding2D(d_model)

        # Neighbour token: project (Δx, Δy, f_1, ..., f_c) to d_model
        self.neighbour_proj = nn.Linear(2 + n_features, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head for target element concentration
        self.pred_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_elem_id, query_coords, neighbour_features, neighbour_offsets):
        """
        Parameters
        ----------
        target_elem_id : torch.Tensor
            Target element index, shape (batch,).
        query_coords : torch.Tensor
            Query spatial coordinates, shape (batch, 2).
        neighbour_features : torch.Tensor
            Neighbour geochemical features, shape (batch, K, n_features).
        neighbour_offsets : torch.Tensor
            Relative spatial offsets to neighbours, shape (batch, K, 2).

        Returns
        -------
        prediction : torch.Tensor
            Predicted target element concentration, shape (batch,).
        query_repr : torch.Tensor
            Spatial context representation, shape (batch, d_model).
        """
        batch_size = query_coords.size(0)
        K = neighbour_features.size(1)

        # Target element token
        elem_token = self.target_element_embed(target_elem_id)  # (B, d_model)
        elem_token = elem_token.unsqueeze(1)  # (B, 1, d_model)

        # Query token with spatial positional encoding
        query_token = self.query_proj(query_coords) + self.spatial_pe(query_coords)
        query_token = query_token.unsqueeze(1)  # (B, 1, d_model)

        # Neighbour tokens: concatenate offsets with features
        neighbour_input = torch.cat([neighbour_offsets, neighbour_features], dim=-1)
        neighbour_tokens = self.neighbour_proj(neighbour_input)  # (B, K, d_model)

        # Add relative spatial positional encoding to neighbours
        neighbour_pe = self.spatial_pe(neighbour_offsets)
        neighbour_tokens = neighbour_tokens + neighbour_pe

        # Build sequence: [element_token, query_token, neighbour_1, ..., neighbour_K]
        sequence = torch.cat([elem_token, query_token, neighbour_tokens], dim=1)
        sequence = self.norm(sequence)

        # Transformer encoding
        encoded = self.transformer(sequence)

        # Extract query representation (position [1])
        query_repr = encoded[:, 1, :]  # (B, d_model)

        # Predict target element concentration
        prediction = self.pred_head(query_repr).squeeze(-1)  # (B,)

        return prediction, query_repr


class _ElementDependencyEncoder(nn.Module):
    """Element Dependency Encoder for Stage 2.

    Models dependencies among elemental concentrations conditioned on
    spatial context.

    Parameters
    ----------
    n_elements : int
        Number of geochemical elements.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    d_ff : int
        Feed-forward dimension.
    dropout : float
        Dropout rate.
    elem_embed_dim : int
        Element identity embedding dimension.
    """

    def __init__(
        self,
        n_elements: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        elem_embed_dim: int = 64,
        context_dim: int = 128,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.d_model = d_model

        # Geo-context token projection
        self.context_proj = nn.Linear(context_dim, d_model)

        # Element identity embedding
        self.element_embed = nn.Embedding(n_elements, elem_embed_dim)

        # Value projection
        self.value_proj = nn.Linear(1, d_model - elem_embed_dim)

        # Element token combiner
        self.element_token_proj = nn.Linear(d_model, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction head: shared linear decoder
        self.recon_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, context_repr, element_values):
        """
        Parameters
        ----------
        context_repr : torch.Tensor
            Spatial context representation from Stage 1, shape (batch, context_dim).
        element_values : torch.Tensor
            Element concentrations, shape (batch, n_elements).

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed element values, shape (batch, n_elements).
        """
        batch_size = element_values.size(0)

        # Geo-context token
        context_token = self.context_proj(context_repr)  # (B, d_model)
        context_token = context_token.unsqueeze(1)  # (B, 1, d_model)

        # Element tokens
        elem_ids = torch.arange(self.n_elements, device=element_values.device)
        elem_ids = elem_ids.unsqueeze(0).expand(batch_size, -1)  # (B, C)

        id_embed = self.element_embed(elem_ids)  # (B, C, elem_embed_dim)
        val_embed = self.value_proj(element_values.unsqueeze(-1))  # (B, C, d_model - embed_dim)

        elem_tokens = torch.cat([id_embed, val_embed], dim=-1)  # (B, C, d_model)
        elem_tokens = self.element_token_proj(elem_tokens)

        # Build sequence: [context_token, elem_1, ..., elem_C]
        sequence = torch.cat([context_token, elem_tokens], dim=1)
        sequence = self.norm(sequence)

        # Transformer encoding
        encoded = self.transformer(sequence)

        # Reconstruct from element tokens (positions 1 to C+1)
        elem_encoded = encoded[:, 1 : self.n_elements + 1, :]
        reconstructed = self.recon_head(elem_encoded).squeeze(-1)  # (B, C)

        return reconstructed


class _NeighbourDataset(Dataset):
    """Dataset that provides pre-computed neighbour information for each sample."""

    def __init__(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        target_values: np.ndarray,
        neighbour_indices: np.ndarray,
        target_elem_id: int = 0,
    ):
        self.coordinates = torch.FloatTensor(coordinates)
        self.features = torch.FloatTensor(features)
        self.target_values = torch.FloatTensor(target_values)
        self.neighbour_indices = neighbour_indices
        self.target_elem_id = target_elem_id

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        query_coord = self.coordinates[idx]
        target_val = self.target_values[idx]

        # Get neighbour data
        nb_idx = self.neighbour_indices[idx]
        nb_coords = self.coordinates[nb_idx]
        nb_features = self.features[nb_idx]

        # Compute relative offsets
        offsets = nb_coords - query_coord.unsqueeze(0)

        return {
            "query_coords": query_coord,
            "target_value": target_val,
            "neighbour_features": nb_features,
            "neighbour_offsets": offsets,
            "features": self.features[idx],
            "target_elem_id": self.target_elem_id,
        }


class GeoChemFormer(BaseAnomalyDetector):
    """GeoChemFormer: Two-stage transformer for geochemical anomaly detection.

    Stage 1 (Spatial Context Learning): Learns spatial context representations
    by predicting target element concentrations from neighbourhood samples.

    Stage 2 (Element Dependency Modelling): Models inter-element dependencies
    conditioned on spatial context for reconstruction-based anomaly scoring.

    Parameters
    ----------
    k_neighbours : int
        Number of nearest neighbours for spatial context.
    scl_d_model, scl_n_heads, scl_n_layers, scl_d_ff : int
        Spatial context encoder architecture params.
    edm_d_model, edm_n_heads, edm_n_layers, edm_d_ff : int
        Element dependency encoder architecture params.
    scl_epochs, edm_epochs : int
        Training epochs for each stage.
    elem_embed_dim : int
        Element identity embedding dimension.
    device : str
        Compute device.
    """

    def __init__(
        self,
        k_neighbours: int = 128,
        scl_d_model: int = 128,
        scl_n_heads: int = 4,
        scl_n_layers: int = 4,
        scl_d_ff: int = 256,
        scl_dropout: float = 0.1,
        scl_epochs: int = 50,
        scl_learning_rate: float = 1e-3,
        scl_batch_size: int = 64,
        edm_d_model: int = 128,
        edm_n_heads: int = 4,
        edm_n_layers: int = 4,
        edm_d_ff: int = 256,
        edm_dropout: float = 0.1,
        edm_epochs: int = 100,
        edm_learning_rate: float = 1e-3,
        edm_batch_size: int = 128,
        elem_embed_dim: int = 64,
        device: str = "auto",
    ):
        super().__init__()
        self.k = k_neighbours
        self.scl_d_model = scl_d_model
        self.scl_n_heads = scl_n_heads
        self.scl_n_layers = scl_n_layers
        self.scl_d_ff = scl_d_ff
        self.scl_dropout = scl_dropout
        self.scl_epochs = scl_epochs
        self.scl_lr = scl_learning_rate
        self.scl_batch_size = scl_batch_size
        self.edm_d_model = edm_d_model
        self.edm_n_heads = edm_n_heads
        self.edm_n_layers = edm_n_layers
        self.edm_d_ff = edm_d_ff
        self.edm_dropout = edm_dropout
        self.edm_epochs = edm_epochs
        self.edm_lr = edm_learning_rate
        self.edm_batch_size = edm_batch_size
        self.elem_embed_dim = elem_embed_dim
        self.device = self._resolve_device(device)

        self.scl_encoder = None
        self.edm_encoder = None
        self._train_coordinates = None
        self._train_features = None
        self._train_target_values = None
        self._neighbour_indices = None
        self._spatial_contexts = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_neighbours(self, coordinates: np.ndarray) -> np.ndarray:
        """Build KD-tree and find K nearest neighbours for all samples."""
        tree = cKDTree(coordinates)
        k_query = min(self.k + 1, len(coordinates))
        _, indices = tree.query(coordinates, k=k_query)
        # Exclude self
        return indices[:, 1 : self.k + 1]

    def _train_scl(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        target_values: np.ndarray,
        neighbour_indices: np.ndarray,
    ) -> np.ndarray:
        """Train Stage 1: Spatial Context Learning.

        Returns spatial context representations for all training samples.
        """
        n_features = features.shape[1]

        self.scl_encoder = _SpatialContextEncoder(
            n_features=n_features,
            d_model=self.scl_d_model,
            n_heads=self.scl_n_heads,
            n_layers=self.scl_n_layers,
            d_ff=self.scl_d_ff,
            dropout=self.scl_dropout,
        ).to(self.device)

        dataset = _NeighbourDataset(
            coordinates, features, target_values, neighbour_indices, target_elem_id=0
        )
        loader = DataLoader(
            dataset, batch_size=self.scl_batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(self.scl_encoder.parameters(), lr=self.scl_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scl_epochs
        )

        self.scl_encoder.train()
        for epoch in range(self.scl_epochs):
            total_loss = 0
            n_batches = 0
            for batch in loader:
                query_coords = batch["query_coords"].to(self.device)
                target_val = batch["target_value"].to(self.device)
                nb_features = batch["neighbour_features"].to(self.device)
                nb_offsets = batch["neighbour_offsets"].to(self.device)
                elem_id = torch.zeros(
                    query_coords.size(0), dtype=torch.long, device=self.device
                )

                prediction, _ = self.scl_encoder(
                    elem_id, query_coords, nb_features, nb_offsets
                )

                loss = nn.functional.mse_loss(prediction, target_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

        # Extract spatial context representations for all samples
        self.scl_encoder.eval()
        all_contexts = []
        eval_dataset = _NeighbourDataset(
            coordinates, features, target_values, neighbour_indices, target_elem_id=0
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.scl_batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in eval_loader:
                query_coords = batch["query_coords"].to(self.device)
                nb_features = batch["neighbour_features"].to(self.device)
                nb_offsets = batch["neighbour_offsets"].to(self.device)
                elem_id = torch.zeros(
                    query_coords.size(0), dtype=torch.long, device=self.device
                )

                _, context_repr = self.scl_encoder(
                    elem_id, query_coords, nb_features, nb_offsets
                )
                all_contexts.append(context_repr.cpu().numpy())

        return np.concatenate(all_contexts, axis=0)

    def _train_edm(
        self,
        features: np.ndarray,
        spatial_contexts: np.ndarray,
    ) -> None:
        """Train Stage 2: Element Dependency Modelling."""
        n_elements = features.shape[1]

        self.edm_encoder = _ElementDependencyEncoder(
            n_elements=n_elements,
            d_model=self.edm_d_model,
            n_heads=self.edm_n_heads,
            n_layers=self.edm_n_layers,
            d_ff=self.edm_d_ff,
            dropout=self.edm_dropout,
            elem_embed_dim=self.elem_embed_dim,
            context_dim=self.scl_d_model,
        ).to(self.device)

        dataset = TensorDataset(
            torch.FloatTensor(spatial_contexts),
            torch.FloatTensor(features),
        )
        loader = DataLoader(
            dataset, batch_size=self.edm_batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.edm_encoder.parameters(), lr=self.edm_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.edm_epochs
        )

        self.edm_encoder.train()
        for epoch in range(self.edm_epochs):
            for context_batch, feature_batch in loader:
                context_batch = context_batch.to(self.device)
                feature_batch = feature_batch.to(self.device)

                reconstructed = self.edm_encoder(context_batch, feature_batch)

                loss = nn.functional.mse_loss(reconstructed, feature_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

    def fit(self, features: np.ndarray, coordinates: np.ndarray = None) -> "GeoChemFormer":
        """Fit GeoChemFormer on training data.

        Parameters
        ----------
        features : np.ndarray
            Geochemical concentration matrix, shape (n_samples, n_elements).
        coordinates : np.ndarray
            Spatial coordinates, shape (n_samples, 2). Required for GeoChemFormer.
        """
        if coordinates is None:
            raise ValueError("GeoChemFormer requires spatial coordinates")

        self._train_coordinates = coordinates.copy()
        self._train_features = features.copy()

        # Use first element as target for SCL (can be configured)
        self._train_target_values = features[:, 0].copy()

        # Build neighbour graph
        self._neighbour_indices = self._build_neighbours(coordinates)

        # Stage 1: Spatial Context Learning
        self._spatial_contexts = self._train_scl(
            coordinates, features, self._train_target_values,
            self._neighbour_indices
        )

        # Stage 2: Element Dependency Modelling
        self._train_edm(features, self._spatial_contexts)

        self.is_fitted = True
        return self

    def _get_spatial_context(
        self, coordinates: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Compute spatial context representations for new samples.

        Uses training samples as the neighbourhood reference.
        """
        tree = cKDTree(self._train_coordinates)
        k_query = min(self.k, len(self._train_coordinates))
        _, nb_indices = tree.query(coordinates, k=k_query)

        dataset = _NeighbourDataset(
            coordinates,
            features,
            np.zeros(len(coordinates)),  # dummy targets
            nb_indices,
            target_elem_id=0,
        )
        # Override features to use training set for neighbours
        dataset.features = torch.FloatTensor(self._train_features)
        dataset.coordinates = torch.FloatTensor(
            np.vstack([coordinates, self._train_coordinates])
        )

        # Manually build contexts
        self.scl_encoder.eval()
        all_contexts = []

        coords_t = torch.FloatTensor(coordinates)
        train_coords_t = torch.FloatTensor(self._train_coordinates)
        train_features_t = torch.FloatTensor(self._train_features)

        with torch.no_grad():
            for i in range(0, len(coordinates), self.scl_batch_size):
                end = min(i + self.scl_batch_size, len(coordinates))
                batch_coords = coords_t[i:end].to(self.device)
                batch_nb_idx = nb_indices[i:end]

                # Get neighbour data from training set
                nb_coords_list = train_coords_t[batch_nb_idx]
                nb_features_list = train_features_t[batch_nb_idx]

                offsets = nb_coords_list - batch_coords.cpu().unsqueeze(1)
                offsets = offsets.to(self.device)
                nb_features_list = nb_features_list.to(self.device)

                elem_id = torch.zeros(
                    batch_coords.size(0), dtype=torch.long, device=self.device
                )

                _, context_repr = self.scl_encoder(
                    elem_id, batch_coords, nb_features_list, offsets
                )
                all_contexts.append(context_repr.cpu().numpy())

        return np.concatenate(all_contexts, axis=0)

    def score(self, features: np.ndarray, coordinates: np.ndarray = None) -> np.ndarray:
        """Compute anomaly scores.

        Parameters
        ----------
        features : np.ndarray
            Geochemical features, shape (n_samples, n_elements).
        coordinates : np.ndarray
            Spatial coordinates, shape (n_samples, 2). Required.

        Returns
        -------
        np.ndarray
            Mean squared reconstruction error across elements.
        """
        if coordinates is None:
            raise ValueError("GeoChemFormer requires spatial coordinates for scoring")

        # Check if scoring training data (use pre-computed contexts)
        if (
            self._spatial_contexts is not None
            and len(features) == len(self._train_features)
            and np.allclose(coordinates, self._train_coordinates)
        ):
            contexts = self._spatial_contexts
        else:
            contexts = self._get_spatial_context(coordinates, features)

        # Score through EDM
        self.edm_encoder.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(features), self.edm_batch_size):
                end = min(i + self.edm_batch_size, len(features))
                ctx_batch = torch.FloatTensor(contexts[i:end]).to(self.device)
                feat_batch = torch.FloatTensor(features[i:end]).to(self.device)

                reconstructed = self.edm_encoder(ctx_batch, feat_batch)
                errors = ((feat_batch - reconstructed) ** 2).mean(dim=1)
                all_scores.append(errors.cpu().numpy())

        return np.concatenate(all_scores, axis=0)
