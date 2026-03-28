"""Spatial utility functions for GeoChemAD.

Provides KD-tree construction, nearest neighbour queries,
and spatial interpolation methods (IDW, Kriging).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def build_kdtree(coordinates: np.ndarray) -> cKDTree:
    """Build a KD-tree from spatial coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_points, 2) with (x, y) coordinates.

    Returns
    -------
    cKDTree
        Spatial index.
    """
    return cKDTree(coordinates)


def get_k_nearest_neighbours(
    tree: cKDTree,
    query_point: np.ndarray,
    k: int = 128,
    exclude_self: bool = True,
) -> tuple:
    """Retrieve K nearest neighbours for a query point.

    Parameters
    ----------
    tree : cKDTree
        Pre-built spatial index.
    query_point : np.ndarray
        Query coordinates of shape (2,) or (n_queries, 2).
    k : int
        Number of neighbours to retrieve.
    exclude_self : bool
        If True and the query point is in the tree, retrieve k+1 and drop self.

    Returns
    -------
    tuple of (distances, indices)
        distances: shape (k,) or (n_queries, k)
        indices: shape (k,) or (n_queries, k)
    """
    k_query = k + 1 if exclude_self else k
    k_query = min(k_query, tree.n)

    distances, indices = tree.query(query_point, k=k_query)

    if exclude_self:
        if query_point.ndim == 1:
            # Single query: skip first if distance is ~0
            if distances[0] < 1e-10:
                distances = distances[1:]
                indices = indices[1:]
            else:
                distances = distances[:k]
                indices = indices[:k]
        else:
            # Batch query
            mask = distances[:, 0] < 1e-10
            result_d = np.zeros((len(query_point), k))
            result_i = np.zeros((len(query_point), k), dtype=int)
            for i in range(len(query_point)):
                if mask[i]:
                    result_d[i] = distances[i, 1 : k + 1]
                    result_i[i] = indices[i, 1 : k + 1]
                else:
                    result_d[i] = distances[i, :k]
                    result_i[i] = indices[i, :k]
            distances = result_d
            indices = result_i

    return distances, indices


def get_all_neighbours(
    coordinates: np.ndarray,
    k: int = 128,
) -> tuple:
    """Get K nearest neighbours for all points in a dataset.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_points, 2).
    k : int
        Number of neighbours.

    Returns
    -------
    tuple of (distances, indices)
        Both of shape (n_points, k).
    """
    tree = build_kdtree(coordinates)
    k_query = min(k + 1, tree.n)
    distances, indices = tree.query(coordinates, k=k_query)

    # Exclude self (first column)
    return distances[:, 1:k + 1], indices[:, 1:k + 1]


def idw_interpolation(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    query_coords: np.ndarray,
    power: float = 2.0,
    k_neighbours: int = 12,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation.

    Parameters
    ----------
    known_coords : np.ndarray
        Known sample coordinates, shape (n_known, 2).
    known_values : np.ndarray
        Known values, shape (n_known,) or (n_known, n_vars).
    query_coords : np.ndarray
        Query coordinates, shape (n_query, 2).
    power : float
        Distance weighting power (default 2).
    k_neighbours : int
        Number of nearest neighbours to use.

    Returns
    -------
    np.ndarray
        Interpolated values at query locations.
    """
    tree = build_kdtree(known_coords)
    distances, indices = tree.query(query_coords, k=k_neighbours)

    # Handle coincident points
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / (distances ** power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_norm = weights / weights_sum

    if known_values.ndim == 1:
        neighbour_vals = known_values[indices]
        return (weights_norm * neighbour_vals).sum(axis=1)
    else:
        n_query = query_coords.shape[0]
        n_vars = known_values.shape[1]
        result = np.zeros((n_query, n_vars))
        for v in range(n_vars):
            neighbour_vals = known_values[indices, v]
            result[:, v] = (weights_norm * neighbour_vals).sum(axis=1)
        return result


def kriging_interpolation(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    query_coords: np.ndarray,
    variogram_model: str = "linear",
    k_neighbours: int = 12,
) -> np.ndarray:
    """Simple Ordinary Kriging interpolation.

    Uses a basic variogram model for spatial interpolation. For production use,
    consider using a dedicated geostatistics library.

    Parameters
    ----------
    known_coords : np.ndarray
        Known sample coordinates, shape (n_known, 2).
    known_values : np.ndarray
        Known values, shape (n_known,).
    query_coords : np.ndarray
        Query coordinates, shape (n_query, 2).
    variogram_model : str
        Variogram model type: "linear" or "spherical".
    k_neighbours : int
        Number of nearest neighbours.

    Returns
    -------
    np.ndarray
        Interpolated values at query locations.
    """
    tree = build_kdtree(known_coords)

    def variogram(h, sill=1.0, range_param=1.0, nugget=0.0):
        if variogram_model == "linear":
            return nugget + sill * h / range_param
        elif variogram_model == "spherical":
            ratio = np.minimum(h / range_param, 1.0)
            return nugget + sill * (1.5 * ratio - 0.5 * ratio ** 3)
        return h

    # Estimate variogram parameters from data
    n_known = len(known_values)
    if n_known > 500:
        subset_idx = np.random.choice(n_known, 500, replace=False)
    else:
        subset_idx = np.arange(n_known)

    subset_coords = known_coords[subset_idx]
    subset_vals = known_values[subset_idx] if known_values.ndim == 1 else known_values[subset_idx, 0]
    dists = cdist(subset_coords, subset_coords)
    range_param = np.percentile(dists[dists > 0], 50)
    sill = np.var(subset_vals)

    results = np.zeros(len(query_coords))
    for qi in range(len(query_coords)):
        _, idx = tree.query(query_coords[qi], k=min(k_neighbours, n_known))
        if np.isscalar(idx):
            idx = np.array([idx])

        local_coords = known_coords[idx]
        if known_values.ndim == 1:
            local_vals = known_values[idx]
        else:
            local_vals = known_values[idx, 0]

        n = len(idx)
        # Build Kriging system
        K = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                K[i, j] = variogram(
                    np.linalg.norm(local_coords[i] - local_coords[j]),
                    sill=sill, range_param=range_param
                )
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        k_vec = np.zeros(n + 1)
        for i in range(n):
            k_vec[i] = variogram(
                np.linalg.norm(query_coords[qi] - local_coords[i]),
                sill=sill, range_param=range_param
            )
        k_vec[n] = 1.0

        try:
            weights = np.linalg.solve(K, k_vec)
            results[qi] = np.dot(weights[:n], local_vals)
        except np.linalg.LinAlgError:
            # Fallback to IDW
            d = np.linalg.norm(local_coords - query_coords[qi], axis=1)
            d = np.maximum(d, 1e-10)
            w = 1.0 / (d ** 2)
            results[qi] = np.dot(w / w.sum(), local_vals)

    return results
