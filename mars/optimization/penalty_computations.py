from __future__ import annotations

from dataclasses import dataclass, field
import typing as tp
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler


def detect_local_minima(X: np.ndarray, losses: np.ndarray, k_neighbors: int = None) ->\
        tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify local minima from observed points (parameter vectors + loss values).

    A point is considered a local minimum if its loss is strictly lower than the loss
    of all its k nearest neighbours (Euclidean distance in parameter space).

    :param X:
        (N, D) numpy array of parameter vectors (D‑dimensional points).
    :param losses:
        (N,) numpy array of loss values corresponding to each row in X.
    :param k_neighbors:
        Number of nearest neighbours to consider for comparison.
        If None, default = min(20, N-1). If the resulting k_neighbors < 1,
        all points are returned as minima.
    :return:
        List of tuples (min_point, min_loss), where min_point is a numpy array
        of shape (D,) and min_loss is a float.
    """
    N = len(losses)

    if N == 0:
        return np.array([]), np.empty((0, X.shape[1])), np.array([])
    if k_neighbors is None:
        k_neighbors = min(20, N - 1)
    k_neighbors = min(k_neighbors, N - 1)
    if k_neighbors < 1:
        return np.arange(N), X, losses

    tree = KDTree(X)
    dist, idx = tree.query(X, k=k_neighbors + 1)
    neighbor_losses = losses[idx[:, 1:]]
    is_minima = np.all(losses[:, np.newaxis] < neighbor_losses, axis=1)
    minima_indices = np.where(is_minima)[0]
    return minima_indices, X[minima_indices], losses[minima_indices]


def _radii_loss_based(
    minima: list[tp.Tuple[np.ndarray, float]],
    all_X: np.ndarray,
    all_losses: np.ndarray,
    k_neighbors: int,
    min_radius: float = 1e-6
) -> np.ndarray:
    """
    Compute basin radii for all local minima using a quadratic loss model.

    For each minimum, assume near the minimum the loss behaves as::

        L(p) = L_min + α * ||p - p_min||^2

    where:
        - ||·|| is Euclidean distance in the D‑dimensional scaled parameter space.
        - L_min is the loss at the minimum.
        - α (>0) is a local curvature (steepness) parameter.

    The curvature α is estimated from the k nearest neighbours
    (among all recorded points) using least squares forced through zero:

        α = Σ(Δ_j * d_j^2) / Σ(d_j^4)

    where:
        - d_j = Euclidean distance from the minimum to neighbour j.
        - Δ_j = L_neigh,j - L_min (loss increase).

    Then the basin radius r is the distance at which the loss reaches the
    average neighbour loss increase Δ_avg:

        r = sqrt(Δ_avg / α)

    This radius provides a natural scale for repulsion penalties: moving
    farther than r typically lifts the loss above the average of nearby points.

    :param minima:
        List of tuples (min_point, min_loss), where min_point is a D‑dimensional
        numpy array (the parameter vector) and min_loss is a float.
    :param all_X:
        (N, D) numpy array of all recorded parameter points (all trials).
    :param all_losses:
        (N,) numpy array of loss values corresponding to all_X.
    :param k_neighbors:
        Number of nearest neighbours to consider for each minimum.
        For robust estimation, it is recommended that k_neighbors > D
        (the number of parameters/dimensions). A warning is issued otherwise.
    :param min_radius:
        Fallback minimum radius used when the quadratic fit is degenerate
        (e.g., all neighbour losses equal to min_loss). Default 1e-6.

    :return:
        radii : (n_minima,) numpy array of estimated basin radii (one per minimum).
    """
    n_min = len(minima)
    if n_min == 0:
        return np.array([])

    N, D = all_X.shape
    if N < 2:
        return np.full(n_min, min_radius)

    min_points = np.array([m[0] for m in minima])
    min_losses = np.array([m[1] for m in minima])

    tree_all = KDTree(all_X)
    dist, idx = tree_all.query(min_points, k=k_neighbors + 1)

    neigh_dist = dist[:, 1:]
    neigh_idx = idx[:, 1:]
    neigh_loss = all_losses[neigh_idx]

    delta_neigh = neigh_loss - min_losses[:, np.newaxis]
    delta_avg = np.mean(delta_neigh, axis=1)

    d2 = neigh_dist ** 2

    numerator = np.sum(delta_neigh * d2, axis=1)
    denominator = np.sum(d2 ** 2, axis=1)

    valid = (denominator > 1e-12) & (delta_avg > 0) & (numerator > 0)

    radii = np.full(n_min, min_radius, dtype=float)
    if np.any(valid):
        alpha = numerator[valid] / denominator[valid]
        radii[valid] = np.sqrt(delta_avg[valid] / alpha)

        max_neigh_dist = np.max(neigh_dist[valid], axis=1)
        radii[valid] = np.minimum(radii[valid], max_neigh_dist)

    invalid = ~valid
    if np.any(invalid):
        avg_neigh_dist = np.mean(neigh_dist[invalid], axis=1)
        radii[invalid] = np.maximum(avg_neigh_dist, min_radius)

    return radii


@dataclass
class MinimumRecord:
    """
    Stores information about a single local minimum.

    :param params: Original (unscaled) parameter dictionary.
    :param raw_loss: True objective value (no penalty).
    :param scaled_coords: (D,) numpy array of scaled parameters.
    :param radius: Basin radius estimated from the loss landscape.
    :param metadata: Optional additional diagnostics.
    """
    params: tp.Dict[str, float]
    raw_loss: float
    scaled_coords: np.ndarray
    radius: float
    metadata: tp.Dict[str, tp.Any] = field(default_factory=dict)


class MinimaArchive:
    """
    Stores the current set of discovered minima.

    Provides:
    - storage of minima records (with scaled coordinates and radii)
    - easy retrieval of all minima for penalty computation
    - clearing and adding new minima
    """

    def __init__(self):
        """Initialize empty archive."""
        self._minima: tp.List[MinimumRecord] = []

    def clear(self) -> None:
        """Remove all stored minima."""
        self._minima.clear()

    def add(self, record: MinimumRecord) -> None:
        """
        Add a new minimum to the archive.

        :param record: MinimumRecord instance.
        """
        self._minima.append(record)

    def get_all(self) -> tp.List[MinimumRecord]:
        """
        Return a copy of the list of all minima.

        :return: List of MinimumRecord objects.
        """
        return list(self._minima)

    def get_scaled_centers_and_radii(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Return arrays of scaled coordinates and radii of all minima.

        :return: (centers, radii) where centers shape (n_min, D), radii shape (n_min,).
        """
        if not self._minima:
            return np.empty((0, 0)), np.empty(0)
        centers = np.array([m.scaled_coords for m in self._minima])
        radii = np.array([m.radius for m in self._minima])
        return centers, radii

    def __len__(self) -> int:
        """Number of minima stored."""
        return len(self._minima)


class RepulsivePenalty:
    """
    Repulsion penalty that periodically updates the archive by refitting a scaler,
    detecting local minima, and recomputing their basin radii.

    The penalty for a candidate point x is:
        penalty = strength * Σ_i exp( -||x_scaled - center_i||^2 / (2 * σ_i^2) )

    where center_i and σ_i are the scaled coordinates and radius of the i‑th minimum.
    """

    def __init__(
        self,
        archive: tp.Optional[MinimaArchive] = None,
        k_neighbors: int = 5,
        min_radius: float = 1e-6,
        sigma_scale: float = 1.0
    ):
        """
        :param archive: MinimaArchive instance where discovered minima will be stored.
        :param k_neighbors: Number of neighbours used for local minimum detection and radius estimation.
        :param min_radius: Fallback minimum radius for degenerate cases.
        :param sigma_scale: The factor which scale radius for minima distance computations
            That is, sigma for minimum equal to (radius / sigma_scale) ** 2
        """
        self.archive = MinimaArchive() if archive is None else archive
        self.k_neighbors = k_neighbors
        self.min_radius = min_radius
        self._scaler = None
        self._force_factor = None
        self._sigma_scale: float = sigma_scale

    def update(self, X: np.ndarray, losses: np.ndarray, param_dicts: tp.List[tp.Dict[str, float]]) -> None:
        """
        Refit the scaler on X, detect local minima, compute their radii, and replace the archive.

        This method should be called periodically (e.g., every N optimization steps).

        :param X: (N, D) numpy array of unscaled parameter vectors.
        :param losses: (N,) numpy array of loss values.
        :param param_dicts: List of length N, each a dictionary mapping parameter names to values.
        """
        if len(X) == 0:
            self.archive.clear()
            self._scaler = None
            return

        self._scaler = StandardScaler()
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        minima_indices, min_points_scaled, min_losses = detect_local_minima(
            X_scaled, losses, k_neighbors=self.k_neighbors
        )
        if len(minima_indices) == 0:
            self.archive.clear()
            return

        self._force_factor = np.mean(min_losses)
        minima_for_radii = [(min_points_scaled[i], min_losses[i]) for i in range(len(minima_indices))]
        radii = _radii_loss_based(
            minima=minima_for_radii,
            all_X=X_scaled,
            all_losses=losses,
            k_neighbors=self.k_neighbors,
            min_radius=self.min_radius
        )

        new_records = []
        for idx, scaled_pt, loss, r in zip(minima_indices, min_points_scaled, min_losses, radii):
            orig_params = param_dicts[idx]
            record = MinimumRecord(
                params=orig_params,
                raw_loss=loss,
                scaled_coords=scaled_pt,
                radius=r,
            )
            new_records.append(record)

        self.archive.clear()
        for rec in new_records:
            self.archive.add(rec)

    def compute_penalty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the total repulsion penalty for candidate parameter vectors.

        Conceptually, this sums Gaussian repulsion terms from all archived minima
        for each input point.

        :param X: (N, D) numpy array of unscaled parameter vectors, where N is
                  the number of points and D is the dimensionality.
        :return: (N,) numpy array of penalty values (>= 0) for each input point.
        """
        if self._scaler is None or len(self.archive) == 0:
            return np.zeros(X.shape[0])
        #print(X.shape)
        scaled_point = self._scaler.transform(X)
        centers, radii = self.archive.get_scaled_centers_and_radii()
        diff = centers[:, np.newaxis, :] - scaled_point[np.newaxis, :, :]

        distances = np.linalg.norm(diff, axis=2)

        sigmas = np.maximum(radii, 1e-12) / self._sigma_scale
        gaussians = np.exp(-(distances ** 2) / (2 * sigmas[:, np.newaxis] ** 2))

        penalty = self._force_factor * np.sum(gaussians, axis=0)
        return penalty

    def clear(self):
        self.archive.clear()
        self._scaler = None
        self._force_factor = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Alias for compute_penalty."""
        return self.compute_penalty(X)