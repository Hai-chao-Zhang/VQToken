"""Small, dependency-free spherical K-means implementations used by VQToken."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class KMeansTorch:
    """Spherical K-means with K-means++ initialization.

    Clustering is deliberately performed in float32. Vision features commonly
    arrive as float16/bfloat16, for which distance calculations and centroid
    accumulation are both inaccurate.
    """

    def __init__(self, num_clusters: int, max_iteration: int = 300, tol: float = 1e-4, seed: int | None = 0):
        if isinstance(num_clusters, bool) or not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError("num_clusters must be a positive integer")
        if isinstance(max_iteration, bool) or not isinstance(max_iteration, int) or max_iteration < 1:
            raise ValueError("max_iteration must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")

        self.num_clusters = num_clusters
        self.max_iteration = max_iteration
        self.tol = float(tol)
        self.seed = seed
        self.centroids: torch.Tensor | None = None

    def _validate_input(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            raise ValueError("X must be a 2D torch.Tensor with shape [samples, features]")
        if not X.is_floating_point():
            raise TypeError("X must use a floating-point dtype")
        if X.shape[0] < self.num_clusters:
            raise ValueError(
                f"num_clusters ({self.num_clusters}) cannot exceed the number of samples ({X.shape[0]})"
            )
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if not torch.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X.to(dtype=torch.float32)

    def sample_init_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centers with squared-distance K-means++ sampling."""

        n_samples = X.shape[0]
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=X.device)
            generator.manual_seed(self.seed)
        first_idx = torch.randint(n_samples, (1,), device=X.device, generator=generator).item()
        centroids = [X[first_idx]]
        closest_squared_distance = (X - centroids[0]).square().sum(dim=1)

        for _ in range(1, self.num_clusters):
            total = closest_squared_distance.sum()

            if not torch.isfinite(total) or total <= 0:
                next_idx = torch.randint(n_samples, (1,), device=X.device, generator=generator).item()
            else:
                next_idx = torch.multinomial(closest_squared_distance / total, 1, generator=generator).item()
            centroids.append(X[next_idx])
            new_squared_distance = (X - centroids[-1]).square().sum(dim=1)
            closest_squared_distance = torch.minimum(closest_squared_distance, new_squared_distance)

        return torch.stack(centroids)

    @staticmethod
    def compute_cosine_distances(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        X = F.normalize(X, dim=-1, eps=1e-12)
        centroids = F.normalize(centroids, dim=-1, eps=1e-12)
        return 1 - X @ centroids.T

    @staticmethod
    def find_clusters_index(distances: torch.Tensor) -> torch.Tensor:
        return torch.argmin(distances, dim=1)

    def update_centroids(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("fit must initialize centroids before they are updated")

        sums = torch.zeros_like(self.centroids)
        sums.index_add_(0, labels, X)
        counts = torch.bincount(labels, minlength=self.num_clusters)
        means = sums / counts.clamp_min(1).to(X.dtype)[:, None]
        return torch.where(counts[:, None] > 0, means, self.centroids)

    def _canonicalize_clusters(self, labels: torch.Tensor) -> torch.Tensor:
        """Order cluster ids by their first assigned token for stable LLM input."""

        assert self.centroids is not None
        first_indices = torch.full(
            (self.num_clusters,),
            labels.numel(),
            device=labels.device,
            dtype=torch.long,
        )
        for cluster_idx in range(self.num_clusters):
            assigned = torch.nonzero(labels == cluster_idx, as_tuple=False)
            if assigned.numel() > 0:
                first_indices[cluster_idx] = assigned[0, 0]

        # The cluster id is a deterministic tie-breaker for empty clusters.
        tie_breaker = torch.arange(self.num_clusters, device=labels.device)
        order = torch.argsort(first_indices * self.num_clusters + tie_breaker)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(self.num_clusters, device=labels.device)
        self.centroids = self.centroids[order]
        return inverse[labels]

    def fit(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = self._validate_input(X)
        self.centroids = self.sample_init_centroids(X)

        for _ in range(self.max_iteration):
            labels = self.find_clusters_index(self.compute_cosine_distances(X, self.centroids))
            new_centroids = self.update_centroids(X, labels)
            shift = torch.linalg.vector_norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift <= self.tol:
                break

        # Labels must correspond to the final (possibly just-updated) centers.
        labels = self.find_clusters_index(self.compute_cosine_distances(X, self.centroids))
        labels = self._canonicalize_clusters(labels)
        return labels, self.centroids

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("fit must be called before predict")
        if not isinstance(X, torch.Tensor) or X.ndim != 2 or not X.is_floating_point():
            raise ValueError("X must be a floating-point 2D torch.Tensor")
        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError("X and fitted centroids must have the same feature dimension")
        X = X.to(device=self.centroids.device, dtype=torch.float32)
        return self.find_clusters_index(self.compute_cosine_distances(X, self.centroids))


class AdaptiveKMeansTorch:
    """Choose a spherical K-means codebook size with elbow or silhouette scoring."""

    SUPPORTED_METHODS = {"elbow", "silhouette"}

    def __init__(
        self,
        max_clusters: int = 20,
        method: str = "silhouette",
        max_iteration: int = 100,
        tol: float = 1e-4,
        min_clusters: int = 12,
        seed: int | None = 0,
    ):
        if isinstance(min_clusters, bool) or not isinstance(min_clusters, int) or min_clusters < 1:
            raise ValueError("min_clusters must be a positive integer")
        if isinstance(max_clusters, bool) or not isinstance(max_clusters, int) or max_clusters < min_clusters:
            raise ValueError("max_clusters must be an integer greater than or equal to min_clusters")
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {sorted(self.SUPPORTED_METHODS)}, got {method!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.method = method
        self.max_iteration = max_iteration
        self.tol = tol
        self.seed = seed
        self.best_K: int | None = None
        self.centroids: torch.Tensor | None = None
        self._best_labels: torch.Tensor | None = None

    def fit_kmeans(self, X: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        kmeans = KMeansTorch(
            num_clusters=K,
            max_iteration=self.max_iteration,
            tol=self.tol,
            seed=self.seed,
        )
        return kmeans.fit(X)

    @staticmethod
    def compute_silhouette_torch(
        X: torch.Tensor,
        labels: torch.Tensor,
        centroids: torch.Tensor,
    ) -> float:
        """Return the mean cosine-distance silhouette coefficient.

        Cluster distance sums are computed from normalized cluster-vector sums,
        so this is the exact pairwise cosine silhouette in O(NKD) rather than
        materializing an O(N^2) distance matrix.
        """

        if X.ndim != 2 or labels.ndim != 1 or centroids.ndim != 2 or X.shape[0] != labels.shape[0]:
            raise ValueError("X, labels, and centroids have incompatible shapes")
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long")
        if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= centroids.shape[0]:
            raise ValueError("labels must index the supplied centroids")

        normalized = F.normalize(X.to(dtype=torch.float32), dim=-1, eps=1e-12)
        counts = torch.bincount(labels, minlength=centroids.shape[0])
        populated = counts > 0
        if int(populated.sum()) < 2:
            return -1.0

        cluster_sums = torch.zeros(
            centroids.shape[0],
            normalized.shape[1],
            device=normalized.device,
            dtype=normalized.dtype,
        )
        cluster_sums.index_add_(0, labels, normalized)
        similarity_sums = normalized @ cluster_sums.T
        distance_sums = (counts.to(normalized.dtype)[None, :] - similarity_sums).clamp_min(0)

        rows = torch.arange(X.shape[0], device=X.device)
        self_distance = 1 - normalized.square().sum(dim=1)
        within = (distance_sums[rows, labels] - self_distance).clamp_min(0)
        within = within / (counts[labels] - 1).clamp_min(1).to(normalized.dtype)

        other = distance_sums / counts.clamp_min(1).to(normalized.dtype)[None, :]
        other[:, ~populated] = torch.inf
        other[rows, labels] = torch.inf
        nearest_other = other.amin(dim=1)
        denominator = torch.maximum(within, nearest_other).clamp_min(1e-12)
        score = (nearest_other - within) / denominator
        # As in the standard silhouette definition, singleton clusters have a
        # score of zero rather than a spuriously perfect score.
        score = torch.where(counts[labels] == 1, torch.zeros_like(score), score)
        return float(score.mean().item())

    @staticmethod
    def compute_elbow(distortions: Sequence[float], cluster_counts: Sequence[int]) -> int:
        """Return the candidate K farthest from the endpoints' straight line."""

        if len(distortions) != len(cluster_counts) or not distortions:
            raise ValueError("distortions and cluster_counts must be non-empty and have equal length")
        if len(cluster_counts) < 3:
            return int(cluster_counts[0])

        points = torch.tensor(list(zip(cluster_counts, distortions)), dtype=torch.float64)
        span = points.amax(dim=0) - points.amin(dim=0)
        points = (points - points.amin(dim=0)) / span.clamp_min(1e-12)
        line = points[-1] - points[0]
        line_norm = torch.linalg.vector_norm(line)
        if line_norm <= 1e-12:
            return int(cluster_counts[0])

        relative = points - points[0]
        # 2D perpendicular distance to the line through the first/last points.
        distances = torch.abs(relative[:, 0] * line[1] - relative[:, 1] * line[0]) / line_norm
        return int(cluster_counts[int(torch.argmax(distances).item())])

    def find_best_K(self, X: torch.Tensor) -> None:
        max_candidate = min(self.max_clusters, X.shape[0])
        if self.method == "silhouette" and X.shape[0] > 1:
            max_candidate = min(max_candidate, X.shape[0] - 1)
        min_candidate = min(self.min_clusters, max_candidate)
        cluster_counts = list(range(min_candidate, max_candidate + 1))

        candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        distortions: list[float] = []
        best_score = -float("inf")
        best_K = cluster_counts[0]

        for K in cluster_counts:
            labels, centroids = self.fit_kmeans(X, K)
            candidates[K] = (labels, centroids)

            if self.method == "silhouette":
                score = self.compute_silhouette_torch(X, labels, centroids)
                if score > best_score:
                    best_score = score
                    best_K = K
            else:
                distances = KMeansTorch.compute_cosine_distances(X, centroids)
                rows = torch.arange(X.shape[0], device=X.device)
                distortions.append(float(distances[rows, labels].clamp_min(0).sum().item()))

        if self.method == "elbow":
            best_K = self.compute_elbow(distortions, cluster_counts)

        self.best_K = best_K
        self._best_labels, self.centroids = candidates[best_K]

    def fit(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(X, torch.Tensor) or X.ndim != 2 or not X.is_floating_point():
            raise ValueError("X must be a floating-point 2D torch.Tensor")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be non-empty")
        if not torch.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        X = X.to(dtype=torch.float32)
        self.find_best_K(X)
        assert self._best_labels is not None and self.centroids is not None
        return self._best_labels, self.centroids
