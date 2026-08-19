"""Token-to-codebook helpers for VQToken."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from VQToken.cluster import AdaptiveKMeansTorch, KMeansTorch


def _flatten_normalized_tokens(tokens: torch.Tensor) -> tuple[int, int, int, torch.Tensor]:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ValueError("tokens must be a 3D torch.Tensor with shape [frames, tokens, features]")
    if not tokens.is_floating_point():
        raise TypeError("tokens must use a floating-point dtype")
    if any(size == 0 for size in tokens.shape):
        raise ValueError("tokens must be non-empty in every dimension")
    if not torch.isfinite(tokens).all():
        raise ValueError("tokens must contain only finite values")

    frames, tokens_per_frame, token_dim = tokens.shape
    flat = tokens.reshape(-1, token_dim).to(dtype=torch.float32)
    normalized = F.normalize(flat, dim=-1, eps=1e-12)
    return frames, tokens_per_frame, token_dim, normalized


def _cluster_means(
    normalized_tokens: torch.Tensor,
    labels: torch.Tensor,
    num_clusters: int,
    output_dtype: torch.dtype,
    fallback_centroids: torch.Tensor,
) -> torch.Tensor:
    """Accumulate code vectors in float32, then restore the model dtype.

    Degenerate inputs can leave a K-means cluster empty. In that case the
    fitted center is a safer code token than an unrelated all-zero vector.
    """

    token_dim = normalized_tokens.shape[1]
    sums = torch.zeros(
        num_clusters,
        token_dim,
        device=normalized_tokens.device,
        dtype=torch.float32,
    )
    sums.scatter_add_(0, labels[:, None].expand(-1, token_dim), normalized_tokens)

    counts = torch.zeros(num_clusters, device=normalized_tokens.device, dtype=torch.float32)
    counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    means = sums / counts.clamp_min(1)[:, None]
    fallback_centroids = fallback_centroids.to(device=means.device, dtype=means.dtype)
    means = torch.where(counts[:, None] > 0, means, fallback_centroids)
    return means.to(dtype=output_dtype)


def adaptive_kmeans_clustering_tokens_torch(
    tokens: torch.Tensor,
    max_K: int = 20,
    method: str = "elbow",
    min_K: int = 12,
    canonicalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster tokens and select the codebook size automatically.

    Args:
        tokens: Tensor shaped ``[num_frames, tokens_per_frame, token_dim]``.
        max_K: Maximum candidate codebook size.
        method: ``"elbow"`` or ``"silhouette"``.
        min_K: Minimum candidate codebook size. If there are fewer samples,
            the sample count is used instead.
        canonicalize: Reorder cluster IDs by first occurrence. Disable this
            for checkpoints trained against raw K-means assignment IDs.

    Returns:
        A ``[num_frames, tokens_per_frame]`` assignment tensor and a
        ``[selected_K, token_dim]`` normalized codebook.
    """

    frames, tokens_per_frame, _, normalized = _flatten_normalized_tokens(tokens)
    num_samples = normalized.shape[0]
    if isinstance(max_K, bool) or not isinstance(max_K, int) or max_K < 1:
        raise ValueError("max_K must be a positive integer")
    if isinstance(min_K, bool) or not isinstance(min_K, int) or min_K < 1:
        raise ValueError("min_K must be a positive integer")
    if max_K < min_K:
        raise ValueError("max_K must be greater than or equal to min_K")

    effective_max = min(max_K, num_samples)
    effective_min = min(min_K, effective_max)
    kmeans = AdaptiveKMeansTorch(
        max_clusters=effective_max,
        min_clusters=effective_min,
        method=method,
        canonicalize=canonicalize,
    )
    # Assignments are nondifferentiable; detaching prevents an unnecessary
    # graph through every K-means iteration. Means below still carry gradients.
    labels, centroids = kmeans.fit(normalized.detach())
    assert kmeans.best_K is not None
    means = _cluster_means(normalized, labels, kmeans.best_K, tokens.dtype, centroids)
    return labels.reshape(frames, tokens_per_frame), means


def kmeans_clustering_tokens_torch(
    tokens: torch.Tensor,
    K: int,
    max_iteration: int = 50,
    canonicalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster tokens into a fixed-size normalized codebook.

    ``canonicalize=False`` preserves raw K-means IDs for compatibility with
    released attention checkpoints that consume assignment IDs directly.
    """

    frames, tokens_per_frame, _, normalized = _flatten_normalized_tokens(tokens)
    num_samples = normalized.shape[0]
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer")
    if K > num_samples:
        raise ValueError(f"K ({K}) cannot exceed the number of tokens ({num_samples})")

    kmeans = KMeansTorch(
        num_clusters=K,
        max_iteration=max_iteration,
        canonicalize=canonicalize,
    )
    labels, centroids = kmeans.fit(normalized.detach())
    means = _cluster_means(normalized, labels, K, tokens.dtype, centroids)
    return labels.reshape(frames, tokens_per_frame), means
