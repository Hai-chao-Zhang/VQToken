import torch
from VQToken.cluster import KMeansTorch
from VQToken.cluster import AdaptiveKMeansTorch


def adaptive_kmeans_clustering_tokens_torch(tokens: torch.Tensor, max_K: int = 20, method="elbow"):
    """
    Cluster a sequence of tokens with Adaptive K-means and return per-token labels and
    empirical (data-averaged) cluster means.

    Args:
        tokens (torch.Tensor): Shape [num_frames, tokens_per_frame, token_dim].
        max_K (int): Upper bound on the number of clusters considered.
        method (str): Strategy to select K (e.g., "elbow" or "silhouette").

    Returns:
        cluster_indices (torch.Tensor): Shape [num_frames, tokens_per_frame] with assignments.
        cluster_means (torch.Tensor): Shape [adaptive_K, token_dim] with per-cluster means.
    """
    # Flatten to a 2D matrix [T*N, D]
    T, N, D = tokens.shape
    flat = tokens.view(-1, D)

    # L2-normalize each token vector
    x_norm = flat / flat.norm(dim=-1, keepdim=True)

    # Run Adaptive K-means to pick K automatically
    km = AdaptiveKMeansTorch(max_clusters=max_K, method=method)
    labels, _ = km.fit(x_norm)

    # Compute empirical means from assignments rather than relying on centroids
    C = km.best_K
    cluster_means = torch.zeros(C, D, device=tokens.device, dtype=tokens.dtype)
    counts = torch.zeros(C, device=tokens.device, dtype=tokens.dtype)

    # Accumulate token sums per cluster via scatter_add_
    sum_by_cluster = torch.zeros_like(cluster_means).scatter_add_(
        0, labels.view(-1, 1).expand(-1, D), x_norm
    )

    # Count tokens per cluster
    counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=tokens.dtype))

    # Avoid divide-by-zero: treat empty clusters as having count 1 (sum will be 0 so mean stays 0)
    counts = counts.clamp_min(1)

    # Final per-cluster means
    cluster_means = sum_by_cluster / counts.view(-1, 1)

    # Restore label shape to [T, N]
    cluster_indices = labels.view(T, N)

    return cluster_indices, cluster_means


def kmeans_clustering_tokens_torch(tokens: torch.Tensor, K: int):
    """
    Cluster a sequence of tokens using standard K-means and return per-token labels and
    empirical (data-averaged) cluster means.

    Args:
        tokens (torch.Tensor): Shape [num_frames, tokens_per_frame, token_dim].
        K (int): Number of clusters.

    Returns:
        cluster_indices (torch.Tensor): Shape [num_frames, tokens_per_frame] with assignments.
        cluster_means (torch.Tensor): Shape [K, token_dim] with per-cluster means.
    """
    # Flatten to a 2D matrix [T*N, D]
    T, N, D = tokens.shape
    flat = tokens.view(-1, D)

    # L2-normalize tokens (helpful for cosine-like behavior)
    x_norm = flat / flat.norm(dim=-1, keepdim=True)

    # Run fixed-K K-means
    km = KMeansTorch(n_clusters=K, max_iter=50)
    labels, _ = km.fit(x_norm)

    # Reshape labels back to [T, N]
    cluster_indices = labels.view(T, N)

    # Compute empirical means from assignments
    cluster_means = torch.zeros(K, D, device=tokens.device, dtype=tokens.dtype)
    counts = torch.zeros(K, device=tokens.device, dtype=tokens.dtype)

    # Sum tokens per cluster
    sum_by_cluster = torch.zeros_like(cluster_means).scatter_add_(
        0, labels.view(-1, 1).expand(-1, D), x_norm
    )

    # Count tokens per cluster
    counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=tokens.dtype))

    # Safe division
    counts = counts.clamp_min(1)
    cluster_means = sum_by_cluster / counts.view(-1, 1)

    return cluster_indices, cluster_means
