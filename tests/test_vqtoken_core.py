from __future__ import annotations

import pytest
import torch

from VQToken.cluster import AdaptiveKMeansTorch, KMeansTorch
from VQToken.vq_attn import VQAttn
from VQToken.vq_token import (
    adaptive_kmeans_clustering_tokens_torch,
    kmeans_clustering_tokens_torch,
)


def test_fixed_kmeans_contract_and_gradients():
    torch.manual_seed(0)
    tokens = torch.randn(3, 10, 8, requires_grad=True)

    labels, means = kmeans_clustering_tokens_torch(tokens, K=4)

    assert labels.shape == (3, 10)
    assert means.shape == (4, 8)
    assert labels.dtype == torch.long
    assert torch.isfinite(means).all()
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 4

    means.square().sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_centroid_accumulation_uses_float32(dtype):
    # Low-precision counters/sums lose unit increments for clusters this large.
    tokens = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).repeat(4096).reshape(1, 4096, 3)
    _, means = kmeans_clustering_tokens_torch(tokens, K=1)
    expected = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)

    assert means.dtype == dtype
    torch.testing.assert_close(means.float(), expected, atol=5e-3, rtol=5e-3)


def test_zero_vectors_remain_finite():
    tokens = torch.zeros(2, 8, 4)
    labels, means = kmeans_clustering_tokens_torch(tokens, K=3)

    assert labels.shape == (2, 8)
    assert torch.isfinite(means).all()
    assert torch.count_nonzero(means) == 0


def test_fixed_kmeans_rejects_invalid_k():
    tokens = torch.randn(1, 4, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        kmeans_clustering_tokens_torch(tokens, K=5)
    with pytest.raises(ValueError, match="positive integer"):
        kmeans_clustering_tokens_torch(tokens, K=0)


def test_silhouette_score_depends_on_assignments():
    X = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    centroids = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    good = torch.tensor([0, 0, 1, 1])
    bad = torch.tensor([0, 1, 0, 1])

    good_score = AdaptiveKMeansTorch.compute_silhouette_torch(X, good, centroids)
    bad_score = AdaptiveKMeansTorch.compute_silhouette_torch(X, bad, centroids)
    assert good_score > bad_score


def test_silhouette_singletons_are_not_scored_as_perfect():
    X = torch.eye(3)
    labels = torch.arange(3)

    score = AdaptiveKMeansTorch.compute_silhouette_torch(X, labels, X)

    assert score == 0.0


def test_empty_clusters_use_fitted_centroids_instead_of_invalid_zero_codes():
    tokens = torch.tensor([1.0, 2.0, 3.0]).repeat(8).reshape(1, 8, 3)

    _, means = kmeans_clustering_tokens_torch(tokens, K=3)

    assert torch.isfinite(means).all()
    assert torch.count_nonzero(means, dim=1).tolist() == [3, 3, 3]


def test_elbow_returns_the_candidate_k_not_an_offset_index():
    counts = [20, 21, 22, 23, 24, 25, 26]
    distortions = [100.0, 78.0, 58.0, 41.0, 37.0, 35.0, 34.0]
    selected = AdaptiveKMeansTorch.compute_elbow(distortions, counts)

    assert selected in counts
    assert selected >= 23


@pytest.mark.parametrize("method", ["elbow", "silhouette"])
def test_adaptive_kmeans_contract(method):
    torch.manual_seed(1)
    tokens = torch.randn(2, 16, 6)
    labels, means = adaptive_kmeans_clustering_tokens_torch(tokens, min_K=2, max_K=5, method=method)

    assert labels.shape == (2, 16)
    assert 2 <= means.shape[0] <= 5
    assert means.shape[1] == 6
    assert torch.isfinite(means).all()


def test_kmeans_predict_requires_fit():
    model = KMeansTorch(num_clusters=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(torch.randn(3, 4))


def test_kmeans_initialization_is_reproducible_by_default():
    X = torch.randn(24, 5)

    labels_a, centers_a = KMeansTorch(num_clusters=4).fit(X)
    torch.rand(100)
    labels_b, centers_b = KMeansTorch(num_clusters=4).fit(X)

    torch.testing.assert_close(labels_a, labels_b)
    torch.testing.assert_close(centers_a, centers_b)


def test_vq_attention_preserves_sequence_axes():
    torch.manual_seed(2)
    module = VQAttn(query_dim=5, context_dim=8, num_heads=2)
    query = torch.randn(7, 5)
    codebook = torch.randn(3, 8)

    output = module.cross_attention_weighted_clusters(query, codebook)
    assert output.shape == (7, 8)
    assert torch.isfinite(output).all()

    batched = module.cross_attention_weighted_clusters(query.unsqueeze(0).expand(2, -1, -1), codebook)
    assert batched.shape == (2, 7, 8)


def test_vq_attention_initializes_layer_norm_scales_to_one():
    module = VQAttn(query_dim=5, context_dim=8, num_heads=2)
    layer_norm_weights = [m.weight for m in module.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert layer_norm_weights
    for weight in layer_norm_weights:
        torch.testing.assert_close(weight, torch.ones_like(weight))


def test_vq_attention_rejects_incompatible_batches():
    module = VQAttn(query_dim=5, context_dim=8, num_heads=2)
    with pytest.raises(ValueError, match="batch sizes"):
        module.cross_attention_weighted_clusters(torch.randn(2, 4, 5), torch.randn(3, 6, 8))
