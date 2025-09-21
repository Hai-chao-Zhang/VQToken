import torch
import torch.nn as nn
import faiss

import torch

class KMeansTorch:
    def __init__(self, num_clusters: int, max_iteration: int = 300, tol: float = 1e-4):
        self.num_clusters = num_clusters
        self.max_iteration = max_iteration
        self.tol = tol
        self.centroids = None

    def sample_init_centroids(self, X):
        """
        K-Means++ style init with:
        - squared-distance weighting
        - small ε floor to avoid zero-sum
        - multinomial fallback
        """
        n_samples = X.size(0)
        # 1) first centroid uniform at random
        first_idx = torch.randint(0, n_samples, (1,)).item()
        centroids = [X[first_idx]]

        eps = 1e-6  # ensures weights.sum() > 0

        for _ in range(1, self.num_clusters):
            # compute squared Euclidean distance to nearest centroid
            dists = torch.cdist(X, torch.stack(centroids), p=2).min(dim=1)[0]
            weights = dists.pow(2).clamp(min=0.0) + eps

            # normalize into probabilities
            probs = weights / weights.sum()

            # sample next centroid, fallback to uniform on error
            try:
                next_idx = torch.multinomial(probs, 1).item()
            except RuntimeError:
                next_idx = torch.randint(0, n_samples, (1,)).item()

            centroids.append(X[next_idx])

        return torch.stack(centroids)

    def compute_cosine_distances(self, X, centroids):
        X = X / X.norm(dim=-1, keepdim=True)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True)
        return 1 - torch.mm(X, centroids.T)

    def find_clusters_index(self, distances):
        return torch.argmin(distances, dim=1)

    def update_centroids(self, X, labels):
        new_c = []
        for i in range(self.num_clusters):
            pts = X[labels == i]
            new_c.append(self.centroids[i] if pts.numel()==0 else pts.mean(dim=0))
        return torch.stack(new_c)

    def fit(self, X):
        self.centroids = self.sample_init_centroids(X)
        for _ in range(self.max_iteration):
            dists = self.compute_cosine_distances(X, self.centroids)
            labels = self.find_clusters_index(dists)
            new_c = self.update_centroids(X, labels)
            if torch.norm(self.centroids - new_c) < self.tol:
                break
            self.centroids = new_c
        return labels, self.centroids

    def predict(self, X):
        dists = self.compute_cosine_distances(X, self.centroids)
        return self.find_clusters_index(dists)



class AdaptiveKMeansTorch:
    def __init__(self, max_clusters: int = 20, method: str = "silhouette", max_iteration: int = 100, tol: float = 1e-4):
        self.max_clusters = max_clusters
        self.method = method
        self.max_iteration = max_iteration
        self.tol = tol
        self.best_K = None
        self.centroids = None

    def approximate_cosine_similarity(self, X, top_k=50):
        """Use FAISS for fast nearest neighbors instead of full pairwise cosine similarity."""
        X_np = X.cpu().numpy().astype('float32')
        index = faiss.IndexFlatIP(X_np.shape[1])  # Inner product = cosine similarity
        index.add(X_np)

        distances, _ = index.search(X_np, top_k)
        return distances.mean(axis=1)  # Approximate silhouette score

    def fit_kmeans(self, X, K):
        """Perform standard K-Means clustering using cosine similarity."""
        kmeans = KMeansTorch(num_clusters=K, max_iteration=self.max_iteration, tol=self.tol)
        return kmeans.fit(X)

    def compute_silhouette_torch(self, X, labels):
        """Compute Silhouette Score using FAISS to avoid full similarity matrix."""
        return self.approximate_cosine_similarity(X).mean()

    def compute_elbow(self, distortions):
        """Find the best `K` using the Elbow Method."""
        diffs = torch.tensor(distortions[1:]) - torch.tensor(distortions[:-1])
        diffs2 = diffs[1:] - diffs[:-1]
        return torch.argmax(diffs2) + 2

    def find_best_K(self, X):
        """Determine the best K using Silhouette Score or Elbow Method."""
        best_score = -float("inf")
        best_K = 16  # Enforce larger K
        distortions = []

        for K in range(12, self.max_clusters + 1):
            labels, centroids = self.fit_kmeans(X, K)

            if self.method == "silhouette":
                score = self.compute_silhouette_torch(X, labels)
                if score > best_score:
                    best_score, best_K, self.centroids = score, K, centroids

            elif self.method == "elbow":
                distortion = ((X - centroids[labels])**2).sum().item()
                distortions.append(distortion)

        if self.method == "elbow":
            best_K = self.compute_elbow(distortions) + 2  # Increase K

        self.best_K = max(best_K, 12)  # Ensure K is not too small

    def fit(self, X):
        """Main function to fit Adaptive K-Means."""
        X = X.to(torch.float32)
        self.find_best_K(X)
        labels, centroids = self.fit_kmeans(X, self.best_K)
        return labels, centroids

