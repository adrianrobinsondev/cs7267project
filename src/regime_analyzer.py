import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class RegimeAnalyzer:
    """
    Analyzes market regimes using unsupervised learning techniques.
    Implements PCA and K-Means from scratch as per project requirements.
    """

    def __init__(self, cluster_count: int = 3):
        self.k = cluster_count
        self.pca_components = None
        self.mean_vec = None
        self.centroids = None

    def perform_pca(self, X: pd.DataFrame, variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, float]:
        """
        Performs Principal Component Analysis from scratch.
        Returns reduced dataframe and total explained variance.
        """
        if X.shape[1] < 2:
            print("Only one feature available. PCA is trivial; returning original data.")
            return X, 1.0

        # Step 1: Center the data
        self.mean_vec = X.mean(axis=0).values
        X_centered = X.values - self.mean_vec

        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Ensure cov_matrix is 2D (handle case where np.cov returns a scalar)
        if np.ndim(cov_matrix) == 0:
            cov_matrix = np.array([[cov_matrix]])
        elif np.ndim(cov_matrix) == 1:
            cov_matrix = cov_matrix.reshape(1, 1)

        # Step 3: Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Determine number of components based on explained variance
        total_var = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_var
        cumulative_variance = np.cumsum(explained_variance_ratio)

        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Store selected components
        self.pca_components = sorted_eigenvectors[:, :n_components]

        # Project data
        reduced_data = np.dot(X_centered, self.pca_components)

        column_names = [f"PC{i+1}" for i in range(n_components)]
        reduced_df = pd.DataFrame(reduced_data, index=X.index, columns=column_names)

        return reduced_df, cumulative_variance[n_components-1]

    def perform_kmeans(self, X: pd.DataFrame, max_iters: int = 100) -> pd.Series:
        """
        Implements K-Means clustering from scratch.
        Returns a Series containing cluster labels.
        """
        data = X.values
        np.random.seed(42)

        # Initialize centroids randomly from data points
        initial_indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[initial_indices]

        for _ in range(max_iters):
            # Euclidean distance from each point to every centroid
            # Shape: (n_samples, k)
            distances = np.sqrt(np.sum((data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :])**2, axis=-1))

            # Assign points to nearest centroid
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.k):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return pd.Series(labels, index=X.index, name="market_state")

    def calculate_silhouette_score(self, X: pd.DataFrame, labels: pd.Series) -> float:
        """
        Calculates the silhouette score from scratch.
        """
        data = X.values
        num_samples = data.shape[0]
        a = np.zeros(num_samples)
        b = np.zeros(num_samples)

        # Precompute pairwise distances
        pairwise_distances = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :])**2, axis=-1))

        unique_labels = np.unique(labels)

        for i in range(num_samples):
            current_label = labels.iloc[i]

            # a(i): Mean distance to points in same cluster
            same_cluster_mask = (labels == current_label)
            same_cluster_mask.iloc[i] = False
            if same_cluster_mask.sum() > 0:
                a[i] = np.mean(pairwise_distances[i, same_cluster_mask.values])
            else:
                a[i] = 0

            # b(i): Mean distance to points in nearest neighbor cluster
            neighbor_distances = []
            for label in unique_labels:
                if label == current_label:
                    continue
                other_cluster_mask = (labels == label)
                if other_cluster_mask.sum() > 0:
                    mean_dist = np.mean(pairwise_distances[i, other_cluster_mask.values])
                    neighbor_distances.append(mean_dist)

            b[i] = min(neighbor_distances) if neighbor_distances else 0

        silhouettes = (b - a) / np.maximum(a, b)
        # Handle NaNs where a=b=0
        silhouettes = np.nan_to_num(silhouettes)

        return np.mean(silhouettes)
