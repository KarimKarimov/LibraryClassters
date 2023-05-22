import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        # Initialize each point as a separate cluster
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)

        # Compute distance matrix
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                dist_matrix[j, i] = dist_matrix[i, j]

        # Perform agglomerative clustering
        for k in range(n_samples - self.n_clusters):
            # Find the two closest clusters
            i, j = np.unravel_index(np.argmin(dist_matrix), (n_samples, n_samples))
            
            # Merge the two clusters
            if self.linkage == 'single':
                dist_matrix[i, j] = np.inf
                dist_matrix[j, i] = np.inf
                self.labels_[self.labels_ == self.labels_[j]] = self.labels_[i]
            elif self.linkage == 'complete':
                dist_matrix[i, :] = np.maximum(dist_matrix[i, :], dist_matrix[j, :])
                dist_matrix[:, i] = np.maximum(dist_matrix[:, i], dist_matrix[:, j])
                dist_matrix[i, i] = np.inf
                dist_matrix[j, :] = np.inf
                dist_matrix[:, j] = np.inf
                self.labels_[self.labels_ == self.labels_[j]] = self.labels_[i]
            else:
                raise ValueError("Invalid linkage parameter")

        # Update cluster labels to start from 0
        unique_labels = np.unique(self.labels_)
        for i, label in enumerate(unique_labels):
            self.labels_[self.labels_ == label] = i

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
