import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, linkage="single"):
        self.linkage = linkage

    def fit(self, X):
        # Calculate pairwise distances between samples
        pairwise_distances = pdist(X)

        # Initialize each sample as a cluster
        clusters = [{i} for i in range(X.shape[0])]

        # Merge clusters until only one remains
        while len(clusters) > 1:
            # Calculate pairwise distances between clusters
            distances = squareform(pdist(clusters, metric=self.linkage))

            # Find closest pair of clusters
            i, j = np.unravel_index(np.argmin(distances), distances.shape)

            # Merge closest pair of clusters
            clusters[i].update(clusters[j])
            del clusters[j]

        # Store cluster labels for each sample
        self.labels = np.zeros(X.shape[0], dtype=int)
        for i, c in enumerate(clusters):
            for j in c:
                self.labels[j] = i

    def predict(self, X):
        # Predict clusters for new data
        return np.array([self.labels[i] for i in range(X.shape[0])])
