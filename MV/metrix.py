
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score

class DavisBouldinIndex:
    def __init__(self):
        pass

    def calculate(self, X, labels):
        n_clusters = len(set(labels))
        cluster_centers = np.zeros((n_clusters, X.shape[1]))

        for i in range(n_clusters):
            cluster_centers[i] = np.mean(X[labels == i], axis=0)

        cluster_distances = pairwise_distances(cluster_centers)
        intra_cluster_distances = np.zeros(n_clusters)
        for i in range(n_clusters):
            intra_cluster_distances[i] = np.mean(pairwise_distances(X[labels == i], cluster_centers[i].reshape(1, -1)))

        inter_cluster_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    inter_cluster_distances[i][j] = intra_cluster_distances[i] + intra_cluster_distances[j]

        db_index = np.zeros(n_clusters)
        for i in range(n_clusters):
            db_index[i] = np.max((intra_cluster_distances + intra_cluster_distances[i]) / inter_cluster_distances[i])

        return np.mean(db_index)

class SilhouetteCoefficient:
    def __init__(self):
        pass

    def calculate(self, X, labels):
        n_clusters = len(set(labels))

        if n_clusters == 1:
            return 0

        cluster_silhouette = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a_i = np.mean(pairwise_distances(X[i].reshape(1, -1), X[labels == labels[i]]))
            b_i = np.zeros(n_clusters)
            for j in range(n_clusters):
                if j != labels[i]:
                    b_i[j] = np.mean(pairwise_distances(X[i].reshape(1, -1), X[labels == j]))
            cluster_silhouette[i] = (np.min(b_i) - a_i) / np.max([a_i, np.min(b_i)])

        return np.mean(cluster_silhouette)

class SilhouetteScore:
    def __init__(self):
        pass

    def compute(self, X, labels):
        return silhouette_score(X, labels)