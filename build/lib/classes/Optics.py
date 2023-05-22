from sklearn.cluster import OPTICS

class OpticsClusterer:
    def __init__(self, min_samples=5, xi=0.05, min_cluster_size=0.1):
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.clusterer = None

    def fit(self, X):
        self.clusterer = OPTICS(min_samples=self.min_samples, xi=self.xi, min_cluster_size=self.min_cluster_size)
        self.clusterer.fit(X)

    def predict(self):
        labels = self.clusterer.labels_
        return labels