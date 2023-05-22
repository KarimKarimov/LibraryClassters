from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, n_clusters=8, n_init=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = n_init
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.max_iter,
            random_state=self.random_state
        ).fit(X)

    def predict(self, X):
        return self.model.predict(X)
 
    def get_centers(self):
        return self.model.cluster_centers_

