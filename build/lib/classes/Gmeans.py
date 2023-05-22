from pyclustering.cluster.gmeans import gmeans

class GMeansClustering:
    def __init__(self, tolerance=0.001):
        self.tolerance = tolerance
        self.model = None

    def fit(self, X):
        self.model = gmeans(X, tolerance=self.tolerance)
        self.model.process()

    def predict(self, X):
        return self.model.predict(X)

    def get_centers(self):
        return self.model.get_centers()
