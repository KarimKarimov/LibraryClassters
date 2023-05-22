from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

class ClusterVisualizer:
    def __init__(self):
        pass

    def plot_clusters(self, X, labels):
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.show()

    def plot_silhouette_score(self, X, labels):
        silhouette_vals = silhouette_samples(X, labels)
        y_lower, y_upper = 0, 0
        fig, ax = plt.subplots()
        for i in np.unique(labels):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            y_lower += len(cluster_silhouette_vals)
        silhouette_avg = silhouette_score(X, labels)
        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Silhouette Coefficients")
        ax.set_ylabel("Cluster labels")
        ax.set_yticks([])
        plt.show()
