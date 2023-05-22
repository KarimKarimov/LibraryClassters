import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering

class CURE(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, alpha=0.5, c_size=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.c_size = c_size
        self.linkage = linkage

    def fit(self, X, y=None):
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ = []
        
        # Поиск начальных центроидов
        for i in range(self.n_clusters):
            centroid_idx = np.random.choice(X.shape[0])
            centroid = X[centroid_idx]
            self.centroids_.append(centroid)
            self.labels_[centroid_idx] = i + 1
            
        # Кластеризация с помощью иерархической кластеризации
        ac = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        ac.fit(X)
        self.labels_ = ac.labels_
        
        # Кластеризация с помощью CURE
        while len(self.centroids_) < self.n_clusters:
            # Нахождение наиболее удаленных точек
            distances = pdist(self.centroids_, metric='euclidean')
            max_dist_idx = np.argmax(distances)
            idx_1, idx_2 = np.unravel_index(max_dist_idx, (len(self.centroids_), len(self.centroids_)))
            
            # Нахождение ближайших точек к выбранным центроидам
            cluster_1 = X[self.labels_ == idx_1]
            cluster_2 = X[self.labels_ == idx_2]
            closest_1_idx = np.argmin(squareform(pdist(np.vstack([self.centroids_[idx_1], cluster_1]), metric='euclidean'))[:-1])
            closest_2_idx = np.argmin(squareform(pdist(np.vstack([self.centroids_[idx_2], cluster_2]), metric='euclidean'))[:-1])
            
            # Объединение кластеров и обновление центроидов
            new_cluster = np.vstack([cluster_1, cluster_2])
            new_centroid = (self.alpha * self.centroids_[idx_1] + (1 - self.alpha) * new_cluster[closest_1_idx]) / 2
            self.centroids_[idx_1] = new_centroid
            self.centroids_[idx_2] = new_cluster[closest_2_idx]
        
            # Обновление меток кластеров
            self.labels_[self.labels_ == idx_2] = idx_1
        
        return self

    def predict(self, X):
        labels = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids_]
            labels[i] = np.argmin(distances)
        return labels

    def save_model(self, filename):
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            
    
    def load_model(filename):
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            return model
