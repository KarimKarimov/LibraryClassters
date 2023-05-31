import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from classes.Optics import OpticsClusterer
from MV.vizualizer import ClusterVisualizer
# Загрузим набор данных iris
iris = load_iris()
X = iris.data

# Выполним стандартизацию данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Создадим экземпляр класса OpticsClusterer
clusterer = OpticsClusterer(min_samples=5, xi=0.05, min_cluster_size=0.1)

# Выполним кластеризацию данных
clusterer.fit(X_scaled)
labels = clusterer.predict()

# Выведем результаты
print("Кластеры:")
print(labels)

visualizer = ClusterVisualizer()
# Визуализируем данные и метки кластеров
visualizer.plot_clusters(X_scaled, labels)

# Визуализируем значение метрики качества кластеризации
visualizer.plot_silhouette_score(X_scaled, labels)