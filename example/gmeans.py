from classes.Gmeans import GMeansClustering
from MV.vizualizer import ClusterVisualizer
from sklearn.datasets import load_iris
# Создание объекта класса GMeansClustering
gmeans_clustering = GMeansClustering(tolerance=0.001)

# Генерация данных для кластеризации
X = [[1.0, 2.0], [1.5, 1.8], [5.0, 6.0], [5.5, 5.7], [9.0, 10.0], [9.5, 9.8]]
iris = load_iris()
X = iris.data
# Обучение модели G-Means
gmeans_clustering.fit(X)

# Получение меток кластеров для новых данных
new_data = [[2.0, 2.0], [8.0, 8.0]]
labels = gmeans_clustering.predict(X)

# Получение центров кластеров
centers = gmeans_clustering.get_centers()

print(labels)
print(centers) 


visualizer = ClusterVisualizer()
# Визуализируем данные и метки кластеров
visualizer.plot_clusters(X, labels)

# Визуализируем значение метрики качества кластеризации
visualizer.plot_silhouette_score(X, labels)