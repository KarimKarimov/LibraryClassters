from sklearn.datasets import make_blobs
from MV.metrix import SilhouetteScore
from clustering.KMeans import KMeansClustering
from MV.vizualizer import ClusterVisualizer
from MV.metrix import SilhouetteScore
from clustering.Gmeans import GMeansClustering
# Создаем случайные данные
X, y = make_blobs(n_samples=1000, centers=8, n_features=2, random_state=42)

# Инициализируем объекты классов
clustering = KMeansClustering(n_clusters=8, random_state=42)
metric = SilhouetteScore()
visualizer = ClusterVisualizer()

# Обучаем модель на данных
clustering.fit(X)

# Получаем метки кластеров для данных
labels = clustering.predict(X)

# Вычисляем значение метрики качества кластеризации
score = metric.compute(X, labels)
print("Silhouette score:", score)

# Визуализируем данные и метки кластеров
visualizer.plot_clusters(X, labels)

# Визуализируем значение метрики качества кластеризации
visualizer.plot_silhouette_score(X, labels)