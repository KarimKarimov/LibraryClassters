from classes.MeanShift import MeanShift
from MV.vizualizer import ClusterVisualizer
from sklearn.datasets import load_iris

import numpy as np

# Создание объекта класса MeanShift
mean_shift = MeanShift(kernel_bandwidth=5.5)

# Загрузим набор данных iris
iris = load_iris()
X = iris.data


# Кластеризация с использованием метода MeanShift
mean_shift.fit(X)

# Получение меток кластеров для новых данных
new_data = np.random.rand(20, 2)
labels = mean_shift.predict(X)

print(labels)

visualizer = ClusterVisualizer()
# Визуализируем данные и метки кластеров
visualizer.plot_clusters(X, labels)

# Визуализируем значение метрики качества кластеризации
visualizer.plot_silhouette_score(X, labels)