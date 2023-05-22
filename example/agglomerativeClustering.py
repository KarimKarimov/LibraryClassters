import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from classes.AgglomerativeClustering import AgglomerativeClustering
# Создание синтетических данных
X, y = make_blobs(n_samples=200, centers=4, random_state=42)
X = StandardScaler().fit_transform(X)

# Создание экземпляра класса AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='single')

# Обучение модели и предсказание меток
labels = agg_clustering.fit_predict(X)

print(labels)  # Выводит предсказанные метки
