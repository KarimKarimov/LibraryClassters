import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, linkage="single"):
        """
        Класс для иерархической кластеризации.

        Параметры:
            linkage (str): Метод объединения кластеров. Возможные значения:
                           - "single" (по умолчанию): Метод одиночного связывания.
                             Определяет расстояние между кластерами как минимальное
                             расстояние между точками в этих кластерах.
                           - "complete": Метод полного связывания. Определяет расстояние
                             между кластерами как максимальное расстояние между точками
                             в этих кластерах.
        """
        self.linkage = linkage

    def fit(self, X):
        """
        Обучает модель иерархической кластеризации на заданном наборе данных.

        Параметры:
            X (ndarray): Массив данных размерности (n_samples, n_features),
                         где n_samples - количество образцов, n_features - количество
                         признаков.
        """
        # Вычисление попарных расстояний между образцами
        pairwise_distances = pdist(X)

        # Инициализация каждого образца как отдельного кластера
        clusters = [{i} for i in range(X.shape[0])]

        # Объединение кластеров до тех пор, пока не останется только один
        while len(clusters) > 1:
            # Вычисление попарных расстояний между кластерами
            distances = squareform(pdist(clusters, metric=self.linkage))

            # Поиск двух наиболее близких кластеров
            i, j = np.unravel_index(np.argmin(distances), distances.shape)

            # Объединение двух наиболее близких кластеров
            clusters[i].update(clusters[j])
            del clusters[j]

        # Сохранение меток кластеров для каждого образца
        self.labels = np.zeros(X.shape[0], dtype=int)
        for i, c in enumerate(clusters):
            for j in c:
                self.labels[j] = i

    def predict(self, X):
        """
        Предсказывает кластеры для новых данных.

        Параметры:
            X (ndarray): Массив новых данных размерности (n_samples, n_features),
                         где n_samples - количество образцов, n_features - количество
                         признаков.

        Возвращает:
            ndarray: Массив меток кластеров размерности (n_samples,), где каждая метка
                     указывает на принадлежность образца к соответствующему кластеру.
        """
        # Предсказывает кластеры для новых данных
        return np.array([self.labels[i] for i in range(X.shape[0])])

