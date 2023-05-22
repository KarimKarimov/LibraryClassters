from sklearn.cluster import OPTICS

class OpticsClusterer:
    """
    Класс для выполнения кластеризации методом OPTICS.

    Параметры:
    ----------
    min_samples : int, optional (default=5)
        Минимальное количество соседей для определения ядра кластера.
    xi : float, optional (default=0.05)
        Параметр определения расстояния между кластерами.
    min_cluster_size : float, optional (default=0.1)
        Минимальный размер кластера.

    Атрибуты:
    ---------
    min_samples : int
        Минимальное количество соседей для определения ядра кластера.
    xi : float
        Параметр определения расстояния между кластерами.
    min_cluster_size : float
        Минимальный размер кластера.
    clusterer : sklearn.cluster.OPTICS
        Объект модели OPTICS, обученный на данных.

    Методы:
    -------
    fit(X)
        Обучает модель OPTICS на данных X.
    predict()
        Прогнозирует метки кластеров на основе обученной модели OPTICS.

    """

    def __init__(self, min_samples=5, xi=0.05, min_cluster_size=0.1):
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.clusterer = None

    def fit(self, X):
        """
        Обучает модель OPTICS на данных X.

        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения модели.
        """
        self.clusterer = OPTICS(min_samples=self.min_samples, xi=self.xi, min_cluster_size=self.min_cluster_size)
        self.clusterer.fit(X)

    def predict(self):
        """
        Прогнозирует метки кластеров на основе обученной модели OPTICS.

        Возвращает:
        -----------
        labels : array-like, shape (n_samples,)
            Метки кластеров для каждой точки данных.
        """
        labels = self.clusterer.labels_
        return labels
