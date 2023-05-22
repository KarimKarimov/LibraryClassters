from pyclustering.cluster.gmeans import gmeans

class GMeansClustering:
    """
    Класс для выполнения кластеризации методом G-Means.

    Параметры:
    ----------
    tolerance : float, optional (default=0.001)
        Параметр для определения остановки алгоритма.

    Атрибуты:
    ---------
    tolerance : float
        Параметр для определения остановки алгоритма.
    model : pyclustering.cluster.gmeans.gmeans
        Объект модели G-Means.

    Методы:
    -------
    fit(X)
        Обучает модель G-Means на данных X.
    predict(X)
        Прогнозирует метки кластеров на основе обученной модели G-Means.
    get_centers()
        Возвращает центры кластеров, найденные моделью G-Means.

    """

    def __init__(self, tolerance=0.001):
        self.tolerance = tolerance
        self.model = None

    def fit(self, X):
        """
        Обучает модель G-Means на данных X.

        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения модели.
        """
        self.model = gmeans(X, tolerance=self.tolerance)
        self.model.process()

    def predict(self, X):
        """
        Прогнозирует метки кластеров на основе обученной модели G-Means.

        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Данные для прогнозирования меток кластеров.

        Возвращает:
        -----------
        labels : list
            Метки кластеров для каждой точки данных.
        """
        return self.model.predict(X)

    def get_centers(self):
        """
        Возвращает центры кластеров, найденные моделью G-Means.

        Возвращает:
        -----------
        centers : array-like, shape (n_clusters, n_features)
            Центры кластеров.
        """
        return self.model.get_centers()

