from sklearn.cluster import KMeans

class KMeansClustering:
    """
    Класс для выполнения кластеризации методом KMeans.

    Параметры:
    ----------
    n_clusters : int, optional (default=8)
        Количество кластеров для формирования.
    n_init : int, optional (default=300)
        Количество итераций для поиска начальных центров кластеров.
    random_state : int, optional (default=None)
        Зерно для инициализации генератора случайных чисел.

    Атрибуты:
    ---------
    n_clusters : int
        Количество кластеров для формирования.
    max_iter : int
        Количество итераций для поиска начальных центров кластеров.
    random_state : int
        Зерно для инициализации генератора случайных чисел.
    model : sklearn.cluster.KMeans
        Объект модели KMeans, обученный на данных.

    Методы:
    -------
    fit(X)
        Обучает модель KMeans на данных X.
    predict(X)
        Прогнозирует метки кластеров для данных X на основе обученной модели KMeans.
    get_centers()
        Возвращает центры кластеров, найденные моделью KMeans.
    """

    def __init__(self, n_clusters=8, n_init=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = n_init
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        """
        Обучает модель KMeans на данных X.

        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения модели.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.max_iter,
            random_state=self.random_state
        ).fit(X)

    def predict(self, X):
        """
        Прогнозирует метки кластеров для данных X на основе обученной модели KMeans.

        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для прогнозирования меток кластеров.

        Возвращает:
        -----------
        labels : array-like, shape (n_samples,)
            Метки кластеров для каждой точки данных.
        """
        return self.model.predict(X)

    def get_centers(self):
        """
        Возвращает центры кластеров, найденные моделью KMeans.

        Возвращает:
        -----------
        centers : array-like, shape (n_clusters, n_features)
            Координаты центров кластеров.
        """
        return self.model.cluster_centers_


