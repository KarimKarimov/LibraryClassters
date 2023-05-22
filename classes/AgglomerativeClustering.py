import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        """
        Агломеративная кластеризация.

        Параметры:
        - n_clusters (int): Количество кластеров для поиска.
        - linkage (str): Критерий объединения кластеров. Допустимые значения: 'single' и 'complete'.

        Атрибуты:
        - labels_ (ndarray): Метки кластеров, назначенные каждой точке данных после обучения.

        """

        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        """
        Обучение модели агломеративной кластеризации на входных данных.

        Параметры:
        - X (ndarray): Входные данные размерности (n_samples, n_features).

        """
        # Инициализация каждой точки как отдельного кластера
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)

        # Вычисление матрицы расстояний
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                dist_matrix[j, i] = dist_matrix[i, j]

        # Выполнение агломеративной кластеризации
        for k in range(n_samples - self.n_clusters):
            # Поиск двух ближайших кластеров
            i, j = np.unravel_index(np.argmin(dist_matrix), (n_samples, n_samples))
            
            # Объединение двух кластеров
            if self.linkage == 'single':
                dist_matrix[i, j] = np.inf
                dist_matrix[j, i] = np.inf
                self.labels_[self.labels_ == self.labels_[j]] = self.labels_[i]
            elif self.linkage == 'complete':
                dist_matrix[i, :] = np.maximum(dist_matrix[i, :], dist_matrix[j, :])
                dist_matrix[:, i] = np.maximum(dist_matrix[:, i], dist_matrix[:, j])
                dist_matrix[i, i] = np.inf
                dist_matrix[j, :] = np.inf
                dist_matrix[:, j] = np.inf
                self.labels_[self.labels_ == self.labels_[j]] = self.labels_[i]
            else:
                raise ValueError("Недопустимое значение параметра linkage")

        # Обновление меток кластеров, начиная с 0
        unique_labels = np.unique(self.labels_)
        for i, label in enumerate(unique_labels):
            self.labels_[self.labels_ == label] = i

    def fit_predict(self, X):
        """
        Обучение модели агломеративной кластеризации на входных данных и возврат меток кластеров.

        Параметры:
        - X (ndarray): Входные данные размерности (n_samples, n_features).

        Возвращает:
        - labels (ndarray): Метки кластеров, назначенные каждой точке данных.

        """
        self.fit(X)
        return self.labels_
