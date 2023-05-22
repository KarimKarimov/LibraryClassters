import numpy as np

class MeanShift:

    """
    Класс для выполнения кластеризации методом Mean Shift.

    Parameters:
    -----------
    kernel_bandwidth : float, optional (default=1)
        Ширина ядра, используемая для вычисления весов точек.

    Attributes:
    -----------
    centers : array-like, shape (n_clusters, n_features)
        Центры кластеров, полученные после выполнения кластеризации.

    Methods:
    --------
    fit(X):
        Обучает модель на данных X и находит центры кластеров.
    
    predict(X):
        Прогнозирует метки кластеров для данных X на основе найденных центров.

    _radial_kernel(distances, bandwidth):
        Вычисляет радиальное ядро для весов точек на основе расстояний и ширины ядра.

    """
    def __init__(self, kernel_bandwidth=1):
        """
        Инициализация объекта MeanShift.

        Parameters:
        -----------
        kernel_bandwidth : float, optional (default=1)
            Ширина ядра, используемая для вычисления весов точек.
        """
        self.kernel_bandwidth = kernel_bandwidth
        
    def fit(self, X):
        """
        Обучает модель на данных X и находит центры кластеров.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения модели.
        """
        bandwidth = self.kernel_bandwidth
        
        while True:
            old_centers = X.copy()
            
            for i, x in enumerate(X):
                # Вычисление расстояний от текущей точки до всех других точек
                distances = np.linalg.norm(X - x, axis=1)
                # Вычисление весов для всех точек, используя радиальное ядро
                weights = self._radial_kernel(distances, bandwidth)
                # Обновление текущей точки
                X[i] = np.sum(X * weights.reshape(-1, 1), axis=0) / np.sum(weights)
                
            # Если центры не изменились, то завершаем алгоритм
            if np.allclose(X, old_centers):
                break
                
        self.centers = X
        
    def predict(self, X):
        """
        Прогнозирует метки кластеров для данных X на основе найденных центров.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Входные данные для прогнозирования меток кластеров.

        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Метки кластеров для каждой точки данных.
        """
        bandwidth = self.kernel_bandwidth
        labels = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # Вычисление расстояний от текущей точки до всех центров
            distances = np.linalg.norm(self.centers - x, axis=1)
            # Вычисление весов для всех центров, используя радиальное ядро
            weights = self._radial_kernel(distances, bandwidth)
            # Нахождение индекса центра с наибольшим весом
            index = np.argmax(weights)
            # Присваивание метки класса текущей точке
            labels[i] = index
            
        return labels
    
    def _radial_kernel(self, distances, bandwidth):
        """
        Вычисляет радиальное ядро для весов точек на основе расстояний и ширины ядра.

        Parameters:
        -----------
        distances : array-like, shape (n_samples,)
            Расстояния от текущей точки до всех других точек.
        
        bandwidth : float
            Ширина ядра.

        Returns:
        --------
        weights : array-like, shape (n_samples,)
            Веса точек, вычисленные с использованием радиального ядра.
        """
        return np.exp(-0.5 * (distances / bandwidth) ** 2)