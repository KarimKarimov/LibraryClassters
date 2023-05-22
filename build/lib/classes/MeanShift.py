import numpy as np

class MeanShift:
    def __init__(self, kernel_bandwidth=1):
        self.kernel_bandwidth = kernel_bandwidth
        
    def fit(self, X):
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
        return np.exp(-0.5 * (distances / bandwidth) ** 2)