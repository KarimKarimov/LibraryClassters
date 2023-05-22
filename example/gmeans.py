from classes.Gmeans import GMeansClustering

# Создание объекта класса GMeansClustering
gmeans_clustering = GMeansClustering(tolerance=0.001)

# Генерация данных для кластеризации
X = [[1.0, 2.0], [1.5, 1.8], [5.0, 6.0], [5.5, 5.7], [9.0, 10.0], [9.5, 9.8]]

# Обучение модели G-Means
gmeans_clustering.fit(X)

# Получение меток кластеров для новых данных
new_data = [[2.0, 2.0], [8.0, 8.0]]
labels = gmeans_clustering.predict(new_data)

# Получение центров кластеров
centers = gmeans_clustering.get_centers()

print(labels)
print(centers) 