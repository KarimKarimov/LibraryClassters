from classes.MeanShift import MeanShift

import numpy as np

# Создание объекта класса MeanShift
mean_shift = MeanShift(kernel_bandwidth=0.5)

# Генерация случайных данных для кластеризации
X = np.random.rand(100, 2)
print(X)
# Кластеризация с использованием метода MeanShift
mean_shift.fit(X)

# Получение меток кластеров для новых данных
new_data = np.random.rand(20, 2)
labels = mean_shift.predict(new_data)

print(labels)
