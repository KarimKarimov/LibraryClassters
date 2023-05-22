import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from classes.CURE import CURE

# Генерация синтетических данных
X, y = make_blobs(n_samples=100, n_features=3, centers=4, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создание и обучение модели CURE
cure = CURE(n_clusters=4, alpha=0.5, c_size=3, linkage='ward')
cure.fit(X_scaled)
# Прогнозирование меток кластеров для новых данных
new_data = np.array([[0.5, 1.0, -1.5], [-2.0, 3.0, 0.0]])
new_data_scaled = scaler.transform(new_data)

labels = cure.predict(new_data_scaled)


print("Predicted labels:", labels)

# Оценка качества кластеризации
silhouette_avg = silhouette_score(X_scaled, cure.labels_)
print("Silhouette Score:", silhouette_avg)

# Визуализация результатов кластеризации
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cure.labels_, cmap='viridis')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.colorbar(scatter)
plt.show()
