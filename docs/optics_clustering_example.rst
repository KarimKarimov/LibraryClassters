================================================================================
OPTICS (Ordering Points To Identify the Clustering Structure) Clustering Example
================================================================================


**Код**

.. literalinclude:: ..\example\optics.py
   :language: python

.. admonition:: Описание шагов

   Шаг 1:
   Загрузка набора данных iris с использованием `load_iris()` из модуля `sklearn.datasets`. Загруженные данные представлены в переменной `iris`, а их признаки находятся в `X`.   
   
   Шаг 2:
   Стандартизация данных с помощью `StandardScaler()` из модуля `sklearn.preprocessing`. Вызов `scaler.fit_transform(X)` выполняет стандартизацию признаков данных `X` и сохраняет результат в `X_scaled`.

   Шаг 3:
   Создание экземпляра класса `OpticsClusterer` с указанием параметров `min_samples=5`, `xi=0.05` и `min_cluster_size=0.1`. Эти параметры определяют поведение алгоритма OPTICS.

   Шаг 4:
   Кластеризация данных с использованием метода `fit` экземпляра `clusterer`. Вызов `clusterer.fit(X_scaled)` применяет алгоритм OPTICS к стандартизованным данным `X_scaled` и находит кластеры.
   
   Шаг 5:
   Прогнозирование меток кластеров без указания новых данных. Вызов `clusterer.predict()` возвращает метки кластеров для каждой точки в исходных данных `X_scaled`.

   Шаг 6:
   Вывод результатов на экран с помощью функции `print`. В данном примере выводятся метки кластеров для каждой точки.


**Результат**

.. code-block:: python

   Кластеры:
   [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0 -1 -1 -1  1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1  1  1 -1  1 -1 -1
   -1 -1 -1 -1 -1 -1 -1  1  1  1  1 -1 -1 -1 -1 -1  1  1  1 -1  1 -1  1  1
   1 -1 -1  1 -1 -1  2  2  2 -1 -1 -1 -1 -1 -1 -1  2 -1 -1 -1  2 -1 -1 -1
   2 -1 -1 -1  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  2 -1  2  2  2 -1  2
   2  2 -1  2 -1 -1]]