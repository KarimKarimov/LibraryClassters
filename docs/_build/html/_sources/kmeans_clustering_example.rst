=======================================================
KMeans Clustering Example
=======================================================


**Код**

.. literalinclude:: ..\example\kmeans.py
   :language: python

.. admonition:: Описание шагов

   Шаг 1:
   Создается набор данных `X`, содержащий координаты точек, которые будут кластеризованы.

   Шаг 2:
   Создается объект `kmeans` класса KMeansClustering с указанием количества кластеров.
   Метод `fit` вызывается для обучения модели на данных `X`. Затем вызывается метод `predict`,
   чтобы получить метки кластеров для данных `X`.

   Шаг 3:
   Выводятся метки кластеров для данных `X`.
   
   *Пожалуйста, убедитесь, что у вас установлена библиотека NumPy, и что класс KMeansClustering находится в модуле "clustering".*

**Результат**

.. code-block:: python

   Silhouette score: 0.4958856269356724

.. image:: _static//Figure_1.png
   :width: 500
   :height: 400
   :alt: Alternative text
   :align: center

.. image:: _static//Figure_2.png
   :width: 500
   :height: 400
   :alt: Alternative text
   :align: center
