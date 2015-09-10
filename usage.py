# -*- coding: utf-8 -*-
# Ниже приведен пример использования класса Processor

from Processor import Processor

# Загрузка обучающей выборки
proc = Processor('data/Learning_sample.txt')

# Выполнение PCA
proc.perform_pca(n_components=4)

# Анализ PCA спектра
proc.plot_pca_spectrum()

# Сокращение размерности - переход в новое пространство признаков
proc.reduce_dimension()

# Кластеризация методом k-means
proc.fit_kmeans()
proc.show_labels()

# Построение проекции кластеров на плоскости
proc.plot_clusters()

# Вычисление расстояния между кластерами и внутрикластерного среднего
print('Рассояние между кластерами: %f' % proc.cluster_distance())
print('Внутрикластерное среднее: %f' % proc.intra_cluster_average())

# Прогноз по файлу Test_sample.txt
from numpy import loadtxt, savetxt, transpose
data = transpose(loadtxt('data/Test_sample.txt'))
forecast = proc.predict(data)

# Запись результатов в файл Forecast.txt
savetxt('data/Forecast.txt', forecast, delimiter=' ', fmt='%i')
