# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from numpy import loadtxt, transpose
from numpy.linalg import norm
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Processor:
	"""
	Класс Processor выполняет основные действия по обработке данных. Он
	принимает имя файла обучающей выборки в конструкторе и помещает ее в
	ndarray.

	Промежуточные результаты обработки являются состоянием класса Processor. Во
	избежании действий над пустым состоянием, важно запускать методы в
	правильном порядке. Пример работы с классом можно найти в файле usage.py
	"""

	def __init__(self, file_name):
		"""
		Загрузка обучающей выборки и инициализация состояния

		Args:
			file_name (string): имя файла с обучающей выборкой
		"""

		self.data = transpose(loadtxt(file_name, delimiter=' '))
		self.pca = None
		self.estimator = None
		self.projector = None
		self.reduced_data = None
		self.projected_data = None
		self.labels = None


	def perform_pca(self, n_components=None):
		"""
		Выполнение PCA над данными.

		Args:
			n_components (int): количество анализируемых главных компонент.
		"""

		if n_components:
			self.pca = PCA(n_components=n_components)
		else:
			self.pca = PCA(n_components='mle')
		
		self.pca.fit(self.data)


	def plot_pca_spectrum(self):
		"""
		Построения графика PCA спектра.
		"""

		plt.figure(figsize=(6, 4))
		plt.clf()
		plt.axes([.2, .2, .8, .8])
		plt.plot(self.pca.explained_variance_ratio_, linewidth=2)
		plt.xlabel('n_components')
		plt.ylabel('explained_variance_ratio_')
		plt.grid()
		plt.show()


	def reduce_dimension(self):
		"""
		Выполнение операции понижения размерности над данными. Преобразованные
		данные заносятся в состояние класса.
		"""

		self.reduced_data = self.pca.transform(self.data)


	def fit_kmeans(self):
		"""
		Кластеризация методом k-means.
		"""

		self.estimator = KMeans(n_clusters=2)
		self.estimator.fit(self.reduced_data)
		self.labels = self.estimator.labels_


	def show_labels(self):
		"""
		Отображает принадлежность кластерам объектов обучающей выборки.
		"""

		print(self.labels)


	def plot_clusters(self):
		"""
		Изображение кластеров на плоскости. Позволяет оценить разделимость
		кластеров а также качество кластеризации. Плоскость проекции получается
		с помощью PCA при n_components = 2.
		"""

		if self.projected_data is None:
			self.projector = PCA(n_components=2)
			self.projected_data = self.projector.fit_transform(self.data)

		self.labels = self.estimator.labels_

		plt.figure(figsize=(6, 4))
		plt.clf()
		plt.axes([.2, .2, .8, .8])

		plt.scatter(
			self.projected_data[self.labels == 0, 0],
			self.projected_data[self.labels == 0, 1],
			s=40, c='yellow'
		)

		plt.scatter(
			self.projected_data[self.labels == 1, 0],
			self.projected_data[self.labels == 1, 1],
			s=100,
			c='black',
			marker='+',
			linewidth=2
		)

		plt.legend(['cluster 0', 'cluster 1'])
		plt.grid()
		plt.show()


	def predict(self, data):
		"""
		Предсказание принадлежности кластеру.

		Args:
			data (ndarray, [n_samples, n_features]): данные для предсказания.
		"""

		reduced_data = self.pca.transform(data)
		return self.estimator.predict(reduced_data)


	def intra_cluster_average(self):
		"""
		Вычисление внутрикластерного среднего. Вместе с информацией о расстоянии
		между кластерами дает представление о качестве кластеризации.
		"""

		intra_distances = [pdist(self.reduced_data[self.labels == label])
			for label in [0, 1]]

		return sum(map(sum, intra_distances)) / sum(map(len, intra_distances))


	def cluster_distance(self):
		"""
		Вычисление расстояния между кластерами.
		"""

		centers = self.estimator.cluster_centers_
		return norm(centers[0, :] - centers[1, :])
