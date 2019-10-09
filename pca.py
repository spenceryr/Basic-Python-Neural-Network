################################################################################
# CSE 154: Programming Assignment 1
# Code snippet by Michael Liu
# Fall 2019
################################################################################

import numpy as np
from dataloader import *
from matplotlib import pyplot as plt

'''
Principal Components Analysis using Turk's Trick

Ask a TA or go for discussion session if you still have question regarding PCA
'''
class PCA:

	def __init__(self, k):
		"""
		k: number of principal components we want to take
		mean: variable save the average value of train images
		std: variable save the standard deviation of the train images
		img_dim: variable save the dimension of a single train image (height, width)
		p_components: variable save the computed principal components on train images
		s_vals: the corresponding singular values
		"""
		self.k = k
		self.img_dim = None
		self.mean = None
		self.std = None
		self.s_vals = None
		self.p_components = None

	def fit(self, data):
		"""
		Because the image dimension (height * width) is so huge, we cannot compute SVD on matrix A (data) directly.

		A work around is to:
		1. compute the covariance matrix of the data
		2. compute the eigenvectors and eigenvalues of (covariance matrix) AAt = data @ data^T 
		3. find its (AAt) eigenvectors (evecs) corresponding to the top k eigenvalues
		4. compute the principal components by left matrix multiply (evecs) by (data.T)
		5. normalize to make sure each of the components is a unit vector

		Args:
			data: numpy image array with dimension of (number of image, height, width)
		"""
		n, h, w = data.shape

		# quick check for the right dimension
		assert n < 100
		self.img_dim = (h, w)
		data = data.reshape(n, -1)

		# generate a canonical face with the average image data and use it to center all image information
		# you can refer this process similar to z-score
		self.mean = np.mean(data, axis=0)
		self.std = np.std(data, axis=0)
		# center the image data
		data = (data - self.mean) / self.std

		# AA^T shape: m x m (covariance matrix)
		AAt = (data @ data.T) / (n - 1)
		evals, evecs = np.linalg.eigh(AAt)
		# sorting eigenvectors in accordance with eigenvalues, from large to small
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:,idx]
		evals = evals[idx]
		# eigen vectors shape: n x k
		evecs = evecs[:, :self.k]
		evals = evals[:self.k]
		v_evecs = data.T @ evecs
		# normalize principal components with singular values to make unit vectors
		self.s_vals = np.sqrt(evals.reshape(1, -1))
		v_evecs = v_evecs / np.linalg.norm(v_evecs, axis=0)
		# quick check if all principal components are unit vectors, should be all 1
		assert np.allclose(np.linalg.norm(v_evecs, axis=0), 1)
		self.p_components = v_evecs

	def transform(self, data):
		"""
		Transform a single image into k dimension 
		By first subtract the train mean and then project onto the principal components

		Args:
			data: numpy representation of a single image with the shape of (x, y) or (1, x, y)
		Ret:
			transformed data: data after projecting images onto the k principal components with the shape of (1, k)
		"""
		if self.p_components is None:
			print("[WARNING] PCA has not been fitted with train images")
			return data
		
		# make sure it is a single image
		assert len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[0] == 1)

		# center the data with trained center
		data = (data.reshape(1, -1) - self.mean) / self.std
		
		# project the data onto principal components and normalize it with singular values
		data = data @ self.p_components / self.s_vals

		return data

	def inverse_transform(self, data):
		"""
		Inverse transform a dimension reduced vector representation back to the dimension reduced image representation

		Args:
			data: pca transformed image representation
		Ret:
			img: dimension reduced image restoration
		"""
		img = (data * self.s_vals) @ self.p_components.T
		# add back the canonical face
		img *= self.std
		img += self.mean
		return img.reshape(self.img_dim)

	def display(self, save_path = './pca_display.png', only_show=6):
		"""
		Display top k principal components, the image should resemble ghostly looking faces
		"""
		x, y = self.img_dim
		assert only_show <= self.k
		pca_imgs = self.p_components.reshape(x, y, self.k)
		pca_imgs = pca_imgs[:, :, :only_show]
		pca_imgs = np.transpose(pca_imgs, (2, 1, 0))
		pca_imgs = np.concatenate(pca_imgs, axis=0)

		plt.tight_layout()
		plt.imshow(pca_imgs.T, cmap='gray')
		plt.title('Visualization of top {} principal components'.format(only_show))
		print('Save PCA image to {}'.format(save_path))
		plt.savefig(save_path)

"""
A demo script on how to use PCA class
"""
def main():
	images, labels = load_data(data_dir="./CAFE/")
	# k is the number of principal components you want 
	pca = PCA(k=50)
	# choose your training images here
	pca.fit(np.array(images[:]))

	# transform and inverse transform a single image
	index = 10 # change this to try on more images
	projected_image = pca.transform(np.array(images[index]))
	inverse = pca.inverse_transform(projected_image)
	plt.imshow(np.array(images[index]), cmap='gray')
	plt.show()
	plt.imshow(inverse.reshape(np.array(images[index]).shape), cmap='gray')
	plt.show()
	print("Projected image with a shape of {}".format(projected_image.shape))

	pca.display()

if __name__ == '__main__':
	main()