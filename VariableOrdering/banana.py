import numpy as np
from scipy.stats import norm

class banana_posterior():

	""" Banana (x1,x2) prior density and local observation model:

			Prior:
			x1 ~ N(x1_mean, x1_std)
			x2 | x1 ~ N(x1^2 + x2_mean, x2_std)

			Observation model:
			y ~ N(x2, y_std)

	"""

	def __init__(self):
		self.x1_mean = 0
		self.x1_std  = 1
		self.x2_mean = 1
		self.x2_std  = 0.5
		self.y_std   = 0.5

	def sample_prior(self, N):
		x1 = self.x1_std * np.random.randn(N,1) + self.x1_mean
		x2 = self.x2_std * np.random.randn(N,1) + x1**2 + self.x2_mean 
		return np.hstack((x1,x2))

	def sample_data(self, x):
		N = x.shape[0]
		y = x[:,1] + self.y_std * np.random.randn(1,N)
		return y.T

	def sample_joint(self, N):
		x = self.sample_prior(N)
		y = self.sample_data(x)
		return np.hstack((x,y))

	def prior_pdf(self, x):
		pi = norm.pdf(x[:,0], loc=self.x1_mean, scale=self.x1_std)
		pi *= norm.pdf(x[:,1], loc=x[:,0]**2 + self.x2_mean, scale=self.x2_std)
		return pi

	def likelihood_function(self, x, y):
		return norm.pdf(y, loc=x[:,1], scale=self.y_std)

	def joint_pdf(self, x, y):
		prior = self.prior_pdf(x)
		lik = self.likelihood_function(x, y)
		return prior * lik
