import numpy as np
from scipy.stats import gamma, norm

class tanh_example():

    """ Tanh model:
    Prior: x ~ U[-3,3]
    Observation model: specified in sub-class
    """

    def __init__(self):
        self.x_a     = -3.
        self.x_b     = 3.

    def sample_prior(self, N):
        return (self.x_b - self.x_a) * np.random.rand(N,1) + self.x_a

    def sample_joint(self, N):
        x = self.sample_prior(N)
        y = self.sample_data(x)
        return np.hstack((x,y))

    def sample_data(self, x):
        raise ValueError('Not implemented here')

    def likelihood_function(self, x, y):
        raise ValueError('Not Implemented here')

    def prior_pdf(self, x):
        supp = np.ones((x.shape[0],1))
        supp[x[:,0] < self.x_a] = 0
        supp[x[:,0] > self.x_b] = 0
        pi = 1./(self.x_b - self.x_a) * supp
        return pi

    def joint_pdf(self, x, y):
        prior = self.prior_pdf(x)
        lik = self.likelihood_function(x, y)
        return prior * lik

class tanh_v1(tanh_example):

    """ Observation model:
    y = tanh(x) + Gamma[1,0.3]
    """

    def __init__(self):
        super(tanh_v1, self).__init__()
        self.y_alpha = 1.
        self.y_beta  = 1./0.3

    def sample_data(self, x):
        N = x.shape[0]
        g = gamma.rvs(self.y_alpha, loc=0, scale=1./self.y_beta, size=(N,1))
        return np.tanh(x) + g

    def likelihood_function(self, x, y):
        g = y - np.tanh(x)
        return gamma.pdf(g, self.y_alpha, loc=0, scale=1./self.y_beta)

class tanh_v2(tanh_example):

    """ Observation model:
    y = tanh(x) + Normal(0,0.05)
    """

    def __init__(self):
        super(tanh_v2, self).__init__()
        self.n_mean  = 0.
        self.n_std   = np.sqrt(0.05)

    def sample_data(self, x):
        N = x.shape[0]
        n = norm.rvs(loc=self.n_mean, scale=self.n_std, size=(N,1))
        return np.tanh(x + n)

    def likelihood_function(self, x, y):
        n = np.arctanh(y) - x
        lik = norm.pdf(n, loc=self.n_mean, scale=self.n_std)
        lik[np.isnan(lik)] = 0.
        return lik

class tanh_v3(tanh_example):

    """ Observation model:
    y = gamma*tanh(x), gamma ~ Gamma(1,0.3)
    """

    def __init__(self):
        super(tanh_v3, self).__init__()
        self.y_alpha = 1.
        self.y_beta  = 1./0.3

    def sample_data(self, x):
        N = x.shape[0]
        g = gamma.rvs(self.y_alpha, loc=0, scale=1./self.y_beta, size=(N,1))
        return np.tanh(x) * g

    def likelihood_function(self, x, y):
        g = y / np.tanh(x)
        return gamma.pdf(g, self.y_alpha, loc=0, scale=1./self.y_beta)
