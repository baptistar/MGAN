import numpy as np
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from scipy.stats import multivariate_normal

# define class for element in chain
class MCSample():
    def __init__(self):
        self.x
        self.logpdf
        self.logprop
        self.grad_x_logpdf

# define Gaussian rv proposal
class GaussianProposal():
    def __init__(self, cov):
        self.cov = cov
    def sample(self, x):
        return multivariate_normal.rvs(mean=x, cov=self.cov)
    def logpdf(self, xp, x):
        return multivariate_normal.logpdf(xp, mean=x, cov=self.cov)

class AdaptiveMetropolisSampler():
    
    def __init__(self, pi, prop, min_iter_adapt=100, iter_step_adapt=5):
        # check definitions of pi and prop
        assert hasattr(pi, 'logpdf') and hasattr(prop, 'logpdf'), 'define classes with logpdf function'
        #assert prop == GaussianProposal()
        # assign inputs
        self.pi = pi
        self.prop = prop
        self.min_iter_adapt = min_iter_adapt
        self.iter_step_adapt = iter_step_adapt
        
    def sample(self, x0, n_steps: int, bounds=None):
        """ Sample from target density self.pi using a MCMC chain of 
            length n_samps starting from x0 """ 
        
        # define array to store samples
        dim = x0.size
        samps = np.zeros((n_steps+1, dim))
        logpdfs = np.zeros((n_steps+1,))

        # define bounds and check dimensions
        #if any(bounds==None):
        #    bounds = np.array([[-np.inf]*dim, [np.inf]*dim])
        #assert(bounds.shape == (2,dim))

        # define counter of accepted samples
        n_accept = 0
        
        # define xold at x0
        xold = x0
        samps[0,:] = xold

        # evaluate target at xold
        logpdf_old = self.pi.logpdf(xold)
        logpdfs[0] = logpdf_old

        for i in tqdm(range(1,n_steps+1), ascii=True, ncols=100):

            # sample from proposal and evaluate under pi
            xnew = self.prop.sample(xold)

            # check if sample is inside bounds
            if any(xnew < bounds[0,:]) or any(xnew > bounds[1,:]):
                logpdf_new = -np.inf 
            else:
                logpdf_new = self.pi.logpdf(xnew)
            
            # adapt proposal starting at iteration self.n_adapt
            if (i > self.min_iter_adapt) and np.mod(i, self.iter_step_adapt):
                emp_pert = samps[:i,:] - np.mean(samps[:i,:],axis=0)
                emp_cov = np.dot(emp_pert.T, emp_pert)/(i-1)
                #cov = EmpiricalCovariance().fit(samps[:i,:])
                #emp_cov = cov.covariance_
                self.prop.cov = emp_cov + 1e-6*np.eye(dim)

            # evaluate density under proposal
            logprop_old = self.prop.logpdf(xold, xnew)
            logprop_new = self.prop.logpdf(xnew, xold)
            
            # compute acceptance probability
            detBalance = logpdf_new - logpdf_old + logprop_old - logprop_new
            logAcceptProb = np.min((0., detBalance))
            
            # accept or reject samle
            if np.log(np.random.random()) < logAcceptProb:
                xold = xnew
                logpdf_old = logpdf_new
                n_accept += 1
            # save sample
            samps[i,:] = xold
            logpdfs[i] = logpdf_old

        # print acceptance probability
        accept_rate = float(n_accept)/float(n_steps) * 100
        print("Overall acceptance rate: %3.1f\n" % accept_rate)
        
        # return samples 
        return (samps, logpdfs)
