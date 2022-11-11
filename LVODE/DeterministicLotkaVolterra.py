import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm
from scipy.optimize import minimize

class DeterministicLotkaVolterra:
    def __init__(self, T):
        # number of unknown (prior) parameters
        self.d = 4
        # prior parameters
        self.alpha_mu  = -0.125
        self.alpha_std = 0.5
        self.beta_mu   = -3
        self.beta_std  = 0.5
        self.gamma_mu  = -0.125
        self.gamma_std = 0.5
        self.delta_mu  = -3
        self.delta_std = 0.5
        # initial condition
        self.x0 = [30,1];
        # length of integration window
        self.T = T
        # observation parameters
        self.obs_std = np.sqrt(0.1)

    def sample_prior(self, N):
        # generate Normal samples
        alpha = lognorm.rvs(scale=np.exp(self.alpha_mu), s=self.alpha_std, size=(N,))
        beta  = lognorm.rvs(scale=np.exp(self.beta_mu),  s=self.beta_std,  size=(N,))
        gamma = lognorm.rvs(scale=np.exp(self.gamma_mu), s=self.gamma_std, size=(N,))
        delta = lognorm.rvs(scale=np.exp(self.delta_mu), s=self.delta_std, size=(N,))
        # join samples
        return np.vstack((alpha, beta, gamma, delta)).T

    def ode_rhs(self, z, t, theta):
        # extract parameters
        alpha, beta, gamma, delta = theta
        # compute RHS of 
        fz1 = alpha * z[0] - beta * z[0]*z[1]
        fz2 = -gamma * z[1] + delta * z[0]*z[1]
        return np.array([fz1, fz2])

    def simulate_ode(self, theta, tt):
        # check dimension of theta
        assert(theta.size == self.d)
        # numerically intergate ODE
        return odeint(self.ode_rhs, self.x0, tt, args=(theta,))

    def sample_data(self, theta):
        # check inputs
        if len(theta.shape) == 1:
            theta = theta[np.newaxis,:]
        assert(theta.shape[1] == self.d)
        # define observation locations
        tt = np.arange(0, self.T, step=2)
        nt = 2*(len(tt)-1)
        # define arrays to store results
        xt = np.zeros((theta.shape[0], nt))
        # run ODE for each parameter value
        for j in range(theta.shape[0]):
            yobs = self.simulate_ode(theta[j,:], tt);
            # extract observations, flatten, and add noise
            yobs = np.abs(yobs[1:,:]).ravel()
            #xt[j,:] = lognorm.rvs(scale=np.exp(np.log(yobs)), s=self.obs_std, size=(1,))
            xt[j,:] = np.array([lognorm.rvs(scale=x, s=self.obs_std) for x in yobs])
        return (xt, tt)

    def log_prior_pdf(self, theta):
        # check dimensions of inputs
        assert(theta.shape[1] == self.d)
        # compute mean and variance
        prior_mean = [self.alpha_mu, self.beta_mu, self.gamma_mu, self.delta_mu]
        prior_std = [self.alpha_std, self.beta_std, self.gamma_std, self.delta_std]
        # evaluate product of PDFs for independent variables
        return np.sum(lognorm.logpdf(theta, scale=np.exp(prior_mean), s=prior_std), axis=1)

    def prior_pdf(self, theta):
        return np.exp( self.log_prior_pdf(theta) )
    
    def log_likelihood(self, theta, yobs):
        # check dimension of inputs
        assert(theta.shape[1] == self.d)
        assert(yobs.size == (self.T-2))
        # define observation locations
        tt = np.arange(0, self.T, step=2)
        # define array to store log-likelihood
        loglik = np.zeros(theta.shape[0],)
        # simulate dynamics for each theta
        for j in range(theta.shape[0]):
            xt = self.simulate_ode(theta[j,:], tt)
            xt = np.abs(xt[1:,:]).ravel()
            # compare observations under LogNormal(G(theta),obs_var)
            loglik[j] = np.sum([lognorm.logpdf(yobs, scale=xt, s=self.obs_std)])
        return loglik

    def likelihood(self, theta, yobs):
        return np.exp( self.log_likelihood(theta, yobs) )

if __name__ == '__main__':

    # define model
    T = 20;
    LV = DeterministicLotkaVolterra(T)

    # define true parameters and observation
    xtrue = np.array([0.6859157, 0.10761319, 0.88789904, 0.116794825])
    tt = np.linspace(0,LV.T,1000)
    ytrue = LV.simulate_ode(xtrue, tt)
    yobs,tobs = LV.sample_data(xtrue)
    nobs = int(yobs.size/2.)
    yobs_plot = yobs.reshape((nobs,2))

    # plot single simulation
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(tt,ytrue)
    plt.plot(tobs[1:],yobs_plot,'o','MarkerSize',8)
    plt.xlabel('$t$')
    plt.ylabel('Observations')
    plt.show()

    # generate many training data
    Ntrain = 1000
    X = LV.sample_prior(Ntrain)
    Y,_ = LV.sample_data(X)
    X = np.real(X); Y = np.real(Y)

    # evaluate densities
    pi_prior = LV.prior_pdf(X)
    print(pi_prior)
    pi_lik = LV.likelihood(X, yobs)
    print(pi_lik)

    # Run Inference
    from MCMCSamplers import AdaptiveMetropolisSampler, GaussianProposal

    # define target density and bounds
    class Posterior():
        def __init__(self, yobs):
            self.yobs = yobs
        def logpdf(self, x):
            if len(x.shape) == 1:
                x = x.reshape((1,LV.d))
            return LV.log_likelihood(x, self.yobs) + LV.log_prior_pdf(x)
    pi = Posterior(yobs[0,:])
    bounds = np.array([[0.]*LV.d,[np.inf]*LV.d])

    # find MAP point
    x0 = np.random.rand(LV.d,)
    neg_post = lambda x : -1*pi.logpdf(x)
    xmap = minimize(neg_post, x0)

    # define Gaussian proposal
    prop_std = 0.1;
    prop = GaussianProposal(cov=prop_std**2*np.eye(LV.d));

    # run MCMC
    n_steps = int(1e5)
    mcmc = AdaptiveMetropolisSampler(pi, prop)
    x_samps, logpdf_samps = mcmc.sample(xmap.x, n_steps, bounds)

    # save results
    data_file = 'DeterministicLV_mcmc'
    import scipy.io
    scipy.io.savemat(name + '.mat', mdict={'samples':x_samps, 'yobs':yobs, 'xtrue':xtrue})

    # plot results
    import pandas as pd
    df = pd.DataFrame(x_samps, columns=['x1','x2','x3','x4'])
    plt.figure()
    pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.show()
