import torch
import torch.nn as nn

import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

from scipy import stats

import sys
sys.path.append('../')
from utilities import FCFFNet, UnitGaussianNormalizer, MatReader, LpLoss, kde_2d
from torch_two_sample.statistics_diff import MMDStatistic
from banana import banana_posterior

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--monotone_param", type=float, default=0.01, help="monote penalty constant")
parser.add_argument("--normalize", type=float, default=1.0, help="wether to normalize the data")
parser.add_argument("--n_train", type=int, default=10000, help="number of training samples")
parser.add_argument("--learning_rate", type=float, default=0.00005, help="learning rate")
opt = parser.parse_args()

#Pick device: cuda, cpu
device = torch.device('cpu')

Ntrain = opt.n_train

#Load the data
loader = MatReader('banana.mat')
y_train = loader.read_field('y_prime')[0:Ntrain,:].contiguous()
y_train = torch.fliplr(y_train)
y_train_test = loader.read_field('y_primetest')[0:Ntrain,:].contiguous()
y_train_test = torch.fliplr(y_train_test)
ban = banana_posterior()

#Dimensions of y
dy = 2

#Normalize the marginals to have zero mean and unit std
#Fixed affine transformation is fully invertible
if opt.normalize > 0.0:
    y_norm = UnitGaussianNormalizer(y_train)
    y_train = y_norm.encode(y_train)
    y_train_test = y_norm.encode(y_train_test)

#Data loaders for training
bsize = 100 #Should divide Ntest
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train), batch_size=bsize, shuffle=True)

#Transport map and discriminator
F = FCFFNet([dy, 32, 64, 32, 2], nn.LeakyReLU, nonlinearity_params=[0.2, True]).to(device)
D = FCFFNet([dy, 32, 64, 32, 1], nn.LeakyReLU, nonlinearity_params=[0.2, True], out_nonlinearity=nn.Sigmoid).to(device)
#F = FCFFNet([dy, 128, 256, 128, 2], nn.LeakyReLU, nonlinearity_params=[0.2, True]).to(device)
#D = FCFFNet([dy, 128, 256, 128, 1], nn.LeakyReLU, nonlinearity_params=[0.2, True], out_nonlinearity=nn.Sigmoid).to(device)

#Optimizers
optimizer_F = torch.optim.Adam(F.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

#Monotonicity penalty
mon_lambda = opt.monotone_param

# define name
name = 'mgan_rev_banana_l' + str(mon_lambda)[2:]

epochs = 300

#Loss
mse_loss = torch.nn.MSELoss()

monotonicity = np.zeros((epochs, ))
kl_error     = np.zeros((epochs, ))
train_loss   = np.zeros((epochs, ))
test_loss    = np.zeros((epochs, ))

for ep in range(epochs):

    F.train()
    D.train()

    mon_percent = 0.0
    for y in train_loader:
        
        #Data batch
        y = y[0].to(device)

        ones = torch.ones(bsize, 1, device=device)
        zeros = torch.zeros(bsize, 1, device=device)

        ###Loss for transport map###

        optimizer_F.zero_grad()

        #Draw from reference
        z2 = torch.randn(bsize, dy, device=device)

        #Transport reference to conditional y|x
        Fz = F(z2)

        #Transport of reference z1 to x marginal is by identity map
        #Discriminator error of generated joint 
        F_loss = mse_loss(D(Fz), ones)

        #Draw new reference sample
        z2_prime = torch.randn(bsize, dy, device=device)
        Fz_prime = F(z2_prime)

        #Monotonicity constraint
        mon_penalty = torch.sum(((Fz - Fz_prime).view(bsize,-1))*((z2 - z2_prime).view(bsize,-1)), 1)
        if mon_lambda > 0.0:
            F_loss = F_loss - mon_lambda*torch.mean(mon_penalty)

        F_loss.backward()
        optimizer_F.step()

        #Percent of examples in batch with monotonicity satisfied
        mon_penalty = mon_penalty.detach()
        mon_percent += float((mon_penalty>=0).sum().item())/bsize

        ###Loss for discriminator###

        optimizer_D.zero_grad()

        D_loss = 0.5*(mse_loss(D(y), ones) + mse_loss(D(Fz.detach()), zeros))
        D_loss.backward()

        optimizer_D.step()

    F.eval()
    D.eval()

    #Average monotonicity percent over batches
    mon_percent = mon_percent/math.ceil(float(Ntrain)/bsize)
    monotonicity[ep] = mon_percent

    ##Training loss
    zt = torch.randn(y_train.shape[0], dy, device=device)
    yt = y_train.to(device)
    #Transport reference to conditionals
    with torch.no_grad():
        Fz = F(zt)
    ones = torch.ones(y_train.shape[0], 1, device=device)
    zeros = torch.zeros(y_train.shape[0], 1, device=device)
    D_loss_train = 0.5*(mse_loss(D(yt), ones) + mse_loss(D(Fz.detach()), zeros))
    train_loss[ep] = D_loss_train.detach()

    ##Test loss
    zt = torch.randn(y_train_test.shape[0], dy, device=device)
    yt = y_train_test.to(device)
    #Transport reference to conditionals
    with torch.no_grad():
        Fz = F(zt)
    ones = torch.ones(y_train_test.shape[0], 1, device=device)
    zeros = torch.zeros(y_train_test.shape[0], 1, device=device)
    D_loss_test = 0.5*(mse_loss(D(yt), ones) + mse_loss(D(Fz.detach()), zeros))
    test_loss[ep] = D_loss_test.detach()

    #Sample conditional
    Ntest = int(5000)
    z_t = torch.randn(Ntest, dy, device=device)
    with torch.no_grad():
        Fz = F(z_t)

    #Undo normalizations
    if opt.normalize > 0.0:
        Fz = y_norm.decode(Fz.cpu())
    else:
        Fz = Fz.cpu()
    
    Fz = torch.fliplr(Fz).numpy()

    #Relative KL error between pdfs
    x1_dom = [-2,2]
    x2_dom = [0,5]
    x_grid, y_grid = np.mgrid[x1_dom[0]:x1_dom[1]:128j, x2_dom[0]:x2_dom[1]:128j]
    my_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_kernel = stats.gaussian_kde(Fz.T)
    
    if opt.normalize > 0.0:
        y_train_temp = y_norm.decode(y_train_test).numpy()
    else:
        y_train_temp = np.copy(y_train_test)
    y_train_temp = np.fliplr(y_train_temp)
    #true_pdf   = ban.prior_pdf(y_train_test)
    true_pdf   = ban.prior_pdf(y_train_temp)
    approx_pdf = kde_kernel(y_train_temp.T)
    kl_error[ep] = np.mean(np.log(true_pdf) - np.log(approx_pdf))
    
    print(ep, 'Monotonicity: %f, KL: %f, D test loss: %f' % (monotonicity[ep], kl_error[ep], test_loss[ep]))

    
if opt.normalize > 0.0:
    name += '_norm'

F.eval()
F.cpu()

#Save model
torch.save(F.state_dict(), name + '_F.pt')

#Save stats
scipy.io.savemat(name + '.mat', mdict={'kl_error': kl_error,'monotonicity': monotonicity,'train_loss':train_loss, 'test_loss':test_loss})
