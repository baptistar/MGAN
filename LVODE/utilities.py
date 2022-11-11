import torch
import torch.nn as nn

import numpy as np
import scipy.io
import h5py

from scipy import stats

import operator
from functools import reduce

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


#Fully Connected Feed Forward Network
class FCFFNet(nn.Module):
    def __init__(self, layers, nonlinearity, nonlinearity_params=None, 
                 out_nonlinearity=None, out_nonlinearity_params=None, normalize=False):
    
        super(FCFFNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                if nonlinearity_params is not None:
                    self.layers.append(nonlinearity(*nonlinearity_params))
                else:
                    self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            if out_nonlinearity_params is not None:
                self.layers.append(out_nonlinearity(*out_nonlinearity_params))
            else:
                self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c



class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x=None, size=1, eps=1e-8):
        super(UnitGaussianNormalizer, self).__init__()

        assert (x is not None) or (size is not None), "Input or size must be specified."

        if x is not None:
            mean = torch.mean(x, 0).view(-1)
            std = torch.std(x, 0).view(-1)
        else:
            mean =  torch.zeros(size)
            std = torch.ones(size)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)

        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x * (self.std + self.eps)) + self.mean
        x = x.view(s)

        return x

    def forward(self, x):
        return self.encode(x)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x_size.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)



def kde_2d(x, y, grid=None, s=None):

    if grid is not None:
        my_grid = grid
        shape = my_grid.shape[1]
        shape = (int(np.sqrt(shape)), int(np.sqrt(shape)))
    else:
        x_min = x.min().item()
        x_max = x.max().item()
        y_min = y.min().item()
        y_max = y.max().item()

        if s is not None:
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, s), np.linspace(y_min, y_max, s), indexing='ij')
        else:
            x_grid, y_grid = np.mgrid[x_min:x_max:512j, y_min:y_max:512j]

        my_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
        shape = x_grid.shape

    kde_kernel = stats.gaussian_kde(torch.cat((x, y), 1).t().numpy())
    kde_eval = np.reshape(kde_kernel(my_grid).T, shape)

    if grid is not None:
        return kde_eval
    else:
        return kde_eval, my_grid


def synthethic_out_pdf(x):
    y = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
    ind = torch.logical_and(torch.abs(x[:,0]) <= 3.0,  x[:,1] > torch.tanh(x[:,0]))
    y[ind,0] = (0.3/6)*torch.exp(-0.3*(x[ind,1] - torch.tanh(x[ind,0])))

    return y