# This is modified from https://github.com/Physics-aware-AI/Symplectic-ODENet/blob/master/nn_models.py

import torch
import numpy as np
from se3hamneuralode import choose_nonlinearity


class MLP(torch.nn.Module):
    '''Multilayer perceptron'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True, init_gain = 1.0):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight, gain=init_gain) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity( self.linear1(x) )
        h = self.nonlinearity( self.linear2(h) )
        return self.linear3(h)


class PSD(torch.nn.Module):
    '''A positive semi-definite matrix of the form LL^T + epsilon where L is a neural network'''
    def __init__(self, input_dim, hidden_dim, diag_dim, nonlinearity='tanh', init_gain = 1.0):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization
            
            self.nonlinearity = choose_nonlinearity(nonlinearity)
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight, gain=init_gain) # use a principled initialization
                #torch.nn.init.constant_(l.weight, 0.1)  # use a principled initialization
            
            self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            return h*h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            for i in range(self.diag_dim):
                D[:, i, i] = D[:, i, i] + 0.01
            return D


class MatrixNet(torch.nn.Module):
    ''' A neural net which outputs a matrix'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True, shape=(2,2), init_gain = 1.0):
        super(MatrixNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, nonlinearity, bias_bool, init_gain=init_gain)
        self.shape = shape

    def forward(self, x):
        flatten = self.mlp(x)
        return flatten.view(-1, *self.shape)

