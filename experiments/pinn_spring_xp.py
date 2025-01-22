""" Starting point of the script is a saving of all singular values and vectors
in mnist_save/

We perform the 100-optimization implemented in optim_nn_pca_greedy
"""
import sys
sys.path.append(r"D:\3ACS\TDL\lipEstimation")
import math
import torch
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

from models.pinn_spring import pinn_spring
from seqlip import optim_nn_pca_greedy

n_sv = 200

clf = pinn_spring()

for p in clf.parameters():
    p.requires_grad = False

compute_module_input_sizes(clf, [1, 1])

lip = 1
lip_spectral = 1


# Indices of convolutions and linear layers
layers = [2, 3]

for i in range(len(layers) - 1):
    print('Dealing with linear layer {}'.format(i))
    U = torch.load('pinn_spring_save/layer{}-left-singular'.format(layers[i]))
    
    U = torch.cat(U[:n_sv], dim=0)
    print(U.shape)
    # U = U.view(min(n_sv, U.shape[0] * U.shape[1]), -1)
    
    su = torch.load('pinn_spring_save/layer{}-singular'.format(layers[i]))
    su = su[:n_sv]
    print(su)

    V = torch.load('pinn_spring_save/layer{}-right-singular'.format(layers[i+1]))
    V = torch.cat(V[:n_sv], dim=0)
    # V = V.view(min(n_sv, V.shape[0] * V.shape[1]), -1)
    sv = torch.load('pinn_spring_save/layer{}-singular'.format(layers[i+1]))
    sv = sv[:n_sv]
    print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
    print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

    U, V = U.cpu(), V.cpu()


    if i == 0:
        sigmau = torch.diag(torch.Tensor(su)) #+ torch.diag(torch.ones(torch.Tensor(su).shape))
    else:
        sigmau = torch.diag(torch.sqrt(torch.Tensor(su))) #+ torch.diag(torch.ones(torch.Tensor(su).shape))

    if i == len(layers) - 2:
        sigmav = torch.diag(torch.Tensor(sv))
    else:
        sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))

    expected = sigmau[0,0] * sigmav[0,0]
    print('Expected: {}'.format(expected))

    lip_spectral *= expected
    print(sigmau)
    curr, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau)
    print('Approximation: {}'.format(curr))
    lip *= float(curr)
    print("Total Lip approx: {}".format(lip))


print('Lipschitz spectral: {}'.format(lip_spectral))
print('Lipschitz approximation: {}'.format(lip))
