# Compute AlexNet 50 highest singular vectors for every convolutions
import torch
import os
import numpy as np
import sys
sys.path.append(r"D:\3ACS\TDL\lipEstimation")
from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

from models.pinn_spring import pinn_spring

n_sv = 100

def spec_pinn(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=500,
                use_cuda=True)
        self.spectral_norm = s
        print(s)
        self.u = u
        self.v = v
    else:
        print("Doing nothing, not conv or linear")

def save_singular(pinn):
    # Save for convolutions
    os.makedirs('pinn_spring_save', exist_ok=True)
    i=1
    for layer in pinn.layers :
        if "relu" not in layer.__class__.__name__.lower():
            torch.save(layer.spectral_norm, open(f'pinn_spring_save/layer{i}-singular', 'wb'))
            torch.save(layer.u, open(f'pinn_spring_save/layer{i}-left-singular', 'wb'))
            torch.save(layer.v, open(f'pinn_spring_save/layer{i}-right-singular', 'wb'))
            i+=1


if __name__ == '__main__':
    clf = pinn_spring()
    clf = clf.cuda()
    clf = clf.eval()

    for p in clf.parameters():
        p.requires_grad = False

    compute_module_input_sizes(clf, [1, 1])
    execute_through_model(spec_pinn, clf)

    save_singular(clf)
