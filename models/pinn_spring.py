import sys
sys.path.append(r"D:\3ACS\TDL\lipEstimation")

import torch
import torch.nn as nn
from pinn_utils.spring.physics_utils import *

from utils import load_model, save_model

def pinn_spring(loc=None):
    """
    So far, LB for Lipschitz constant on this model is 25.5323 obtained by
    annealing
    """
    mass = 1
    mu = 1
    k = 10
    system = DampenSpringSystem(1,0,mass, k, mu)

    pinn = MLP(system,1,1,[32,32,32,32,32])
    # if loc is None:
        # loc = 'models_ckpts/pinn_spring.pth'
    load_model(pinn, loc)
    return pinn

class MLP(nn.Module):
    def __init__(self, system, input_size = 1, output_size = 1,  sizes = [32,32]):
        super(MLP,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        list_layers = [nn.Linear(input_size, sizes[0]),
                       nn.ReLU()]

        self.system = system
        self.criterion = nn.MSELoss()

        for i in range(1,len(sizes)-1):

            list_layers.append(nn.Linear(sizes[i], sizes[i+1]))
            list_layers.append(nn.ReLU())

        list_layers.append(nn.Linear(sizes[-1],output_size))
        self.layers = nn.Sequential(*list_layers)

    def forward(self,t):
        x = self.layers(t)
        return x
    
    def physics_loss(self,pred, truth, t_max, M, alpha = 1, device = "cuda"):
        loss = self.criterion(pred, truth.unsqueeze(1))

        times = torch.linspace(0,t_max,M).unsqueeze(1).to(device)
        # print(pred, truth.unsqueeze(1))
        p_preds = self.forward(times).squeeze(1)
        p_preds_dot = torch.gradient(p_preds,spacing = (times.squeeze(1),))[0]
        p_preds_dotdot = torch.gradient(p_preds_dot,spacing = (times.squeeze(1),))[0]
        p_loss = (self.system.equation(p_preds,p_preds_dot,p_preds_dotdot)**2).mean() * alpha
        # print(loss,p_loss)
        return loss.float() + p_loss.float()

    

