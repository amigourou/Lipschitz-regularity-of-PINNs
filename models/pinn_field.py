import sys
sys.path.append(r"D:\3ACS\TDL\lipEstimation")

import torch
import torch.nn as nn
from pinn_utils.spring.physics_utils import *

from utils import load_model, save_model

def pinn_field(loc=None, size = "tiny"):
    """
    So far, LB for Lipschitz constant on this model is 25.5323 obtained by
    annealing
    """

    pinn = NewPlanetsModel(2, size)
    # if loc is None:
        # loc = 'models_ckpts/pinn_spring.pth'
    load_model(pinn, loc)
    return pinn

class NewPlanetsModel(nn.Module):
    """
    This class defines the neural network used to encode the location of the planets. \\
    It also includes the method to load the weights.

    Attributes:
      hidden_size: Size of the intermediate layers.

    Methods:
      __init__:
        Args:
          nb_dimensions: Number of dimensions in the hyperspace.
      forward:
        Args:
          inputs: Inputs to the neural network.
        Returns:
          potential: Gravitational potential output by the model.
      load_weights:
        Args:
          weight_file: Path to the weight file.
          device: Device on which we load the neural network (GPU, or CPU it not available).

    """

    

    def __init__(self, nb_dimensions, size = "tiny"):
        super().__init__()
        if size == "tiny" :
          self.hidden_size = 64
          self.fc = nn.Sequential(
              nn.Linear(nb_dimensions, self.hidden_size),
              nn.Tanh(),
              nn.Linear(self.hidden_size, self.hidden_size),
              nn.Tanh(),
              nn.Linear(self.hidden_size, self.hidden_size),
              nn.Tanh(),
              nn.Linear(self.hidden_size, self.hidden_size),
              nn.Tanh(),
              nn.Linear(self.hidden_size, 1))
        
        else:
           self.hidden_size = 128
           self.fc = nn.Sequential(
            nn.Linear(nb_dimensions, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1))

    def forward(self, x):
        return self.fc(x)

    def load_weights(self, weight_file, device):
        self.load_state_dict(torch.load(weight_file, map_location=device))
        
    def compute_residual(self, x, R, true_lapl, planets):
      # Ensure input is on the correct device
      x = x.to(device)

      # Forward pass to compute potential
      phi = self.forward(x).reshape(30, 30)  # Assuming a 30x30 grid for simplicity

      # Compute gradients using torch.gradient
      # phi.shape = (30, 30), so we calculate gradients along both dimensions
      dphi_dx, dphi_dy = torch.gradient(phi, spacing=(1/29, 1/29))  # Adjust spacing based on grid resolution

      # Compute second derivatives (Laplacian terms)
      d2phi_dx2, _ = torch.gradient(dphi_dx, spacing=(1/29, 1/29))
      _, d2phi_dy2 = torch.gradient(dphi_dy, spacing=(1/29, 1/29))

      # Laplacian as the sum of second derivatives
      laplacian = d2phi_dx2 + d2phi_dy2

      # Flatten grid for distance computations
      flat_phi = phi.flatten()
      laplacian_flat = laplacian.flatten()
      x_flat = x

      # Compute distances from planets
      distances = torch.norm(planets.unsqueeze(1) - x_flat.unsqueeze(0), dim=-1)

      # Identify points inside any planet
      inside_planet = distances < 0.1
      inside_planet = inside_planet.any(dim=0)

      # Compute the residual
      residual = laplacian_flat.unsqueeze(0)
      residual = torch.where(inside_planet, residual - 4 / (3 * R**2), residual)

      # Return residual, Laplacian, and gradient magnitude
      gradient_magnitude = torch.sqrt(dphi_dx**2 + dphi_dy**2).flatten()
      return residual, laplacian_flat, gradient_magnitude.to(device)
