from torch import nn
import torch

from .base import NeuralField, FourierFeatures


class FFNeuralField(NeuralField):
    def __init__(self, input_dim, output_dim, n_features=64, sigma=10.0,
                 hidden_dims=(256, 256), activation=nn.SiLU):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.ff = FourierFeatures(input_dim, n_features, sigma=sigma)
        layers = []
        prev = 2 * n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        z = self.ff(coords)
        return self.net(z) # + 1.0 / (torch.norm(coords - 0.5, dim=-1, keepdim=True) + 0.5)
