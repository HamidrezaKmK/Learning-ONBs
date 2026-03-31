from torch import nn

from .base import NeuralField, FourierFeatures


class FFNeuralField(NeuralField):
    def __init__(self, input_dim, output_dim, n_features=64, sigma=10.0,
                 hidden_dims=(256, 256), activation=nn.ReLU):
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
        return self.net(z)
