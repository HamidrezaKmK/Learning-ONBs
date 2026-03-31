import math

from torch import nn

from .base import NeuralField, FourierFeatures


class NTKMLPNeuralField(NeuralField):
    """MLP with NTK parameterization for lazy-training / kernel stability.

    Weights W_l ~ N(0, 1); each pre-activation is divided by sqrt(fan_in).
    This keeps every hidden layer output O(1), making each gradient step change
    f by O(1/sqrt(m)), so the NTK remains approximately constant during training.

    Use smooth activations (default: Tanh) — ReLU causes discrete Jacobian jumps
    when activation patterns flip, which destabilises the kernel.

    Plain MLPs on raw coordinates suffer from spectral bias: the NTK has
    exponentially small eigenvalues for high frequencies, so the network can only
    learn smooth low-frequency functions.  Setting ``n_fourier_features > 0``
    prepends a frozen random Fourier embedding γ(x) = [sin(Bx), cos(Bx)] before
    the MLP.  The NTK of the combined network K(x,x') ≈ K_MLP(γ(x), γ(x')) has
    large eigenvalues across all frequencies covered by B, enabling high-frequency
    reconstruction while the frozen features keep the MLP weights in the NTK regime.

    Args:
        input_dim:         coordinate dimension.
        output_dim:        output channels.
        hidden_dims:       dict or list of hidden widths.
        activation:        activation constructor (default nn.Tanh).
        n_fourier_features: number of random Fourier feature pairs (0 = disabled).
        fourier_sigma:     bandwidth of the random Fourier features.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims,
        activation=nn.Tanh,
        n_fourier_features: int = 128,
        fourier_sigma: float = 10.0,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        # Explicitly cast to Python int: OmegaConf IntegerNode values are not
        # accepted by PyTorch's nn.Linear when Hydra instantiates the class.
        # Use hasattr(.values) to handle both plain dict and OmegaConf DictConfig
        # (isinstance(x, dict) returns False for DictConfig).
        if hasattr(hidden_dims, 'values'):
            widths = [int(v) for v in hidden_dims.values()]
        else:
            widths = [int(v) for v in hidden_dims]

        if n_fourier_features > 0:
            self.ff = FourierFeatures(input_dim, n_fourier_features, sigma=fourier_sigma)
            mlp_input_dim = 2 * n_fourier_features
        else:
            self.ff = None
            mlp_input_dim = input_dim

        dims = [int(mlp_input_dim)] + widths + [int(output_dim)]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = activation()
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, coords):
        x = self.ff(coords) if self.ff is not None else coords
        for layer in self.layers[:-1]:
            # Divide the pre-activation by sqrt(fan_in), not sqrt(current width).
            # This keeps every hidden layer output O(1) regardless of the previous
            # layer's width — critical for the first layer where fan_in = input_dim
            # (e.g. 2 or 2*n_fourier_features), not the hidden width (e.g. 1024).
            x = self.activation(layer(x) / math.sqrt(layer.in_features))
        return self.layers[-1](x) / math.sqrt(self.layers[-1].in_features)
