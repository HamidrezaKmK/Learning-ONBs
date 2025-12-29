import math
import torch
import torch.nn.functional as F
from torch import nn
from nflows.transforms import Transform, CompositeTransform, Sigmoid, IdentityTransform, PiecewiseRationalQuadraticCouplingTransform, ActNorm
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform as MAF
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet

class LogitTransform(Transform):
    def __init__(self, alpha=0.0005, eps=1e-7):
        super().__init__()
        if alpha <= eps or alpha >= 0.5:
            raise ValueError("alpha must be in (eps, 0.5).")
        self.alpha = alpha
        self.eps = eps

    @staticmethod
    def _stable_logit(x):
        # log(x) - log1p(-x) is a stable logit when x∈(0,1)
        return torch.log(x) - torch.log1p(-x)

    def forward(self, inputs, context=None):
        dims = list(range(1, inputs.ndim))
        # Map [0,1] → [α, 1−α]
        pre_logit = self.alpha + (1.0 - 2.0*self.alpha) * inputs

        y = self._stable_logit(pre_logit)

        logdets = torch.sum(
            math.log(1.0 - 2.0*self.alpha)
            - torch.log(pre_logit).to(inputs.device)
            - torch.log1p(-pre_logit).to(inputs.device),
            dim=dims
        )
        return y, logdets

    def inverse(self, inputs, context=None):
        dims = list(range(1, inputs.ndim))
        sigm = torch.sigmoid(inputs)                
        x = (sigm - self.alpha) / (1.0 - 2.0*self.alpha)

        logdets = torch.sum(
            torch.log(sigm) + torch.log1p(-sigm) - math.log(1.0 - 2.0*self.alpha),
            dim=dims
        )
        return x, logdets

def build_flow_Rd(d, hidden_features=64, num_layers=5, num_blocks=2):
    layers = []
    for i in range(num_layers):
        layers.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=((torch.arange(d) + i) % 2 == 0),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks
                ),
                tails='linear',        # <-- avoids inputOutsideDomain
                tail_bound=5.0,        # widen if needed
                num_bins=8,
                min_bin_width=1e-3,
                min_bin_height=1e-3
            )
        )
        layers.append(ActNorm(features=d))
    return CompositeTransform(layers)


def build_unitcube_diffeo(d, hidden_features=64, num_layers=5, num_blocks=2, eps=1e-7):
    """
    Diffeomporphism from the (0, 1)^d cube to itself constructed as
    logit (inverse sigmoid) composed with a flow on R^d composed with sigmoid.
    """
    flow_Rd = build_flow_Rd(d, hidden_features, num_layers, num_blocks)

    logit = LogitTransform(eps=eps)
    sigmoid = Sigmoid(learn_temperature=True)

    return CompositeTransform([
        logit,                  
        flow_Rd,                
        sigmoid                 
    ])

def build_identity_diffeo(d, eps=1e-7):
    """
    Identity diffeomporphism from the (0, 1)^d cube to itself constructed as
    logit (inverse sigmoid) composed with identity on R^d composed with sigmoid.
    """
    identity_Rd = IdentityTransform()

    logit = LogitTransform(eps=eps)
    sigmoid = Sigmoid(learn_temperature=True)

    return CompositeTransform([
        logit,                  
        identity_Rd,                
        sigmoid                 
    ])
