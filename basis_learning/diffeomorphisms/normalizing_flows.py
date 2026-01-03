import math
import torch
import torch.nn.functional as F
from torch import nn
from nflows.transforms import Transform, CompositeTransform, Sigmoid, IdentityTransform, PiecewiseRationalQuadraticCouplingTransform, ActNorm
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform as MAF
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet

from .base import Diffeomorphism

class LogitTransform(Transform):
    """
    This transform models a coordinate-wise logit transform as follows: 

    f(x) = logit(alpha + (1-2 alpha)x) = log(alpha + (1-2 alpha)x) - log(1 - alpha - (1-2 alpha)x)

    This maps values from [0, 1] to [-log ((1 - alpha)/alpha), log((1 - alpha) / alpha)] which
    is [-infty, infty] when alpha -> 0 and [0, 0] when alpha -> 0.5.
    """
    @property
    def alpha(self):
        return torch.nn.functional.sigmoid(self._alpha) * 0.5 * self.alpha_scale
    
    def __init__(self, alpha=0.0005):
        super().__init__()
        self.alpha_scale = alpha * 2
        # randomly initialize self._alpha to a value that is normal
        self._alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)

    @staticmethod
    def _stable_logit(x):
        # log(x) - log1p(-x) is a stable logit when x∈(0,1)
        return torch.log(x) - torch.log1p(-x)

    def forward(self, inputs, context=None):
        dims = list(range(1, inputs.ndim))
        pre_logit = self.alpha + (1.0 - 2.0*self.alpha) * inputs
        y = self._stable_logit(pre_logit)

        logdets = torch.sum(
            torch.log(1.0 - 2.0*self.alpha)
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
            torch.log(sigm) + torch.log1p(-sigm) - torch.log(1.0 - 2.0*self.alpha),
            dim=dims
        )
        return x, logdets

class CubeFlow(Diffeomorphism):
    def __init__(self, d):
        super().__init__()
        self.d = d

class UnitCubeNeuralSplineFlow(CubeFlow):
    """
    This is a diffeomorphism designed to map between [0, 1]^d and itself.
    First off, an inverse sigmoid (LogitTransform) is applied to map inputs from (0, 1) to R.
    then, a series of neural spline flow coupling layers are applied to transform the data on R^d,
    and finally, a sigmoid with learnable temperature is applied to map back to (0, 1)^d.
    """
    def __init__(self, d, hidden_features=64, num_layers=5, num_blocks=2, alpha=0.0005):
        super().__init__(d=d)
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
                    tails='linear',
                    tail_bound=5.0,
                    num_bins=8,
                    min_bin_width=1e-3,
                    min_bin_height=1e-3
                )
            )
            layers.append(ActNorm(features=d))
        self.transform = CompositeTransform(layers)
        self.logit = LogitTransform(alpha=alpha)
        
    def forward(self, x):
        y, logabsdet1 = self.logit.forward(x)
        z, logabsdet2 = self.transform.forward(y)
        w, logabsdet3 = self.logit.inverse(z)
        return w, logabsdet1 + logabsdet2 + logabsdet3

    def inverse(self, w):
        z, logabsdet1 = self.logit.forward(w)
        y, logabsdet2 = self.transform.inverse(z)
        x, logabsdet3 = self.logit.inverse(y)
        return x, logabsdet1 + logabsdet2 + logabsdet3


class UnitSquareKumaraswamy(CubeFlow):
    """
    Extremely simple, numerically stable diffeomorphism [0,1]^2 -> [0,1]^2.
    Coordinatewise monotone Kumaraswamy warps with closed-form inverse.

    Forward (per coord):
      x_eps = eps + (1-2eps)*x
      y = 1 - (1 - x_eps**a)**b

    Inverse (per coord):
      x_eps = (1 - (1-y)**(1/b))**(1/a)
      x = (x_eps - eps) / (1-2eps)
    """
    def __init__(self, d: int, eps: float = 1e-6, a_min: float = 1e-3, b_min: float = 1e-3):
        super().__init__(d=d)
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must be in (0, 0.5).")
        self.eps = eps
        self.a_min = a_min
        self.b_min = b_min

        # Learnable unconstrained params (2 dims). Initialize near identity: a=b=1.
        # softplus(0) ~ 0.693, so set raw so that softplus(raw) ~ 1 -> raw ~ softplus^{-1}(1) ~ 0.541.
        init_raw = 0.541324854612918  # approx inverse softplus of 1.0
        self.a_raw = nn.Parameter(torch.tensor([init_raw, init_raw], dtype=torch.float32))
        self.b_raw = nn.Parameter(torch.tensor([init_raw, init_raw], dtype=torch.float32))

    def _ab(self):
        a = F.softplus(self.a_raw) + self.a_min  # > 0
        b = F.softplus(self.b_raw) + self.b_min  # > 0
        return a, b

    def _to_open_unit(self, x):
        # map [0,1] -> [eps, 1-eps], stable and invertible
        return self.eps + (1.0 - 2.0*self.eps) * x

    def _from_open_unit(self, x_eps):
        return (x_eps - self.eps) / (1.0 - 2.0*self.eps)

    def forward(self, x: torch.Tensor):
        """
        x: (N,2) in [0,1]
        returns y: (N,2) in (0,1), logabsdet: (N,)
        """
        if x.shape[-1] != 2:
            raise ValueError("Expected x with last dim = 2.")

        a, b = self._ab()  # (2,), (2,)
        x = x.clamp(0.0, 1.0)
        x_eps = self._to_open_unit(x)

        # Kumaraswamy forward
        x_pow_a = x_eps.pow(a)                     # (N,2) via broadcasting
        one_minus = (1.0 - x_pow_a).clamp_min(1e-12)
        y = 1.0 - one_minus.pow(b)

        # log|det J| = sum_i log dy_i/dx_i (diagonal Jacobian)
        # dy/dx = (1-2eps) * a*b*x_eps^(a-1) * (1 - x_eps^a)^(b-1)
        log_scale = torch.log(1.0 - 2.0*self.eps)

        log_dy_dx = (
            log_scale
            + torch.log(a) + torch.log(b)
            + (a - 1.0) * torch.log(x_eps.clamp_min(1e-12))
            + (b - 1.0) * torch.log(one_minus)
        )  # (N,2)

        logabsdet = log_dy_dx.sum(dim=-1)  # (N,)
        return y, logabsdet

    def inverse(self, y: torch.Tensor):
        """
        y: (N,2) in [0,1]
        returns x: (N,2) in (0,1), logabsdet: (N,)
        """
        if y.shape[-1] != 2:
            raise ValueError("Expected y with last dim = 2.")

        a, b = self._ab()
        y = y.clamp(0.0, 1.0)

        # Kumaraswamy inverse
        one_minus_y = (1.0 - y).clamp_min(1e-12)
        inner = 1.0 - one_minus_y.pow(1.0 / b)
        inner = inner.clamp_min(1e-12)
        x_eps = inner.pow(1.0 / a)

        x = self._from_open_unit(x_eps).clamp(0.0, 1.0)

        # log|det J_{inverse}| = -log|det J_{forward}| evaluated at x
        # We can compute forward logdet at x and negate (stable, consistent).
        _, logabsdet_fwd = self.forward(x)
        logabsdet_inv = -logabsdet_fwd
        return x, logabsdet_inv


class DiskFlow(Diffeomorphism):

    def __init__(
        self,
        cubeflow: CubeFlow,
    ):
        super().__init__()
        if cubeflow.d != 2:
            raise ValueError("cubeflow must have d=2 for DiskFlow.")
        self.cubeflow = cubeflow
        
    def _cart_to_polar(self, xy: torch.Tensor):
        r = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2) # (N,)
        theta = torch.atan2(xy[:, 1], xy[:, 0])  # (N,)
        # scale theta to [0,1]
        theta_scaled = (theta + math.pi) / (2.0 * math.pi)  # (N,) 
        rtheta = torch.stack([r, theta_scaled]).transpose(0, 1)  # (N,2)

        # log|det d(r,u)/d(x,y)| = -log r - log(2pi)
        r_safe = r.clamp_min(1e-12)
        logabsdet = (-torch.log(r_safe) - math.log(2.0 * math.pi)).squeeze(-1)  # (N,)
        return rtheta, logabsdet
    
    def _polar_to_cart(self, rtheta: torch.Tensor):
        r, theta_scaled = rtheta[:, 0], rtheta[:, 1]
        theta = theta_scaled * (2.0 * math.pi) - math.pi  # scale back to [-pi, pi]
        xy = torch.stack([
            r * torch.cos(theta),
            r * torch.sin(theta),
        ]).transpose(0, 1)  # (N,2)

        # log|det d(x,y)/d(r,u)| = log r + log(2pi)
        r_safe = r.clamp_min(1e-12)
        logabsdet = (torch.log(r_safe) + math.log(2.0 * math.pi)).squeeze(-1)  # (N,)
        return xy, logabsdet
    
    def forward(self, xy: torch.Tensor):
        if xy.shape[-1] != 2:
            raise ValueError("Expected coordinates with last dim = 2.")
        # compute the log absolute determinent of transforming xy to rtheta
        rtheta, logabsdet_1 = self._cart_to_polar(xy)  # (N,2), (N,)
        rtheta_transformed, logabsdet_mid = self.cubeflow.forward(rtheta)  # (N,2), (N,)
        xy_transformed, logabsdet_2 = self._polar_to_cart(rtheta_transformed)  # (N,2), (N,)
        return xy_transformed, logabsdet_1 + logabsdet_mid + logabsdet_2


    def inverse(self, xy: torch.Tensor):
        if xy.shape[-1] != 2:
            raise ValueError("Expected xy with last dim = 2.")
        rtheta, logabsdet_1 = self._cart_to_polar(xy)  # (N,2), (N,)
        rtheta_inv, logabsdet_mid = self.cubeflow.inverse(rtheta)  # (N,2), (N,)
        xy_inv, logabsdet_2 = self._polar_to_cart(rtheta_inv)  # (N,2), (N,)
        return xy_inv, logabsdet_1 + logabsdet_mid + logabsdet_2
