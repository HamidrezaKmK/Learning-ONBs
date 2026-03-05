"""
Normalizing-flow diffeomorphisms that map the unit hypercube to itself.

All classes here produce diffeomorphisms on ``[0, 1]^d``, making them suitable for
domains with a natural hypercube structure (e.g. the unit square).

Overview
--------
``LogitTransform``
    Coordinate-wise logit with a learnable softness parameter α.  Maps (0,1) ↔ ℝ.
    Used internally to "open up" the bounded cube before applying unconstrained flows.

``CubeFlow``
    Abstract base for ``[0,1]^d`` → ``[0,1]^d`` diffeomorphisms.

``UnitCubeNeuralSplineFlow``
    Full learnable diffeomorphism on ``[0,1]^d`` built from
    LogitTransform → rational-quadratic neural spline coupling layers → LogitTransform⁻¹.

``UnitSquareKumaraswamy``
    Lightweight, closed-form diffeomorphism on ``[0,1]^2`` using coordinate-wise
    Kumaraswamy warps.  Has a stable analytic inverse and log-det.
"""

from typing import Literal
import math
import torch
import torch.nn.functional as F
from torch import nn
from nflows.transforms import Transform, CompositeTransform, PiecewiseRationalQuadraticCouplingTransform, ActNorm
from nflows.nn.nets import ResidualNet

from .base import Diffeomorphism, IdentityFlow, ChainDiffeomorphism, InverseDiffeomorphism


class LogitTransform(Transform):
    """Coordinate-wise logit transform with a learnable softness parameter α.

    The forward map is:

    .. math::

        f(x) = \\operatorname{logit}(\\alpha + (1 - 2\\alpha)\\, x)

    which maps :math:`x \\in [0, 1]` to :math:`\\mathbb{R}`, with α controlling how
    aggressively the boundaries are compressed.  As α → 0 the map approaches the
    standard logit; as α → 0.5 it collapses to a constant.

    α is parameterized as ``sigmoid(_alpha) * 0.5 * alpha_scale`` so that it stays in
    ``(0, 0.5)`` regardless of the raw unconstrained parameter ``_alpha``.

    Args:
        alpha: Initial upper bound for α (the learnable parameter is initialized so
            that α ≈ alpha / 2 at construction time).
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
        """Apply the logit transform: ``[0,1]^d`` → ℝ^d.

        Args:
            inputs: Tensor of shape ``(N, d)`` with values in ``[0, 1]``.
            context: Unused; present for API compatibility with ``nflows``.

        Returns:
            Tuple ``(y, logdets)`` where ``y`` is in ℝ^d and ``logdets`` has shape ``(N,)``.
        """
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
        """Apply the inverse logit (sigmoid) transform: ℝ^d → ``(0,1)^d``.

        Args:
            inputs: Tensor of shape ``(N, d)`` with unconstrained real values.
            context: Unused; present for API compatibility with ``nflows``.

        Returns:
            Tuple ``(x, logdets)`` where ``x`` is in ``(0,1)^d`` and ``logdets``
            has shape ``(N,)``.
        """
        dims = list(range(1, inputs.ndim))
        sigm = torch.sigmoid(inputs)
        x = (sigm - self.alpha) / (1.0 - 2.0*self.alpha)

        logdets = torch.sum(
            torch.log(sigm) + torch.log1p(-sigm) - torch.log(1.0 - 2.0*self.alpha),
            dim=dims
        )
        return x, logdets


class CubeFlow(Diffeomorphism):
    """Abstract base class for diffeomorphisms that map ``[0, 1]^d`` to itself.

    Subclasses implement the concrete forward and inverse maps.  The unit-cube
    constraint is important: it ensures the diffeomorphism respects the boundary
    of the domain so that samples drawn from the domain stay in the domain after
    the map is applied.

    Args:
        d: Dimensionality of the unit hypercube.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d


class UnitCubeNeuralSplineFlow(CubeFlow):
    """Learnable diffeomorphism on ``[0, 1]^d`` built from rational-quadratic neural splines.

    The full pipeline is:

    1. **LogitTransform** — maps ``[0,1]^d`` → ℝ^d (opens up the bounded domain).
    2. **Rational-quadratic spline coupling layers** — alternating masked coupling
       transforms with ActNorm, parameterized by ``ResidualNet`` sub-networks.
    3. **LogitTransform⁻¹ (sigmoid)** — maps ℝ^d → ``(0,1)^d``.

    Both forward and inverse accumulate log-absolute-determinants from all three stages.

    Args:
        d: Dimensionality of the unit cube.
        hidden_features: Width of the residual networks inside each coupling layer.
        num_layers: Number of coupling-layer + ActNorm pairs.
        num_blocks: Number of residual blocks in each coupling-layer network.
        alpha: Softness parameter for the LogitTransform boundary handling.
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
        """Map ``x ∈ [0,1]^d`` to ``w ∈ (0,1)^d`` through logit → spline → sigmoid.

        Args:
            x: Tensor of shape ``(N, d)`` with values in ``[0, 1]``.

        Returns:
            Tuple ``(w, logabsdet)`` where ``logabsdet`` has shape ``(N,)``.
        """
        y, logabsdet1 = self.logit.forward(x)
        z, logabsdet2 = self.transform.forward(y)
        w, logabsdet3 = self.logit.inverse(z)
        return w, logabsdet1 + logabsdet2 + logabsdet3

    def inverse(self, w):
        """Map ``w ∈ (0,1)^d`` back to ``x ∈ [0,1]^d`` through logit → spline⁻¹ → sigmoid.

        Args:
            w: Tensor of shape ``(N, d)`` with values in ``(0, 1)``.

        Returns:
            Tuple ``(x, logabsdet)`` where ``logabsdet`` has shape ``(N,)``.
        """
        z, logabsdet1 = self.logit.forward(w)
        y, logabsdet2 = self.transform.inverse(z)
        x, logabsdet3 = self.logit.inverse(y)
        return x, logabsdet1 + logabsdet2 + logabsdet3


class UnitSquareKumaraswamy(CubeFlow):
    """Closed-form, coordinate-wise diffeomorphism on ``[0, 1]^2`` via Kumaraswamy warps.

    Each coordinate is independently transformed by a Kumaraswamy CDF with learnable
    shape parameters ``a`` and ``b``.  The Kumaraswamy distribution has a tractable
    closed-form inverse CDF, making both forward and inverse analytically stable.

    Forward (per coordinate ``i``):

    .. math::

        x_\\varepsilon = \\varepsilon + (1 - 2\\varepsilon)\\, x_i

        y_i = 1 - (1 - x_\\varepsilon^{a_i})^{b_i}

    Inverse (per coordinate ``i``):

    .. math::

        x_\\varepsilon = \\bigl(1 - (1 - y_i)^{1/b_i}\\bigr)^{1/a_i}

        x_i = \\frac{x_\\varepsilon - \\varepsilon}{1 - 2\\varepsilon}

    The boundary-softening factor ``ε`` keeps inputs away from 0 and 1, improving
    numerical stability of the power operations.

    Args:
        d: Dimensionality (must be 2; enforced in forward/inverse).
        eps: Boundary softening margin.  Maps ``[0,1]`` to ``[eps, 1-eps]`` before
            applying the warp.
        a_min: Minimum value for the ``a`` shape parameter (added after softplus).
        b_min: Minimum value for the ``b`` shape parameter (added after softplus).
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
        """Return positive shape parameters ``(a, b)`` via softplus + minimum offset."""
        a = F.softplus(self.a_raw) + self.a_min  # > 0
        b = F.softplus(self.b_raw) + self.b_min  # > 0
        return a, b

    def _to_open_unit(self, x):
        # map [0,1] -> [eps, 1-eps], stable and invertible
        return self.eps + (1.0 - 2.0*self.eps) * x

    def _from_open_unit(self, x_eps):
        return (x_eps - self.eps) / (1.0 - 2.0*self.eps)

    def forward(self, x: torch.Tensor):
        """Apply the Kumaraswamy warp: ``[0,1]^2`` → ``(0,1)^2``.

        Args:
            x: Tensor of shape ``(N, 2)`` with values in ``[0, 1]``.

        Returns:
            Tuple ``(y, logabsdet)`` where ``y`` is in ``(0,1)^2`` and
            ``logabsdet`` has shape ``(N,)``.

        Raises:
            ValueError: If ``x.shape[-1] != 2``.
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
        log_scale = math.log(1.0 - 2.0*self.eps)

        log_dy_dx = (
            log_scale
            + torch.log(a) + torch.log(b)
            + (a - 1.0) * torch.log(x_eps.clamp_min(1e-12))
            + (b - 1.0) * torch.log(one_minus)
        )  # (N,2)

        logabsdet = log_dy_dx.sum(dim=-1)  # (N,)
        return y, logabsdet

    def inverse(self, y: torch.Tensor):
        """Apply the inverse Kumaraswamy warp: ``(0,1)^2`` → ``[0,1]^2``.

        The inverse log-det is computed by evaluating the forward log-det at the
        recovered ``x`` and negating it, which is numerically consistent.

        Args:
            y: Tensor of shape ``(N, 2)`` with values in ``[0, 1]``.

        Returns:
            Tuple ``(x, logabsdet)`` where ``x`` is in ``[0,1]^2`` and
            ``logabsdet`` has shape ``(N,)``.

        Raises:
            ValueError: If ``y.shape[-1] != 2``.
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
