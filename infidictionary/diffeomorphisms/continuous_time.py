"""
Continuous-time (neural ODE) diffeomorphisms.

These classes define diffeomorphisms whose forward and inverse passes are computed
by integrating an ODE from ``t=start_time`` to ``t=end_time`` (forward) or in reverse
(inverse).  The log-absolute-determinant is tracked simultaneously via the
instantaneous change-of-variables formula (the *Liouville equation*):

.. math::

    \\frac{d}{dt} \\log |\\det J| = \\operatorname{div}\\, v(t, x)

where ``v`` is the velocity field.  This turns volume tracking into a scalar ODE
that is integrated alongside the position ODE, avoiding an explicit Jacobian computation.

Overview
--------
``CTDiffeomorphism``
    Abstract base class.  Subclasses provide ``velocity_field`` and
    ``project_to_domain``; this class handles the augmented ODE integration.

``CTDynamics``
    Internal ``torch.nn.Module`` that packages the combined position + log-det
    dynamics for ``torchdiffeq``.

``CTCubeFlow``
    Concrete neural ODE on ``[0, 1]^d``.  The velocity field is attenuated near
    boundaries by a ``(x(1-x))^γ`` factor, keeping trajectories inside the cube.

``CTRadialFlow``
    Variant of ``CTCubeFlow`` operating on the unit disk.  The velocity field is
    projected onto the tangential direction so trajectories stay inside the disk.
"""

import torch
import math
from torch import nn
from abc import abstractmethod

from torchdiffeq import odeint, odeint_adjoint
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.utils import SinusoidalTimeEmbedding


class CTDiffeomorphism(Diffeomorphism):
    """Abstract base for ODE-based diffeomorphisms with tracked log-determinant.

    The forward pass integrates from ``start_time`` to ``end_time``; the inverse
    integrates in the opposite direction.  Both return the final coordinates *and*
    the accumulated ``log |det J|`` computed via the Liouville equation.

    Subclasses must implement:

    * :meth:`velocity_field` — the time-dependent vector field ``v(t, x)``.
    * :meth:`project_to_domain` — a projection step applied after each ODE step
      to keep points inside the valid domain (a temporary workaround for ODE
      solvers that do not natively respect domain constraints).

    Args:
        use_adjoint: If ``True``, use ``torchdiffeq.odeint_adjoint`` for
            memory-efficient gradient computation via the adjoint method.
            If ``False``, use standard ``odeint`` (stores full trajectory).
        method: ODE solver method passed to ``torchdiffeq``
            (e.g. ``'dopri5'``, ``'euler'``, ``'rk4'``).
        rtol: Relative tolerance for adaptive-step solvers.
        atol: Absolute tolerance for adaptive-step solvers.
    """

    def __init__(
        self,
        use_adjoint: bool = True,
        method: str = 'dopri5',
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ):
        super().__init__()
        self.use_adjoint = use_adjoint
        self.method = method
        self.rtol = rtol
        self.atol = atol

    @abstractmethod
    def velocity_field(self, t: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Compute the velocity field at time t and position xy.

        Args:
            t: Tensor of shape (B,) representing time.
            xy: Tensor of shape (B, d) representing positions in d-dimensional space.

        Returns:
            Tensor of shape (B, d) representing the velocity field at the given time and positions.
        """
        pass

    @abstractmethod
    def project_to_domain(self, xy_t: torch.Tensor) -> torch.Tensor:
        """Project the points xy_t back to the valid domain if they go out of bounds.

        Args:
            xy_t: Tensor of shape (B, d) representing positions in d-dimensional space.

        Returns:
            Tensor of shape (B, d) representing the projected positions.
        """
        pass

    def _run_ode(
        self,
        xy: torch.Tensor,
        forward: bool,
        start_time: float,
        end_time: float,
    ):
        """Integrate the augmented ODE (position + log-det) from start to end time.

        The state is augmented with a scalar log-det accumulator initialized to zero.
        After integration the log-det accumulator holds ``log |det J|`` for the full
        trajectory.

        Args:
            xy: Starting positions of shape ``(N, d)``.
            forward: If ``True``, integrate from ``start_time`` to ``end_time``;
                if ``False``, integrate in reverse.
            start_time: Lower bound of the time interval.
            end_time: Upper bound of the time interval.

        Returns:
            Tuple ``(xT, sT)`` where ``xT`` has shape ``(N, d)`` (final positions,
            projected to domain) and ``sT`` has shape ``(N,)`` (accumulated log-det).
        """
        # initial augmented state
        s0 = torch.zeros((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
        y0 = torch.cat([xy, s0], dim=-1)

        if forward:
            tspan = torch.tensor([start_time, end_time], device=xy.device, dtype=xy.dtype)
        else:
            tspan = torch.tensor([end_time, start_time], device=xy.device, dtype=xy.dtype)

        func = CTDynamics(self)

        if self.use_adjoint:
            yt = odeint_adjoint(func, y0, tspan, method=self.method, rtol=self.rtol, atol=self.atol, adjoint_params=tuple(self.parameters()))
        else:
            yt = odeint(func, y0, tspan, method=self.method, rtol=self.rtol, atol=self.atol)

        yT = yt[-1]                 # (B, d+1)
        xT = yT[:, :-1]
        sT = yT[:, -1]              # (B,)

        xT = self.project_to_domain(xT)
        # TODO: hacky! the ODE solver should respect the domain

        return xT, sT

    def forward(
        self,
        xy: torch.Tensor,
        start_time: float,
        end_time: float,
    ):
        """Push coordinates forward from ``start_time`` to ``end_time``.

        Args:
            xy: Tensor of shape ``(N, d)`` with source-domain coordinates.
            start_time: Start of the integration interval.
            end_time: End of the integration interval.

        Returns:
            Tuple ``(xT, logabsdet)`` where ``xT`` has shape ``(N, d)`` and
            ``logabsdet`` has shape ``(N,)``.
        """
        return self._run_ode(
            xy,
            forward=True,
            start_time=start_time,
            end_time=end_time,
        )

    def inverse(
        self,
        xy: torch.Tensor,
        start_time: float,
        end_time: float,
    ):
        """Pull coordinates back from ``end_time`` to ``start_time``.

        Args:
            xy: Tensor of shape ``(N, d)`` with target-domain coordinates.
            start_time: Start of the integration interval (the destination time).
            end_time: End of the integration interval (the source time for inversion).

        Returns:
            Tuple ``(x0, logabsdet)`` where ``x0`` has shape ``(N, d)`` and
            ``logabsdet`` has shape ``(N,)``.
        """
        return self._run_ode(
            xy,
            forward=False,
            start_time=start_time,
            end_time=end_time,
        )


class CTDynamics(torch.nn.Module):
    """ODE dynamics module for ``torchdiffeq``, combining position and log-det evolution.

    Packages the velocity field and the Liouville divergence term into a single
    ``forward(t, y)`` callable expected by ``torchdiffeq.odeint``.

    The augmented state ``y`` has shape ``(N, d+1)`` where the last column is the
    running log-det accumulator ``s``.  The dynamics are:

    .. math::

        \\dot{x} = v(t, x), \\quad \\dot{s} = \\operatorname{div}\\, v(t, x)

    Divergence is computed via a sum of scalar Jacobian-vector products (one per
    output dimension), which scales as O(d) backward passes rather than O(d²).

    Args:
        flow: The parent :class:`CTDiffeomorphism` that provides
            :meth:`~CTDiffeomorphism.velocity_field` and
            :meth:`~CTDiffeomorphism.project_to_domain`.
    """

    def __init__(self, flow: CTDiffeomorphism):
        super().__init__()
        self.flow = flow

    def divergence(self, v, x, create_graph):
        """Compute the divergence of ``v`` with respect to ``x`` via per-component autograd.

        Args:
            v: Velocity tensor of shape ``(N, d)``.
            x: Position tensor of shape ``(N, d)``, must have ``requires_grad=True``.
            create_graph: Whether to build the autograd graph through the divergence
                (needed during training for second-order gradients / adjoint method).

        Returns:
            Scalar divergence tensor of shape ``(N,)``.
        """
        div = 0.0
        for i in range(x.shape[1]):
            div_i = torch.autograd.grad(
                v[:, i].sum(), x,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0][:, i]
            div = div + div_i
        return div

    def forward(self, t, y: torch.Tensor):
        """Evaluate the augmented dynamics ``(v, div v)`` at time ``t`` and state ``y``.

        Args:
            t: Scalar time tensor (broadcast to a batch by ``torchdiffeq``).
            y: Augmented state tensor of shape ``(N, d+1)``; last column is log-det.

        Returns:
            Tensor of shape ``(N, d+1)`` containing ``[v(t, x), div v(t, x)]``.
        """
        # y: (B, d+1) where last dim is s
        x = y[:, :-1]

        # torchdiffeq passes scalar t; make batch
        t_batch = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * t

        # TODO: this is a bit hacky, but we need x to be in the domain
        def _inner_forward(x: torch.Tensor, create_graph: bool):
            x_proj = self.flow.project_to_domain(x)
            x_proj.requires_grad_(True)
            v = self.flow.velocity_field(t_batch, x_proj)
            div = self.divergence(v, x_proj, create_graph=create_graph)
            ds = div.unsqueeze(-1)
            return v, ds

        if torch.is_grad_enabled():
            v, ds = _inner_forward(x, create_graph=True)
        else:
            with torch.enable_grad():
                v, ds = _inner_forward(x, create_graph=False)

        return torch.cat([v, ds], dim=-1)         # (B, d+1)


class CTCubeFlow(CTDiffeomorphism):
    """Neural ODE diffeomorphism on the unit hypercube ``[0, 1]^d``.

    The velocity field is parameterized by a small MLP with sinusoidal time embedding.
    To keep trajectories inside ``[0, 1]^d`` the raw MLP output ``v_raw`` is multiplied
    by ``(x(1-x))^γ``, which vanishes at the boundaries:

    .. math::

        v(t, x) = v_{\\text{raw}}(t, x) \\cdot \\prod_i x_i^\\gamma (1-x_i)^\\gamma

    A hard clamp :meth:`project_to_domain` is also applied after each ODE step as
    a safety measure.

    Args:
        dimensions: Dimensionality ``d`` of the unit cube.
        use_adjoint: Passed to :class:`CTDiffeomorphism`.
        method: ODE solver name passed to ``torchdiffeq``.
        rtol: Relative ODE tolerance.
        atol: Absolute ODE tolerance.
        hidden_features: Width of the MLP hidden layers.
        num_layers: Number of hidden layers (each followed by ReLU).
        num_frequencies: Number of sinusoidal frequencies in the time embedding.
        gamma: Exponent controlling how quickly the velocity field decays to zero
            at the cube boundaries.  Higher values create a stronger boundary effect.
    """

    def __init__(
        self,
        dimensions: int,
        use_adjoint: bool = True,
        method: str = 'dopri5',
        rtol: float = 1e-6,
        atol: float = 1e-6,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_frequencies: int = 8,
        gamma: float = 1.0,
    ):
        super().__init__(
            use_adjoint=use_adjoint,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        self.d = dimensions

        # register a module called pre_velocity_field that takes in
        # time and a d-dimensional space and outputs a d-dimensional velocity field

        self.time_embedding = SinusoidalTimeEmbedding(num_frequencies=num_frequencies)
        layers = [
            nn.Linear(dimensions + self.time_embedding.out_dim, hidden_features),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
            ]
        layers.append(nn.Linear(hidden_features, dimensions))
        self.pre_velocity_field = nn.Sequential(*layers)
        self.gamma = gamma

    def velocity_field(self, t, xy):
        """Evaluate the boundary-attenuated velocity field.

        Args:
            t: Time tensor of shape ``(N,)``.
            xy: Position tensor of shape ``(N, d)`` inside ``[0, 1]^d``.

        Returns:
            Velocity tensor of shape ``(N, d)`` that vanishes at the cube boundaries.
        """
        # xy: (B, d)
        # t: (B,)
        # Compute the velocity field using the pre_velocity_field network
        embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
        # pass through inverse
        # pass xy through inverse sigmoid to map to R^d
        input = torch.cat([xy, embedded], dim=-1)
        v = self.pre_velocity_field(input)

        # make the velocity tangential to the cube boundaries
        velocity = (xy) ** self.gamma * (1 - xy) ** self.gamma * v

        return velocity

    def project_to_domain(self, xy_t, eps: float = 1e-2):
        """Clamp positions to the interior of ``[0, 1]^d``.

        Args:
            xy_t: Position tensor of shape ``(N, d)``.
            eps: Margin from the boundary (positions are clamped to ``[eps, 1-eps]``).

        Returns:
            Clamped tensor of shape ``(N, d)``.
        """
        return torch.clamp(xy_t, eps, 1 - eps)


class CTRadialFlow(CTCubeFlow):
    """Neural ODE diffeomorphism on the unit disk.

    Inherits the MLP architecture from :class:`CTCubeFlow` but overrides the
    velocity field and domain projection for the disk geometry:

    * **Velocity field** — the MLP output is projected onto the tangent plane of
      the unit circle (i.e. the component along the radial direction is removed),
      so trajectories cannot escape the disk.
    * **Domain projection** — points that drift outside the disk are renormalized
      back onto the boundary circle.
    """

    def velocity_field(self, t, xy):
        """Evaluate the tangentially-projected velocity field on the disk.

        Args:
            t: Time tensor of shape ``(N,)``.
            xy: Position tensor of shape ``(N, 2)`` inside the unit disk.

        Returns:
            Velocity tensor of shape ``(N, 2)`` with no radial component.
        """
        # t: (B,)
        # Compute the velocity field using the pre_velocity_field network
        embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
        input = torch.cat([xy, embedded], dim=-1)
        v = self.pre_velocity_field(input)
        v = v - (xy * (v * xy).sum(dim=1, keepdim=True))  # make tangential to circle
        return v

    def project_to_domain(self, xy_t, eps: float = 1e-2):
        """Renormalize points that have drifted outside the unit disk.

        Args:
            xy_t: Position tensor of shape ``(N, 2)``.
            eps: Points with norm greater than ``1 - eps`` are projected back to
                the circle of radius ``1 - eps``.

        Returns:
            Projected tensor of shape ``(N, 2)`` with all points inside the disk.
        """
        norms = torch.norm(xy_t, dim=1, keepdim=True)
        return torch.where(
            norms > (1 - eps),
            xy_t / norms * (1 - eps),  # project back to unit cube
            xy_t
        )
