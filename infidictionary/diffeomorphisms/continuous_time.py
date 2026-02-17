
import torch
import math
from torch import nn
from abc import abstractmethod

from torchdiffeq import odeint, odeint_adjoint
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.utils import SinusoidalTimeEmbedding

class CTDiffeomorphism(Diffeomorphism):

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
        """
        Compute the velocity field at time t and position xy.

        Args:
            t: Tensor of shape (B,) representing time.
            xy: Tensor of shape (B, d) representing positions in d-dimensional space.

        Returns:
            Tensor of shape (B, d) representing the velocity field at the given time and positions.
        """
        pass
    
    @abstractmethod
    def project_to_domain(self, xy_t: torch.Tensor) -> torch.Tensor:
        """
        Project the points xy_t back to the valid domain if they go out of bounds.

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
        return self._run_ode(
            xy,
            forward=False,
            start_time=start_time,
            end_time=end_time,
        )


class CTDynamics(torch.nn.Module):
    def __init__(self, flow: CTDiffeomorphism):
        super().__init__()
        self.flow = flow

    def divergence(self, v, x, create_graph):
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
        return torch.clamp(xy_t, eps, 1 - eps)
            
class CTRadialFlow(CTCubeFlow):

    def velocity_field(self, t, xy):
        # t: (B,)
        # Compute the velocity field using the pre_velocity_field network
        embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
        input = torch.cat([xy, embedded], dim=-1)
        v = self.pre_velocity_field(input)
        v = v - (xy * (v * xy).sum(dim=1, keepdim=True))  # make tangential to circle
        return v
    
    def project_to_domain(self, xy_t, eps: float = 1e-2):
        norms = torch.norm(xy_t, dim=1, keepdim=True)
        return torch.where(
            norms > (1 - eps),
            xy_t / norms * (1 - eps),  # project back to unit cube
            xy_t
        )
