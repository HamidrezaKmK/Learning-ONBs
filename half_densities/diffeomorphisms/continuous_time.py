import math
import torch
import torch.nn.functional as F
from torch import nn
from nflows.transforms import Transform, CompositeTransform, Sigmoid, IdentityTransform, PiecewiseRationalQuadraticCouplingTransform, ActNorm
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform as MAF
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet

class SinusoidalTimeEmbedding(nn.Module):
    """
    Scalar t  ->  [sin(w_k t), cos(w_k t)]_k  (Fourier features)
    Similar in spirit to what diffusion models use.
    """
    def __init__(self, num_frequencies: int = 8, max_log_freq: float = 3.0):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Frequencies: 2^0, 2^{max_log_freq} on a log scale
        freqs = torch.exp(torch.linspace(0.0, max_log_freq, num_frequencies) * math.log(2.0))
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)              # (B, 1)

        # (B, 1, num_freqs)
        angles = t[..., None] * self.freqs[None, None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 1, 2*num_freqs)

        return emb.view(t.size(0), -1)       # (B, 2 * num_freqs)


class CTCubeFlow(nn.Module):
    def __init__(
        self,
        dimensions: int,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_frequencies: int = 8,   
        gamma: float = 1.0,
        normalize_velocity: bool = False,
    ):
        super().__init__()
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
        self.normalize_velocity = normalize_velocity
        
    def velocity_field(self, t, xy):
        # xy: (B, d)
        # t: (B,)
        # Compute the velocity field using the pre_velocity_field network
        embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
        # pass through inverse 
        # pass xy through inverse sigmoid to map to R^d
        xy_transformed = torch.logit(xy.clamp(1e-6, 1 - 1e-6))  
        # xy_transformed = xy
        input = torch.cat([xy_transformed, embedded], dim=-1)
        v = self.pre_velocity_field(input)

        # make the velocity tangential to the cube boundaries
        velocity = (xy) ** self.gamma * (1 - xy) ** self.gamma * v

        if self.normalize_velocity:
            velocity = velocity / (torch.norm(velocity, dim=1, keepdim=True) + 1e-10)  # normalize to unit norm
        return velocity

    def evaluate(
        self,
        xy: torch.Tensor,
        start_time: float,
        end_time: float,
        initial_basis_fn: callable,
        n_steps: int = 10,
    ):
        timesteps = torch.linspace(start_time, end_time, n_steps + 1).to(xy.device)
        dt = (end_time - start_time) / n_steps

        xy_t = xy.clone().detach().requires_grad_(True)
        divs_cumsum = torch.zeros(xy_t.size(0), device=xy_t.device)

        for i in range(n_steps):
            t = timesteps[i]
            t_batch = t * torch.ones(xy_t.size(0), device=xy_t.device)

            v = self.velocity_field(t_batch, xy_t)

            grad_xy = torch.autograd.grad(
                v.sum(),
                xy_t,
                create_graph=False, 
                retain_graph=False,
            )[0] 
            div = grad_xy.sum(dim=1) 

            divs_cumsum = divs_cumsum + div * dt
            # Euler update for positions; detach so next step starts fresh
            with torch.no_grad():
                xy_t = xy_t + v * dt
            xy_t.requires_grad_(True)
        
        return initial_basis_fn(xy_t) * torch.exp(0.5 * divs_cumsum)

        
class CTRadialFlow(CTCubeFlow):

    def velocity_field(self, t, xy):
        # t: (B,)
        # Compute the velocity field using the pre_velocity_field network
        embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
        # perform logit transform on xy
        xy_transformed = torch.logit(xy.clamp(1e-6, 1 - 1e-6))
        
        input = torch.cat([xy_transformed, embedded], dim=-1)
        v = self.pre_velocity_field(input)
        v = v - (xy * (v * xy).sum(dim=1, keepdim=True))  # make tangential to circle
        return v
    
    # NOTE: this is a bit of a weird one:
    # def velocity_field(self, t, xy):
    #     # xy: (B, d)
    #     # t: (B,)
    #     # Compute the velocity field using the pre_velocity_field network
    #     embedded = self.time_embedding(t.unsqueeze(-1))  # (B, 16)
    #     # pass through inverse 
    #     # pass xy through inverse sigmoid to map to R^d
    #     transformed_r = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2).unsqueeze(-1) + 1e-10
    #     transformed_theta = torch.atan2(xy[:, 1], xy[:, 0]).unsqueeze(-1)
        
    #     xy_transformed = torch.logit(xy.clamp(1e-6, 1 - 1e-6))

    #     input = torch.cat([xy_transformed, embedded], dim=-1)
    #     radial_v = self.pre_velocity_field(input)
    #     v_r = radial_v[:, 0:1] * (1 - transformed_r)  # (B, 1)
    #     v_theta = radial_v[:, 1:2] * transformed_r # (B, 1)
    #     # convert to Cartesian coordinates
    #     v_x = v_r * torch.sin(v_theta)
    #     v_y = v_r * torch.cos(v_theta)
    #     v = torch.cat([v_x, v_y], dim=1)
        
    #     v = v - (xy * (v * xy).sum(dim=1, keepdim=True))  # make tangential to circle

    #     if self.normalize_velocity:
    #         v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-10)  # normalize to unit norm
    #     return v
    