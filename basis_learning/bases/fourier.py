
import torch
import math
from typing import Literal

from .base import BaseFunction
from basis_learning.utils import sample_from_disk

def cos1d(k, t):
    return math.sqrt(2.0) * torch.cos(2.0 * math.pi * k * t) if k > 0 else torch.ones_like(t)

def sin1d(k, t):
    return math.sqrt(2.0) * torch.sin(2.0 * math.pi * k * t) if k > 0 else torch.zeros_like(t)

class FourierBasis(BaseFunction):

    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2)).to(self.device)

    def __call__(
        self, 
        xy: torch.Tensor, 
        kx: int, 
        ky: int,
        kind: Literal["cc", "cs", "sc", "ss"] = "cc",
    ):
        if kx < 0 or ky < 0:
            raise ValueError("kx, ky must be >= 0")
        if kind not in ("cc", "cs", "sc", "ss"):
            raise ValueError("kind must be one of {'cc','cs','sc','ss'}")

        x, y = xy[:, 0], xy[:, 1]


        fx_c = cos1d(kx, x); fx_s = sin1d(kx, x)
        fy_c = cos1d(ky, y); fy_s = sin1d(ky, y)

        if kind == "cc":
            vals = fx_c * fy_c
        elif kind == "cs":
            vals = fx_c * fy_s
        elif kind == "sc":
            vals = fx_s * fy_c
        else:
            vals = fx_s * fy_s

        return vals.to(self.device)  # shape (N,)


class RadialFourierBasis(FourierBasis):
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return sample_from_disk(N).to(self.device)
    
    def __call__(
        self, 
        xy: torch.Tensor, 
        kx: int, 
        ky: int,
        kind: Literal["cc", "cs", "sc", "ss"] = "cc",
    ):
        r = xy[:, 0]**2 + xy[:, 1]**2
        theta = torch.atan2(xy[:, 1], xy[:, 0]) + math.pi
        theta = theta / (2 * math.pi)  # normalize to [-1,1]
        
        rtheta = torch.stack([r, theta], dim=1)
        return super().__call__(rtheta, kx, ky, kind)
