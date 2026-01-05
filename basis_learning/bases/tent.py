
import torch
import math

from .base import BaseFunction
from basis_learning.utils import sample_from_disk

class TentBasis(BaseFunction):

    def __init__(
        self,
        total_rows: int,
        total_cols: int,
    ):
        self.total_rows = total_rows
        self.total_cols = total_cols 
        super().__init__(
            num_basis_elements=total_rows * total_cols,
        )

    
    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2))
    
    def __call__(self, xy: torch.Tensor, row: int, col: int):

        w = 1.0 / float(self.total_cols)
        h = 1.0 / float(self.total_rows)
        xc = (col + 0.5) * w
        yc = (row + 0.5) * h

        u = torch.abs(xy[:, 0] - xc) / (0.5 * w)
        v = torch.abs(xy[:, 1] - yc) / (0.5 * h)
        tent = torch.clamp(1.0 - torch.maximum(u, v), min=0.0)
        scale = math.sqrt(6.0 / (w * h)) 

        return (tent * scale)

    def _get(self, xy: torch.Tensor, idx: int):
        row = idx // self.total_cols
        col = idx % self.total_cols
        return self.__call__(xy, row=row, col=col)

class RadialTentBasis(TentBasis):

    def sample_from_domain(
        self,
        N: int,
    ):
        return sample_from_disk(N)

    def __call__(self, xy, row, col):
        r = xy[:, 0]**2 + xy[:, 1]**2
        theta = torch.atan2(xy[:, 1], xy[:, 0]) + math.pi
        theta = theta / (2 * math.pi)
        
        rtheta = torch.stack([r, theta], dim=1)
        return super().__call__(rtheta, row, col)
