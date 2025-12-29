
import torch
import math

from .base import BaseFunction

class TentBasis(BaseFunction):

    def __init__(
        self,
        total_rows: int,
        total_cols: int,
        device: torch.device,
    ):
        self.total_rows = total_rows
        self.total_cols = total_cols 
        self.device = device
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2)).to(self.device)
    
    def __call__(self, xy: torch.Tensor, row: int, col: int):

        w = 1.0 / float(self.total_cols)
        h = 1.0 / float(self.total_rows)
        xc = (col + 0.5) * w
        yc = (row + 0.5) * h

        u = torch.abs(xy[:, 0] - xc) / (0.5 * w)
        v = torch.abs(xy[:, 1] - yc) / (0.5 * h)
        tent = torch.clamp(1.0 - torch.maximum(u, v), min=0.0)
        scale = math.sqrt(6.0 / (w * h)) 

        return (tent * scale).to(self.device)
