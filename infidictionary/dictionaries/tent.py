
import torch
import math

from .base import InfiDictionary
from .utils import sample_from_disk

class TentDictionary(InfiDictionary):

    def __init__(
        self,
        total_rows: int,
        total_cols: int,
    ):
        self.total_rows = total_rows
        self.total_cols = total_cols 
        super().__init__(
            num_atoms=total_rows * total_cols,
        )

    
    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2))
    
    def _get_atoms(self, xy: torch.Tensor, idx: torch.Tensor):
        # idx: (K,) long tensor of atom indices
        idx = idx.to(device=xy.device, dtype=torch.long).view(-1)  # (K,)

        row = idx // self.total_cols                      # (K,)
        col = idx % self.total_cols                       # (K,)

        w = 1.0 / float(self.total_cols)
        h = 1.0 / float(self.total_rows)

        xc = (col + 0.5) * w                              # (K,)
        yc = (row + 0.5) * h                              # (K,)

        # Broadcast over (K, N)
        u = (xy[:, 0].unsqueeze(0) - xc.unsqueeze(1)).abs() / (0.5 * w)  # (K, N)
        v = (xy[:, 1].unsqueeze(0) - yc.unsqueeze(1)).abs() / (0.5 * h)  # (K, N)

        tent = torch.clamp(1.0 - torch.maximum(u, v), min=0.0)           # (K, N)
        scale = math.sqrt(6.0 / (w * h))
        return tent * scale                                              # (K, N)

    def gram_solve(self, atom_indices: torch.Tensor, inner_products: torch.Tensor):
        return inner_products[atom_indices]

        
class RadialTentDictionary(TentDictionary):

    def sample_from_domain(
        self,
        N: int,
    ):
        return sample_from_disk(N)

    def _get_atoms(self, xy, idx):
        r = xy[:, 0]**2 + xy[:, 1]**2
        theta = torch.atan2(xy[:, 1], xy[:, 0]) + math.pi
        theta = theta / (2 * math.pi)
        
        rtheta = torch.stack([r, theta], dim=1)
        return super()._get_atoms(rtheta, idx)
