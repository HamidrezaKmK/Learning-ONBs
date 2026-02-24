
import torch
import math

from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product

class BoxDictionary(InfiDictionary):
    # TODO: switch this with Haar wavelet basis
    
    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        resolution: int,
    ):
        super().__init__()
        self.domain_dim = domain_dim  
        self.num_channels = num_channels
        self.resolution = resolution
        self.is_orthonormal = True

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        idx = torch.zeros((num_samples, self.domain_dim), dtype=torch.long)
        for d in range(self.domain_dim):
            idx[:, d] = torch.randint(0, self.resolution, (num_samples,))
        return idx
    
    def get_atoms(self, coords: torch.Tensor, idx: torch.Tensor):
        vals = torch.ones(
            (idx.shape[0], coords.shape[0], self.num_channels), 
            device=coords.device, dtype=coords.dtype,
        )
        for d in range(self.domain_dim):
            d_idx = idx[:, d] # (A, )
            # create a box function centered at d_idx with width 1
            box_component = torch.where(
                (coords[:, d] * self.resolution >= d_idx[:, None]) & (coords[:, d] * self.resolution < d_idx[:, None] + 1),
                torch.ones_like(coords[:, d]),
                torch.zeros_like(coords[:, d]),
            ) # (A, N_coords)
            vals *= box_component[:, :, None] # (A, N_coords, 1) broadcast to (A, N_coords, C)
        return vals * math.sqrt(self.resolution ** self.domain_dim) / math.sqrt(self.num_channels) # shape (A, N_coords)

    
    def estimate_captured_energy( 
        # TODO: add a fast method for Haar wavelet basis
        # TODO: check if I need to do the reconstruction trick or not
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        num_samples: int,
    ) -> torch.Tensor: # (B, )
        idx = self.sample_indices(num_samples).to(coords.device) # (A, ...)
        atoms = self.get_atoms(coords, idx) # (A, N, C)
        energy = pairwise_inner_product(values, atoms, logabsdet) ** 2 # (B, A)
        return energy.sum(dim=-1) / num_samples
    
    def get_truncated_indices(self, num_truncated: int):
        num_truncated = min(num_truncated, self.resolution)
        grids = [torch.arange(num_truncated) for _ in range(self.domain_dim)]
        return torch.cartesian_prod(*grids)
