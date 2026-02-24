from typing import Literal
from sklearn.naive_bayes import abstractmethod
import torch
import math
import pytorch_finufft.functional as finufft
from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product

class FourierDictionary(InfiDictionary):

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        index_geom_p: float,
    ):
        super().__init__()
        self.domain_dim = domain_dim
        self.index_geom_p = index_geom_p
        self.num_channels = num_channels
        self.is_orthonormal = True

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        idx = torch.zeros((num_samples, self.domain_dim), dtype=torch.long)
        for d in range(self.domain_dim):
            # sample num_samples from a geometric distribution with p = self.index_geom_p
            geom_samples = torch.distributions.Geometric(self.index_geom_p).sample((num_samples,))
            # randomly assign half of the samples to be sine and half to be cosine
            kind_samples = torch.randint(0, 2, (num_samples,))
            idx[:, d] = torch.where(
                kind_samples == 0, 
                geom_samples,
                - geom_samples,
            )
        return idx
    
    def get_atoms(
        self, 
        coords: torch.Tensor, 
        idx: torch.Tensor, # (A, domain_dim)
    ):
        vals = torch.ones((idx.shape[0], coords.shape[0], self.num_channels), device=coords.device, dtype=coords.dtype)
        for d in range(self.domain_dim):
            d_idx = idx[:, d]
            kind = d_idx >= 0 # 1 for cosine, 0 for sine
            freq = torch.abs(d_idx).float() # (A, )
            kt = 2.0 * math.pi * freq[:, None] * coords[:, d][None, :]   # (N_idx, N_coords)
            cos_component = math.sqrt(2.0) * torch.cos(kt)
            sin_component = math.sqrt(2.0) * torch.sin(kt)
            component = torch.where(
                d_idx[:, None] == 0, 
                torch.ones_like(cos_component),
                torch.where(kind[:, None] == 0, cos_component, sin_component)
            ) # (A, N_coords)
            vals *= component[:, :, None] # (A, N_coords, 1) broadcast to (A, N_coords, C)
        return vals / math.sqrt(self.num_channels) # shape (A, N_coords)

    def monte_carlo_captured_energy(
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        num_samples: int,
    ) -> torch.Tensor: # (B, )
        # TODO: reuse the unique operation
        idx = self.sample_indices(num_samples).to(coords.device) # (A, ...)
        atoms = self.get_atoms(coords, idx) # (A, N, C)
        energy = pairwise_inner_product(values, atoms, logabsdet) ** 2 # (B, A)
        return energy.sum(dim=-1) / num_samples
    
    def compute_grid_probas(self, nyquist: int) -> torch.Tensor:
        single_tensor = self.index_geom_p * (1.0 - self.index_geom_p) ** torch.arange(0, nyquist + 1)
        single_tensor = single_tensor / (1 - (1 - self.index_geom_p) ** (nyquist + 1))

        # now do a tensor product to get d_dimensional weights
        weights = torch.ones(*[(nyquist+1) for _ in range(self.domain_dim)], device=single_tensor.device, dtype=single_tensor.dtype) # (nyquist, nyquist, ...)
        for d in range(self.domain_dim):
            shape = [1 for _ in range(self.domain_dim)]
            shape[d] = nyquist + 1
            weights *= single_tensor.view(shape) # (1, 1, ..., nyquist+1, ..., 1) where the nyquist+1 is in the d-th dimension
        
        all_probas = torch.zeros((2*nyquist+1, 2*nyquist+1))
        all_probas[nyquist:, nyquist:] += weights
        all_probas[nyquist:, :nyquist+1] += torch.flip(weights, dims=[1])
        all_probas[:nyquist+1, nyquist:] += torch.flip(weights, dims=[0])
        all_probas[:nyquist+1, :nyquist+1] += torch.flip(weights, dims=[0, 1])
        all_probas /= 4

        return all_probas
    
    def nufft_captured_energy(
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        nyquist: int,
        return_dft: bool = False,
    ) -> torch.Tensor: # (B, )
        points = coords.transpose(0, 1).contiguous().to(coords.device).to(dtype=torch.float32)# (d, N)
        values = values.permute(0, 2, 1).contiguous().to(coords.device).to(dtype=torch.complex64)
        # normalize values
        values = values / math.sqrt(nyquist) # (B, C, N)
        dft = finufft.finufft_type1(
            points=points,  # Transpose to shape (d, N) as expected by finufft
            values=values,  # Reshape to (B*C, N)
            output_shape=(nyquist * 2 + 1, nyquist * 2 + 1),
            modeord=0,
        ).permute(0, 2, 3, 1) / points.shape[1] # (B, A, C) where A is the number of frequencies in the grid
        all_probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device) # (2 * nyquist + 1, 2 * nyquist + 1 )
        energy = (dft.abs() ** 2 * all_probas[None, None, :, :]).sum(dim=(1, 2, 3)) # (B, )
        # energy_normalized = energy / nyquist
        return energy if not return_dft else (energy, dft, all_probas)
    
    def estimate_captured_energy( 
        # TODO: check if I need to do the reconstruction trick or not
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        method: Literal['monte_carlo', 'nufft'],
        *args,
        **kwargs,
    ) -> torch.Tensor: # (B, )
        if method == 'monte_carlo':
            return self.monte_carlo_captured_energy(coords, logabsdet, values, *args, **kwargs)
        else:
            return self.nufft_captured_energy(coords, logabsdet, values, *args, **kwargs)

    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        idx = torch.arange(-num_truncated + 1, num_truncated)
        grid = torch.stack(
            torch.meshgrid(*[idx for _ in range(self.domain_dim)], indexing='ij'), 
            dim=-1,
        ).view(-1, self.domain_dim) 
        return grid

    # TODO: add a nufft based reconstruction scheme maybe?
