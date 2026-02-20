import torch
import math

from abc import ABC, abstractmethod

class DomainSampler(ABC):

    @abstractmethod
    def sample(
        self, 
        n_per_dim: int,
    ) -> torch.Tensor:
        """
        Sample n_per_dim**d coordinates from the domain according to some probability measure.
        Returns a tensor of shape (N, d) where d is the dimension of the domain.
        """
        raise NotImplementedError("Subclasses must implement this method")

class LineSegmentSampler(DomainSampler):
    
    def __init__(
        self, 
        stratified: bool = False,
        add_noise: bool = True,
        length: float = 1.0,
    ):
        super().__init__()
        self.stratified = stratified
        self.length = length
        self.add_noise = add_noise

    def sample(self, n_per_dim: int) -> torch.Tensor:
        """
        Sample n_per_dim coordinates uniformly from the line segment [0, length].
        Returns a tensor of shape (N, 1) for the coordinates.
        """
        if self.stratified:
            linspace = (torch.linspace(0, 1, n_per_dim + 1) + 0.5 / n_per_dim)[:-1]
            coords = linspace.unsqueeze(-1)
            if self.add_noise:
                coords += (torch.rand_like(coords) - 0.5) / n_per_dim
            coords *= self.length
            return coords
        else:
            N = n_per_dim
            coords = torch.rand(N, 1) * self.length
            return coords
        
class SquareSampler(DomainSampler):
    
    def __init__(
        self, 
        stratified: bool = False,
        add_noise: bool = True,
        height: float = 1.0,
        width: float = 1.0,
    ):
        super().__init__()
        self.stratified = stratified
        self.height = height
        self.width = width
        self.add_noise = add_noise

    def sample(self, n_per_dim: int) -> torch.Tensor:
        """
        Sample n_per_dim**2 coordinates uniformly from the unit square [0, 1]^2.
        Returns a tensor of shape (N, 2) for the coordinates.
        """
        if self.stratified:
            linspace = (torch.linspace(0, 1, n_per_dim + 1) + 0.5 / n_per_dim)[:-1]
            grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing='ij')
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            # add some noise to avoid perfect grid which can cause issues for some models
            if self.add_noise:
                coords += (torch.rand_like(coords) - 0.5) / n_per_dim
            # rescale to desired width and height
            coords[:, 0] *= self.width
            coords[:, 1] *= self.height
            return coords
        else: 
            N = n_per_dim ** 2
            coords = torch.rand(N, 2)
            coords[:, 0] *= self.width
            coords[:, 1] *= self.height
            return coords

class DiskSampler(DomainSampler):

    def __init__(
        self,
        stratified: bool = True,
        add_noise: bool = True,
        radius: float = 1.0,
    ):
        super().__init__()
        self.square_sampler = SquareSampler(
            stratified=stratified, 
            add_noise=add_noise,
            height=1.0, width=1.0,
        )
        self.radius = radius

    def sample(self, n_per_dim: int) -> torch.Tensor:
        """
        Sample n_per_dim**2 coordinates uniformly from the unit disk in 2D.
        Returns a tensor of shape (N, 2).
        """
        u = self.square_sampler.sample(n_per_dim)
        r = torch.sqrt(u[:, 0])  # radius
        theta = 2 * math.pi * u[:, 1]  # angle
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        coords = torch.stack([x, y], dim=1) * self.radius
        return coords
    
