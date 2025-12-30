import torch
import math

import numpy as np
import torch
from basis_learning.diffeomorphisms.base import Diffeomorphism

def sample_from_disk(N: int):
    u = torch.rand(N, 2)
    r = torch.sqrt(u[:, 0])  # radius
    theta = 2 * math.pi * u[:, 1]  # angle
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)  # shape (N, 2)

def sample_stratified(stratified_gridsize, device):
    g = stratified_gridsize
    # Create grid cell coordinates
    xs, ys = torch.meshgrid(
        torch.arange(g, device=device),
        torch.arange(g, device=device),
        indexing="ij"
    )
    base = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # (g^2, 2)

    # Random offsets inside each cell
    jitter = torch.rand(g * g, 2, device=device)

    # Combine base + jitter, normalize to [0,1]
    xy = (base + jitter) / g
    return xy

def deform_vals(xy, diffeomorphism: Diffeomorphism, basis_fn: callable, **kwargs):
    """The functionality that takes a diffeomorphism and applies the isometric operator"""
    transformed_xy, logabsdet = diffeomorphism.forward(xy)
    basis_fn_vals = basis_fn(transformed_xy, **kwargs)
    return basis_fn_vals * torch.exp(0.5 * logabsdet)
