import torch
import math

import numpy as np
import torch
from basis_learning.diffeomorphisms.base import Diffeomorphism
from basis_learning.bases.base import BaseFunction

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

def gram_projection(
    coords: torch.Tensor,
    warped_coords: torch.Tensor,
    logabsdets: torch.Tensor,
    vals: torch.Tensor,
    basis: BaseFunction,
    device: torch.device,
):
    inner_products = []
    all_deformed_vals = []
    for idx in range(basis.num_basis_elements):
        deformed_vals = basis.get(warped_coords, idx).to(device)
        deformed_vals = deformed_vals * torch.exp(0.5 * logabsdets)
        all_deformed_vals.append(deformed_vals)
        inner_product = torch.mean(deformed_vals * vals)
        inner_products.append(inner_product)
    all_deformed_vals = torch.stack(all_deformed_vals, dim=0)  # shape (n_basis, N)
    inner_products = torch.stack(inner_products)  # shape (n_basis,)
    gram_matrix_inv = basis.gram_matrix_inv.to(device)
    coeffs = gram_matrix_inv @ inner_products  # shape (n_basis,)
    projection = torch.sum(coeffs[:, None] * all_deformed_vals, dim=0)  # shape (N,)
    return projection
