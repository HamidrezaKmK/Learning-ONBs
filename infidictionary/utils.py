import torch
import math
import numpy as np
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.dictionaries.base import InfiDictionary

def gram_projection(
    atom_indices: torch.Tensor,
    coords: torch.Tensor, # (N, d)
    vals: torch.Tensor, # (B, N)
    deformed_coords: torch.Tensor, # (N, d)
    deformed_vals: torch.Tensor, # (B, N)
    logabsdets: torch.Tensor, # (N,)
    initial_dictionary: InfiDictionary,
    device: torch.device,
):
    N = coords.shape[0]
    
    all_deformed_vals_theta = initial_dictionary.get_atom(deformed_coords, atom_indices).to(device) # (A, N)
    all_deformed_vals_theta = all_deformed_vals_theta * torch.exp(0.5 * logabsdets) # (A, N)
    inner_products_1 = torch.einsum("an,bn->ab", all_deformed_vals_theta, vals) / N # (A, B)
    
    all_deformed_vals_pullback = initial_dictionary.get_atom(coords, atom_indices).to(device)   # (A, N)
    all_deformed_vals_pullback = all_deformed_vals_pullback * torch.exp(-0.5 * logabsdets) # (A, N)
    inner_products_2 = torch.einsum("an,bn->ab", all_deformed_vals_pullback, deformed_vals) / N # (A, B)
    
    inner_products = 0.5 * (inner_products_1 + inner_products_2)  # shape (A, B)
    coeffs = initial_dictionary.gram_solve(atom_indices, inner_products)
    projection = torch.einsum("ai,an->in", coeffs, all_deformed_vals_theta)  # shape (B, N)
    return projection
