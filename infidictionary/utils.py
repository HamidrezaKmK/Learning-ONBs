import torch
import math
import numpy as np
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.dictionaries.base import InfiDictionary

def gram_projection(
    coords: torch.Tensor,
    warped_coords: torch.Tensor,
    logabsdets: torch.Tensor,
    vals: torch.Tensor,
    initial_dictionary: InfiDictionary,
    device: torch.device,
):
    inner_products = []
    all_deformed_vals = []
    for idx in range(initial_dictionary.num_atoms):
        deformed_vals = initial_dictionary.get_atom(warped_coords, idx).to(device)
        deformed_vals = deformed_vals * torch.exp(0.5 * logabsdets)
        all_deformed_vals.append(deformed_vals)
        inner_product = torch.mean(deformed_vals * vals)
        inner_products.append(inner_product)
    all_deformed_vals = torch.stack(all_deformed_vals, dim=0)  # shape (n_dictionary, N)
    inner_products = torch.stack(inner_products)  # shape (n_dictionary,)
    gram_matrix_inv = initial_dictionary.gram_matrix_inv.to(device)
    coeffs = gram_matrix_inv @ inner_products  # shape (n_dictionary,)
    projection = torch.sum(coeffs[:, None] * all_deformed_vals, dim=0)  # shape (N,)
    return projection
