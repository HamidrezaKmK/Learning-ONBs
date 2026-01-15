import torch
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.linear_synthesis.base import OrthogonalSynthesis

def dictionary_pullback(
    orthogonal_synthesis: OrthogonalSynthesis,
    initial_dictionary: InfiDictionary,
    atom_indices: torch.Tensor,
    deformed_coords: torch.Tensor, # (N, d)
    logabsdets: torch.Tensor, # (N,)
    device: torch.device,
):
    pullback = initial_dictionary.get_atom(deformed_coords, atom_indices).to(device) # (A, N)
    pullback = pullback * torch.exp(0.5 * logabsdets) # (A, N)
    return orthogonal_synthesis.pullback(pullback, atom_indices)

def dictionary_pushforward(
    orthogonal_synthesis: OrthogonalSynthesis,
    initial_dictionary: InfiDictionary,
    atom_indices: torch.Tensor,
    coords: torch.Tensor, # (N, d)
    logabsdets: torch.Tensor, # (N,)
    device: torch.device,
):
    pushforward = initial_dictionary.get_atom(coords, atom_indices).to(device)   # (A, N)
    pushforward = pushforward * torch.exp(-0.5 * logabsdets) # (A, N)
    return orthogonal_synthesis.pushforward(pushforward, atom_indices)

def gram_projection(
    orthogonal_synthesis: OrthogonalSynthesis,
    atom_indices: torch.Tensor,
    coords: torch.Tensor, # (N, d)
    vals: torch.Tensor, # (B, N)
    deformed_coords: torch.Tensor, # (N, d)
    deformed_vals: torch.Tensor, # (B, N)
    logabsdets: torch.Tensor, # (N,)
    initial_dictionary: InfiDictionary,
    device: torch.device,
    n_truncation: int | None = None,
):
    N = coords.shape[0]
    
    all_deformed_vals_pullback = dictionary_pullback(
        orthogonal_synthesis,
        initial_dictionary,
        atom_indices,
        deformed_coords,
        logabsdets,
        device,
    )  # (A, N)
    inner_products_1 = torch.einsum("an,bn->ab", all_deformed_vals_pullback, vals) / N # (A, B)
    
    all_deformed_vals_pushforward = dictionary_pushforward(
        orthogonal_synthesis,
        initial_dictionary,
        atom_indices,
        coords,
        logabsdets,
        device,
    )  # (A, N)
    inner_products_2 = torch.einsum("an,bn->ab", all_deformed_vals_pushforward, deformed_vals) / N # (A, B)
    
    inner_products = 0.5 * (inner_products_1 + inner_products_2)  # shape (A, B)
    coeffs = initial_dictionary.gram_solve(atom_indices, inner_products)
    if n_truncation is not None:
        coeffs = coeffs[:n_truncation, :]
        all_deformed_vals_pullback = all_deformed_vals_pullback[:n_truncation, :]
    projection = torch.einsum("ai,an->in", coeffs, all_deformed_vals_pullback)  # shape (B, N)
    return projection
