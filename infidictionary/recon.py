import torch
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries.base import NeuralIsometry
from infidictionary.networks import NeuralField

def get_reconstructions(
    coords: torch.Tensor, # (N, d)
    functions: torch.Tensor, # (B, N)
    neural_isometry: NeuralIsometry,
    mean_function: NeuralField,
    dictionary: InfiDictionary,
    truncation_factor: int,
    **transform_kwargs,
):
    """
    Get the reconstructions of the given functions using the provided
    neural isometry and dictionary at the specified atom indices (prefixes).
    """
    device = coords.device
    mean_function_eval = mean_function(coords) # (N, C)
    func_centered = functions - mean_function_eval.unsqueeze(0)  # (B, N, C)
    src_coords, src_logabsdet, src_field = neural_isometry.pullback(
        tgt_coords=coords,
        tgt_logabsdet=torch.zeros(coords.shape[0], device=device),
        tgt_field=func_centered,
        **transform_kwargs,
    )
    recon_pullback = dictionary.get_reconstructions(
        coords=src_coords,
        # logabsdet=src_logabsdet,
        functions=src_field,
        truncation_factor=truncation_factor,
    ) # (B, N, C)
    _, _, recon = neural_isometry.pushforward(
        src_coords=src_coords,
        src_logabsdet=src_logabsdet,
        src_field=recon_pullback,
        **transform_kwargs,
    )
    return recon + mean_function_eval.unsqueeze(0) # (B, N, C)