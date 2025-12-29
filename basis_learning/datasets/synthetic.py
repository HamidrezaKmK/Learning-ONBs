import numpy as np
import torch
import math

def random_bandpass(
    xy: torch.Tensor,
    datum_idx: int, 
    l_sin: int,
    l_cos: int,
    omega_lo: float = 5.,
    omega_hi: float = 10.,
    r_lo: float = 0.5,
    r_hi: float = 1.0,
    dset_seed: int = 42,
):
    """
    Computes the value of the random function evaluated at points xy.

    Parameters
    ----------
    xy : torch.Tensor
        Tensor of shape (N, 2), where each row contains the (x, y) coordinates.
    datum_idx : int
        Index of the datum to determine the random function. This only determines
        the weights of the linear combination of the basis functions and the basis
        functions themselves are determined by dset_seed.
    omega_lo, omega_hi : float
        Lower/upper bounds for the frequencies.
    r_lo, r_hi : float
        Lower/upper bounds for the amplitudes.
    l_sin, l_cos : int
        Number of sine and cosine basis functions.
    dset_seed : int
        Seed to determine the basis functions.
    """
    device = xy.device

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")

    rng_dset = torch.Generator().manual_seed(dset_seed)
    rng = torch.Generator().manual_seed(datum_idx + torch.randint(0, 10000, (1,), generator=rng_dset).item())

    omega_sin = torch.rand(size=(l_sin, 2), generator=rng_dset) * (omega_hi - omega_lo) + omega_lo
    omega_cos = torch.rand(size=(l_cos, 2), generator=rng_dset) * (omega_hi - omega_lo) + omega_lo
    phase_lo = 0.0
    phase_hi = 2.0 * math.pi
    phase_sin = torch.rand(size=(l_sin,), generator=rng_dset) * (phase_hi - phase_lo) + phase_lo
    phase_cos = torch.rand(size=(l_cos,), generator=rng_dset) * (phase_hi - phase_lo) + phase_lo
    r_sin = torch.rand(size=(l_sin,), generator=rng_dset) * (r_hi - r_lo) + r_lo
    r_cos = torch.rand(size=(l_cos,), generator=rng_dset) * (r_hi - r_lo) + r_lo

    # move everything to appropriate device
    omega_sin = omega_sin.to(device)
    omega_cos = omega_cos.to(device)
    phase_sin = phase_sin.to(device)
    phase_cos = phase_cos.to(device)
    r_sin = r_sin.to(device)
    r_cos = r_cos.to(device)

    # evaluate basis functions
    inner_product_cos = torch.einsum("ij,kj->ik", omega_cos, xy)  # (l_X, N)
    inner_product_cos = inner_product_cos + phase_cos.unsqueeze(1)  # (l_X, N)
    e_cos = torch.cos(inner_product_cos) * r_cos.unsqueeze(1)  # (l_X, N)
    
    inner_product_sin = torch.einsum("ij,kj->ik", omega_sin, xy)  # (l_Y, N)
    inner_product_sin = inner_product_sin + phase_sin.unsqueeze(1)  # (l_Y, N)
    e_sin = torch.sin(inner_product_sin) * r_sin.unsqueeze(1)  # (l_Y, N)

    all_e = torch.cat([e_cos, e_sin], dim=0)  # (l_X + l_Y, N)

    # sample random weights and combine
    weights = torch.randn(size=(l_sin + l_cos,), generator=rng) / math.sqrt(l_sin + l_cos)  # (l_X + l_Y,)
    weights = weights.to(device)

    val = (weights.unsqueeze(1) * all_e).sum(axis=0)  # (N,)
    return val
