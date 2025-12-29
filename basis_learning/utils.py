import torch
import math

import numpy as np
import torch

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
