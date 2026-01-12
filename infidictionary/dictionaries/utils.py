import torch
import math

def sample_from_disk(N: int):
    u = torch.rand(N, 2)
    r = torch.sqrt(u[:, 0])  # radius
    theta = 2 * math.pi * u[:, 1]  # angle
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)  # shape (N, 2)
