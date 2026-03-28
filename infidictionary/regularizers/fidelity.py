from .base import PushforwardRegularizer
import torch

class FidelityRegularizer(PushforwardRegularizer):
    """MSE fidelity: E_a = mean_x ||Qφ_a(x) - φ_a(x)||²."""

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        return torch.mean((pushed - init) ** 2, dim=(1, 2))
