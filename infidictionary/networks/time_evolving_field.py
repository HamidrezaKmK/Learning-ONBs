from typing import Callable, Dict, Any

import torch
from torch import nn

from .base import NeuralField
from .time_embedding import SinusoidalTimeEmbedding


class TimeEvolvingField(NeuralField):
    def __init__(
        self,
        base_field_partial: Callable[[Dict[str, Any]], NeuralField],
        coords_dim: int = 2,
        output_dim: int = 1,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.time_embedding = SinusoidalTimeEmbedding()
        self.time_evolving_field = base_field_partial(
            input_dim=self.time_embedding.out_dim + coords_dim,
            output_dim=output_dim,
        )
        self.coords_dim = coords_dim
        self.output_dim = output_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (B, )
        # x: (B, d)
        t_emb = self.time_embedding(t.unsqueeze(-1))  # (B, time_embedding_dim)
        inp = torch.cat([t_emb, x], dim=-1)  # (B, time_embedding_dim + d)
        return self.time_evolving_field(inp)  # (B, output_dim)
