import torch
from abc import ABC, abstractmethod


class IrregularDataset(ABC):

    @abstractmethod
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of functions evaluated on a set of coordinates.

        Returns:
            coords: (N, d)    — coordinate locations (possibly irregular)
            F:      (B, N, C) — B function evaluations at those N coordinates
        """
        raise NotImplementedError
