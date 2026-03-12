import torch
from abc import ABC, abstractmethod


class IrregularDataset(ABC):

    def reset_buffer(self) -> None:
        """Reset the internal sampling buffer.

        For synthetic datasets this is a no-op.  Datasets with a finite image
        pool (e.g. CelebADataset) override this to shuffle the image order and
        reset the cyclic pointer, ensuring non-overlapping batches across
        gradient-accumulation steps.
        """
        pass

    @abstractmethod
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of functions evaluated on a set of coordinates.

        Returns:
            coords: (N, d)    — coordinate locations (possibly irregular)
            F:      (B, N, C) — B function evaluations at those N coordinates
        """
        raise NotImplementedError
