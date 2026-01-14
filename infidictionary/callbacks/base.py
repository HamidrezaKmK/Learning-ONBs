import torch

from abc import ABC, abstractmethod
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.linear_synthesis import OrthogonalSynthesis

class Callback(ABC):

    @abstractmethod
    def __call__(
        self,
        epoch: int,
        orthogonal_synthesis: OrthogonalSynthesis,
        diffeomorphism: Diffeomorphism,
        wandb_enabled: bool,
        device: torch.device,
    ):
        raise NotImplementedError("Callback is an abstract base class.")
