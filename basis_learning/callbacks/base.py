import torch

from abc import ABC, abstractmethod
from basis_learning.diffeomorphisms.base import Diffeomorphism

class Callback(ABC):

    @abstractmethod
    def __call__(
        self,
        epoch: int,
        diffeomorphism: Diffeomorphism,
        wandb_enabled: bool,
        device: torch.device,
    ):
        raise NotImplementedError("Callback is an abstract base class.")
