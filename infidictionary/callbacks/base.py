import torch

from abc import ABC, abstractmethod
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.networks import NeuralField

class Callback(ABC):

    @abstractmethod
    def __call__(
        self,
        epoch: int,
        neural_isometry: NeuralIsometry,
        mean_function: NeuralField,
        wandb_enabled: bool,
        device: torch.device,
    ):
        raise NotImplementedError("Callback is an abstract base class.")
