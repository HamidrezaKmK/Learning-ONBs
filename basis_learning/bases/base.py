# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from basis_learning.diffeomorphisms.base import Diffeomorphism

import torch

class BaseFunction(ABC):

    @abstractmethod
    def __call__(self, coords: torch.Tensor, *args, **kwds):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def sample_from_domain(self, N: int):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get(self, coords: torch.Tensor, idx: int):
        raise NotImplementedError("Subclasses must implement this method if needed")
