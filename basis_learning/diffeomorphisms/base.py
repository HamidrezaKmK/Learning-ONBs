from abc import ABC, abstractmethod
import torch
from torch import nn

class Diffeomorphism(ABC, nn.Module):

    @abstractmethod    
    def forward(self, coords: torch.Tensor):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def inverse(self, coords: torch.Tensor):
        raise NotImplementedError("Subclasses must implement this method")
