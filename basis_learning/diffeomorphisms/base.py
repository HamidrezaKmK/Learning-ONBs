from abc import ABC, abstractmethod
from torch import nn

class Diffeomorphism(ABC, nn.Module):

    @abstractmethod    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def inverse(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
