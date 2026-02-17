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

class IdentityFlow(Diffeomorphism):
    
    def forward(self, x: torch.Tensor):
        return x, torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor):
        return y, torch.zeros(y.shape[0], device=y.device)
    
class ChainDiffeomorphism(Diffeomorphism):
    def __init__(self, diffeomorphisms: list[Diffeomorphism]):
        super().__init__()
        self.diffeomorphisms = nn.ModuleList(diffeomorphisms)

    def to(self, *args, **kwargs):
        for diffeo in self.diffeomorphisms:
            diffeo.to(*args, **kwargs)
        return self
    
    def forward(self, x: torch.Tensor):
        logabsdet_total = torch.zeros(x.shape[0], device=x.device)
        for diffeo in self.diffeomorphisms:
            x, logabsdet = diffeo.forward(x)
            logabsdet_total += logabsdet
        return x, logabsdet_total

    def inverse(self, y: torch.Tensor):
        logabsdet_total = torch.zeros(y.shape[0], device=y.device)
        for diffeo in reversed(self.diffeomorphisms):
            y, logabsdet = diffeo.inverse(y)
            logabsdet_total += logabsdet
        return y, logabsdet_total
    
class InverseDiffeomorphism(Diffeomorphism):
    def __init__(self, diffeomorphism: Diffeomorphism):
        super().__init__()
        self.diffeomorphism = diffeomorphism

    def forward(self, x: torch.Tensor):
        return self.diffeomorphism.inverse(x)

    def inverse(self, y: torch.Tensor):
        return self.diffeomorphism.forward(y)
