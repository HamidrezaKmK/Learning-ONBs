from .base import Regularizer
from .fidelity import FidelityRegularizer
from .fourierer import FouriererRegularizer
from .heat import HeatQuadraticFormRegularizer, DirichletEnergyRegularizer
from .ml_based import ClassActivationRegularizer, CLIPRegularizer, ImageTargetRegularizer
from .ntk import NTKRegularizer
from .pwc import GraphLaplacianRegularizer, EntropyRegularizer, TVMaterialRegularizer
