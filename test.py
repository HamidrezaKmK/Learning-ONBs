
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import functools

from basis_learning.diffeomorphisms.continuous_time import CTRadialFlow
from basis_learning.bases import RadialTentBasis
from basis_learning.utils import deform_vals


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)


torch.manual_seed(0)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
N = 1000
T = CTRadialFlow(
    dimensions=2,
    gamma=1.,
    use_adjoint=False,
    method='dopri5',
    rtol=1e-6,
    atol=1e-6,
    start_time=0.,
    end_time=1.0,
).to(device)

tent_basis = RadialTentBasis(
    total_rows=3,
    total_cols=3,
    device=device,
)

# seed everything for reproducibility
torch.manual_seed(0)
N = 500
end_time = 5.5
all_deformed_vals = []
all_original_vals = []
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
for row in range(3):
    for col in range(3):
        ax = axes[row, col]
        basis_func = functools.partial(tent_basis, row=row, col=col)
        xy = tent_basis.sample_from_domain(N).to(device)
        vals = basis_func(xy).detach().cpu().numpy()
        all_original_vals.append(vals)
        deformed_vals = deform_vals(xy, T, basis_func)
        deformed_vals = deformed_vals.detach().cpu().numpy()
        all_deformed_vals.append(deformed_vals)
        # approximate integral by monte carlo
        # ax.hexbin(xy[:,0].detach().cpu(), xy[:,1].detach().cpu(), C=deformed_vals.detach().cpu(), gridsize=50, cmap='viridis') 
        # plt.colorbar(mappable=ax.collections[0], ax=ax)
        # ax.set_title(f'Basis Function (row={row}, col={col})')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
# plt.show()