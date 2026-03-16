import torch
from torch.func import functional_call
from tqdm import tqdm

from infidictionary.utils import NeuralField


def estimate_ntk(
    model: NeuralField,
    coords: torch.Tensor,  # (N, d)
    n_samples: int,
    sigma: float,
) -> torch.Tensor:
    """
    Monte Carlo estimate of the Neural Tangent Kernel centered on the model's
    *current* parameters theta:

        K_flat[i*C+c, j*C+c'] = E_{delta ~ N(0, sigma^2 I)} [
            d MLP_c(x_i; theta+delta)/d_theta  ·  d MLP_{c'}(x_j; theta+delta)/d_theta
        ]

    With sigma=0 (and n_samples=1) this collapses to the exact empirical NTK
    J(theta)^T J(theta) at the current weights.  With a freshly-initialised
    model (theta ~ default init) and large sigma it approximates the standard
    random-feature NTK.  This unified form lets the same function serve both
    a random-init model and a pre-trained model loaded from disk.

    Args:
        model:     neural field; NTK is centred on its current parameter vector.
        coords:    (N, d) evaluation points.
        n_samples: number of MC samples to average.  Use 1 when sigma=0.
        sigma:     std of the Gaussian perturbation around the current weights.

    Returns:
        K_flat: (N*C, N*C) NTK matrix, detached from the computation graph.
    """
    device = coords.device
    N = coords.shape[0]
    param_names = [name for name, _ in model.named_parameters()]
    base_params  = [p.detach().clone() for p in model.parameters()]

    with torch.no_grad():
        dummy = model(coords[:1])
    C = dummy.shape[-1]

    K_flat = torch.zeros(N * C, N * C, device=device)

    for _ in tqdm(range(n_samples), desc="Estimating NTK", leave=False):
        # Sample theta ~ N(current_params, sigma^2 I)
        theta_tuple = tuple(
            base + sigma * torch.randn_like(base)
            for base in base_params
        )

        def f_params(*thetas):
            theta_dict = dict(zip(param_names, thetas))
            return functional_call(model, theta_dict, (coords,)).reshape(-1)  # (N*C,)

        # Jacobian J[k, :] = d(output_k)/d(theta), shape (N*C, total_params)
        J_tuple = torch.autograd.functional.jacobian(f_params, theta_tuple)
        J = torch.cat([j.reshape(N * C, -1) for j in J_tuple], dim=-1)  # (N*C, n_params)
        K_flat = K_flat + (J @ J.T) / n_samples

    return K_flat.detach()
