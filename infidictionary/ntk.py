import torch
from torch.func import functional_call
from tqdm import tqdm
import math as _math

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

def fft_ntk_eigenvalues(
    ntk_model: NeuralField,
    neural_isometry,
    nyquist: int,
    pushforward_kwargs: dict,
) -> torch.Tensor:
    """Fast NTK eigenvalue grid via FFT and vectorised Jacobian (2-D domains only).

    For each spatial frequency k on a (2*nyquist+1)^2 grid and each output
    channel c, computes:

        lambda_k_c = ||grad_theta <Q* f_theta, phi_k>_{L^2(src)}||^2

    The trick: start from a *uniform source grid*, pushforward to get target
    coordinates, evaluate f_theta there, and pull back.  The round-trip
    returns Q*f_theta sampled on the original uniform source grid, so a
    standard FFT correctly computes its Fourier coefficients — no NUFFT or
    measure weighting needed.

    Args:
        ntk_model:          Trained neural field.
        neural_isometry:    Trained NeuralIsometry in eval mode with tspan set.
        nyquist:            Frequency half-bandwidth; grid is (2*nyquist+1)^2.
        pushforward_kwargs: Forwarded to pushforward / pullback.

    Returns:
        Eigenvalue grid of shape ``(2*nyquist+1, 2*nyquist+1, C)``, with
        zero-frequency centred: ``grid[nyquist, nyquist, c]`` = lambda_{0,0,c}.
    """
    assert (
        hasattr(neural_isometry, "coords_dim")
        and neural_isometry.coords_dim == 2
    ), "fft_ntk_eigenvalues only supports 2-D domains"

    device = next(ntk_model.parameters()).device
    param_names = [name for name, _ in ntk_model.named_parameters()]

    with torch.no_grad():
        dummy = ntk_model(torch.zeros(1, 2, device=device))
    C = dummy.shape[-1]

    N = 2 * nyquist + 1
    xs = torch.linspace(0.0, 1.0 - 1.0 / N, N, device=device)
    x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
    coords_src = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)  # (N^2, 2)
    logabsdet_src = torch.zeros(N * N, device=device)

    # ── Pushforward uniform source grid to get target coordinates ────────
    with torch.no_grad():
        coords_tgt, logabsdet_tgt, _ = neural_isometry.pushforward(
            src_coords=coords_src,
            src_logabsdet=logabsdet_src,
            src_field=torch.zeros(1, N * N, C, device=device),
            **pushforward_kwargs,
        )

    M = C * N * N

    # ── Pre-compute 2-D trig extraction indices ──────────────────────────
    signed = torch.arange(-nyquist, nyquist + 1, device=device)  # (N,)
    j = signed.abs()           # DFT bin indices
    neg_j = (-j) % N           # reflected DFT bin indices

    j1_2d     = j[:, None].expand(N, N)
    j2_2d     = j[None, :].expand(N, N)
    neg_j1_2d = neg_j[:, None].expand(N, N)

    # Convention from FourierDictionary._get_spatial_atoms:
    #   k < 0  →  cosine atom   sqrt(2) cos(2π|k|x)
    #   k = 0  →  constant      1
    #   k > 0  →  sine   atom   sqrt(2) sin(2πk x)
    is_sin  = (signed > 0).float()
    is_sin1 = is_sin[:, None]  # (N, 1)
    is_sin2 = is_sin[None, :]  # (1, N)

    scale    = torch.where(signed != 0, _math.sqrt(2), 1.0)
    scale_2d = scale[:, None] * scale[None, :]  # (N, N)

    def fft_of_pullback(*params):
        """params -> real product-trig coefficients of Q*f_theta via 2-D FFT.

        Because coords_src is a uniform grid and the pushforward/pullback
        round-trip returns Q*f_theta at those same uniform source points,
        the standard FFT correctly computes the source-domain Fourier
        coefficients without any measure weighting.
        """
        ntk_dict = dict(zip(param_names, params))
        f_vals = functional_call(ntk_model, ntk_dict, (coords_tgt,))  # (N^2, C)

        _, _, qsf = neural_isometry.pullback(
            tgt_coords=coords_tgt,
            tgt_logabsdet=logabsdet_tgt,
            tgt_field=f_vals.unsqueeze(0),  # (1, N^2, C)
            **pushforward_kwargs,
        )
        # qsf: (1, N^2, C) evaluated at the uniform source coordinates
        qsf_2d = qsf.squeeze(0).reshape(N, N, C).permute(2, 0, 1)  # (C, N, N)

        # ── Single 2-D FFT on the uniform source grid ───────────────
        F = torch.fft.fft2(qsf_2d, dim=(-2, -1)) / (N * N)  # (C, N, N) complex

        # Gather the DFT entries at (+j1, +j2) and (-j1, +j2):
        Fp = F[:, j1_2d,     j2_2d]  # F[+|k1|, +|k2|]   (C, N, N)
        Fm = F[:, neg_j1_2d, j2_2d]  # F[-|k1|, +|k2|]   (C, N, N)

        # Decompose into product-trig components.
        #
        # Writing  CC = <f, cos(k1)·cos(k2)>,  SS = <f, sin(k1)·sin(k2)>,
        #          SC = <f, sin(k1)·cos(k2)>,  CS = <f, cos(k1)·sin(k2)>:
        #
        #   F[+j1, +j2] = (CC - SS) - i(SC + CS)
        #   F[-j1, +j2] = (CC + SS) + i(SC - CS)
        #
        # Solving:
        CC = (Fp.real + Fm.real) / 2
        SS = (Fm.real - Fp.real) / 2
        SC = (-Fp.imag + Fm.imag) / 2
        CS = (-Fp.imag - Fm.imag) / 2

        # Select the right component for each signed (k1, k2)
        not_sin1 = 1.0 - is_sin1
        not_sin2 = 1.0 - is_sin2
        coeff = (
              not_sin1 * not_sin2 * CC
            + not_sin1 * is_sin2  * CS
            + is_sin1  * not_sin2 * SC
            + is_sin1  * is_sin2  * SS
        )  # (C, N, N)

        result = coeff * scale_2d  # sqrt(2) per non-DC dimension
        return result.reshape(-1)  # (M,) = (C*N*N,)

    # ── Jacobian & squared-norm ──────────────────────────────────────────
    J_tuple = torch.autograd.functional.jacobian(
        fft_of_pullback,
        tuple(ntk_model.parameters()),
        create_graph=False,
        vectorize=True,
    )

    lambda_flat = sum(
        j.reshape(M, -1).pow(2).sum(-1) for j in J_tuple
    )  # (M,)

    # Reshape: result[k1+nyquist, k2+nyquist, c] = lambda_{k1,k2,c}
    lambda_grid = lambda_flat.reshape(C, N, N).permute(1, 2, 0)
    return lambda_grid.detach()
