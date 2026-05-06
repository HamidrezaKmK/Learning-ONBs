import torch
from torch.func import functional_call
import math as _math

from infidictionary.networks import NeuralField


def estimate_ntk(
    model: NeuralField,
    coords: torch.Tensor,  # (N, d)
    batch_size: int | None = None,
) -> torch.Tensor:
    """Empirical Neural Tangent Kernel at the model's current parameters.

        K(x, y) = <grad_theta f(x; theta), grad_theta f(y; theta)>

    which is exactly J(theta)^T J(theta), where J is the Jacobian of the
    flattened model output with respect to its parameters.  No averaging,
    no perturbation — just the kernel at the current weights.

    Args:
        model:      neural field; NTK is centred on its current parameter vector.
        coords:     (N, d) evaluation points.
        batch_size: if set, the Jacobian is computed in chunks of this many
                    points, with each chunk detached and moved to CPU before
                    the next chunk starts.  Reduces peak VRAM at the cost of
                    more sequential Jacobian calls.  None uses the original
                    single-pass behaviour (backward compatible).

    Returns:
        K: (N*C, N*C) NTK matrix, detached from the computation graph,
        where C is the model's output channel count.
    """
    device = coords.device
    N = coords.shape[0]
    param_names = [name for name, _ in model.named_parameters()]
    theta_tuple = tuple(p.detach().clone() for p in model.parameters())

    with torch.no_grad():
        dummy = model(coords[:1])
    C = dummy.shape[-1]

    if batch_size is None:
        def f_params(*thetas):
            theta_dict = dict(zip(param_names, thetas))
            return functional_call(model, theta_dict, (coords,)).reshape(-1)  # (N*C,)

        J_tuple = torch.autograd.functional.jacobian(f_params, theta_tuple)
        J = torch.cat([j.reshape(N * C, -1) for j in J_tuple], dim=-1)  # (N*C, n_params)
        return (J @ J.T).detach()

    # ── Batched path ──────────────────────────────────────────────────────
    # Compute J in row-batches, offload each block to CPU, then assemble K.
    Js = []  # list of CPU tensors, each (n_b*C, n_params)
    for start in range(0, N, batch_size):
        batch = coords[start:start + batch_size]
        n_b = batch.shape[0]

        def f_batch(*thetas, _b=batch, _n=n_b):
            theta_dict = dict(zip(param_names, thetas))
            return functional_call(model, theta_dict, (_b,)).reshape(-1)  # (n_b*C,)

        J_tuple = torch.autograd.functional.jacobian(f_batch, theta_tuple)
        J_block = torch.cat([j.reshape(n_b * C, -1) for j in J_tuple], dim=-1)
        Js.append(J_block.detach().cpu())

    # Assemble K = J_all @ J_all.T block by block on the GPU.
    K = torch.zeros(N * C, N * C)
    row = 0
    for J_i in Js:
        n_i = J_i.shape[0]
        J_i_gpu = J_i.to(device)
        col = 0
        for J_j in Js:
            n_j = J_j.shape[0]
            K[row:row + n_i, col:col + n_j] = (J_i_gpu @ J_j.to(device).T).cpu()
            col += n_j
        row += n_i

    return K


def fft_ntk_eigenvalues(
    ntk_model: NeuralField,
    neural_isometry,
    nyquist: int,
    pushforward_kwargs: dict,
) -> torch.Tensor:
    """Fast NTK eigenvalue grid via FFT and vectorized Jacobian (2-D domains only).

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
