"""Shared visualization and utility helpers for learning-projections notebooks.

Convention used everywhere: visualisation grids come from
``SquareSampler(stratified=True, add_noise=False).sample(N)``, which lays
coordinates out as ``coords[i*N + j] = (x_i, y_j)`` (i = x-index, j =
y-index). With this layout, ``field_to_img`` and ``to_rgb_image`` both
transpose the reshape so that ``imshow(arr, origin='lower')`` puts y on the
vertical axis and x on the horizontal axis.

For data stored in pixel-grid (yx) layout — e.g. ``CelebADataset.images``
laid out at ``CelebADataset._full_coords`` with y outer, x inner — use
``yx_to_ij`` (and the matching ``coords_yx_to_ij``) to permute into the
SquareSampler convention before passing to any of the helpers below.
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt


# ── Layout conversion (CelebA yx layout ↔ SquareSampler ij layout) ──────────

def yx_to_ij(signal_yx: torch.Tensor, n: int) -> torch.Tensor:
    """Permute a `(..., N=n*n, ...)` signal from yx (y outer) to ij (x outer).

    Inverse of `coords_yx_to_ij` and self-inverse — use the same call to
    convert ij back to yx.

    Works for any tensor whose ``N``-axis sits in position ``-2`` (e.g.
    ``(B, N, C)`` or ``(N, C)``) by collapsing leading and trailing dims.
    """
    s = signal_yx
    pre, post = s.shape[:-2], s.shape[-1:]
    return (s.reshape(*pre, n, n, *post)
              .transpose(-3, -2)
              .reshape(*pre, n * n, *post))


def coords_yx_to_ij(coords_yx: torch.Tensor, n: int) -> torch.Tensor:
    """Same permutation as `yx_to_ij`, but on a ``(N, d)`` coordinate tensor."""
    return coords_yx.reshape(n, n, -1).transpose(0, 1).reshape(n * n, -1)


# ── Display helpers (ij layout from SquareSampler) ─────────────────────────

def field_to_img(f_N1, N: int) -> np.ndarray:
    """(N²,1) or (N²,) tensor → (N, N) float array for imshow(origin='lower').

    ij-indexing: first dim = x (columns), second = y (rows), so we transpose.
    """
    return f_N1.detach().cpu().float().squeeze(-1).reshape(N, N).T.numpy()


def to_rgb_image(field_N3, N: int) -> np.ndarray:
    """(N², 3) tensor → (N, N, 3) float array, globally normalised to [0, 1].

    Same ij-layout convention as ``field_to_img`` — the reshape is followed
    by a (axis 0, axis 1) transpose so that ``imshow(arr, origin='lower')``
    puts the y axis vertical.
    """
    img = field_N3.detach().cpu().float().reshape(N, N, 3).transpose(0, 1)
    lo = img.amin(dim=(0, 1, 2), keepdim=True)
    hi = img.amax(dim=(0, 1, 2), keepdim=True)
    return ((img - lo) / (hi - lo + 1e-8)).numpy()


def render(v_img: np.ndarray, cmap: str = "RdBu_r"):
    """Normalise a (H, W, C) field snapshot for imshow.

    C == 1 → diverging cmap colourmap centred at zero.
    C >  1 → each channel normalised to [0, 1] for RGB display.

    Returns (image, cmap, vmin, vmax) — unpack directly into ax.imshow().
    """
    C = v_img.shape[-1]
    if C == 1:
        vmax = float(np.abs(v_img[:, :, 0]).max()) + 1e-8
        return v_img[:, :, 0], cmap, -vmax, vmax
    lo  = v_img.min(axis=(0, 1), keepdims=True)
    hi  = v_img.max(axis=(0, 1), keepdims=True)
    rng = hi - lo
    display = np.divide(v_img - lo, rng, out=np.full_like(v_img, 0.5), where=rng > 1e-6)
    return np.clip(display, 0, 1), None, None, None


def show_field(ax, signal, n: int, title=None, cmap: str = "RdBu_r"):
    """One-call display of a (N², C) signal in ij layout.

    C=1 → ``field_to_img`` + cmap; C≥2 → first three channels via
    ``to_rgb_image``. Same axes/extent/origin as the rest of the helpers.
    """
    C = signal.shape[-1]
    if C == 1:
        ax.imshow(field_to_img(signal, n), origin="lower",
                  extent=[0, 1, 0, 1], cmap=cmap)
    else:
        ax.imshow(to_rgb_image(signal[..., :3], n), origin="lower",
                  extent=[0, 1, 0, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title, fontsize=8)


# ── PSNR + reconstruction-grid plot ─────────────────────────────────────────

def psnr(recon: torch.Tensor, target: torch.Tensor, peak: float = 1.0) -> float:
    """Peak-signal-to-noise ratio in dB. Assumes signals are in `[0, peak]`."""
    mse = (recon - target).pow(2).mean().item()
    return 10.0 * math.log10((peak ** 2) / (mse + 1e-12))


def plot_recon_grid(methods, val_imgs_cpu, n_atoms_list, n: int, cmap: str = "RdBu_r"):
    """One figure with a `(K+1) x N_VAL` grid per method.

    All signal tensors are expected in ij (SquareSampler) layout.

    methods       : list of `(name, list_of_(L, N, C)_tensors_per_truncation)`
    val_imgs_cpu  : `(L, N, C)` originals
    n_atoms_list  : list of K atom counts (used as row labels)
    n             : grid side length

    Returns ``dict[name -> list[(img_idx, n_atoms, psnr_db)]]``.
    """
    L  = val_imgs_cpu.shape[0]
    K  = len(n_atoms_list)
    M  = len(methods)
    fig = plt.figure(figsize=(L * 2.0 * M + 0.6, (K + 1) * 2.0 + 0.4))
    outer = fig.add_gridspec(1, M, wspace=0.08)
    metrics_out: dict = {}
    for m_idx, (name, recons) in enumerate(methods):
        inner = outer[m_idx].subgridspec(K + 1, L, hspace=0.06, wspace=0.03)
        axes  = np.array([[fig.add_subplot(inner[i, j]) for j in range(L)] for i in range(K + 1)])
        axes[0, L // 2].set_title(name, fontsize=10, pad=6, fontweight="bold")
        for l in range(L):
            show_field(axes[0, l], val_imgs_cpu[l], n, cmap=cmap)
        if m_idx == 0:
            axes[0, 0].set_ylabel("original", fontsize=8, rotation=0, labelpad=24, va="center")
        per_method = []
        for k, (recon_batch, n_atoms) in enumerate(zip(recons, n_atoms_list)):
            if m_idx == 0:
                axes[k + 1, 0].set_ylabel(str(n_atoms), fontsize=7, rotation=0, labelpad=24, va="center")
            for l in range(L):
                ax = axes[k + 1, l]
                p = psnr(recon_batch[l], val_imgs_cpu[l])
                per_method.append((l, n_atoms, p))
                show_field(ax, recon_batch[l], n, cmap=cmap)
                ax.set_title(f"{p:.1f} dB", fontsize=6, pad=1)
        metrics_out[name] = per_method
    plt.tight_layout(); plt.show()
    return metrics_out


# ── Corpus helpers (work for any IrregularDataset) ──────────────────────────

def fetch_validation_signals(function_generator, coords_full: torch.Tensor,
                             n_val: int, device, val_seed_offset: int = 0) -> torch.Tensor:
    """Return `(n_val, N, C)` validation signals on `coords_full`.

    Uses the cached ``.images`` buffer when present (image-style generators
    such as CelebA) — note: those images are stored at the generator's
    ``_full_coords`` (yx) layout. Otherwise evaluates
    ``_eval(coords_full, seed)`` for ``seed`` in
    ``[val_seed_offset, val_seed_offset + n_val)``.
    """
    if hasattr(function_generator, "images"):
        # n_corpus = function_generator.images.shape[0]
        # idx = torch.arange(max(0, n_corpus - n_val), n_corpus)
        idx = torch.arange(0, n_val)
        return function_generator.images[idx].to(device)
    sigs = []
    for seed in range(val_seed_offset, val_seed_offset + n_val):
        v = function_generator._eval(coords_full.cpu(), seed)
        if v.ndim == 1:
            v = v.unsqueeze(-1)
        sigs.append(v.to(device))
    return torch.stack(sigs, dim=0)


def build_corpus(function_generator, coords_full: torch.Tensor,
                 fallback_n: int = 256) -> torch.Tensor:
    """Return an `(M, N, C)` corpus of signals on ``coords_full`` (CPU)."""
    if hasattr(function_generator, "images"):
        return function_generator.images.detach().cpu()
    M = getattr(function_generator, "n_functions", fallback_n)
    sigs = []
    for seed in range(M):
        v = function_generator._eval(coords_full.cpu(), seed)
        if v.ndim == 1:
            v = v.unsqueeze(-1)
        sigs.append(v.detach().cpu())
    return torch.stack(sigs, dim=0)


def pca_reconstruct(val_imgs: torch.Tensor, mean_flat: torch.Tensor,
                    components: torch.Tensor, n_atoms_list, n_full: int,
                    channels: int, device) -> list[torch.Tensor]:
    """Project ``val_imgs`` onto the top-k PCA components for each k in
    ``n_atoms_list``."""
    recons = []
    mean_dev = mean_flat.to(device)
    L = val_imgs.shape[0]
    for n_atoms in n_atoms_list:
        k      = min(n_atoms, components.shape[0])
        comps  = components[:k].to(device)
        flat   = val_imgs.reshape(L, -1)
        coeffs = (flat - mean_dev) @ comps.T
        recon  = coeffs @ comps + mean_dev
        recons.append(recon.reshape(L, n_full, channels).cpu())
    return recons


# ── Atom (k1, k2) mosaic — shared across notebooks ─────────────────────────

def atom_grid_layout(indices, *, per_channel: bool = False):
    """Return ``(k1_vals, k2_vals, c_vals, lookup)`` for an `(A, d+1)` index tensor.

    ``per_channel=False`` → ``lookup[(k1, k2)] = row_idx``, restricted to the
    first channel value found in ``indices[:, -1]``. Used by single-channel
    visualisations (Euclidean, vibrational modes, NTK).

    ``per_channel=True``  → ``lookup[(k1, k2, c)] = row_idx`` over all
    channel values present. Used by RGB / multi-channel mosaics (FPCA RGB).
    """
    idx_np  = (indices.detach().cpu().numpy()
               if isinstance(indices, torch.Tensor) else np.asarray(indices))
    k1_vals = sorted(set(int(v) for v in idx_np[:, 0]))
    k2_vals = sorted(set(int(v) for v in idx_np[:, 1]))
    if per_channel:
        c_vals = sorted(set(int(v) for v in idx_np[:, -1]))
        lookup = {(int(idx_np[i, 0]), int(idx_np[i, 1]), int(idx_np[i, -1])): i
                  for i in range(len(idx_np))}
    else:
        c0     = int(min(int(v) for v in idx_np[:, -1]))
        c_vals = [c0]
        lookup = {(int(idx_np[i, 0]), int(idx_np[i, 1])): i
                  for i in range(len(idx_np)) if int(idx_np[i, -1]) == c0}
    return k1_vals, k2_vals, c_vals, lookup


def plot_atom_mosaic(
    atoms_arr,
    indices,
    title: str,
    *,
    n_vis: int | None = None,
    coords=None,
    hexgridsize: int = 45,
    tile_size: float = 0.6,
    cmap: str = "RdBu_r",
):
    """Render a `(2K-1) × (2K-1)` mosaic of atoms by `(k1, k2)` index.

    Channel handling — fixed across notebooks:
      • ``C == 1`` → single panel, cmap tiles centred at zero.
      • ``C  > 1`` → ``C`` panels side-by-side, one per *input* channel
        (e.g. R/G/B). Each tile is RGB-normalised via ``to_rgb_image``.
        Imshow display only.

    Display modes — pass exactly one:
      • ``n_vis``  → regular ij-grid imshow. Tiles are ``n_vis × n_vis``.
        Used wherever atoms are evaluated on
        ``SquareSampler(stratified=True, add_noise=False).sample(n_vis)``.
      • ``coords`` → hexbin scatter on the supplied ``(N, 2)`` evaluation
        coordinates. Only ``C == 1`` is supported. Used for non-rectangular
        domains (disk material, NTK over an irregular grid).

    This is the **canonical** atom-mosaic helper for all notebooks; please
    reuse it rather than reimplementing per-notebook variants.
    """
    if (n_vis is None) == (coords is None):
        raise ValueError("Pass exactly one of `n_vis` (imshow) or `coords` (hexbin).")

    atoms_t = (atoms_arr if isinstance(atoms_arr, torch.Tensor)
               else torch.from_numpy(np.asarray(atoms_arr)))
    A, N, C = atoms_t.shape

    if coords is not None:
        if C > 1:
            raise ValueError("hexbin mode only supports C = 1 atoms.")
        coords_np = (coords.detach().cpu().numpy()
                     if isinstance(coords, torch.Tensor) else np.asarray(coords))
        gx, gy = coords_np[:, 0], coords_np[:, 1]

    k1_vals, k2_vals, c_vals, lookup = atom_grid_layout(indices, per_channel=(C > 1))

    nk1, nk2, nC = len(k1_vals), len(k2_vals), len(c_vals)
    k1_rev       = list(reversed(k1_vals))

    fig = plt.figure(figsize=(nC * nk2 * tile_size + 0.6, nk1 * tile_size + 0.6))
    subfigs = fig.subfigures(1, nC, wspace=0.04) if nC > 1 else [fig]

    for c_idx, c_input in enumerate(c_vals):
        sf = subfigs[c_idx] if nC > 1 else fig
        if nC > 1:
            ch_label = ["R", "G", "B"][c_input] if C == 3 else f"atom set {c_input}"
            sf.suptitle(f"atom set {c_input}", fontsize=10)
        axes = sf.subplots(nk1, nk2, gridspec_kw={"wspace": 0.03, "hspace": 0.03})
        for ri, k1 in enumerate(k1_rev):
            for ci, k2 in enumerate(k2_vals):
                ax = axes[ri, ci]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                key = (int(k1), int(k2), c_input) if C > 1 else (int(k1), int(k2))
                if key in lookup:
                    a = atoms_t[lookup[key]]                     # (N, C)
                    if coords is not None:
                        vals_np = a.detach().cpu().numpy().squeeze(-1)
                        vmax    = float(np.abs(vals_np).max()) + 1e-8
                        ax.hexbin(gx, gy, C=vals_np, reduce_C_function=np.mean,
                                  gridsize=hexgridsize, cmap=cmap,
                                  vmin=-vmax, vmax=vmax)
                    elif C == 1:
                        img  = field_to_img(a, n_vis)
                        vmax = float(np.abs(img).max()) + 1e-8
                        ax.imshow(img, origin="lower", cmap=cmap,
                                  vmin=-vmax, vmax=vmax)
                    else:
                        ax.imshow(to_rgb_image(a[..., :3], n_vis), origin="lower")
                else:
                    ax.set_facecolor("#ddd")
                ax.set_aspect("equal")
                if ri == nk1 - 1:
                    ax.set_xlabel(str(k2), fontsize=6, labelpad=1)
                if ci == 0:
                    ax.set_ylabel(str(k1), fontsize=6, rotation=0, labelpad=8, va="center")

    fig.suptitle(title, fontsize=12, y=0.995)
    plt.show()


def plot_atoms_comparison(
    atoms_src,
    atoms_tgt,
    n_vis: int,
    n_show: int = 32,
    tile_size: float = 0.55,
    cmap: str = "RdBu_r",
    row_labels: tuple = ("Fourier  $\\varphi_k$", "Learned  $Q\\varphi_k$"),
    title: str = "Top atoms ordered by $\\nu$ (PMF descending)",
):
    """2-row atom comparison: source basis (top) vs. pushed-forward basis (bottom).

    Atoms are displayed left-to-right in the order they are supplied; sort by
    PMF descending before calling to reproduce the standard ν-ordered layout.
    The first ``n_show`` atoms are shown.

    C == 1: tiles rendered with *cmap* centred at zero.
    C  > 1: tiles rendered as normalised RGB images.
    """
    def _as_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))

    atoms_src = _as_tensor(atoms_src)
    atoms_tgt = _as_tensor(atoms_tgt)
    A, N, C = atoms_src.shape
    n_show = min(n_show, A)

    fig, axes = plt.subplots(
        2, n_show,
        figsize=(n_show * tile_size + 1.2, 2 * tile_size + 0.8),
        gridspec_kw={"wspace": 0.03, "hspace": 0.06},
        squeeze=False,
    )

    for row_idx, (atoms, label) in enumerate(
        [(atoms_src, row_labels[0]), (atoms_tgt, row_labels[1])]
    ):
        axes[row_idx, 0].set_ylabel(
            label, fontsize=8, rotation=0, ha="right", va="center", labelpad=5,
        )
        for col_idx in range(n_show):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
            a = atoms[col_idx]   # (N, C)
            if C == 1:
                img  = field_to_img(a, n_vis)
                vmax = float(np.abs(img).max()) + 1e-8
                ax.imshow(img, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(to_rgb_image(a[..., :3], n_vis), origin="lower")
            ax.set_aspect("equal")
            if row_idx == 0:
                ax.set_title(str(col_idx + 1), fontsize=6, pad=2)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def scatter_signal(ax, coords, vals, *, title=None, size_c1: float = 80, size_cgt1: float = 20, cmap: str = "RdBu_r"):
    """Scatter a `(N', C)` signal at `coords (N', 2)` on an existing Axes.

    C=1 → diverging cmap centred at zero; C≥3 → first three channels as RGB
    (clipped to [0, 1]). Used as the irregular-grid fallback wherever the
    notebooks need to display a sparse drop-pattern signal.
    """
    coords_np = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    vals_t    = vals if isinstance(vals, torch.Tensor) else torch.as_tensor(vals)
    v_np      = vals_t.detach().cpu().float().numpy()
    if vals_t.shape[-1] == 1:
        vmax = float(np.abs(v_np).max()) + 1e-8
        ax.scatter(coords_np[:, 0], coords_np[:, 1], c=v_np.squeeze(-1),
                   cmap=cmap, vmin=-vmax, vmax=vmax, s=size_c1)
    else:
        ax.scatter(coords_np[:, 0], coords_np[:, 1],
                   c=np.clip(v_np[:, :3], 0, 1), s=size_cgt1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


# ── Physics helpers ──────────────────────────────────────────────────────────

def periodic_gaussian_mixture(coords, specs, weights=None):
    """Mixture of anisotropic, periodic (torus-aware) Gaussians.

    coords  : (N, 2) coordinates in [0, 1]²
    specs   : list of dicts with keys:
                'center' (x, y), 'angle' (radians), 'sigmas' (sigma_x, sigma_y)
    weights : list of floats (default: uniform 1.0)

    Returns (N, 1) field values (not normalised, max ≈ 1 per component).
    """
    if weights is None:
        weights = [1.0] * len(specs)
    result = torch.zeros(coords.shape[0], 1, device=coords.device, dtype=coords.dtype)
    for spec, w in zip(specs, weights):
        mu = torch.tensor(spec['center'], device=coords.device, dtype=coords.dtype)
        theta = spec['angle']
        sx, sy = spec['sigmas']
        c, s = math.cos(theta), math.sin(theta)
        R_mat = torch.tensor([[c, -s], [s, c]], device=coords.device, dtype=coords.dtype)
        prec  = R_mat @ torch.diag(
            torch.tensor([1.0 / sx**2, 1.0 / sy**2], device=coords.device, dtype=coords.dtype)
        ) @ R_mat.T
        delta = coords - mu
        delta = delta - torch.round(delta)  # shortest path on torus
        maha  = (delta @ prec * delta).sum(dim=-1, keepdim=True)
        result = result + w * torch.exp(-0.5 * maha)
    return result


# ── EulerianIsometry visualisation ──────────────────────────────────────────

def visualize_generator_U(isometry, coords_vis, N_vis: int, timesteps, device, cmap: str = "RdBu_r", trunc: int = 5):
    """(R+1) × len(timesteps) figure: U columns and exp(K_R − K_Rᵀ).

    Rows 0..R-1 : spatial plot of u_r(t, ·) at each timestep.
                  C=1  → grayscale cmap imshow + colorbar.
                  C>1  → RGB imshow (channels globally normalised to [0, 1]).
    Row R       : R×R heatmap of exp(K_R − K_Rᵀ) — the finite rotation
                  generated by K_R at each timestep.

    The displayed columns are the **full** generator field
    ``U(t, x) = function_field(t, x) + Σ_a w_a(t) · φ_a(x)``,
    i.e. ``isometry._assemble_U(t_emb, coords)`` — the same quantity that
    drives the Cayley rotation inside ``_run_euler``. Either path may be
    None, but at least one is always present.

    Parameters
    ----------
    isometry   : EulerianIsometry (.rank, .channels_dim, .time_embedding,
                 ._assemble_U, ._compute_kr)
    coords_vis : (N_vis², d) tensor on `device`, from a no-noise stratified grid
                 in **ij-layout** (i.e. `SquareSampler(stratified=True, add_noise=False)`).
    N_vis      : grid size per dimension
    timesteps  : 1-D tensor of time values to visualise
    device     : torch device
    """
    R   = min(isometry.rank, trunc)
    C   = isometry.channels_dim
    N   = coords_vis.shape[0]
    n_t = len(timesteps)

    fig, axes = plt.subplots(
        nrows=R + 1, ncols=n_t,
        figsize=(4 * n_t, 4 * (R + 1)),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if n_t == 1:
        axes = axes[:, None]

    with torch.no_grad():
        for j, t_val in enumerate(timesteps):
            t_sc    = float(t_val)
            t_batch = torch.full((N,), t_sc, device=device, dtype=coords_vis.dtype)
            t_emb   = isometry.time_embedding(t_batch)            # (N, emb_dim)
            U       = isometry._assemble_U(t_emb, coords_vis)     # (N, R, C)
            if U.shape[-2] > R:
                U = U[:, :R, :]

            t_emb_1 = isometry.time_embedding(t_batch[:1])        # (1, emb_dim)
            K_R     = isometry._compute_kr(t_emb_1)[0]            # (R, R)
            O       = torch.matrix_exp(K_R - K_R.T)               # (R, R) orthogonal

            for r in range(R):
                ax  = axes[r, j]
                u_r = U[:, r, :]  # (N, C)
                if C == 1:
                    im = ax.imshow(
                        field_to_img(u_r, N_vis),
                        origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                    )
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.imshow(to_rgb_image(u_r, N_vis), origin='lower', extent=[0, 1, 0, 1])
                ax.set_title(f'$u_{{{r}}}$,  t={t_sc:.2f}', fontsize=8)
                ax.set_xlabel('x'); ax.set_ylabel('y')
                ax.set_aspect('equal')

            ax = axes[R, j]
            im = ax.imshow(O.cpu().float().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'$\\exp(K_R\\!-\\!K_R^\\top)$,  t={t_sc:.2f}', fontsize=8)
            ax.set_xlabel('col'); ax.set_ylabel('row')

    plt.suptitle(
        'Rows $0..R\\!-\\!1$: columns of $U(t,\\cdot)$;  '
        'Row $R$: finite rotation $\\exp(K_R - K_R^\\top)$',
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


# ── PSNR line chart (mean ± std across signals) ──────────────────────────────

def psnr_matrix(recons_list, ref_cpu):
    """Compute a (L, K) PSNR array.

    Args:
        recons_list: K tensors, each ``(L, N, C)`` — one per truncation level.
        ref_cpu:     ``(L, N, C)`` reference signals on CPU.
    Returns:
        ``np.ndarray`` of shape ``(L, K)``.
    """
    L = ref_cpu.shape[0]
    arr = np.zeros((L, len(recons_list)))
    for k, rb in enumerate(recons_list):
        for l in range(L):
            arr[l, k] = psnr(rb[l], ref_cpu[l])
    return arr


def plot_psnr_line_chart(
    in_methods,
    n_atoms_list,
    ood_methods=None,
    in_label="In-distribution",
    ood_label="Out-of-distribution",
    suptitle="Reconstruction PSNR: in-distribution vs out-of-distribution",
):
    """Side-by-side mean ± std PSNR line plots.

    Args:
        in_methods:    List of ``(name, psnr_mat)`` where ``psnr_mat`` is
                       ``(L, K)`` from :func:`psnr_matrix`.  Optionally a
                       3-tuple ``(name, psnr_mat, color)``.
        n_atoms_list:  K atom counts (x-axis tick labels).
        ood_methods:   Same format as ``in_methods``, or ``None`` to omit the
                       OOD panel.
        in_label:      Subtitle for the in-distribution panel.
        ood_label:     Subtitle for the OOD panel.
        suptitle:      Figure super-title.
    """
    colors = plt.cm.tab10(np.arange(len(in_methods)))
    n_cols = 2 if ood_methods is not None else 1
    fig, axes_row = plt.subplots(1, n_cols, figsize=(8 * n_cols, 5), sharey=False)
    axes_row = np.atleast_1d(axes_row)

    panels = [(axes_row[0], in_methods, in_label)]
    if ood_methods is not None:
        panels.append((axes_row[1], ood_methods, ood_label))

    xs = np.arange(len(n_atoms_list))
    for ax, methods, panel_label in panels:
        N_sig = methods[0][1].shape[0]
        for i, entry in enumerate(methods):
            name, pm = entry[0], entry[1]
            color = entry[2] if len(entry) > 2 else colors[i]
            mean = pm.mean(axis=0)
            std  = pm.std(axis=0)
            ax.plot(xs, mean, marker='o', color=color, label=name, linewidth=2, markersize=5)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels(n_atoms_list)
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(fr"Mean $\pm$ std PSNR — {panel_label}  (N={N_sig})")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Spectral compactness ─────────────────────────────────────────────────────

def plot_spectral_compactness(
    methods,
    title=None,
    xlabel="K  (number of atoms)",
    xscale="log",
    figsize=(12, 3.6),
    ref_lines=(0.50, 0.75, 0.90),
    axes=None,
):
    """Two-panel spectral compactness plot: cumulative energy and log residual.

    Matches the style of the 1-D FPCA notebook (section 7). Both panels share
    a log x-axis; the right panel uses a log y-axis so power-law decay appears
    linear. Optional per-method std bands via ``fill_between``.

    Parameters
    ----------
    methods : list of dicts, each with keys
        label  — legend label
        xs     — 1-D array of x values (atom counts / ranks)
        mean   — mean captured fraction in [0, 1], same length as xs
        std    — std of captured fraction, same length, or None / absent
        color  — matplotlib colour
        marker — marker char ('o', 's', '^', …) or None for line-only
    title    : figure super-title; omitted when None.
    xlabel   : shared x-axis label.
    xscale   : 'log' (default) or 'linear'.
    figsize  : figure size used only when axes=None.
    ref_lines: fraction values in (0, 1) drawn as dashed guides on the left panel.
    axes     : (ax_left, ax_right) to plot into, or None to create a new figure.

    Returns
    -------
    (fig, (ax_left, ax_right))
    ``fig`` is None when *axes* was supplied.
    Call ``plt.tight_layout(); plt.show()`` yourself in both cases.
    """
    from matplotlib.transforms import blended_transform_factory as _btf
    eps = 1e-9

    standalone = axes is None
    if standalone:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = None
        ax_l, ax_r = axes

    for is_right, ax in [(False, ax_l), (True, ax_r)]:
        for m in methods:
            xs    = np.asarray(m["xs"])
            mean  = np.asarray(m["mean"])
            std   = np.asarray(m["std"]) if m.get("std") is not None else None
            color = m["color"]
            marker = m.get("marker")
            ms     = 5 if marker else 0

            if not is_right:
                ax.plot(xs, mean, color=color, marker=marker, lw=1.5, ms=ms,
                        label=m["label"])
                if std is not None:
                    ax.fill_between(xs,
                                    np.clip(mean - std, 0.0, 1.0),
                                    np.clip(mean + std, 0.0, 1.0),
                                    alpha=0.20, color=color)
            else:
                ys = np.clip(1.0 - mean, eps, 1.0)
                ax.plot(xs, ys, color=color, marker=marker, lw=1.5, ms=ms,
                        label=m["label"])
                if std is not None:
                    lo_res = np.clip(1.0 - np.clip(mean + std, 0.0, 1.0), eps, 1.0)
                    hi_res = np.clip(1.0 - np.clip(mean - std, 0.0, 1.0), eps, 1.0)
                    ax.fill_between(xs, lo_res, hi_res, alpha=0.20, color=color)

        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Left: guide lines placed at axes-x=0.98, data-y=frac
    _trans = _btf(ax_l.transAxes, ax_l.transData)
    for frac in ref_lines:
        ax_l.axhline(frac, ls='--', c='grey', alpha=0.45, lw=0.8)
        ax_l.text(0.98, frac + 0.005, f"{frac * 100:.0f}%",
                  transform=_trans, ha='right', va='bottom', fontsize=7, color='grey')
    ax_l.set_ylabel("captured energy fraction")
    ax_l.set_title("Cumulative captured energy  (linear-y)")

    # Right: log y axis
    ax_r.set_yscale("log")
    ax_r.grid(alpha=0.3, which='both')
    ax_r.set_ylabel(r"residual energy  (1 $-$ captured fraction)  $\downarrow$ lower is better")
    ax_r.set_title(r"Residual energy  (log–log)  $\leftarrow$ large-$K$ / Nyquist regime")

    if title is not None:
        _fig = fig if fig is not None else ax_l.get_figure()
        _fig.suptitle(title, fontsize=11)

    return fig, (ax_l, ax_r)
