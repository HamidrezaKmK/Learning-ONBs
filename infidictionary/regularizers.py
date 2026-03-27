import torch
from abc import ABC, abstractmethod

class Regularizer(ABC):
    def update_coordinates(self, coords: torch.Tensor) -> None:
        """Called whenever the spatial coordinates change.

        Subclasses may override this to pre-compute any coordinate-dependent
        quantities (e.g. KNN graphs, weight matrices) that are expensive to
        rebuild every time ``compute_energy`` is called.  The default
        implementation is a no-op.

        Args:
            coords: (N, d) the new spatial coordinates.
        """
        pass

    @abstractmethod
    def compute_energy(self, coords: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords:  (N, d) spatial coordinates.
            values:  (A, N, C) atom values at those coordinates.
            indices: (A, ...) atom multi-indices (e.g. Fourier multi-indices).
        Returns:
            Per-atom energy, shape (A,).
        """
        pass

class EntropyRegularizer(Regularizer):
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def compute_energy(self, coords: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # values: (A, N, C)
        # pairwise squared L2 distances in value space: (A, N, N)
        val_diffs = values.unsqueeze(2) - values.unsqueeze(1)  # (A, N, N, C)
        val_norms_sq = val_diffs.pow(2).sum(dim=-1)  # (A, N, N)

        kde = torch.exp(-val_norms_sq / (2 * self.sigma**2))  # (A, N, N)
        p_x = kde.mean(dim=-1) + 1e-8  # (A, N)

        entropy = -torch.mean(torch.log(p_x), dim=-1)  # (A,)
        return entropy

class GraphLaplacianRegularizer(Regularizer):
    def __init__(
        self,
        sigma: float = 0.1,
        neighbourhood_r: float = 0.1,
    ):
        self.sigma = sigma
        self.neighbourhood_r = neighbourhood_r

    def compute_energy(self, coords: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # values: (A, N, C)
        distances = torch.cdist(coords, coords)
        weights = torch.where(
            distances < self.neighbourhood_r,
            torch.exp(-distances.pow(2) / (2 * self.sigma ** 2)),
            torch.zeros_like(distances)
        )
        weights.fill_diagonal_(0)

        laplacian = torch.diag(weights.sum(dim=1)) - weights  # (N, N)
        # sum Rayleigh quotients across channels
        energy = torch.einsum("anc,nm,amc->a", values, laplacian, values) / laplacian.shape[0]  # (A,)
        return energy

class TVMaterialRegularizer(Regularizer):
    """Regularizer that applies a sparse TV penalty inside a material mask.

    Uses a ``Material`` to derive an inside mask from its mass map, then
    penalises variation of atom values at coordinates *inside* the material
    (mass >= mass_threshold).  The penalty is uniform across all atoms —
    there is no index-parity selection.

    The TV penalty is the mean weighted |v_i - v_j| over a sparse KNN graph
    of spatial neighbours, restricted to edges where both endpoints lie inside
    the mask.  The KNN graph is built with FAISS on detached coordinates so the
    full N×N distance matrix is never materialised.

    Using a ``ConstantMaterial`` (mass = 1 everywhere) makes the penalty apply
    globally, promoting piecewise-constant atom functions on the whole domain.

    Args:
        material:        Any ``Material`` used to derive the inside mask via its
                         mass map.
        mass_threshold:  Points with ``mass >= mass_threshold`` are *inside*.
        sigma:           Gaussian bandwidth for the neighbourhood edge weights.
        k:               Number of nearest neighbours per point in the KNN graph.
    """

    def __init__(
        self,
        material,
        mass_threshold: float = 0.5,
        sigma: float = 0.05,
        k: int = 8,
    ):
        self.material = material
        self.mass_threshold = mass_threshold
        self.sigma = sigma
        self.k = k

        # Cached by update_coordinates; None until first call.
        self._edge_w: torch.Tensor | None = None
        self._w_inside: torch.Tensor | None = None
        self._src: torch.Tensor | None = None
        self._dst: torch.Tensor | None = None

    @staticmethod
    def _build_knn_edges(coords: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        import faiss
        import numpy as np

        xy = coords.detach().cpu().float().contiguous().numpy()  # (N, d)
        N, d = xy.shape
        index = faiss.IndexFlatL2(d)
        index.add(xy)
        _, I = index.search(xy, k + 1)   # (N, k+1); first column is self
        I = I[:, 1:]                      # (N, k) — exclude self
        src = np.repeat(np.arange(N), k)  # (N*k,)
        dst = I.reshape(-1)               # (N*k,)
        device = coords.device
        return (
            torch.from_numpy(src).long().to(device),
            torch.from_numpy(dst).long().to(device),
        )

    def update_coordinates(self, coords: torch.Tensor) -> None:
        """Pre-compute and cache the KNN graph and masked edge weights.

        Builds the FAISS KNN graph, computes Gaussian edge weights, queries
        the material for the inside mask, and stores the masked weight vector
        ``_w_inside``.  Called once per coordinate update so that
        ``compute_energy`` only indexes into the pre-built edge set.

        Args:
            coords: (N, d) the new spatial coordinates.
        """
        with torch.no_grad():
            src, dst = self._build_knn_edges(coords, self.k)

            edge_sq_dist = (coords[src] - coords[dst]).pow(2).sum(dim=-1)  # (E,)
            edge_w = torch.exp(-edge_sq_dist / (2.0 * self.sigma ** 2))    # (E,)

            _, mass = self.material(coords)                  # (N,)
            inside = (mass >= self.mass_threshold).float()   # (N,)

            self._src     = src
            self._dst     = dst
            self._edge_w  = edge_w
            self._w_inside = edge_w * inside[src] * inside[dst]  # (E,)

    def compute_energy(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-atom sparse TV penalty restricted to the material mask.

        Relies on the KNN graph and edge weights cached by the last call to
        ``update_coordinates``.  Raises if ``update_coordinates`` has not been
        called yet.

        Args:
            coords:  (N, d) spatial coordinates — the actual graph is read from
                     the cache.
            values:  (A, N, C) pushed-forward atom values.
            indices: (A, ...) atom multi-indices (unused; present for API
                     compatibility).
        Returns:
            energy: (A,) per-atom TV penalty.
        """
        if self._src is None:
            raise RuntimeError(
                "TVMaterialRegularizer.update_coordinates must be called before "
                "compute_energy."
            )

        src, dst = self._src, self._dst

        # ── Sparse TV energy, fully vectorised over atoms ─────────────────────
        edge_norms = (values[:, src, :] - values[:, dst, :]).norm(dim=-1)  # (A, E)
        return (self._w_inside[None] * edge_norms).mean(dim=-1)  # (A,)


class FouriererRegularizer(Regularizer):
    """Channel-and-region-constancy regularizer for two-channel Fourier dictionaries.

    Atoms are indexed by ``[kx, ky, ..., c]`` where ``c ∈ {0, 1}`` is the
    dictionary channel.  The regularizer promotes the following structure on the
    pushed-forward atoms (shape ``(N, 2)``):

      * **c = 0**: channel 1 is zero everywhere; channel 0 is zero *outside*
        the material mask.  The atom lives only inside the portrait, in channel 0.
      * **c = 1**: channel 0 is zero everywhere; channel 1 is zero *inside*
        the material mask.  The atom lives only outside the portrait, in channel 1.

    The masking objective uses a binary mask (mass >= mass_threshold).

    Optionally, a graph-Laplacian Dirichlet-energy term can be added to encourage
    smooth (low-frequency) behaviour in the active region of each atom.  The
    Dirichlet energy is normalised by the region's signal energy (Rayleigh-quotient
    style) so that its scale is invariant to atom amplitude and grid size.

    Args:
        material:        A ``Material`` used to derive the inside mask.
        mass_threshold:  Points with ``mass >= mass_threshold`` are *inside*.
        smooth_lambda:   Weight of the Dirichlet smoothness penalty.  Set to 0
                         to disable (no FAISS graph is built in that case).
        sigma:           Gaussian bandwidth for edge weights: exp(-dist²/σ²).
        k:               Number of nearest neighbours per point in the KNN graph.
    """

    def __init__(
        self,
        material,
        mass_threshold: float = 0.5,
        smooth_lambda: float = 0.0,
        sigma: float = 0.05,
        k: int = 8,
    ):
        self.material = material
        self.mass_threshold = mass_threshold
        self.smooth_lambda = smooth_lambda
        self.sigma = sigma
        self.k = k

        self._inside:    torch.Tensor | None = None
        self._outside:   torch.Tensor | None = None
        self._src:       torch.Tensor | None = None
        self._dst:       torch.Tensor | None = None
        self._w_inside:  torch.Tensor | None = None  # edge weights for c=0 smoothing
        self._w_outside: torch.Tensor | None = None  # edge weights for c=1 smoothing

    def update_coordinates(self, coords: torch.Tensor) -> None:
        with torch.no_grad():
            diffusivity, mass = self.material(coords)
            self._inside  = (mass >= self.mass_threshold).float()   # (N,)
            self._outside = 1.0 - self._inside                      # (N,)

            if self.smooth_lambda > 0.0:
                # TODO: move build_knn outside of the TVMaterialRegularizer code
                src, dst = TVMaterialRegularizer._build_knn_edges(coords, self.k)
                edge_sq_dist = (coords[src] - coords[dst]).pow(2).sum(dim=-1)  # (E,)
                # Gaussian kernel scaled by geometric mean of diffusivities
                # TODO: maybe harmonic mean is better
                D_edge = (diffusivity[src] * diffusivity[dst]).sqrt()           # (E,)
                edge_w = torch.exp(-edge_sq_dist / self.sigma ** 2) * D_edge   # (E,)
                self._src       = src
                self._dst       = dst
                self._w_inside  = edge_w * self._inside[src]  * self._inside[dst]   # (E,)
                self._w_outside = edge_w * self._outside[src] * self._outside[dst]  # (E,)

    # TODO: remove coords from this and just set coordinates in the update_coordinates function
    def compute_energy(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-atom masking + optional Laplacian smoothness penalty.

        Args:
            coords:  (N, d) spatial coordinates.
            values:  (A, N, 2) pushed-forward atom values (two channels).
            indices: (A, D+1) atom multiindices; last entry is the channel (0 or 1).
        Returns:
            energy: (A,) per-atom penalty.
        """
        from infidictionary.utils import norm2

        if self._inside is None:
            raise RuntimeError(
                "FouriererRegularizer.update_coordinates must be called before "
                "compute_energy."
            )

        inside  = self._inside[None, :, None]   # (1, N, 1)
        outside = self._outside[None, :, None]  # (1, N, 1)

        v0 = values[:, :, 0:1]  # (A, N, 1)
        v1 = values[:, :, 1:2]  # (A, N, 1)

        # ── Binary-mask objective ─────────────────────────────────────────────
        # c=0: atom lives in ch0 inside the portrait → penalise ch1 everywhere
        #      and ch0 outside the mask.
        # c=1: atom lives in ch1 outside the portrait → penalise ch0 everywhere
        #      and ch1 inside the mask.
        e_ch0 = norm2(v1) + norm2(v0 * outside)  # (A,)
        e_ch1 = norm2(v0) + norm2(v1 * inside)   # (A,)

        # ── Dirichlet smoothness — Rayleigh-quotient normalised (optional) ────
        # The denominator is detached so gradients flow only through the Dirichlet
        # numerator; without detach the denominator would create conflicting
        # gradients that fight the masking term.
        # raw_ch* uses .sum()/N (not .mean()) to match norm2's 1/N normalisation,
        # making the Dirichlet energy grid-size invariant (independent of N).
        if self.smooth_lambda > 0.0 and self._src is not None:
            src, dst = self._src, self._dst
            N = values.shape[1]

            v0_flat  = values[:, :, 0]
            d_in     = v0_flat[:, src] - v0_flat[:, dst]                         # (A, E)
            raw_ch0  = (self._w_inside[None] * d_in.pow(2)).sum(dim=-1) / N      # (A,)
            # den_ch0  = norm2(v0 * inside).clamp(min=1e-8).detach()               # (A,) inside-region energy

            v1_flat  = values[:, :, 1]
            d_out    = v1_flat[:, src] - v1_flat[:, dst]                         # (A, E)
            raw_ch1  = (self._w_outside[None] * d_out.pow(2)).sum(dim=-1) / N    # (A,)
            # den_ch1  = norm2(v1 * outside).clamp(min=1e-8).detach()              # (A,) outside-region energy

            e_ch0 = e_ch0 + self.smooth_lambda * raw_ch0 #  / den_ch0
            e_ch1 = e_ch1 + self.smooth_lambda * raw_ch1 # / den_ch1

        ch0 = (indices[:, -1] == 0)         # (A,)
        return torch.where(ch0, e_ch0, e_ch1)


# ── Previous FouriererRegularizer (binary-mask, unnormalised smoothness) ──────
#
# class FouriererRegularizer(Regularizer):
#     def __init__(self, material, mass_threshold=0.5, smooth_lambda=0.0,
#                  sigma=0.05, k=8):
#         self.material        = material
#         self.mass_threshold  = mass_threshold
#         self.smooth_lambda   = smooth_lambda
#         self.sigma           = sigma
#         self.k               = k
#         self._inside  = None;  self._outside = None
#         self._src     = None;  self._dst     = None
#         self._w_inside = None; self._w_outside = None
#
#     def update_coordinates(self, coords):
#         with torch.no_grad():
#             _, mass = self.material(coords)
#             self._inside  = (mass >= self.mass_threshold).float()
#             self._outside = 1.0 - self._inside
#             if self.smooth_lambda > 0.0:
#                 src, dst = TVMaterialRegularizer._build_knn_edges(coords, self.k)
#                 edge_sq_dist = (coords[src] - coords[dst]).pow(2).sum(dim=-1)
#                 edge_w = torch.exp(-edge_sq_dist / self.sigma ** 2)
#                 self._src       = src;  self._dst       = dst
#                 self._w_inside  = edge_w * self._inside[src]  * self._inside[dst]
#                 self._w_outside = edge_w * self._outside[src] * self._outside[dst]
#
#     def compute_energy(self, coords, values, indices):
#         from infidictionary.utils import norm2
#         v0 = values[:, :, 0:1];  v1 = values[:, :, 1:2]
#         e_ch0 = norm2(v1) + norm2(v0 * self._outside[None, :, None])
#         e_ch1 = norm2(v0) + norm2(v1 * self._inside[None,  :, None])
#         if self.smooth_lambda > 0.0 and self._src is not None:
#             src, dst = self._src, self._dst
#             v0_flat = values[:, :, 0]
#             d_in    = v0_flat[:, src] - v0_flat[:, dst]
#             s_ch0   = (self._w_inside[None]  * d_in.pow(2)).mean(dim=-1)
#             v1_flat = values[:, :, 1]
#             d_out   = v1_flat[:, src] - v1_flat[:, dst]
#             s_ch1   = (self._w_outside[None] * d_out.pow(2)).mean(dim=-1)
#             e_ch0   = e_ch0 + self.smooth_lambda * s_ch0
#             e_ch1   = e_ch1 + self.smooth_lambda * s_ch1
#         ch0 = (indices[:, -1] == 0)
#         return torch.where(ch0, e_ch0, e_ch1)
