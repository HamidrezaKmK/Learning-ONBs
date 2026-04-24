
from .base import PushforwardRegularizer
import torch


class TVMaterialRegularizer(PushforwardRegularizer):
    """Sparse TV penalty inside a material mask using a KNN graph.

    Using a ``ConstantMaterial`` (mass = 1 everywhere) makes the penalty
    apply globally.

    Args:
        domain_sampler:    A ``DomainSampler`` providing quadrature points.
        domain_sample_size: Passed as n_per_dim to domain_sampler.sample.
        material:          Any ``Material`` used to derive the inside mask.
        mass_threshold:    Points with ``mass >= mass_threshold`` are *inside*.
        sigma:             Gaussian bandwidth for the neighbourhood edge weights.
        k:                 Number of nearest neighbours per point in the KNN graph.
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        material,
        mass_threshold: float = 0.5,
        sigma: float = 0.05,
        k: int = 8,
    ):
        super().__init__(domain_sampler, domain_sample_size)
        self.material = material
        self.mass_threshold = mass_threshold
        self.sigma = sigma
        self.k = k
        self._edge_w:   torch.Tensor | None = None
        self._w_inside: torch.Tensor | None = None
        self._src:      torch.Tensor | None = None
        self._dst:      torch.Tensor | None = None

    @staticmethod
    def _build_knn_edges(coords: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        import faiss
        import numpy as np

        xy = coords.detach().cpu().float().contiguous().numpy()
        N, d = xy.shape
        index = faiss.IndexFlatL2(d)
        index.add(xy)
        _, I = index.search(xy, k + 1)
        I = I[:, 1:]
        src = np.repeat(np.arange(N), k)
        dst = I.reshape(-1)
        return (
            torch.from_numpy(src).long(),
            torch.from_numpy(dst).long(),
        )

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        coords = self._coords
        with torch.no_grad():
            src, dst = self._build_knn_edges(coords, self.k)
            edge_sq_dist = (coords[src] - coords[dst]).pow(2).sum(dim=-1)
            edge_w = torch.exp(-edge_sq_dist / (2.0 * self.sigma ** 2))
            _, mass = self.material(coords)
            inside = (mass >= self.mass_threshold).float()
            self._src      = src
            self._dst      = dst
            self._edge_w   = edge_w
            self._w_inside = edge_w * inside[src] * inside[dst]

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        if self._src is None:
            raise RuntimeError(
                "TVMaterialRegularizer.update_coordinates must be called before compute_energy."
            )
        device = pushed.device
        src      = self._src.to(device)
        dst      = self._dst.to(device)
        w_inside = self._w_inside.to(device)
        edge_norms = (pushed[:, src, :] - pushed[:, dst, :]).norm(dim=-1)  # (A, E)
        return (w_inside[None] * edge_norms).mean(dim=-1)                   # (A,)
