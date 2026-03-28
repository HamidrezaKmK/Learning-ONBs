
from .base import PushforwardRegularizer
import torch

class EntropyRegularizer(PushforwardRegularizer):
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        val_diffs = pushed.unsqueeze(2) - pushed.unsqueeze(1)  # (A, N, N, C)
        val_norms_sq = val_diffs.pow(2).sum(dim=-1)            # (A, N, N)
        kde = torch.exp(-val_norms_sq / (2 * self.sigma ** 2))
        p_x = kde.mean(dim=-1) + 1e-8                          # (A, N)
        return -torch.mean(torch.log(p_x), dim=-1)             # (A,)


class GraphLaplacianRegularizer(PushforwardRegularizer):
    def __init__(self, sigma: float = 0.1, neighbourhood_r: float = 0.1):
        self.sigma = sigma
        self.neighbourhood_r = neighbourhood_r

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        distances = torch.cdist(tgt_coords, tgt_coords)
        weights = torch.where(
            distances < self.neighbourhood_r,
            torch.exp(-distances.pow(2) / (2 * self.sigma ** 2)),
            torch.zeros_like(distances),
        )
        weights.fill_diagonal_(0)
        laplacian = torch.diag(weights.sum(dim=1)) - weights  # (N, N)
        return torch.einsum("anc,nm,amc->a", pushed, laplacian, pushed) / laplacian.shape[0]


class TVMaterialRegularizer(PushforwardRegularizer):
    """Sparse TV penalty inside a material mask using a KNN graph.

    Using a ``ConstantMaterial`` (mass = 1 everywhere) makes the penalty
    apply globally.

    Args:
        material:        Any ``Material`` used to derive the inside mask.
        mass_threshold:  Points with ``mass >= mass_threshold`` are *inside*.
        sigma:           Gaussian bandwidth for the neighbourhood edge weights.
        k:               Number of nearest neighbours per point in the KNN graph.
    """

    def __init__(self, material, mass_threshold: float = 0.5, sigma: float = 0.05, k: int = 8):
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
        device = coords.device
        return (
            torch.from_numpy(src).long().to(device),
            torch.from_numpy(dst).long().to(device),
        )

    def update_coordinates(self, coords: torch.Tensor) -> None:
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
        src, dst = self._src, self._dst
        edge_norms = (pushed[:, src, :] - pushed[:, dst, :]).norm(dim=-1)  # (A, E)
        return (self._w_inside[None] * edge_norms).mean(dim=-1)            # (A,)
