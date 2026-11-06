import torch


class SquareSampler:
    """Sample points from the unit hypercube ``[0, 1]^d``.

    ``n_per_dim`` denotes the number of samples along each axis.  Stratified
    sampling returns the Cartesian product of cell centers (optionally
    jittered); unstratified sampling returns the same total number of i.i.d.
    uniform points.
    """

    def __init__(
        self,
        domain_dim: int = 2,
        stratified: bool = False,
        add_noise: bool = True,
    ):
        if domain_dim < 1:
            raise ValueError(f"domain_dim must be at least 1; got {domain_dim}")
        self.domain_dim = int(domain_dim)
        self.stratified = bool(stratified)
        self.add_noise = bool(add_noise)

    def sample(self, n_per_dim: int) -> torch.Tensor:
        """Return ``n_per_dim**domain_dim`` points in ``[0, 1]^domain_dim``."""
        if n_per_dim < 1:
            raise ValueError(f"n_per_dim must be at least 1; got {n_per_dim}")

        if not self.stratified:
            return torch.rand(n_per_dim**self.domain_dim, self.domain_dim)

        centers = (torch.arange(n_per_dim, dtype=torch.get_default_dtype()) + 0.5) / n_per_dim
        mesh = torch.meshgrid(*((centers,) * self.domain_dim), indexing="ij")
        coords = torch.stack([axis.reshape(-1) for axis in mesh], dim=-1)
        if self.add_noise:
            coords += (torch.rand_like(coords) - 0.5) / n_per_dim
        return coords
