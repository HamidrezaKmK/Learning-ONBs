import torch
from abc import ABC, abstractmethod


def _eval_pushed_atoms(
    neural_isometry,
    initial_dictionary,
    tgt_coords: torch.Tensor,
    indices: torch.Tensor,
    pushforward_kwargs: dict,
    src_coords: torch.Tensor | None = None,
    src_logabsdet: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pullback → get atoms → pushforward.

    The pullback is performed without gradient tracking (src_coords are
    detached); gradients flow only through the pushforward, matching the
    convention used throughout the codebase.

    If ``src_coords`` and ``src_logabsdet`` are provided (pre-computed by
    ``update_coordinates``), the pullback step is skipped entirely.

    Returns:
        init:    (A, N, C) initial atoms at src_coords.
        pushed:  (A, N, C) pushed-forward atoms.
        tgt_pf:  (N, d) output target coordinates from the pushforward.
    """
    from infidictionary.neural_isometries import EulerianIsometry

    if src_coords is None:
        N = tgt_coords.shape[0]
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        if isinstance(neural_isometry, EulerianIsometry):
            src_coords = tgt_coords
            src_logabsdet = torch.zeros(N, device=device, dtype=dtype)
        else:
            with torch.no_grad():
                src_coords, src_logabsdet, _ = neural_isometry.pullback(
                    tgt_coords=tgt_coords,
                    tgt_logabsdet=torch.zeros(N, device=device, dtype=dtype),
                    tgt_field=torch.zeros(1, N, 1, device=device, dtype=dtype),
                    **pushforward_kwargs,
                )
            src_coords = src_coords.detach()
            src_logabsdet = src_logabsdet.detach()

    init = initial_dictionary.get_atoms(src_coords, indices)
    tgt_pf, _, pushed = neural_isometry.pushforward(
        src_coords=src_coords,
        src_logabsdet=src_logabsdet,
        src_field=init,
        **pushforward_kwargs,
    )
    return init, pushed, tgt_pf


class Regularizer(ABC):
    """Base class for all regularizers.

    Each regularizer owns its domain sampler and samples coordinates
    internally when ``update_coordinates`` is called.

    Args:
        domain_sampler:    A ``DomainSampler`` used to draw quadrature points.
        domain_sample_size: Passed as ``n_per_dim`` to ``domain_sampler.sample``;
                            the total number of quadrature points is
                            ``domain_sample_size ** d`` where d is the domain
                            dimension.
    """

    def __init__(self, domain_sampler, domain_sample_size: int) -> None:
        self.domain_sampler = domain_sampler
        self.domain_sample_size = domain_sample_size
        self._coords: torch.Tensor | None = None
        self._pulled_back_coords: torch.Tensor | None = None
        self._pulled_back_logabsdet: torch.Tensor | None = None

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        """Sample fresh quadrature coordinates and pre-compute their pullback.

        The pulled-back coordinates are stored in ``self._pulled_back_coords``
        and ``self._pulled_back_logabsdet`` so that ``compute_energy`` can skip
        the pullback step.

        Subclasses may override to additionally pre-compute
        coordinate-dependent quantities (e.g. KNN graphs, diffusivity maps).
        Overrides must call ``super().update_coordinates(...)`` first.
        """
        from infidictionary.neural_isometries import EulerianIsometry

        self._coords = self.domain_sampler.sample(self.domain_sample_size)

        N = self._coords.shape[0]
        device = self._coords.device
        dtype = self._coords.dtype

        if isinstance(neural_isometry, EulerianIsometry):
            self._pulled_back_coords = self._coords.detach()
            self._pulled_back_logabsdet = torch.zeros(N, device=device, dtype=dtype)
        else:
            with torch.no_grad():
                src_coords, src_logabsdet, _ = neural_isometry.pullback(
                    tgt_coords=self._coords,
                    tgt_logabsdet=torch.zeros(N, device=device, dtype=dtype),
                    tgt_field=torch.zeros(1, N, 1, device=device, dtype=dtype),
                    **pushforward_kwargs,
                )
            self._pulled_back_coords = src_coords.detach()
            self._pulled_back_logabsdet = src_logabsdet.detach()

    @abstractmethod
    def compute_energy(
        self,
        neural_isometry,
        initial_dictionary,
        indices: torch.Tensor,
        pushforward_kwargs: dict,
    ) -> torch.Tensor:
        """Per-atom energy.

        Args:
            neural_isometry:    The isometry Q being optimised.
            initial_dictionary: Source atom dictionary.
            indices:            (A, ...) atom multi-indices.
            pushforward_kwargs: Forwarded to pullback / pushforward.

        Returns:
            energy: (A,) per-atom scalar energy.
        """
        pass


class PushforwardRegularizer(Regularizer):
    """Regularizer that evaluates energy on pushed-forward atoms.

    Subclasses implement ``_energy_from_atoms``; the pullback/pushforward
    pipeline is handled once here.
    """

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        device = indices.device
        coords = self._coords.to(device)
        src_coords = self._pulled_back_coords.to(device)
        src_logabsdet = self._pulled_back_logabsdet.to(device)
        init, pushed, tgt_pf = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, coords, indices, pushforward_kwargs,
            src_coords=src_coords, src_logabsdet=src_logabsdet,
        )
        return self._energy_from_atoms(tgt_pf, init, pushed, indices)

    @abstractmethod
    def _energy_from_atoms(
        self,
        tgt_coords: torch.Tensor,
        init: torch.Tensor,
        pushed: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom energy given initial and pushed-forward atoms.

        Args:
            tgt_coords: (N, d) spatial coordinates.
            init:       (A, N, C) initial atoms at src_coords.
            pushed:     (A, N, C) pushed-forward atoms at tgt_coords.
            indices:    (A, ...) atom multi-indices.

        Returns:
            energy: (A,) per-atom energy.
        """
        pass
