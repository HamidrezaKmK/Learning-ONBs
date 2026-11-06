import pytest
import torch

from infidictionary.domain_samplers import SquareSampler


@pytest.mark.parametrize("domain_dim", [1, 2, 3])
@pytest.mark.parametrize("stratified", [False, True])
def test_shape_and_range(domain_dim, stratified):
    sampler = SquareSampler(domain_dim=domain_dim, stratified=stratified, add_noise=False)
    coords = sampler.sample(4)
    assert coords.shape == (4**domain_dim, domain_dim)
    assert coords.min() >= 0.0 and coords.max() <= 1.0


def test_stratified_grid_has_expected_values_per_dimension():
    coords = SquareSampler(domain_dim=3, stratified=True, add_noise=False).sample(5)
    for dim in range(3):
        assert coords[:, dim].unique().numel() == 5


def test_stratified_noise_jitters_grid_points():
    deterministic = SquareSampler(domain_dim=2, stratified=True, add_noise=False).sample(8)
    jittered = SquareSampler(domain_dim=2, stratified=True, add_noise=True).sample(8)
    assert not torch.allclose(deterministic, jittered)
    assert jittered.min() >= 0.0 and jittered.max() <= 1.0


@pytest.mark.parametrize("n_per_dim", [0, -1])
def test_rejects_nonpositive_resolution(n_per_dim):
    with pytest.raises(ValueError, match="n_per_dim"):
        SquareSampler().sample(n_per_dim)


def test_rejects_invalid_dimension():
    with pytest.raises(ValueError, match="domain_dim"):
        SquareSampler(domain_dim=0)
