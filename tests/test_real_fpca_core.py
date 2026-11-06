from functools import partial
import math

import pytest
import torch

from infidictionary.checkpointing import Checkpointer
from infidictionary.datasets import ImplicitZooCIFARDataset, OneDimDiscontinuousGenerator
from infidictionary.domain_samplers import SquareSampler
from infidictionary.dictionaries import FourierDictionary, HaarWaveletDictionary, VoronoiPWC
from infidictionary.neural_isometries import (
    ChainedIsometry,
    ChannelOrthogonalIsometry,
    EulerianIsometry,
)
from infidictionary.networks import NerfConditionalField, NerfNeuralField
from infidictionary.utils import norm2, pairwise_inner_product


def _eulerian(**kwargs):
    iso = EulerianIsometry(
        1, 1, 4, 10.0, lambda **kw: NerfConditionalField(**kw), **kwargs
    )
    iso.shuffle_model_state(num_steps=8)
    iso.eval()
    return iso


def test_eulerian_pushforward_preserves_the_l2_norm():
    # The defining property: the flow is orthogonal, so the weighted L² norm is
    # unchanged. This is what silently broke when the solve was approximated.
    torch.manual_seed(0)
    iso = _eulerian()
    coords, lad, f = torch.rand(64, 1), torch.zeros(64), torch.randn(3, 64, 1)
    with torch.no_grad():
        _, _, out = iso.pushforward(coords, lad, f, 0.0, 1.0)
    assert (norm2(out, lad) / norm2(f, lad)).tolist() == pytest.approx([1.0] * 3, abs=1e-3)


def test_pullback_inverts_pushforward():
    torch.manual_seed(0)
    iso = _eulerian()
    coords, lad, f = torch.rand(64, 1), torch.zeros(64), torch.randn(2, 64, 1)
    with torch.no_grad():
        _, _, fwd = iso.pushforward(coords, lad, f, 0.0, 1.0)
        _, _, back = iso.pullback(coords, lad, fwd, 0.0, 1.0)
    assert torch.allclose(back, f, atol=1e-3)


def test_real_multichannel_fourier_atoms_and_energy():
    coords = torch.rand(64, 2)
    dictionary = FourierDictionary(domain_dim=2, num_channels=3, steepness=2.0)
    atoms = dictionary.get_atoms(coords, dictionary.get_lowpass_indices(1))
    assert atoms.shape[-1] == 3
    assert not atoms.is_complex()
    values = torch.rand(4, 64, 3)
    energy = dictionary.monte_carlo_captured_energy(coords, torch.zeros(64), values, 8)
    assert energy.shape == (4,)
    assert not energy.is_complex()


def test_real_orthogonal_channel_mixing_preserves_gram():
    coords, field = torch.rand(48, 2), torch.rand(3, 48, 2)
    isometry = ChannelOrthogonalIsometry(coords_dim=2, channels_dim=2)
    _, _, pushed = isometry.pushforward(coords, torch.zeros(48), field)
    assert not pushed.is_complex()
    assert torch.allclose(pairwise_inner_product(field, field), pairwise_inner_product(pushed, pushed), atol=1e-5)


def test_shared_nerf_fields_are_real_and_differentiable():
    coords = torch.rand(12, 2, requires_grad=True)
    mean_field = NerfNeuralField(
        coords_dim=2, output_dim=3, hidden_dims=(8,), nerf_n_levels=2, nerf_n_base=4
    )
    conditional_field = NerfConditionalField(
        coords_dim=2,
        output_dim=3,
        rank=2,
        cond_dim=5,
        hidden_dims=(8,),
        nerf_n_levels=2,
        nerf_n_base=4,
    )

    mean = mean_field(coords)
    generated = conditional_field(torch.rand(12, 5), coords)
    (mean.square().mean() + generated.square().mean()).backward()

    assert mean.shape == (12, 3) and not mean.is_complex()
    assert generated.shape == (12, 2, 3) and not generated.is_complex()
    assert coords.grad is not None


def test_chained_isometry_forwards_only_supported_state_and_time_arguments():
    chain = ChainedIsometry(
        coords_dim=1,
        channels_dim=2,
        isometries_partial=(
            partial(ChannelOrthogonalIsometry),
            partial(
                EulerianIsometry,
                rank=2,
                base_acceleration=1.0,
                scalar_field_partial=lambda **kwargs: NerfConditionalField(**kwargs),
            ),
        ),
    )
    chain.shuffle_model_state(num_steps=2)
    assert chain.isometries[1]._num_steps == 2

    coords, field = torch.rand(16, 1), torch.rand(3, 16, 2)
    pushed = chain.pushforward(
        coords, torch.zeros(16), field, rot_start_time=0.0, rot_end_time=1.0
    )
    assert pushed[2].shape == field.shape


def test_channel_mixer_starts_at_a_coordinate_independent_haar_rotation():
    torch.manual_seed(7)
    isometry = ChannelOrthogonalIsometry(coords_dim=2, channels_dim=3)
    rotations = isometry._orthogonal(torch.rand(8, 2))
    expected = isometry.base_orthogonal.expand_as(rotations)
    assert torch.allclose(rotations, expected)
    assert torch.allclose(
        isometry.base_orthogonal.T @ isometry.base_orthogonal,
        torch.eye(3),
        atol=1e-6,
    )


def test_real_eulerian_isometry_and_synthetic_dataset_batch():
    dataset = OneDimDiscontinuousGenerator(domain_sample_size=32, n_functions=16)
    coords, values = dataset.get_batch(3)
    assert coords.shape == (32, 1) and values.shape == (3, 32, 1)
    isometry = EulerianIsometry(1, 1, 2, 1.0, lambda **kwargs: NerfConditionalField(**kwargs))
    isometry.shuffle_model_state(num_steps=2)
    _, _, moved = isometry.pullback(coords, torch.zeros(32), values, 0.0, 1.0)
    assert moved.shape == values.shape and not moved.is_complex()


def test_one_dimensional_synthetic_dataset_uses_shared_sampler():
    dataset = OneDimDiscontinuousGenerator(domain_sample_size=8)
    assert isinstance(dataset.domain_sampler, SquareSampler)
    assert dataset.domain_sampler.domain_dim == 1


def test_cifar_inr_dataset_prefers_extracted_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / "cifar_inrs_dataset" / "train" / "ckpts" / "00"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "params": {
                "seq.0.weight": torch.ones(3, 2),
                "seq.0.bias": torch.zeros(3),
            }
        },
        checkpoint_dir / "000000_0000042.ckpt",
    )

    dataset = ImplicitZooCIFARDataset(
        extracted_dir=str(tmp_path / "cifar_inrs_dataset"),
        n_inrs=1,
        domain_sample_size=2,
    )

    coords, values = dataset.get_batch(1)
    assert coords.shape == (4, 2) and values.shape == (1, 4, 3)


def test_real_voronoi_atoms():
    dictionary = VoronoiPWC(resolution=2, dimension=1, num_channels=2, learn_synthesis=False)
    atoms = dictionary.get_atoms(torch.rand(12, 1), dictionary.sample_indices(2, False))
    assert atoms.shape == (2, 12, 2)
    assert not atoms.is_complex()
    indices = dictionary.get_high_probability_indices(0.0)
    pmfs = dictionary.get_index_pmfs(indices)
    assert torch.isclose(pmfs.sum(), torch.tensor(1.0))
    assert torch.all(pmfs[:-1] > pmfs[1:])
    values = torch.rand(3, 12, 2)
    assert dictionary.monte_carlo_captured_energy(torch.rand(12, 1), torch.zeros(12), values, 8).shape == (3,)


def test_fixed_voronoi_has_no_trainable_synthesis():
    dictionary = VoronoiPWC(resolution=2, dimension=1, num_channels=1, learn_synthesis=False)
    assert list(dictionary.parameters()) == []


def test_regular_grid_voronoi_builds_unit_norm_blocks_and_harmonic_prior():
    dictionary = VoronoiPWC(
        resolution=2,
        dimension=2,
        num_channels=1,
        learn_synthesis=False,
    )

    # Blocks are numbered row-major over (x, y); each indicator has height 2 = 2^{2/2}.
    corners = torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    raw = dictionary.get_atoms(corners, torch.tensor([[l, 0] for l in range(4)]), synthesis=False)
    assert torch.equal(raw[..., 0], 2.0 * torch.eye(4))
    pmfs = dictionary.get_index_pmfs(dictionary.get_high_probability_indices(0.0))
    assert torch.all(pmfs[:-1] > pmfs[1:])
    assert torch.isclose(pmfs.sum(), torch.tensor(1.0))


def test_haar_1d_atoms_and_pmf():
    dictionary = HaarWaveletDictionary(domain_dim=1, num_channels=1, learn_synthesis=False)
    coords = torch.tensor([[0.125], [0.375], [0.625], [0.875]])
    idx = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]])
    atoms = dictionary.get_atoms(coords, idx, synthesis=False).squeeze(-1)
    assert torch.equal(atoms[0], torch.ones(4))
    assert torch.equal(atoms[1], torch.tensor([1.0, 1.0, -1.0, -1.0]))
    assert torch.equal(atoms[2], torch.tensor([0.0, 0.0, math.sqrt(2), -math.sqrt(2)]))
    pmfs = dictionary.get_index_pmfs(idx)
    assert torch.allclose(pmfs, torch.tensor([0.5, 0.25, 0.0625]))


def test_haar_2d_orientation_and_threshold_indices():
    dictionary = HaarWaveletDictionary(domain_dim=2, num_channels=1, learn_synthesis=False)
    coords = torch.tensor([[0.25, 0.25], [0.75, 0.25]])
    # Orientation 1 applies the wavelet on x only; orientation 2 on y only.
    atoms = dictionary.get_atoms(
        coords,
        torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 2, 0]]),
        synthesis=False,
    ).squeeze(-1)
    assert torch.equal(atoms[0], torch.tensor([1.0, -1.0]))
    assert torch.equal(atoms[1], torch.tensor([1.0, 1.0]))
    indices = dictionary.get_high_probability_indices(1e-2)
    assert indices.shape[1] == 5
    assert torch.all(dictionary.get_index_pmfs(indices) >= 1e-2)


def test_learnable_synthesis_starts_as_orthogonal_mixing():
    torch.manual_seed(7)
    dictionary = FourierDictionary(domain_dim=1, num_channels=2, synthesis_tail_probability=1e-2)
    weight = dictionary._synthesis.weight.detach()
    assert not torch.allclose(weight, torch.eye(weight.shape[0]))
    assert torch.allclose(weight @ weight.T, torch.eye(weight.shape[0]), atol=1e-6)


def test_fourier_can_bypass_synthesis_and_reconstruct_with_raw_atoms():
    torch.manual_seed(3)
    dictionary = FourierDictionary(
        domain_dim=1,
        num_channels=1,
        synthesis_tail_probability=1e-2,
    )
    coords = torch.linspace(0.0, 1.0, 64).unsqueeze(-1)
    indices = dictionary._synthesis_indices[:4]

    raw_atoms = dictionary.get_atoms(coords, indices, synthesis=False)
    synthesized_atoms = dictionary.get_atoms(coords, indices)

    assert torch.allclose(raw_atoms, dictionary._get_base_atoms(coords, indices))
    assert not torch.allclose(raw_atoms, synthesized_atoms)


def test_voronoi_can_bypass_synthesis_with_cell_channel_indices():
    dictionary = VoronoiPWC(resolution=2, dimension=1, num_channels=2, learn_synthesis=True)
    coords = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
    raw_indices = torch.tensor([[0, 1], [1, 0]])

    raw_atoms = dictionary.get_atoms(coords, raw_indices, synthesis=False)

    expected = 2.0 ** 0.5 * torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    assert torch.allclose(raw_atoms, expected)


@pytest.mark.parametrize(
    "make_dictionary",
    [
        lambda: FourierDictionary(domain_dim=1, num_channels=1, synthesis_tail_probability=1e-2),
        lambda: HaarWaveletDictionary(domain_dim=1, num_channels=1, synthesis_tail_probability=1e-2),
        lambda: VoronoiPWC(resolution=2, dimension=1, num_channels=1, learn_synthesis=True),
    ],
    ids=["fourier", "haar", "voronoi"],
)
def test_trainable_dictionary_synthesis_is_checkpointed(tmp_path, make_dictionary):
    dictionary = make_dictionary()
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=1e-2)
    dictionary._synthesis.weight.sum().backward()
    optimizer.step()
    expected_state = {name: value.detach().clone() for name, value in dictionary.state_dict().items()}

    checkpoint = Checkpointer(
        checkpoint_dir=str(tmp_path / "source"),
        models={"initial_dictionary": dictionary},
        optimizers={"dictionary": optimizer},
        schedulers={"dictionary": None},
    )._build_checkpoint(epoch=0, metric=0.0)

    restored_dictionary = make_dictionary()
    restored_optimizer = torch.optim.Adam(restored_dictionary.parameters(), lr=1e-2)
    Checkpointer(
        checkpoint_dir=str(tmp_path / "restored"),
        models={"initial_dictionary": restored_dictionary},
        optimizers={"dictionary": restored_optimizer},
        schedulers={"dictionary": None},
    ).restore(checkpoint)

    actual_state = restored_dictionary.state_dict()
    assert all(torch.equal(actual_state[name], value) for name, value in expected_state.items())


def test_training_time_grid_spans_the_full_interval():
    # Training must integrate the same [0, 1] flow that eval does; only the
    # interior nodes are random.
    iso = EulerianIsometry(1, 1, 4, 10.0, lambda **kw: NerfConditionalField(**kw))
    iso.train()
    for _ in range(5):
        iso.shuffle_model_state(num_steps=6)
        assert iso.tspan.shape == (6,)
        assert iso.tspan[0] == 0.0 and iso.tspan[-1] == 1.0
        assert torch.all(iso.tspan[1:] >= iso.tspan[:-1])


@pytest.mark.parametrize("domain_dim,steepness,tail", [(1, 2.0, 1e-4), (2, 2.0, 1e-4), (2, 1.01, 1e-4), (3, 2.0, 1e-4)])
def test_fourier_high_probability_set_is_complete(domain_dim, steepness, tail):
    dictionary = FourierDictionary(domain_dim, 1, steepness=steepness, m_max=64, learn_synthesis=False)
    got = dictionary.get_high_probability_indices(tail)
    box = dictionary._box_indices(20)
    expected = box[dictionary.get_index_pmfs(box) >= tail]
    assert got.shape[0] == expected.shape[0]
    assert {tuple(r) for r in got.tolist()} == {tuple(r) for r in expected.tolist()}
