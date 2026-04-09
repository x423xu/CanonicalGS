import torch

from canonicalgs.model import CanonicalReadout, SparseEvidence, SparseEvidenceAccumulator


def make_evidence(order: int) -> SparseEvidence:
    base_indices = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        dtype=torch.long,
    )
    if order == 1:
        base_indices = torch.flip(base_indices, dims=[0])

    return SparseEvidence(
        indices=base_indices,
        surface_evidence=torch.tensor([1.0, 0.2, 0.4]),
        free_evidence=torch.tensor([0.1, 0.8, 0.0]),
        canonical_weight=torch.tensor([0.8, 0.15, 0.35]),
        positional_certainty=torch.tensor([0.9, 0.4, 0.7]),
        appearance_certainty=torch.tensor([0.8, 0.6, 0.9]),
        world_points=torch.tensor(
            [[0.2, 0.4, 1.0], [0.5, 0.7, 1.4], [0.25, 0.45, 0.95]],
            dtype=torch.float32,
        ),
        semantic_features=torch.tensor(
            [[1.0, 0.1], [0.2, 0.8], [0.95, 0.05]],
            dtype=torch.float32,
        ),
    )


def test_accumulator_is_commutative() -> None:
    accumulator = SparseEvidenceAccumulator()
    first = accumulator.accumulate([make_evidence(0), make_evidence(1)])
    second = accumulator.accumulate([make_evidence(1), make_evidence(0)])

    assert torch.equal(first.indices, second.indices)
    assert torch.allclose(first.surface_evidence, second.surface_evidence)
    assert torch.allclose(first.free_evidence, second.free_evidence)
    assert torch.allclose(first.canonical_certainty, second.canonical_certainty)
    assert torch.allclose(first.geometry_mean, second.geometry_mean)
    assert torch.allclose(first.canonical_features, second.canonical_features)


def test_canonical_certainty_grows_with_more_evidence() -> None:
    accumulator = SparseEvidenceAccumulator()
    small = accumulator.accumulate([make_evidence(0)])
    large = accumulator.accumulate([make_evidence(0), make_evidence(1)])

    lookup_small = {
        tuple(index.tolist()): value for index, value in zip(small.indices, small.canonical_certainty)
    }
    lookup_large = {
        tuple(index.tolist()): value for index, value in zip(large.indices, large.canonical_certainty)
    }

    for key, value in lookup_small.items():
        assert lookup_large[key] >= value


def test_readout_produces_canonical_posterior_quantities() -> None:
    accumulator = SparseEvidenceAccumulator()
    state = accumulator.accumulate([make_evidence(0)])
    readout = CanonicalReadout()(state)

    assert readout.support_probability.shape[0] == state.indices.shape[0]
    assert torch.all(readout.support_probability >= 0.0)
    assert torch.all(readout.support_probability <= 1.0)
    assert torch.all(readout.unknown_probability >= 0.0)
    assert torch.all(readout.geometry_variance >= 0.0)
    assert torch.all(readout.canonical_certainty >= 0.0)
    assert torch.all(readout.semantic_consistency >= 0.0)
    assert torch.all(readout.semantic_consistency <= 1.0)
