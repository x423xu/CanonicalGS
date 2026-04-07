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
        confidence=torch.tensor([1.1, 1.0, 0.4]),
        geo_moment_1=torch.tensor([[0.2, 0.4], [0.5, 0.7], [0.2, 0.0]]),
        geo_moment_2=torch.tensor([[0.1, 0.3], [0.5, 0.7], [0.1, 0.0]]),
        app_moment_1=torch.tensor([[0.9, 0.1], [0.3, 0.6], [0.2, 0.5]]),
        app_weight=torch.tensor([1.0, 0.8, 0.4]),
    )


def test_accumulator_is_commutative() -> None:
    accumulator = SparseEvidenceAccumulator()
    first = accumulator.accumulate([make_evidence(0), make_evidence(1)])
    second = accumulator.accumulate([make_evidence(1), make_evidence(0)])

    assert torch.equal(first.indices, second.indices)
    assert torch.allclose(first.surface_evidence, second.surface_evidence)
    assert torch.allclose(first.free_evidence, second.free_evidence)
    assert torch.allclose(first.confidence, second.confidence)
    assert torch.allclose(first.geo_moment_1, second.geo_moment_1)
    assert torch.allclose(first.app_moment_1, second.app_moment_1)


def test_confidence_grows_with_more_evidence() -> None:
    accumulator = SparseEvidenceAccumulator()
    small = accumulator.accumulate([make_evidence(0)])
    large = accumulator.accumulate([make_evidence(0), make_evidence(1)])

    lookup_small = {tuple(index.tolist()): value for index, value in zip(small.indices, small.confidence)}
    lookup_large = {tuple(index.tolist()): value for index, value in zip(large.indices, large.confidence)}

    for key, value in lookup_small.items():
        assert lookup_large[key] >= value


def test_readout_produces_stable_posterior_quantities() -> None:
    accumulator = SparseEvidenceAccumulator()
    state = accumulator.accumulate([make_evidence(0)])
    readout = CanonicalReadout()(state)

    assert readout.support_probability.shape[0] == state.indices.shape[0]
    assert torch.all(readout.support_probability >= 0.0)
    assert torch.all(readout.support_probability <= 1.0)
    assert torch.all(readout.unknown_probability >= 0.0)
    assert torch.all(readout.geometry_variance >= 0.0)
