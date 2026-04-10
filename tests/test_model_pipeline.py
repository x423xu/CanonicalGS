import torch

from canonicalgs.config import ModelConfig
from canonicalgs.model import CanonicalGsPipeline, CanonicalReadout, CanonicalState, LocalGaussianReadout, PosteriorReadout
from canonicalgs.model.evidence_writer import VoxelEvidenceWriter
from canonicalgs.model.view_encoder import CostVolumeDepthPredictor, SymmetricMultiViewEncoder, ViewEncoderOutput


def make_intrinsics(num_views: int) -> torch.Tensor:
    intrinsics = torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1)
    intrinsics[:, 0, 0] = 1.0
    intrinsics[:, 1, 1] = 1.0
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    return intrinsics


def make_extrinsics(num_views: int) -> torch.Tensor:
    extrinsics = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1)
    extrinsics[:, 0, 3] = torch.linspace(0.0, 0.4, num_views)
    extrinsics[:, 1, 3] = torch.linspace(0.0, 0.2, num_views)
    return extrinsics


def _test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_multiview_encoder_is_permutation_invariant() -> None:
    torch.manual_seed(0)
    device = _test_device()
    model = SymmetricMultiViewEncoder(
        feature_dim=16,
        appearance_dim=8,
        dinov2_pretrained=False,
        allow_dinov2_fallback=True,
    ).to(device)
    model.eval()

    images = torch.rand(3, 3, 32, 32, device=device)
    intrinsics = make_intrinsics(3).to(device)
    extrinsics = make_extrinsics(3).to(device)

    reference = model(images, intrinsics, extrinsics)
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    inverse = torch.argsort(permutation)
    permuted = model(images[permutation], intrinsics[permutation], extrinsics[permutation])

    assert torch.allclose(reference.depth, permuted.depth[inverse], atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        reference.positional_certainty,
        permuted.positional_certainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        reference.appearance_certainty,
        permuted.appearance_certainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        reference.combined_certainty,
        permuted.combined_certainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )


def test_appearance_certainty_head_uses_default_random_init() -> None:
    model = SymmetricMultiViewEncoder(
        feature_dim=16,
        appearance_dim=8,
        dinov2_pretrained=False,
        allow_dinov2_fallback=True,
        appearance_uncertainty_init=0.05,
    )
    final_layer = model.appearance_certainty_head[-1]

    assert not torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight))
    probe = torch.zeros(1, final_layer.in_channels, 4, 4)
    certainty = torch.sigmoid(final_layer(probe))
    assert certainty.mean().item() > 0.0
    assert certainty.mean().item() < 1.0


def test_cost_volume_aggregation_preserves_single_strong_match() -> None:
    predictor = CostVolumeDepthPredictor(
        feature_dim=8,
        num_depth_bins=4,
        min_depth=0.25,
        max_depth=8.0,
        min_depth_uncertainty=0.05,
        cost_volume_temperature=0.35,
        cost_volume_visibility_beta=0.2,
    )
    stacked_costs = torch.tensor(
        [
            [[[0.05]]],
            [[[1.5]]],
            [[[1.6]]],
        ],
        dtype=torch.float32,
    )
    stacked_visibilities = torch.ones_like(stacked_costs)

    aggregated = predictor._aggregate_source_costs(stacked_costs, stacked_visibilities)
    visible_mean = stacked_costs.mean(dim=0, keepdim=True)

    assert aggregated.shape == (1, 1, 1, 1)
    assert aggregated.item() < visible_mean.item()


def test_warp_feature_volume_matches_single_source_calls_when_batched() -> None:
    predictor = CostVolumeDepthPredictor(
        feature_dim=4,
        num_depth_bins=3,
        min_depth=0.5,
        max_depth=4.0,
        min_depth_uncertainty=0.05,
    )
    source_features = torch.rand(2, 4, 4, 4, dtype=torch.float32)
    inverse_depth_bins = torch.linspace(1.0 / 4.0, 1.0 / 0.5, 3, dtype=torch.float32)
    reference_intrinsics = make_intrinsics(1).expand(2, -1, -1).clone()
    reference_extrinsics = make_extrinsics(1).expand(2, -1, -1).clone()
    source_intrinsics = make_intrinsics(2).clone()
    source_world_to_camera = torch.linalg.inv(make_extrinsics(2))

    warped_batched, visibility_batched = predictor._warp_feature_volume(
        source_features,
        inverse_depth_bins,
        reference_intrinsics,
        reference_extrinsics,
        source_intrinsics,
        source_world_to_camera,
        (4, 4),
    )

    warped_single = []
    visibility_single = []
    for index in range(2):
        warped, visibility = predictor._warp_feature_volume(
            source_features[index : index + 1],
            inverse_depth_bins,
            reference_intrinsics[index : index + 1],
            reference_extrinsics[index : index + 1],
            source_intrinsics[index : index + 1],
            source_world_to_camera[index : index + 1],
            (4, 4),
        )
        warped_single.append(warped)
        visibility_single.append(visibility)

    assert torch.allclose(warped_batched, torch.cat(warped_single, dim=0), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        visibility_batched,
        torch.cat(visibility_single, dim=0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pair_cost_dot_product_matches_cosine_similarity_for_normalized_features() -> None:
    ref_feature = torch.rand(1, 4, 3, 3, dtype=torch.float32)
    warped = torch.rand(2, 4, 5, 3, 3, dtype=torch.float32)

    ref_normalized = torch.nn.functional.normalize(ref_feature, dim=1)
    warped_normalized = torch.nn.functional.normalize(warped, dim=1)

    cosine_cost = 1.0 - torch.nn.functional.cosine_similarity(
        ref_normalized.unsqueeze(2).expand_as(warped_normalized),
        warped_normalized,
        dim=1,
    )
    dot_cost = 1.0 - (warped_normalized * ref_normalized.unsqueeze(2)).sum(dim=1)

    assert torch.allclose(dot_cost, cosine_cost, atol=1e-5, rtol=1e-5)


def test_local_gaussian_readout_filters_low_certainty_cells() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        appearance_dim=4,
        decoder_hidden_dim=16,
        gaussian_scale_min=0.02,
        gaussian_scale_max=0.25,
    )
    readout = PosteriorReadout(
        indices=torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.long),
        canonical_certainty=torch.tensor([1.2, 0.8, 0.3]),
        positional_certainty=torch.tensor([0.9, 0.8, 0.4]),
        appearance_certainty=torch.tensor([0.8, 0.7, 0.4]),
        semantic_consistency=torch.tensor([0.95, 0.8, 0.3]),
        geometry_mean=torch.tensor(
            [[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
            dtype=torch.float32,
        ),
        geometry_variance=torch.tensor(
            [[0.001, 0.002, 0.003], [0.01, 0.01, 0.01], [0.05, 0.05, 0.05]],
            dtype=torch.float32,
        ),
        canonical_features=torch.tensor(
            [[0.2, 0.4, 0.1, 0.8], [0.3, 0.5, 0.2, 0.1], [0.1, 0.2, 0.4, 0.6]],
            dtype=torch.float32,
        ),
        canonical_color=torch.tensor(
            [[0.9, 0.1, 0.2], [0.2, 0.7, 0.4], [0.1, 0.3, 0.8]],
            dtype=torch.float32,
        ),
        uncertainty=torch.tensor([0.1, 0.2, 0.8]),
    )

    gaussians = LocalGaussianReadout(cfg)(readout)

    assert gaussians.indices.shape[0] == 3
    assert gaussians.covariances.shape == (3, 3, 3)
    assert gaussians.opacities[0] > gaussians.opacities[1]
    assert gaussians.opacities[1] > gaussians.opacities[2]
    assert torch.all((gaussians.opacities > 0.0) & (gaussians.opacities < 1.0))

    soft_gaussians = LocalGaussianReadout(cfg)(readout, prune=False)
    assert soft_gaussians.indices.shape[0] == 3
    assert torch.allclose(soft_gaussians.opacities, gaussians.opacities)
    assert torch.all((gaussians.appearance >= 0.0) & (gaussians.appearance <= 1.0))


def test_canonical_readout_keeps_empty_cells_uncertain() -> None:
    state = CanonicalState(
        indices=torch.tensor([[0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([0.0]),
        free_evidence=torch.tensor([0.0]),
        canonical_certainty=torch.tensor([0.0]),
        positional_certainty=torch.tensor([0.0]),
        appearance_certainty=torch.tensor([0.0]),
        semantic_consistency=torch.tensor([0.0]),
        geometry_mean=torch.zeros((1, 3), dtype=torch.float32),
        geometry_variance=torch.zeros((1, 3), dtype=torch.float32),
        canonical_features=torch.zeros((1, 2), dtype=torch.float32),
        canonical_color=torch.zeros((1, 3), dtype=torch.float32),
        feature_weight=torch.zeros((1,), dtype=torch.float32),
    )

    readout = CanonicalReadout()(state)

    assert torch.allclose(readout.canonical_certainty, torch.tensor([0.0]))
    assert torch.allclose(readout.positional_certainty, torch.tensor([0.0]))
    assert torch.allclose(readout.appearance_certainty, torch.tensor([0.0]))
    assert readout.uncertainty.item() >= 0.0


def test_pipeline_forward_returns_gaussians() -> None:
    torch.manual_seed(1)
    device = _test_device()
    cfg = ModelConfig(
        feature_dim=16,
        appearance_dim=8,
        decoder_hidden_dim=16,
        gaussian_scale_min=0.01,
        gaussian_scale_max=0.3,
        dinov2_pretrained=False,
        allow_dinov2_fallback=True,
    )
    pipeline = CanonicalGsPipeline(cfg).to(device)
    pipeline.eval()

    episode = {
        "images": torch.rand(8, 3, 32, 32, device=device),
        "extrinsics": make_extrinsics(8).to(device),
        "intrinsics": make_intrinsics(8).to(device),
        "context_indices": {
            2: torch.tensor([0, 1], dtype=torch.long, device=device),
            3: torch.tensor([0, 1, 2], dtype=torch.long, device=device),
        },
        "target_indices": torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long, device=device),
    }

    result = pipeline(episode, 2, include_render_payload=True)

    assert result["num_context_views"] == 2
    assert result["num_active_cells"] > 0
    assert result["num_gaussians"] > 0
    assert result["gaussians"].means.shape[1] == 3
    assert result["readout"].canonical_features.shape[-1] == 8
    assert result["appearance_certainty"].shape[-2:] == tuple(result["depth"].shape[-2:])


def test_pipeline_builds_prefix_outputs_from_single_max_context() -> None:
    torch.manual_seed(2)
    device = _test_device()
    cfg = ModelConfig(
        feature_dim=16,
        appearance_dim=8,
        decoder_hidden_dim=16,
        gaussian_scale_min=0.01,
        gaussian_scale_max=0.3,
        dinov2_pretrained=False,
        allow_dinov2_fallback=True,
    )
    pipeline = CanonicalGsPipeline(cfg).to(device)
    pipeline.eval()

    episode = {
        "images": torch.rand(8, 3, 32, 32, device=device),
        "extrinsics": make_extrinsics(8).to(device),
        "intrinsics": make_intrinsics(8).to(device),
        "context_indices": {
            2: torch.tensor([0, 1], dtype=torch.long, device=device),
            3: torch.tensor([0, 1, 2], dtype=torch.long, device=device),
            4: torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
        },
        "target_indices": torch.tensor([4, 5, 6, 7], dtype=torch.long, device=device),
    }

    outputs = pipeline.forward_prefixes(
        episode,
        context_sizes=[2, 3, 4],
        include_render_payload=True,
    )

    assert sorted(outputs) == [2, 3, 4]
    assert outputs[2]["num_context_views"] == 2
    assert outputs[4]["num_context_views"] == 4
    assert torch.equal(
        outputs[3]["context_indices"],
        torch.tensor([0, 1, 2], dtype=torch.long, device=device),
    )


def test_voxel_evidence_writer_outputs_consistent_lengths() -> None:
    cfg = ModelConfig(feature_dim=8, appearance_dim=4)
    writer = VoxelEvidenceWriter(cfg)
    device = _test_device()

    encoder_output = ViewEncoderOutput(
        input_rgb=torch.rand(2, 3, 8, 8, device=device),
        appearance_features=torch.rand(2, 4, 8, 8, device=device),
        geometry_features=torch.rand(2, 8, 8, 8, device=device),
        depth=torch.rand(2, 1, 8, 8, device=device) + 1.0,
        density=torch.rand(2, 1, 8, 8, device=device),
        gaussian_raw_params=torch.rand(2, 84, 8, 8, device=device),
        positional_certainty=torch.rand(2, 1, 8, 8, device=device),
        appearance_certainty=torch.rand(2, 1, 8, 8, device=device),
        combined_certainty=torch.rand(2, 1, 8, 8, device=device),
        depth_confidence=torch.rand(2, 1, 8, 8, device=device),
        confidence=torch.rand(2, 1, 8, 8, device=device),
    )
    extrinsics = make_extrinsics(2).to(device)
    intrinsics = make_intrinsics(2).to(device)

    evidence = writer.write(encoder_output, extrinsics, intrinsics)

    assert len(evidence) == 2
    for item in evidence:
        item.validate()


def test_voxel_evidence_writer_uses_single_surface_sample_per_pixel() -> None:
    cfg = ModelConfig(feature_dim=8, appearance_dim=4, free_space_steps=0)
    writer = VoxelEvidenceWriter(cfg)
    device = _test_device()

    encoder_output = ViewEncoderOutput(
        input_rgb=torch.rand(1, 3, 4, 4, device=device),
        appearance_features=torch.rand(1, 4, 4, 4, device=device),
        geometry_features=torch.rand(1, 8, 4, 4, device=device),
        depth=torch.full((1, 1, 4, 4), 2.0, device=device),
        density=torch.full((1, 1, 4, 4), 0.5, device=device),
        gaussian_raw_params=torch.rand(1, 84, 4, 4, device=device),
        positional_certainty=torch.full((1, 1, 4, 4), 0.8, device=device),
        appearance_certainty=torch.full((1, 1, 4, 4), 0.7, device=device),
        combined_certainty=torch.full((1, 1, 4, 4), 0.5, device=device),
        depth_confidence=torch.full((1, 1, 4, 4), 0.8, device=device),
        confidence=torch.full((1, 1, 4, 4), 0.5, device=device),
    )
    extrinsics = make_extrinsics(1).to(device)
    intrinsics = make_intrinsics(1).to(device)

    item = writer.write(encoder_output, extrinsics, intrinsics)[0]
    surface_count = int((item.surface_evidence > 0).sum().item())
    free_count = int((item.free_evidence > 0).sum().item())

    assert surface_count == 16
    assert free_count == 0


def test_sparse_accumulator_averages_voxel_color() -> None:
    from canonicalgs.model.evidence import SparseEvidence, SparseEvidenceAccumulator

    evidence = SparseEvidence(
        indices=torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([1.0, 1.0]),
        free_evidence=torch.tensor([0.0, 0.0]),
        canonical_weight=torch.tensor([1.0, 1.0]),
        positional_certainty=torch.tensor([0.8, 0.8]),
        appearance_certainty=torch.tensor([0.9, 0.9]),
        world_points=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        semantic_features=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    state = SparseEvidenceAccumulator().accumulate([evidence])

    assert torch.allclose(state.canonical_color, torch.tensor([[0.5, 0.0, 0.5]]), atol=1e-6, rtol=0.0)


def test_sparse_accumulator_can_disable_view_merging() -> None:
    from canonicalgs.model.evidence import SparseEvidence, SparseEvidenceAccumulator

    evidence = SparseEvidence(
        indices=torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([1.0, 2.0]),
        free_evidence=torch.tensor([0.0, 0.0]),
        canonical_weight=torch.tensor([0.4, 0.7]),
        positional_certainty=torch.tensor([0.8, 0.6]),
        appearance_certainty=torch.tensor([0.9, 0.5]),
        world_points=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.2]], dtype=torch.float32),
        semantic_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    state = SparseEvidenceAccumulator(merge_views=False).accumulate([evidence])

    assert state.indices.shape[0] == 2
    assert torch.allclose(state.canonical_certainty, torch.tensor([0.4, 0.7]), atol=1e-6, rtol=0.0)
    assert torch.allclose(state.geometry_mean, evidence.world_points, atol=1e-6, rtol=0.0)
    assert torch.allclose(state.canonical_color, evidence.colors, atol=1e-6, rtol=0.0)
    assert torch.allclose(state.semantic_consistency, torch.ones(2), atol=1e-6, rtol=0.0)


def test_sparse_accumulator_returns_incremental_prefix_states() -> None:
    from canonicalgs.model.evidence import SparseEvidence, SparseEvidenceAccumulator

    evidence_a = SparseEvidence(
        indices=torch.tensor([[0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([1.0]),
        free_evidence=torch.tensor([0.0]),
        canonical_weight=torch.tensor([1.0]),
        positional_certainty=torch.tensor([0.8]),
        appearance_certainty=torch.tensor([0.9]),
        world_points=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        semantic_features=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        colors=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )
    evidence_b = SparseEvidence(
        indices=torch.tensor([[0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([2.0]),
        free_evidence=torch.tensor([0.0]),
        canonical_weight=torch.tensor([2.0]),
        positional_certainty=torch.tensor([0.7]),
        appearance_certainty=torch.tensor([0.95]),
        world_points=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
        semantic_features=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        colors=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    evidence_c = SparseEvidence(
        indices=torch.tensor([[1, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([3.0]),
        free_evidence=torch.tensor([0.0]),
        canonical_weight=torch.tensor([3.0]),
        positional_certainty=torch.tensor([0.6]),
        appearance_certainty=torch.tensor([0.85]),
        world_points=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        semantic_features=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        colors=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )

    states = SparseEvidenceAccumulator().accumulate_prefixes(
        [evidence_a, evidence_b, evidence_c],
        [2, 3],
    )

    assert sorted(states) == [2, 3]
    assert states[2].indices.shape[0] == 1
    assert states[3].indices.shape[0] == 2
    assert torch.isclose(states[2].canonical_certainty[0], torch.tensor(3.0))
