import torch

from canonicalgs.config import ModelConfig
from canonicalgs.model import CanonicalGsPipeline, CanonicalReadout, CanonicalState, LocalGaussianReadout, PosteriorReadout
from canonicalgs.model.view_encoder import CostVolumeDepthPredictor, SymmetricMultiViewEncoder


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
        reference.depth_uncertainty,
        permuted.depth_uncertainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        reference.appearance_uncertainty,
        permuted.appearance_uncertainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        reference.combined_certainty,
        permuted.combined_certainty[inverse],
        atol=1e-5,
        rtol=1e-4,
    )


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


def test_local_gaussian_readout_filters_low_certainty_cells() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        appearance_dim=4,
        decoder_hidden_dim=16,
        support_threshold=0.6,
        confidence_threshold=0.5,
        gaussian_scale_min=0.02,
        gaussian_scale_max=0.25,
    )
    readout = PosteriorReadout(
        indices=torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.long),
        support_probability=torch.tensor([0.9, 0.55, 0.8]),
        free_probability=torch.tensor([0.1, 0.4, 0.15]),
        unknown_probability=torch.tensor([0.05, 0.1, 0.2]),
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
        uncertainty=torch.tensor([0.1, 0.2, 0.8]),
    )

    gaussians = LocalGaussianReadout(cfg)(readout)

    assert gaussians.indices.shape[0] == 3
    assert gaussians.covariances.shape == (3, 3, 3)
    assert gaussians.opacities[0] > gaussians.opacities[1]
    assert gaussians.opacities[1] > gaussians.opacities[2]

    soft_gaussians = LocalGaussianReadout(cfg)(readout, prune=False)
    assert soft_gaussians.indices.shape[0] == 3
    assert torch.all(soft_gaussians.opacities >= gaussians.opacities)


def test_canonical_readout_keeps_empty_cells_unknown() -> None:
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
        feature_weight=torch.zeros((1,), dtype=torch.float32),
    )

    readout = CanonicalReadout()(state)

    assert torch.allclose(readout.support_probability, torch.tensor([0.0]))
    assert torch.allclose(readout.free_probability, torch.tensor([0.0]))
    assert torch.allclose(readout.unknown_probability, torch.tensor([1.0]))


def test_pipeline_forward_returns_gaussians() -> None:
    torch.manual_seed(1)
    device = _test_device()
    cfg = ModelConfig(
        feature_dim=16,
        appearance_dim=8,
        decoder_hidden_dim=16,
        support_threshold=0.0,
        confidence_threshold=0.0,
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
    assert result["appearance_uncertainty"].shape[-2:] == (32, 32)


def test_pipeline_builds_prefix_outputs_from_single_max_context() -> None:
    torch.manual_seed(2)
    device = _test_device()
    cfg = ModelConfig(
        feature_dim=16,
        appearance_dim=8,
        decoder_hidden_dim=16,
        support_threshold=0.0,
        confidence_threshold=0.0,
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
