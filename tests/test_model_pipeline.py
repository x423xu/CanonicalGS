import torch

from canonicalgs.config import ModelConfig
from canonicalgs.model import CanonicalGsPipeline, CanonicalReadout, CanonicalState, LocalGaussianReadout, PosteriorReadout
from canonicalgs.model.view_encoder import SymmetricMultiViewEncoder


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
        reference.confidence,
        permuted.confidence[inverse],
        atol=1e-5,
        rtol=1e-4,
    )


def test_local_gaussian_readout_filters_low_support_cells() -> None:
    cfg = ModelConfig(
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
        confidence=torch.tensor([1.2, 0.8, 0.3]),
        geometry_mean=torch.tensor(
            [[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
            dtype=torch.float32,
        ),
        geometry_variance=torch.tensor(
            [[0.001, 0.002, 0.003], [0.01, 0.01, 0.01], [0.05, 0.05, 0.05]],
            dtype=torch.float32,
        ),
        appearance_mean=torch.tensor(
            [[0.2, 0.4], [0.3, 0.5], [0.1, 0.2]],
            dtype=torch.float32,
        ),
        uncertainty=torch.tensor([0.1, 0.2, 0.8]),
    )

    gaussians = LocalGaussianReadout(cfg)(readout)

    assert gaussians.indices.shape[0] == 1
    assert torch.equal(gaussians.indices[0], torch.tensor([0, 0, 0]))
    assert gaussians.covariances.shape == (1, 3, 3)
    assert gaussians.opacities[0] > 0.0

    soft_gaussians = LocalGaussianReadout(cfg)(readout, prune=False)
    assert soft_gaussians.indices.shape[0] == 3


def test_canonical_readout_keeps_empty_cells_unknown() -> None:
    state = CanonicalState(
        indices=torch.tensor([[0, 0, 0]], dtype=torch.long),
        surface_evidence=torch.tensor([0.0]),
        free_evidence=torch.tensor([0.0]),
        confidence=torch.tensor([0.0]),
        geo_moment_1=torch.zeros((1, 3), dtype=torch.float32),
        geo_moment_2=torch.zeros((1, 3), dtype=torch.float32),
        app_moment_1=torch.zeros((1, 2), dtype=torch.float32),
        app_weight=torch.zeros((1,), dtype=torch.float32),
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
        support_threshold=0.0,
        confidence_threshold=0.0,
        gaussian_scale_min=0.01,
        gaussian_scale_max=0.3,
        dinov2_pretrained=False,
    )
    pipeline = CanonicalGsPipeline(cfg).to(device)
    pipeline.eval()

    episode = {
        "images": torch.rand(8, 3, 32, 32, device=device),
        "extrinsics": make_extrinsics(8).to(device),
        "intrinsics": make_intrinsics(8).to(device),
        "context_indices": {2: torch.tensor([0, 1], dtype=torch.long, device=device)},
        "target_indices": torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long, device=device),
    }

    result = pipeline(episode, 2)

    assert result["num_context_views"] == 2
    assert result["num_active_cells"] > 0
    assert result["num_gaussians"] > 0
    assert result["gaussians"].means.shape[1] == 3
