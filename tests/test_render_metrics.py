import torch

from canonicalgs.model.gaussian_readout import GaussianSet
from canonicalgs.training.render_metrics import compute_render_metrics


def test_compute_render_metrics_runs_with_gaussians() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    image = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsics = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsics[:, 0, 0] = 1.0
    intrinsics[:, 1, 1] = 1.0
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5

    episode = {
        "images": image,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    output = {
        "target_indices": torch.tensor([0], dtype=torch.long, device=device),
        "readout": type("Readout", (), {"support_probability": torch.ones(1, device=device)})(),
        "render_gaussians": GaussianSet(
            indices=torch.tensor([[0, 0, 0]], dtype=torch.long, device=device),
            means=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device),
            covariances=torch.diag_embed(
                torch.tensor([[0.01, 0.01, 0.01]], dtype=torch.float32, device=device)
            ),
            opacities=torch.tensor([1.0], dtype=torch.float32, device=device),
            appearance=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device),
            support=torch.tensor([1.0], dtype=torch.float32, device=device),
            confidence=torch.tensor([1.0], dtype=torch.float32, device=device),
        ),
    }

    metrics = compute_render_metrics(episode, output, max_targets=1)

    assert metrics.num_targets == 1
    assert metrics.coverage > 0.0
    assert metrics.psnr >= 0.0
