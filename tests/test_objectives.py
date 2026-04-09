import torch

from canonicalgs.config import ObjectiveConfig
from canonicalgs.training import objectives as objectives_module
from canonicalgs.training.objectives import CanonicalLossComputer


class _DummyReadout:
    def __init__(self) -> None:
        self.indices = torch.tensor([[0, 0, 0]], dtype=torch.long)
        self.support_probability = torch.tensor([0.9], dtype=torch.float32)
        self.canonical_certainty = torch.tensor([1.2], dtype=torch.float32)
        self.canonical_features = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.geometry_mean = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        self.uncertainty = torch.tensor([0.1], dtype=torch.float32)
        self.semantic_consistency = torch.tensor([0.8], dtype=torch.float32)


def test_objective_uses_render_loss(monkeypatch) -> None:
    cfg = ObjectiveConfig(lambda_rend=1.0, lambda_conv=0.0, lambda_mono=0.0, lambda_null=0.0)
    computer = CanonicalLossComputer(cfg)

    def fake_compute_render_stats(episode, output, max_targets=None):
        return type(
            "RenderStats",
            (),
            {
                "mse": torch.tensor(0.25),
                "psnr": torch.tensor(6.0),
                "coverage": torch.tensor(0.5),
                "num_targets": 1,
            },
        )()

    monkeypatch.setattr(objectives_module, "compute_render_stats", fake_compute_render_stats)

    outputs = {
        2: {
            "state": object(),
            "readout": _DummyReadout(),
            "render_gaussians": object(),
            "target_indices": torch.tensor([0], dtype=torch.long),
        }
    }
    losses = computer(outputs, episode={"images": torch.zeros((1, 3, 4, 4))}, max_render_targets=1)

    assert torch.isclose(losses.render_loss, torch.tensor(0.25))
    assert torch.isclose(losses.total_loss, torch.tensor(0.25))
