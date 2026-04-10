import torch

from canonicalgs.config import ObjectiveConfig
from canonicalgs.training import objectives as objectives_module
from canonicalgs.training.objectives import CanonicalLossComputer


class _DummyReadout:
    def __init__(self) -> None:
        self.indices = torch.tensor([[0, 0, 0]], dtype=torch.long)
        self.canonical_certainty = torch.tensor([1.2], dtype=torch.float32)
        self.canonical_features = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.geometry_mean = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        self.uncertainty = torch.tensor([0.1], dtype=torch.float32)
        self.semantic_consistency = torch.tensor([0.8], dtype=torch.float32)


def test_objective_uses_render_loss(monkeypatch) -> None:
    cfg = ObjectiveConfig(lambda_rend=1.0, lambda_mono=0.0)
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


def test_objective_renders_only_sparse_monotone_pair(monkeypatch) -> None:
    cfg = ObjectiveConfig(lambda_rend=1.0, lambda_mono=1.0)
    computer = CanonicalLossComputer(cfg)
    rendered_contexts = []

    def fake_compute_render_stats(episode, output, max_targets=None):
        rendered_contexts.append(int(output["context_size"]))
        mse = torch.tensor(float(output["context_size"]) / 10.0)
        return type(
            "RenderStats",
            (),
            {
                "mse": mse,
                "psnr": torch.tensor(6.0),
                "coverage": torch.tensor(0.5),
                "num_targets": 1,
            },
        )()

    monkeypatch.setattr(objectives_module, "compute_render_stats", fake_compute_render_stats)

    outputs = {
        2: {
            "context_size": 2,
            "state": object(),
            "readout": _DummyReadout(),
            "render_gaussians": object(),
            "target_indices": torch.tensor([0], dtype=torch.long),
        },
        3: {
            "context_size": 3,
            "state": object(),
            "readout": _DummyReadout(),
            "render_gaussians": object(),
            "target_indices": torch.tensor([0], dtype=torch.long),
        },
        4: {
            "context_size": 4,
            "state": object(),
            "readout": _DummyReadout(),
            "render_gaussians": object(),
            "target_indices": torch.tensor([0], dtype=torch.long),
        },
    }
    losses = computer(outputs, episode={"images": torch.zeros((1, 3, 4, 4))}, max_render_targets=1)

    assert rendered_contexts == [2, 4]
    assert torch.isclose(losses.render_loss, torch.tensor(0.4))
    assert torch.isclose(losses.monotone_loss, torch.tensor(0.2))
