from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("matplotlib")

from canonicalgs.training.eval_visualization import save_evaluation_visualizations


def test_save_evaluation_visualizations_writes_expected_files(tmp_path: Path) -> None:
    images = torch.rand(3, 3, 8, 8)
    intrinsics = torch.tensor(
        [
            [[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]],
            [[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]],
            [[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    extrinsics[1, 0, 3] = 0.25
    extrinsics[2, 1, 3] = 0.25
    episode = {
        "scene_id": "scene_demo",
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    state = SimpleNamespace(
        geometry_mean=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.2],
                [0.3, -0.1, 1.4],
            ],
            dtype=torch.float32,
        )
    )
    readout = SimpleNamespace(canonical_certainty=torch.tensor([0.8, 0.6, 0.4], dtype=torch.float32))
    gaussians = SimpleNamespace(
        means=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.2],
                [0.3, -0.1, 1.4],
            ],
            dtype=torch.float32,
        ),
        appearance=torch.tensor(
            [
                [1.0, 0.2, 0.2],
                [0.2, 1.0, 0.2],
                [0.2, 0.2, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    output = {
        "context_indices": torch.tensor([0, 1], dtype=torch.long),
        "depth": torch.full((2, 1, 8, 8), 1.5, dtype=torch.float32),
        "positional_certainty": torch.full((2, 1, 8, 8), 0.8, dtype=torch.float32),
        "appearance_certainty": torch.full((2, 1, 8, 8), 0.7, dtype=torch.float32),
        "combined_certainty": torch.full((2, 1, 8, 8), 0.56, dtype=torch.float32),
        "state": state,
        "readout": readout,
        "gaussians": gaussians,
    }

    saved = save_evaluation_visualizations(
        output_root=tmp_path,
        episode=episode,
        outputs={2: output},
        phase="eval",
        step=5,
        max_points_per_cloud=32,
    )

    context_dir = tmp_path / "eval" / "step_000005" / "scene_demo" / "ctx_02"
    assert context_dir.exists()
    for file_name in (
        "predicted_depth.png",
        "positional_certainty.png",
        "appearance_certainty.png",
        "combined_certainty.png",
        "certainty_histograms.png",
        "camera_poses.png",
        "per_view_point_clouds.png",
        "aggregated_point_cloud.png",
        "rendered_outputs.png",
        "raw_fields.npz",
        "aggregated_point_cloud.npz",
        "per_view_point_cloud_00.npz",
        "per_view_point_cloud_01.npz",
    ):
        assert (context_dir / file_name).exists()
    assert 2 in saved
