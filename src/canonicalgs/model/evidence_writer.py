from __future__ import annotations

import torch

from canonicalgs.config import ModelConfig

from .evidence import SparseEvidence
from .view_encoder import ViewEncoderOutput


class VoxelEvidenceWriter:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def write(
        self,
        encoder_output: ViewEncoderOutput,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> list[SparseEvidence]:
        evidence = []
        for view_index in range(extrinsics.shape[0]):
            evidence.append(
                self._write_single_view(
                    encoder_output.input_rgb[view_index],
                    encoder_output.appearance_features[view_index],
                    encoder_output.depth[view_index, 0],
                    encoder_output.positional_certainty[view_index, 0],
                    encoder_output.appearance_certainty[view_index, 0],
                    encoder_output.combined_certainty[view_index, 0],
                    extrinsics[view_index],
                    intrinsics[view_index],
                )
            )
        return evidence

    def _write_single_view(
        self,
        rgb: torch.Tensor,
        appearance: torch.Tensor,
        depth: torch.Tensor,
        positional_certainty: torch.Tensor,
        appearance_certainty: torch.Tensor,
        combined_certainty: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> SparseEvidence:
        device = appearance.device
        dtype = appearance.dtype
        height, width = depth.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )

        x_normalized = (grid_x + 0.5) / width
        y_normalized = (grid_y + 0.5) / height
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        camera_dirs = torch.stack(
            (
                (x_normalized - cx) / fx,
                (y_normalized - cy) / fy,
                torch.ones_like(x_normalized),
            ),
            dim=-1,
        )

        rotation = camera_to_world[:3, :3]
        origin = camera_to_world[:3, 3]
        world_dirs = torch.einsum("ij,hwj->hwi", rotation, camera_dirs)

        rgb_flat = rgb.permute(1, 2, 0)
        appearance_flat = appearance.permute(1, 2, 0)
        surface_depth = depth.clamp_min(self.cfg.min_depth)
        surface_weight_map = combined_certainty * self.cfg.surface_weight
        surface_points = origin.view(1, 1, 3) + world_dirs * surface_depth.unsqueeze(-1)
        surface_indices = self._voxelize_points(surface_points)
        surface_points_flat = surface_points.reshape(-1, 3)
        surface_weight = surface_weight_map.reshape(-1)
        surface_positional = positional_certainty.reshape(-1)
        surface_appearance = appearance_certainty.reshape(-1)
        surface_features = appearance_flat.reshape(-1, appearance.shape[0])
        surface_colors = rgb_flat.reshape(-1, rgb.shape[0])
        return SparseEvidence(
            indices=surface_indices,
            surface_evidence=surface_weight,
            free_evidence=torch.zeros_like(surface_weight),
            canonical_weight=surface_weight,
            positional_certainty=surface_positional,
            appearance_certainty=surface_appearance,
            world_points=surface_points_flat,
            semantic_features=surface_features,
            colors=surface_colors,
        )

    def _voxelize_points(self, points: torch.Tensor) -> torch.Tensor:
        # Voxel assignment remains the only intentionally non-differentiable step.
        return torch.floor(points / self.cfg.voxel_size).to(torch.long).reshape(-1, 3)
