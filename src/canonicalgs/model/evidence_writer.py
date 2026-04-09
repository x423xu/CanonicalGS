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
                    encoder_output.appearance_features[view_index],
                    encoder_output.depth[view_index, 0],
                    encoder_output.depth_uncertainty[view_index, 0],
                    encoder_output.appearance_uncertainty[view_index, 0],
                    encoder_output.depth_confidence[view_index, 0],
                    extrinsics[view_index],
                    intrinsics[view_index],
                )
            )
        return evidence

    def _write_single_view(
        self,
        appearance: torch.Tensor,
        depth: torch.Tensor,
        depth_uncertainty: torch.Tensor,
        appearance_uncertainty: torch.Tensor,
        depth_confidence: torch.Tensor,
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
        camera_dirs = torch.nn.functional.normalize(camera_dirs, dim=-1)

        rotation = camera_to_world[:3, :3]
        origin = camera_to_world[:3, 3]
        world_dirs = torch.einsum("ij,hwj->hwi", rotation, camera_dirs)

        appearance_flat = appearance.permute(1, 2, 0)
        relative_depth_uncertainty = depth_uncertainty / depth.clamp_min(self.cfg.min_depth)
        positional_certainty = torch.exp(-relative_depth_uncertainty) * depth_confidence
        appearance_certainty = torch.exp(-appearance_uncertainty)
        combined_certainty = positional_certainty * appearance_certainty

        surface_offsets = depth.new_tensor(self.cfg.surface_band_offsets).view(-1, 1, 1)
        surface_depths = depth.unsqueeze(0) + surface_offsets * depth_uncertainty.unsqueeze(0)
        surface_depths = surface_depths.clamp_min(self.cfg.min_depth)
        surface_band_weights = torch.softmax(
            -0.5 * self.cfg.certainty_temperature * surface_offsets.square(),
            dim=0,
        ).to(dtype)
        surface_weight_map = (
            combined_certainty.unsqueeze(0)
            * self.cfg.surface_weight
            * surface_band_weights
        )
        surface_points = (
            origin.view(1, 1, 1, 3)
            + world_dirs.unsqueeze(0) * surface_depths.unsqueeze(-1)
        )
        surface_indices = self._voxelize_points(surface_points)
        surface_points_flat = surface_points.reshape(-1, 3)
        surface_weight = surface_weight_map.reshape(-1)
        surface_positional = positional_certainty.unsqueeze(0).expand_as(surface_weight_map).reshape(-1)
        surface_appearance = appearance_certainty.unsqueeze(0).expand_as(surface_weight_map).reshape(-1)
        surface_features = appearance_flat.unsqueeze(0).expand(
            surface_depths.shape[0],
            -1,
            -1,
            -1,
        ).reshape(-1, appearance.shape[0])

        free_samples = torch.linspace(
            0.2,
            max(self.cfg.free_space_ratio, 0.2),
            steps=self.cfg.free_space_steps,
            device=device,
            dtype=dtype,
        ).view(-1, 1, 1)
        free_depth_limit = (
            depth - self.cfg.free_space_margin_multiplier * depth_uncertainty
        ).clamp_min(self.cfg.min_depth * 0.5)
        free_depths = free_samples * free_depth_limit.unsqueeze(0)
        free_points = (
            origin.view(1, 1, 1, 3)
            + world_dirs.unsqueeze(0) * free_depths.unsqueeze(-1)
        )
        free_indices = self._voxelize_points(free_points)
        free_weight_map = positional_certainty.unsqueeze(0).expand(
            self.cfg.free_space_steps,
            -1,
            -1,
        )
        free_weight = (
            free_weight_map * self.cfg.free_weight / self.cfg.free_space_steps
        ).reshape(-1)

        zero_surface = torch.zeros_like(free_weight)
        zero_canonical = torch.zeros_like(free_weight)
        zero_app = torch.zeros_like(free_weight)
        zero_features = torch.zeros(
            (free_weight.shape[0], surface_features.shape[-1]),
            dtype=dtype,
            device=device,
        )
        zero_points = torch.zeros((free_weight.shape[0], 3), dtype=dtype, device=device)

        indices = torch.cat([surface_indices, free_indices], dim=0)
        return SparseEvidence(
            indices=indices,
            surface_evidence=torch.cat([surface_weight, zero_surface], dim=0),
            free_evidence=torch.cat([zero_surface, free_weight], dim=0),
            canonical_weight=torch.cat([surface_weight, zero_canonical], dim=0),
            positional_certainty=torch.cat([surface_positional, free_weight], dim=0),
            appearance_certainty=torch.cat([surface_appearance, zero_app], dim=0),
            world_points=torch.cat([surface_points_flat, zero_points], dim=0),
            semantic_features=torch.cat([surface_features, zero_features], dim=0),
        )

    def _voxelize_points(self, points: torch.Tensor) -> torch.Tensor:
        # Voxel assignment remains the only intentionally non-differentiable step.
        return torch.floor(points / self.cfg.voxel_size).to(torch.long).reshape(-1, 3)
