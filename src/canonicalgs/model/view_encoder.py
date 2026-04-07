from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .dpt_head import DptFeatureHead


@dataclass(slots=True)
class ViewEncoderOutput:
    appearance_features: torch.Tensor
    geometry_features: torch.Tensor
    depth: torch.Tensor
    depth_uncertainty: torch.Tensor
    confidence: torch.Tensor


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        norm_groups = max(1, min(8, out_channels))
        while out_channels % norm_groups != 0:
            norm_groups -= 1
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.GELU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class DinoV2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.patch_size = 14
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )
        self.embed_dim = int(getattr(self.backbone, "embed_dim", self.backbone.num_features))
        if "vitl" in model_name:
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            self.intermediate_layer_idx = [2, 5, 8, 11]

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int, tuple[int, int]]:
        images = self._normalize(images)
        _, _, height, width = images.shape
        target_height = max(self.patch_size, math.floor(height / self.patch_size) * self.patch_size)
        target_width = max(self.patch_size, math.floor(width / self.patch_size) * self.patch_size)
        if target_height != height or target_width != width:
            images = F.interpolate(
                images,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=True,
            )

        patch_h = target_height // self.patch_size
        patch_w = target_width // self.patch_size
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone.get_intermediate_layers(
                    images,
                    self.intermediate_layer_idx,
                    return_class_token=True,
                )
        else:
            outputs = self.backbone.get_intermediate_layers(
                images,
                self.intermediate_layer_idx,
                return_class_token=True,
            )
        return outputs, patch_h, patch_w, (height, width)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (images - mean) / std


class CostVolumeDepthPredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_depth_bins: int,
        min_depth: float,
        max_depth: float,
        min_depth_uncertainty: float,
    ) -> None:
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_depth_uncertainty = min_depth_uncertainty

        self.depth_prior = nn.Conv2d(feature_dim, num_depth_bins, kernel_size=1)
        self.cost_refiner = nn.Sequential(
            DepthwiseSeparableBlock(num_depth_bins, num_depth_bins),
            DepthwiseSeparableBlock(num_depth_bins, num_depth_bins),
        )
        self.fullres_refiner = nn.Sequential(
            nn.Conv2d(feature_dim + 3, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
        )
        self.inverse_depth_residual = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.confidence_head = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(
        self,
        lowres_features: torch.Tensor,
        fullres_features: torch.Tensor,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_views, _, low_h, low_w = lowres_features.shape
        full_h, full_w = fullres_features.shape[-2:]
        images_resized = F.interpolate(
            images,
            size=(full_h, full_w),
            mode="bilinear",
            align_corners=True,
        )
        inverse_depth_bins = torch.linspace(
            1.0 / self.max_depth,
            1.0 / self.min_depth,
            self.num_depth_bins,
            device=lowres_features.device,
            dtype=lowres_features.dtype,
        )
        world_to_camera = torch.linalg.inv(extrinsics)

        depth_outputs = []
        uncertainty_outputs = []
        confidence_outputs = []
        for ref_index in range(num_views):
            ref_feature = lowres_features[ref_index : ref_index + 1]
            if num_views == 1:
                aggregated_cost = torch.zeros(
                    (1, self.num_depth_bins, low_h, low_w),
                    dtype=lowres_features.dtype,
                    device=lowres_features.device,
                )
                visibility_score = torch.zeros(
                    (1, 1, low_h, low_w),
                    dtype=lowres_features.dtype,
                    device=lowres_features.device,
                )
            else:
                costs = []
                visibilities = []
                for source_index in range(num_views):
                    if source_index == ref_index:
                        continue
                    warped, visibility = self._warp_feature_volume(
                        lowres_features[source_index : source_index + 1],
                        inverse_depth_bins,
                        intrinsics[ref_index : ref_index + 1],
                        extrinsics[ref_index : ref_index + 1],
                        intrinsics[source_index : source_index + 1],
                        world_to_camera[source_index : source_index + 1],
                        (low_h, low_w),
                    )
                    ref_volume = ref_feature.unsqueeze(2).expand(-1, -1, self.num_depth_bins, -1, -1)
                    pair_cost = 1.0 - F.cosine_similarity(ref_volume, warped, dim=1)
                    pair_cost = pair_cost * visibility + (1.0 - visibility)
                    costs.append(pair_cost)
                    visibilities.append(visibility)

                stacked_costs = torch.cat(costs, dim=0)
                stacked_visibilities = torch.cat(visibilities, dim=0)
                aggregated_cost = (
                    (stacked_costs * stacked_visibilities).sum(dim=0, keepdim=True)
                    / stacked_visibilities.sum(dim=0, keepdim=True).clamp_min(1.0)
                )
                visibility_score = stacked_visibilities.mean(dim=(0, 1)).view(1, 1, low_h, low_w)

            refined_cost = aggregated_cost + self.cost_refiner(aggregated_cost)
            inverse_depth_logits = self.depth_prior(ref_feature) - refined_cost
            depth_probability = torch.softmax(inverse_depth_logits, dim=1)

            inverse_depth_lowres = (
                depth_probability * inverse_depth_bins.view(1, self.num_depth_bins, 1, 1)
            ).sum(dim=1, keepdim=True)
            inverse_depth_variance = (
                depth_probability
                * (
                    inverse_depth_bins.view(1, self.num_depth_bins, 1, 1)
                    - inverse_depth_lowres
                ).square()
            ).sum(dim=1, keepdim=True)
            inverse_depth_uncertainty_lowres = inverse_depth_variance.sqrt().clamp_min(1e-6)
            pdf_max = depth_probability.max(dim=1, keepdim=True).values

            inverse_depth_fullres = F.interpolate(
                inverse_depth_lowres,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            )
            inverse_uncertainty_fullres = F.interpolate(
                inverse_depth_uncertainty_lowres,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            )
            pdf_max_fullres = F.interpolate(
                pdf_max,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            )
            refine_input = torch.cat(
                (
                    fullres_features[ref_index : ref_index + 1],
                    images_resized[ref_index : ref_index + 1],
                ),
                dim=1,
            )
            refined = self.fullres_refiner(refine_input)
            inverse_depth_interval = (1.0 / self.min_depth - 1.0 / self.max_depth) / max(self.num_depth_bins - 1, 1)
            inverse_depth_residual = torch.tanh(self.inverse_depth_residual(refined)) * inverse_depth_interval
            inverse_depth = (inverse_depth_fullres + inverse_depth_residual).clamp(
                1.0 / self.max_depth,
                1.0 / self.min_depth,
            )
            depth = inverse_depth.reciprocal()
            depth_uncertainty = (
                inverse_uncertainty_fullres / inverse_depth.square().clamp_min(1e-6)
            ).clamp_min(self.min_depth_uncertainty)
            confidence = torch.sigmoid(self.confidence_head(refined))
            normalized_inverse_uncertainty = inverse_uncertainty_fullres / max(inverse_depth_interval, 1e-6)
            confidence = confidence * pdf_max_fullres / (1.0 + normalized_inverse_uncertainty)

            depth_outputs.append(depth)
            uncertainty_outputs.append(depth_uncertainty)
            confidence_outputs.append(confidence)

        return (
            torch.cat(depth_outputs, dim=0),
            torch.cat(uncertainty_outputs, dim=0),
            torch.cat(confidence_outputs, dim=0),
        )

    def _warp_feature_volume(
        self,
        source_features: torch.Tensor,
        inverse_depth_bins: torch.Tensor,
        reference_intrinsics: torch.Tensor,
        reference_extrinsics: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_world_to_camera: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image_shape
        device = source_features.device
        dtype = source_features.dtype
        num_depth_bins = inverse_depth_bins.shape[0]
        depth_bins = inverse_depth_bins.reciprocal()

        intrinsics_ref = reference_intrinsics.clone()
        intrinsics_src = source_intrinsics.clone()
        intrinsics_ref[:, 0, :] *= width
        intrinsics_ref[:, 1, :] *= height
        intrinsics_src[:, 0, :] *= width
        intrinsics_src[:, 1, :] *= height

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        homogeneous = torch.stack(
            (grid_x, grid_y, torch.ones_like(grid_x)),
            dim=0,
        ).reshape(1, 3, -1)
        reference_rays = torch.linalg.inv(intrinsics_ref) @ homogeneous
        reference_points = reference_rays.unsqueeze(2) * depth_bins.view(1, 1, num_depth_bins, 1)

        reference_rotation = reference_extrinsics[:, :3, :3]
        reference_origin = reference_extrinsics[:, :3, 3]
        world_points = torch.einsum("bij,bjdn->bidn", reference_rotation, reference_points)
        world_points = world_points + reference_origin[:, :, None, None]

        source_rotation = source_world_to_camera[:, :3, :3]
        source_translation = source_world_to_camera[:, :3, 3]
        source_points = torch.einsum("bij,bjdn->bidn", source_rotation, world_points)
        source_points = source_points + source_translation[:, :, None, None]

        projected = torch.einsum("bij,bjdn->bidn", intrinsics_src, source_points)
        source_xy = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        source_xy = source_xy.reshape(1, 2, num_depth_bins, height, width).permute(0, 2, 3, 4, 1)
        sample_grid = torch.empty_like(source_xy)
        sample_grid[..., 0] = 2.0 * source_xy[..., 0] / max(width - 1, 1) - 1.0
        sample_grid[..., 1] = 2.0 * source_xy[..., 1] / max(height - 1, 1) - 1.0
        sample_grid = sample_grid.reshape(1, num_depth_bins * height, width, 2)

        source_z = projected[:, 2].reshape(1, num_depth_bins, height, width)
        visibility = (
            (source_z > 1e-6)
            & (source_xy[..., 0] >= 0.0)
            & (source_xy[..., 0] <= width - 1)
            & (source_xy[..., 1] >= 0.0)
            & (source_xy[..., 1] <= height - 1)
        ).to(dtype)

        warped = F.grid_sample(
            source_features,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped.reshape(1, source_features.shape[1], num_depth_bins, height, width)
        warped = warped * visibility.unsqueeze(1)
        return warped, visibility


class SymmetricMultiViewEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        appearance_dim: int,
        min_depth: float = 0.25,
        max_depth: float = 8.0,
        min_depth_uncertainty: float = 0.05,
        num_depth_bins: int = 32,
        dpt_output_stride: int = 4,
        dinov2_model_name: str = "dinov2_vits14",
        dinov2_pretrained: bool = True,
        freeze_dinov2: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = DinoV2Backbone(
            model_name=dinov2_model_name,
            pretrained=dinov2_pretrained,
            freeze=freeze_dinov2,
        )
        self.dpt_head = DptFeatureHead(
            in_channels=self.backbone.embed_dim,
            features=feature_dim,
            out_channels=[48, 96, 192, 384],
            lowres_out_channels=feature_dim,
            fullres_out_channels=feature_dim,
            output_stride=dpt_output_stride,
            use_clstoken=True,
        )
        self.depth_predictor = CostVolumeDepthPredictor(
            feature_dim=feature_dim,
            num_depth_bins=num_depth_bins,
            min_depth=min_depth,
            max_depth=max_depth,
            min_depth_uncertainty=min_depth_uncertainty,
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> ViewEncoderOutput:
        dino_features, patch_h, patch_w, output_size = self.backbone(images)
        lowres_geometry, fullres_geometry = self.dpt_head(
            dino_features,
            patch_h,
            patch_w,
            output_size,
        )
        depth, depth_uncertainty, confidence = self.depth_predictor(
            lowres_features=lowres_geometry,
            fullres_features=fullres_geometry,
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        appearance_features = F.interpolate(
            images,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return ViewEncoderOutput(
            appearance_features=appearance_features,
            geometry_features=fullres_geometry,
            depth=depth,
            depth_uncertainty=depth_uncertainty,
            confidence=confidence,
        )
