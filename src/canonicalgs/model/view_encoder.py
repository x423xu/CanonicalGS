from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .dpt_head import DptFeatureHead


@dataclass(slots=True)
class ViewEncoderOutput:
    input_rgb: torch.Tensor
    appearance_features: torch.Tensor
    geometry_features: torch.Tensor
    depth: torch.Tensor
    density: torch.Tensor
    gaussian_raw_params: torch.Tensor
    positional_certainty: torch.Tensor
    appearance_certainty: torch.Tensor
    combined_certainty: torch.Tensor
    depth_confidence: torch.Tensor
    confidence: torch.Tensor


@dataclass(slots=True)
class EncodedViewFeatures:
    images: torch.Tensor
    lowres_geometry: torch.Tensor
    fullres_geometry: torch.Tensor

    def subset(self, count: int) -> "EncodedViewFeatures":
        return EncodedViewFeatures(
            images=self.images[:count],
            lowres_geometry=self.lowres_geometry[:count],
            fullres_geometry=self.fullres_geometry[:count],
        )


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
        allow_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.patch_size = 14
        self.backbone: nn.Module
        self.is_fallback = False
        try:
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
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to load the requested DINOv2 backbone. "
                    "CanonicalGS expects a ViT+DPT feature extractor; "
                    "set model.allow_dinov2_fallback=true only for explicit non-paper fallback."
                ) from exc
            warnings.warn(
                "Falling back to a lightweight patch backbone because DINOv2 could not be loaded. "
                "This is a non-paper test path and should not be used for paper-fidelity experiments.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.backbone = _FallbackPatchBackbone(patch_size=self.patch_size)
            self.embed_dim = self.backbone.embed_dim
            self.intermediate_layer_idx = [2, 5, 8, 11]
            self.is_fallback = True

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor], int, int, tuple[int, int]]:
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
                outputs = self._extract_intermediate_layers(images)
        else:
            outputs = self._extract_intermediate_layers(images)
        return outputs, patch_h, patch_w, (height, width)

    def _extract_intermediate_layers(
        self,
        images: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.is_fallback:
            return self.backbone(images, self.intermediate_layer_idx)
        return self.backbone.get_intermediate_layers(
            images,
            self.intermediate_layer_idx,
            return_class_token=False,
        )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (images - mean) / std


class _FallbackPatchBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _FallbackPatchBackbone(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int = 384, depth: int = 12) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.ModuleList([_FallbackPatchBlock(embed_dim) for _ in range(depth)])

    def forward(
        self,
        images: torch.Tensor,
        intermediate_layer_idx: list[int],
    ) -> list[torch.Tensor]:
        x = self.patch_embed(images)
        outputs: list[torch.Tensor] = []
        target_layers = set(intermediate_layer_idx)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in target_layers:
                tokens = x.flatten(2).transpose(1, 2)
                outputs.append(tokens)
        return outputs


class CostVolumeDepthPredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_depth_bins: int,
        min_depth: float,
        max_depth: float,
        min_depth_uncertainty: float,
        cost_volume_temperature: float = 0.35,
        cost_volume_visibility_beta: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_depth_uncertainty = min_depth_uncertainty
        self.cost_volume_temperature = cost_volume_temperature
        self.cost_volume_visibility_beta = cost_volume_visibility_beta

        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim + num_depth_bins, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, num_depth_bins, kernel_size=1),
        )
        self.cost_refine_head = nn.Sequential(
            nn.Conv2d(num_depth_bins, num_depth_bins, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(num_depth_bins, num_depth_bins),
            nn.Conv2d(num_depth_bins, num_depth_bins, kernel_size=1),
        )
        certainty_hidden_dim = max(16, feature_dim // 2)
        self.positional_certainty_head = nn.Sequential(
            nn.Conv2d(num_depth_bins, certainty_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(certainty_hidden_dim, 1, kernel_size=1),
        )
        self.fullres_refine_head = nn.Sequential(
            nn.Conv2d(feature_dim + 3 + 1, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )

    def _estimate_scene_depth_bounds(
        self,
        extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = extrinsics[:, :3, 3]
        if centers.shape[0] == 1:
            return (
                centers.new_tensor(self.min_depth),
                centers.new_tensor(self.max_depth),
            )

        forward = F.normalize(extrinsics[:, :3, 2], dim=-1, eps=1e-6)
        eye = torch.eye(3, dtype=centers.dtype, device=centers.device)
        projectors = eye.unsqueeze(0) - forward.unsqueeze(-1) * forward.unsqueeze(-2)
        normal_matrix = projectors.sum(dim=0) + 1e-3 * eye
        rhs = torch.einsum("vij,vj->i", projectors, centers)
        focus_point = torch.linalg.solve(normal_matrix, rhs)

        world_to_camera = torch.linalg.inv(extrinsics)
        focus_in_camera = (
            torch.einsum("vij,j->vi", world_to_camera[:, :3, :3], focus_point)
            + world_to_camera[:, :3, 3]
        )
        positive_depths = focus_in_camera[:, 2]
        positive_depths = positive_depths[positive_depths > 1e-3]

        pairwise = torch.cdist(centers, centers)
        nonzero_baselines = pairwise[pairwise > 1e-6]
        baseline = (
            nonzero_baselines.median()
            if nonzero_baselines.numel()
            else centers.new_tensor(1.0)
        )

        if positive_depths.numel() >= 2:
            near = torch.quantile(positive_depths, 0.1) * 0.8
            far = torch.quantile(positive_depths, 0.9) * 1.2
        elif positive_depths.numel() == 1:
            near = positive_depths[0] * 0.8
            far = positive_depths[0] * 1.2 + baseline * 2.0
        else:
            near = baseline * 0.5
            far = baseline * 8.0

        near = near.clamp(
            min=self.min_depth,
            max=max(self.min_depth, self.max_depth - 0.25),
        )
        far = torch.maximum(far, near + baseline.clamp_min(0.5))
        far = far.clamp(min=near + 0.25, max=self.max_depth)
        return near, far

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
        normalized_lowres_features = F.normalize(lowres_features, dim=1)
        scene_near, scene_far = self._estimate_scene_depth_bounds(extrinsics)
        inverse_depth_bins = torch.linspace(
            1.0 / scene_far,
            1.0 / scene_near,
            self.num_depth_bins,
            device=lowres_features.device,
            dtype=lowres_features.dtype,
        )
        inverse_depth_min = inverse_depth_bins[0]
        inverse_depth_max = inverse_depth_bins[-1]
        inverse_depth_span = (inverse_depth_max - inverse_depth_min).clamp_min(1e-6)
        world_to_camera = torch.linalg.inv(extrinsics)

        depth_outputs = []
        positional_certainty_outputs = []
        confidence_outputs = []
        for ref_index in range(num_views):
            ref_feature = lowres_features[ref_index : ref_index + 1]
            ref_feature_normalized = normalized_lowres_features[ref_index : ref_index + 1]
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
                source_indices = torch.cat(
                    [
                        torch.arange(ref_index, device=lowres_features.device),
                        torch.arange(ref_index + 1, num_views, device=lowres_features.device),
                    ]
                )
                num_sources = int(source_indices.numel())
                warped, visibility = self._warp_feature_volume(
                    normalized_lowres_features[source_indices],
                    inverse_depth_bins,
                    intrinsics[ref_index : ref_index + 1].expand(num_sources, -1, -1),
                    extrinsics[ref_index : ref_index + 1].expand(num_sources, -1, -1),
                    intrinsics[source_indices],
                    world_to_camera[source_indices],
                    (low_h, low_w),
                )
                stacked_costs = 1.0 - (
                    warped * ref_feature_normalized.unsqueeze(2)
                ).sum(dim=1)
                stacked_costs = stacked_costs * visibility + (1.0 - visibility)
                stacked_visibilities = visibility
                aggregated_cost = self._aggregate_source_costs(
                    stacked_costs,
                    stacked_visibilities,
                )
            refined_cost = aggregated_cost + self.cost_refine_head(aggregated_cost)
            depth_logits = self.depth_head(torch.cat([ref_feature, refined_cost], dim=1))
            depth_probability = torch.softmax(depth_logits, dim=1)

            inverse_depth_lowres = (
                depth_probability * inverse_depth_bins.view(1, self.num_depth_bins, 1, 1)
            ).sum(dim=1, keepdim=True)
            positional_certainty_lowres = torch.sigmoid(self.positional_certainty_head(refined_cost))

            inverse_depth = F.interpolate(
                inverse_depth_lowres,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            )
            ref_fullres_feature = fullres_features[ref_index : ref_index + 1]
            ref_image = F.interpolate(
                images[ref_index : ref_index + 1],
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            )
            refine_input = torch.cat([ref_fullres_feature, ref_image, inverse_depth], dim=1)
            inverse_depth_residual = 0.25 * inverse_depth_span * torch.tanh(
                self.fullres_refine_head(refine_input)
            )
            inverse_depth = (inverse_depth + inverse_depth_residual).clamp(
                min=inverse_depth_min,
                max=inverse_depth_max,
            )
            depth = inverse_depth.reciprocal().clamp_min(scene_near)
            positional_certainty = F.interpolate(
                positional_certainty_lowres,
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            ).clamp(0.0, 1.0)
            confidence = F.interpolate(
                depth_probability.max(dim=1, keepdim=True)[0],
                size=(full_h, full_w),
                mode="bilinear",
                align_corners=True,
            ).clamp(0.0, 1.0)

            depth_outputs.append(depth)
            positional_certainty_outputs.append(positional_certainty)
            confidence_outputs.append(confidence)

        return (
            torch.cat(depth_outputs, dim=0),
            torch.cat(positional_certainty_outputs, dim=0),
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
        batch_size = source_features.shape[0]
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
        ).reshape(1, 3, -1).expand(batch_size, -1, -1)
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
        source_xy = source_xy.reshape(batch_size, 2, num_depth_bins, height, width).permute(0, 2, 3, 4, 1)
        sample_grid = torch.empty_like(source_xy)
        sample_grid[..., 0] = 2.0 * source_xy[..., 0] / max(width - 1, 1) - 1.0
        sample_grid[..., 1] = 2.0 * source_xy[..., 1] / max(height - 1, 1) - 1.0
        sample_grid = sample_grid.reshape(batch_size, num_depth_bins * height, width, 2)

        source_z = projected[:, 2].reshape(batch_size, num_depth_bins, height, width)
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
        warped = warped.reshape(batch_size, source_features.shape[1], num_depth_bins, height, width)
        warped = warped * visibility.unsqueeze(1)
        return warped, visibility

    def _aggregate_source_costs(
        self,
        stacked_costs: torch.Tensor,
        stacked_visibilities: torch.Tensor,
    ) -> torch.Tensor:
        visibility = stacked_visibilities.clamp(0.0, 1.0)
        matching_confidence = torch.exp(-stacked_costs)
        score = (
            torch.log(matching_confidence.clamp_min(1e-6))
            / max(self.cost_volume_temperature, 1e-6)
            + self.cost_volume_visibility_beta * torch.log(visibility.clamp_min(1e-6))
        )
        score = torch.where(
            visibility > 0.0,
            score,
            torch.full_like(score, torch.finfo(score.dtype).min),
        )
        coefficients = torch.softmax(score, dim=0) * visibility
        coefficients = coefficients / coefficients.sum(dim=0, keepdim=True).clamp_min(1e-6)
        aggregated_cost = (stacked_costs * coefficients).sum(dim=0, keepdim=True)
        return aggregated_cost


class SymmetricMultiViewEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        appearance_dim: int,
        min_depth: float = 0.25,
        max_depth: float = 8.0,
        min_depth_uncertainty: float = 0.05,
        positional_certainty_tau: float = 0.5,
        max_positional_uncertainty: float = 5.0,
        num_depth_bins: int = 32,
        cost_volume_temperature: float = 0.35,
        cost_volume_visibility_beta: float = 0.2,
        dpt_output_stride: int = 4,
        dinov2_model_name: str = "dinov2_vits14",
        dinov2_pretrained: bool = True,
        freeze_dinov2: bool = True,
        allow_dinov2_fallback: bool = False,
        appearance_uncertainty_bias: float = 0.05,
        appearance_uncertainty_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = DinoV2Backbone(
            model_name=dinov2_model_name,
            pretrained=dinov2_pretrained,
            freeze=freeze_dinov2,
            allow_fallback=allow_dinov2_fallback,
        )
        self.dpt_head = DptFeatureHead(
            in_channels=self.backbone.embed_dim,
            features=feature_dim,
            out_channels=[48, 96, 192, 384],
            lowres_out_channels=feature_dim,
            fullres_out_channels=feature_dim,
            output_stride=dpt_output_stride,
            use_clstoken=False,
        )
        self.depth_predictor = CostVolumeDepthPredictor(
            feature_dim=feature_dim,
            num_depth_bins=num_depth_bins,
            min_depth=min_depth,
            max_depth=max_depth,
            min_depth_uncertainty=min_depth_uncertainty,
            cost_volume_temperature=cost_volume_temperature,
            cost_volume_visibility_beta=cost_volume_visibility_beta,
        )
        self.min_depth = min_depth
        self.positional_certainty_tau = positional_certainty_tau
        self.max_positional_uncertainty = max_positional_uncertainty
        self.appearance_adapter = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, appearance_dim, kernel_size=1),
        )
        self.appearance_certainty_head = nn.Sequential(
            nn.Conv2d(feature_dim + appearance_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )
        d_sh = (4 + 1) ** 2
        gaussian_raw_channels = 2 + 7 + 3 * d_sh
        self.gaussian_param_head = nn.Sequential(
            nn.Conv2d(feature_dim + appearance_dim + 3, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
        )
        self.density_head = nn.Sequential(
            nn.Conv2d(feature_dim + appearance_dim + 3 + 1, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            DepthwiseSeparableBlock(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> ViewEncoderOutput:
        encoded = self.encode_views(images)
        return self.decode_views(encoded, intrinsics, extrinsics)

    def encode_views(self, images: torch.Tensor) -> EncodedViewFeatures:
        dino_features, patch_h, patch_w, output_size = self.backbone(images)
        lowres_geometry, fullres_geometry = self.dpt_head(
            dino_features,
            patch_h,
            patch_w,
            output_size,
        )
        return EncodedViewFeatures(
            images=images,
            lowres_geometry=lowres_geometry,
            fullres_geometry=fullres_geometry,
        )

    def decode_views(
        self,
        encoded: EncodedViewFeatures,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> ViewEncoderOutput:
        depth, positional_certainty, confidence = self.depth_predictor(
            lowres_features=encoded.lowres_geometry,
            fullres_features=encoded.fullres_geometry,
            images=encoded.images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        appearance_features = self.appearance_adapter(encoded.fullres_geometry)
        appearance_features = F.normalize(appearance_features, dim=1)
        appearance_features = F.interpolate(
            appearance_features,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        geometry_features = F.interpolate(
            encoded.fullres_geometry,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        input_rgb = F.interpolate(
            encoded.images,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        appearance_context = torch.cat(
            [
                geometry_features,
                appearance_features,
            ],
            dim=1,
        )
        appearance_certainty = torch.sigmoid(self.appearance_certainty_head(appearance_context))
        combined_certainty = (positional_certainty * appearance_certainty).clamp(0.0, 1.0)
        gaussian_context = torch.cat([geometry_features, appearance_features, input_rgb], dim=1)
        gaussian_raw_params = self.gaussian_param_head(gaussian_context)
        density = torch.sigmoid(
            self.density_head(
                torch.cat([gaussian_context, confidence], dim=1),
            )
        ) * confidence
        return ViewEncoderOutput(
            input_rgb=input_rgb,
            appearance_features=appearance_features,
            geometry_features=geometry_features,
            depth=depth,
            density=density,
            gaussian_raw_params=gaussian_raw_params,
            positional_certainty=positional_certainty,
            appearance_certainty=appearance_certainty,
            combined_certainty=combined_certainty,
            depth_confidence=confidence,
            confidence=confidence,
        )
