from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from canonicalgs.config import ModelConfig
from canonicalgs.reference_ffgs.geometry.projection import sample_image_grid
from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from canonicalgs.reference_ffgs.model.encoder.common.gaussians import build_covariance
from canonicalgs.reference_ffgs.model.encoder.mono.gaussian_adapter_mono import MonoGaussianAdapter

from .readout import PosteriorReadout
from .view_encoder import ViewEncoderOutput


@dataclass(slots=True)
class GaussianSet:
    indices: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    opacities: torch.Tensor
    appearance: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor


class LocalGaussianReadout(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_sh = (4 + 1) ** 2
        hidden_dim = max(cfg.decoder_hidden_dim, cfg.appearance_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.appearance_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.scale_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.sh_head = nn.Linear(hidden_dim, 3 * self.d_sh)
        self.opacity_head = nn.Linear(cfg.appearance_dim + 1, 1)
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.zeros_(self.opacity_head.bias)
        with torch.no_grad():
            self.opacity_head.weight[0, 0] = 1.0

    def forward(self, readout: PosteriorReadout, prune: bool = True) -> GaussianSet:
        indices = readout.indices
        if indices.numel() == 0:
            zeros = readout.geometry_mean.new_zeros
            return GaussianSet(
                indices=indices,
                means=zeros((0, 3)),
                covariances=zeros((0, 3, 3)),
                opacities=readout.canonical_certainty.new_zeros((0,)),
                appearance=zeros((0, 3)),
                support=readout.canonical_certainty.new_zeros((0,)),
                confidence=readout.canonical_certainty.new_zeros((0,)),
            )

        canonical_features = readout.canonical_features
        hidden = self.decoder(canonical_features)

        means = readout.geometry_mean + 0.5 * self.cfg.voxel_size * torch.tanh(self.offset_head(hidden))
        scales = self.cfg.gaussian_scale_min + (
            self.cfg.gaussian_scale_max - self.cfg.gaussian_scale_min
        ) * torch.sigmoid(self.scale_head(hidden))
        base_scales = readout.geometry_variance.sqrt().clamp_min(self.cfg.gaussian_scale_min)
        scales = torch.maximum(scales, base_scales).clamp_max(self.cfg.gaussian_scale_max)
        rotations = F.normalize(self.rotation_head(hidden), dim=-1)
        covariances = build_covariance(scales, rotations)

        sh = rearrange(self.sh_head(hidden), "... (xyz d_sh) -> ... xyz d_sh", xyz=3, d_sh=self.d_sh)
        appearance = (0.5 * readout.canonical_color + 0.5 * torch.sigmoid(sh[..., 0])).clamp(0.0, 1.0)

        confidence = readout.canonical_certainty
        density = F.softplus(
            self.opacity_head(torch.cat([confidence.unsqueeze(-1), canonical_features], dim=-1)).squeeze(-1)
        )
        opacities = 1.0 - torch.exp(-density)

        return GaussianSet(
            indices=indices,
            means=means,
            covariances=covariances,
            opacities=opacities,
            appearance=appearance,
            support=readout.semantic_consistency,
            confidence=confidence,
        )


class PerViewMonoGaussianBuilder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_sh = (4 + 1) ** 2
        self.adapter = MonoGaussianAdapter(
            GaussianAdapterCfg(
                gaussian_scale_min=cfg.gaussian_scale_min,
                gaussian_scale_max=cfg.gaussian_scale_max,
                sh_degree=4,
            )
        )

    def forward(
        self,
        encoder_output: ViewEncoderOutput,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> tuple[PosteriorReadout, GaussianSet]:
        num_views, _, height, width = encoder_output.depth.shape
        if num_views == 0:
            empty = encoder_output.depth.new_zeros((0,))
            readout = PosteriorReadout(
                indices=encoder_output.depth.new_zeros((0, 3), dtype=torch.long),
                canonical_certainty=empty,
                positional_certainty=empty,
                appearance_certainty=empty,
                semantic_consistency=empty,
                geometry_mean=encoder_output.depth.new_zeros((0, 3)),
                geometry_variance=encoder_output.depth.new_zeros((0, 3)),
                canonical_features=encoder_output.depth.new_zeros((0, self.cfg.appearance_dim)),
                canonical_color=encoder_output.depth.new_zeros((0, 3)),
                uncertainty=empty,
            )
            gaussians = GaussianSet(
                indices=readout.indices,
                means=readout.geometry_mean,
                covariances=encoder_output.depth.new_zeros((0, 3, 3)),
                opacities=empty,
                appearance=readout.canonical_color,
                support=empty,
                confidence=empty,
            )
            return readout, gaussians

        xy_ray, _ = sample_image_grid((height, width), encoder_output.depth.device)
        xy_ray = rearrange(xy_ray, "h w xy -> () () (h w) () xy")
        pixel_size = encoder_output.depth.new_tensor((1.0 / width, 1.0 / height))

        raw_gaussians = rearrange(
            encoder_output.gaussian_raw_params,
            "v c h w -> () v (h w) c",
        )
        offset_xy = raw_gaussians[..., :2].sigmoid().unsqueeze(-2)
        coordinates = xy_ray.expand(1, num_views, -1, -1, -1) + (offset_xy - 0.5) * pixel_size

        depths = rearrange(encoder_output.depth, "v () h w -> () v (h w) () ()")
        densities = rearrange(encoder_output.density, "v () h w -> () v (h w) () ()")
        raw_tail = rearrange(raw_gaussians[..., 2:], "b v r c -> b v r () () c")

        gaussians = self.adapter.forward(
            rearrange(extrinsics, "v i j -> () v () () () i j"),
            rearrange(intrinsics, "v i j -> () v () () () i j"),
            coordinates.unsqueeze(-2),
            depths,
            densities,
            raw_tail,
            (height, width),
        )

        flat_means = rearrange(gaussians.means, "b v r srf spp xyz -> (b v r srf spp) xyz")
        flat_covariances = rearrange(gaussians.covariances, "b v r srf spp i j -> (b v r srf spp) i j")
        flat_opacities = rearrange(gaussians.opacities, "b v r srf spp -> (b v r srf spp)")
        flat_appearance = rearrange(torch.sigmoid(gaussians.harmonics[..., 0]), "b v r srf spp c -> (b v r srf spp) c")

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=encoder_output.depth.device, dtype=torch.long),
            torch.arange(width, device=encoder_output.depth.device, dtype=torch.long),
            indexing="ij",
        )
        pixel_indices = torch.stack(
            [
                torch.arange(num_views, device=encoder_output.depth.device, dtype=torch.long)[:, None, None].expand(num_views, height, width),
                grid_y.unsqueeze(0).expand(num_views, -1, -1),
                grid_x.unsqueeze(0).expand(num_views, -1, -1),
            ],
            dim=-1,
        ).reshape(-1, 3)

        positional_certainty = encoder_output.positional_certainty.reshape(-1)
        appearance_certainty = encoder_output.appearance_certainty.reshape(-1)
        combined_certainty = encoder_output.combined_certainty.reshape(-1)
        canonical_features = rearrange(encoder_output.appearance_features, "v c h w -> (v h w) c")
        canonical_color = rearrange(encoder_output.input_rgb, "v c h w -> (v h w) c")
        uncertainty = (1.0 - positional_certainty) + (1.0 - appearance_certainty)
        geometry_variance = rearrange(torch.diagonal(flat_covariances, dim1=-2, dim2=-1), "n c -> n c")

        readout = PosteriorReadout(
            indices=pixel_indices,
            canonical_certainty=combined_certainty,
            positional_certainty=positional_certainty,
            appearance_certainty=appearance_certainty,
            semantic_consistency=torch.ones_like(combined_certainty),
            geometry_mean=flat_means,
            geometry_variance=geometry_variance.clamp_min(1e-6),
            canonical_features=canonical_features,
            canonical_color=canonical_color.clamp(0.0, 1.0),
            uncertainty=uncertainty,
        )
        gaussian_set = GaussianSet(
            indices=pixel_indices,
            means=flat_means,
            covariances=flat_covariances,
            opacities=flat_opacities,
            appearance=flat_appearance,
            support=torch.ones_like(combined_certainty),
            confidence=combined_certainty,
        )
        return readout, gaussian_set
