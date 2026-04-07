from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from torch import nn

from canonicalgs.reference_ffgs.geometry.projection import sample_image_grid
from canonicalgs.reference_ffgs.model.decoder.cuda_splatting import render_cuda
from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import (
    GaussianAdapterCfg,
)
from canonicalgs.reference_ffgs.model.encoder.mono.depth_predictor_multiview import (
    DepthPredictorMultiView,
)
from canonicalgs.reference_ffgs.model.encoder.mono.gaussian_adapter_mono import (
    MonoGaussianAdapter,
)
from canonicalgs.reference_ffgs.model.types import Gaussians


@dataclass(slots=True)
class OpacityMappingCfg:
    initial: float = 0.0
    final: float = 0.0
    warm_up: int = 1


@dataclass(slots=True)
class MonoVoxelLiteConfig:
    d_feature: int = 64
    num_depth_candidates: int = 128
    num_surfaces: int = 1
    gaussians_per_pixel: int = 1
    gaussian_adapter: GaussianAdapterCfg = field(
        default_factory=lambda: GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        )
    )
    opacity_mapping: OpacityMappingCfg = field(default_factory=OpacityMappingCfg)
    downscale_factor: int = 4
    multiview_trans_attn_split: int = 2
    costvolume_unet_feat_dim: int = 128
    costvolume_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 1, 1])
    costvolume_unet_attn_res: list[int] = field(default_factory=lambda: [4])
    depth_unet_feat_dim: int = 32
    depth_unet_attn_res: list[int] = field(default_factory=lambda: [16])
    depth_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])


class MonoVoxelLiteModel(nn.Module):
    def __init__(self, cfg: MonoVoxelLiteConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MonoVoxelLiteConfig()
        self.gaussian_adapter = MonoGaussianAdapter(self.cfg.gaussian_adapter)
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=self.cfg.d_feature,
            upscale_factor=self.cfg.downscale_factor,
            num_depth_candidates=self.cfg.num_depth_candidates,
            costvolume_unet_feat_dim=self.cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(self.cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(self.cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=self.cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=self.cfg.gaussians_per_pixel,
            num_views=2,
            depth_unet_feat_dim=self.cfg.depth_unet_feat_dim,
            depth_unet_attn_res=self.cfg.depth_unet_attn_res,
            depth_unet_channel_mult=self.cfg.depth_unet_channel_mult,
            enable_voxel_heads=False,
        )

    def map_pdf_to_opacity(self, pdf: torch.Tensor, global_step: int) -> torch.Tensor:
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        global_step: int = 0,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        b, v, _, h, w = images.shape
        depths, densities, raw_gaussians = self.depth_predictor(
            images,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=self.cfg.gaussians_per_pixel,
            deterministic=deterministic,
        )

        xy_ray, _ = sample_image_grid((h, w), images.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians_raw = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians_raw[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=images.device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / self.cfg.gaussians_per_pixel,
            rearrange(gaussians_raw[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
        )
        out_gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )
        return {
            "depths": depths,
            "densities": densities,
            "raw_gaussians": raw_gaussians,
            "gaussians": out_gaussians,
        }

    def load_active_ffgs_checkpoint(self, checkpoint: str) -> tuple[list[str], list[str]]:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint must resolve to a state_dict-like mapping")

        model_state = self.state_dict()
        filtered = {}
        for key, value in state.items():
            if key.startswith("encoder."):
                key = key[len("encoder.") :]
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                continue
        result = self.load_state_dict(filtered, strict=False)
        return result.missing_keys, result.unexpected_keys


def render_mono_voxel_lite(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    image_shape: tuple[int, int],
    background_color: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, views, _, _ = extrinsics.shape
    if background_color is None:
        background_color = torch.zeros((batch * views, 3), dtype=torch.float32, device=extrinsics.device)
    means = rearrange(
        gaussians.means[:, None].expand(batch, views, *gaussians.means.shape[1:]),
        "b v g xyz -> (b v) g xyz",
    )
    covariances = rearrange(
        gaussians.covariances[:, None].expand(batch, views, *gaussians.covariances.shape[1:]),
        "b v g i j -> (b v) g i j",
    )
    harmonics = rearrange(
        gaussians.harmonics[:, None].expand(batch, views, *gaussians.harmonics.shape[1:]),
        "b v g c d_sh -> (b v) g c d_sh",
    )
    opacities = rearrange(
        gaussians.opacities[:, None].expand(batch, views, *gaussians.opacities.shape[1:]),
        "b v g -> (b v) g",
    )
    return render_cuda(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(intrinsics, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"),
        rearrange(far, "b v -> (b v)"),
        image_shape,
        background_color,
        means,
        covariances,
        harmonics,
        opacities,
        scale_invariant=False,
        use_sh=True,
        vggt_meta=False,
    ).reshape(batch, views, 3, image_shape[0], image_shape[1])
