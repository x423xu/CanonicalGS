import sys
from pathlib import Path

import pytest
import torch
from einops import rearrange

from canonicalgs.model.mono_voxel_lite import (
    MonoVoxelLiteConfig,
    MonoVoxelLiteModel,
    render_mono_voxel_lite,
)
from canonicalgs.reference_ffgs.geometry.projection import sample_image_grid
from canonicalgs.reference_ffgs.model.types import Gaussians as FlatGaussians


def _reference_repo_root() -> Path | None:
    candidates = [
        Path(r"E:\code\Active-FFGS-streaming_multiview_tmp"),
        Path("/data0/xxy/code/Active-FFGS-streaming"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reference_checkpoint() -> Path | None:
    candidates = [
        Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/epoch_9-step_300000.ckpt"),
        Path(r"E:\code\Active-FFGS-streaming_multiview_tmp\checkpoints\mono_voxel_lite\checkpoints\epoch_9-step_300000.ckpt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    batch = 1
    views = 2
    height = 56
    width = 56
    images = torch.rand(batch, views, 3, height, width, device=device)
    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(batch, views, 1, 1)
    intrinsics[:, :, 0, 0] = 0.85
    intrinsics[:, :, 1, 1] = 0.85
    intrinsics[:, :, 0, 2] = 0.5
    intrinsics[:, :, 1, 2] = 0.5
    extrinsics = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch, views, 1, 1)
    extrinsics[:, 1, 0, 3] = 0.1
    near = torch.full((batch, views), 0.5, device=device)
    far = torch.full((batch, views), 8.0, device=device)
    return images, intrinsics, extrinsics, near, far


@pytest.mark.skipif(_reference_repo_root() is None, reason="reference Active-FFGS repo not found")
def test_mono_voxel_lite_matches_reference_modules() -> None:
    repo_root = _reference_repo_root()
    assert repo_root is not None
    sys.path.insert(0, str(repo_root))
    try:
        from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg as RefGaussianAdapterCfg
        from src.model.encoder.mono.depth_predictor_multiview import DepthPredictorMultiView as RefDepthPredictorMultiView
        from src.model.encoder.mono.gaussian_adapter_mono import MonoGaussianAdapter as RefMonoGaussianAdapter
        from src.model.decoder.cuda_splatting import render_cuda as ref_render_cuda
    finally:
        sys.path.pop(0)

    device = _device()
    cfg = MonoVoxelLiteConfig()

    ref_predictor = RefDepthPredictorMultiView(
        feature_channels=cfg.d_feature,
        upscale_factor=cfg.downscale_factor,
        num_depth_candidates=cfg.num_depth_candidates,
        costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
        costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
        costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
        gaussian_raw_channels=cfg.num_surfaces * (7 + 3 * ((cfg.gaussian_adapter.sh_degree + 1) ** 2) + 2),
        gaussians_per_pixel=cfg.gaussians_per_pixel,
        num_views=2,
        depth_unet_feat_dim=cfg.depth_unet_feat_dim,
        depth_unet_attn_res=cfg.depth_unet_attn_res,
        depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        enable_voxel_heads=False,
    ).to(device)
    ref_adapter = RefMonoGaussianAdapter(
        RefGaussianAdapterCfg(
            gaussian_scale_min=cfg.gaussian_adapter.gaussian_scale_min,
            gaussian_scale_max=cfg.gaussian_adapter.gaussian_scale_max,
            sh_degree=cfg.gaussian_adapter.sh_degree,
        )
    ).to(device)

    ours = MonoVoxelLiteModel(cfg).to(device)
    ours.depth_predictor.load_state_dict(ref_predictor.state_dict(), strict=True)
    ours.gaussian_adapter.load_state_dict(ref_adapter.state_dict(), strict=True)
    ours.eval()
    ref_predictor.eval()
    ref_adapter.eval()

    images, intrinsics, extrinsics, near, far = _make_inputs(device)

    with torch.no_grad():
        depths_ref, densities_ref, raw_ref = ref_predictor(
            images,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            deterministic=True,
        )

        height, width = images.shape[-2:]
        xy_ray, _ = sample_image_grid((height, width), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians_raw_ref = rearrange(raw_ref, "... (srf c) -> ... srf c", srf=cfg.num_surfaces)
        offset_xy = gaussians_raw_ref[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((width, height), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gaussians_ref = ref_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths_ref,
            ours.map_pdf_to_opacity(densities_ref, 0) / cfg.gaussians_per_pixel,
            rearrange(gaussians_raw_ref[..., 2:], "b v r srf c -> b v r srf () c"),
            (height, width),
        )
        ours_out = ours(images, intrinsics, extrinsics, near, far, global_step=0, deterministic=True)

        gaussians_ref_flat = FlatGaussians(
            rearrange(gaussians_ref.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians_ref.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians_ref.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians_ref.opacities, "b v r srf spp -> b (v r srf spp)"),
        )

    assert torch.allclose(ours_out["depths"], depths_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["densities"], densities_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["raw_gaussians"], raw_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["gaussians"].means, gaussians_ref_flat.means, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        ours_out["gaussians"].covariances, gaussians_ref_flat.covariances, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        ours_out["gaussians"].harmonics, gaussians_ref_flat.harmonics, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        ours_out["gaussians"].opacities, gaussians_ref_flat.opacities, atol=1e-6, rtol=1e-5
    )

    if device.type == "cuda":
        background = torch.zeros((images.shape[0] * images.shape[1], 3), dtype=torch.float32, device=device)
        means_bv = rearrange(
            gaussians_ref_flat.means[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.means.shape[1:]
            ),
            "b v g xyz -> (b v) g xyz",
        )
        covariances_bv = rearrange(
            gaussians_ref_flat.covariances[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.covariances.shape[1:]
            ),
            "b v g i j -> (b v) g i j",
        )
        harmonics_bv = rearrange(
            gaussians_ref_flat.harmonics[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.harmonics.shape[1:]
            ),
            "b v g c d_sh -> (b v) g c d_sh",
        )
        opacities_bv = rearrange(
            gaussians_ref_flat.opacities[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.opacities.shape[1:]
            ),
            "b v g -> (b v) g",
        )
        ref_render = ref_render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (height, width),
            background,
            means_bv,
            covariances_bv,
            harmonics_bv,
            opacities_bv,
            scale_invariant=False,
            use_sh=True,
            vggt_meta=False,
        )
        ours_render = render_mono_voxel_lite(
            ours_out["gaussians"],
            extrinsics,
            intrinsics,
            near,
            far,
            (height, width),
        )
        assert torch.allclose(
            ours_render, ref_render.reshape(images.shape[0], images.shape[1], 3, height, width), atol=1e-5, rtol=1e-4
        )


@pytest.mark.skipif(
    _reference_repo_root() is None or _reference_checkpoint() is None,
    reason="reference Active-FFGS repo or mono_voxel_lite checkpoint not found",
)
def test_mono_voxel_lite_checkpoint_matches_reference_modules() -> None:
    repo_root = _reference_repo_root()
    checkpoint = _reference_checkpoint()
    assert repo_root is not None
    assert checkpoint is not None
    sys.path.insert(0, str(repo_root))
    try:
        from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg as RefGaussianAdapterCfg
        from src.model.encoder.mono.depth_predictor_multiview import DepthPredictorMultiView as RefDepthPredictorMultiView
        from src.model.encoder.mono.gaussian_adapter_mono import MonoGaussianAdapter as RefMonoGaussianAdapter
        from src.model.decoder.cuda_splatting import render_cuda as ref_render_cuda
    finally:
        sys.path.pop(0)

    device = _device()
    cfg = MonoVoxelLiteConfig()
    ref_predictor = RefDepthPredictorMultiView(
        feature_channels=cfg.d_feature,
        upscale_factor=cfg.downscale_factor,
        num_depth_candidates=cfg.num_depth_candidates,
        costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
        costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
        costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
        gaussian_raw_channels=cfg.num_surfaces * (7 + 3 * ((cfg.gaussian_adapter.sh_degree + 1) ** 2) + 2),
        gaussians_per_pixel=cfg.gaussians_per_pixel,
        num_views=2,
        depth_unet_feat_dim=cfg.depth_unet_feat_dim,
        depth_unet_attn_res=cfg.depth_unet_attn_res,
        depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        enable_voxel_heads=False,
    ).to(device)
    ref_adapter = RefMonoGaussianAdapter(
        RefGaussianAdapterCfg(
            gaussian_scale_min=cfg.gaussian_adapter.gaussian_scale_min,
            gaussian_scale_max=cfg.gaussian_adapter.gaussian_scale_max,
            sh_degree=cfg.gaussian_adapter.sh_degree,
        )
    ).to(device)
    ours = MonoVoxelLiteModel(cfg).to(device)

    missing, unexpected = ours.load_active_ffgs_checkpoint(str(checkpoint))
    assert missing == []
    assert unexpected == []

    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    predictor_state = {
        key[len("encoder.depth_predictor.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.depth_predictor.")
    }
    adapter_state = {
        key[len("encoder.gaussian_adapter.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.gaussian_adapter.")
    }
    ref_predictor.load_state_dict(predictor_state, strict=True)
    ref_adapter.load_state_dict(adapter_state, strict=True)
    ref_predictor.eval()
    ref_adapter.eval()
    ours.eval()

    images, intrinsics, extrinsics, near, far = _make_inputs(device)

    with torch.no_grad():
        depths_ref, densities_ref, raw_ref = ref_predictor(
            images,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            deterministic=True,
        )

        height, width = images.shape[-2:]
        xy_ray, _ = sample_image_grid((height, width), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians_raw_ref = rearrange(raw_ref, "... (srf c) -> ... srf c", srf=cfg.num_surfaces)
        offset_xy = gaussians_raw_ref[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((width, height), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gaussians_ref = ref_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths_ref,
            ours.map_pdf_to_opacity(densities_ref, 0) / cfg.gaussians_per_pixel,
            rearrange(gaussians_raw_ref[..., 2:], "b v r srf c -> b v r srf () c"),
            (height, width),
        )
        ours_out = ours(images, intrinsics, extrinsics, near, far, global_step=0, deterministic=True)

        gaussians_ref_flat = FlatGaussians(
            rearrange(gaussians_ref.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians_ref.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians_ref.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians_ref.opacities, "b v r srf spp -> b (v r srf spp)"),
        )

    assert torch.allclose(ours_out["depths"], depths_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["densities"], densities_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["raw_gaussians"], raw_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ours_out["gaussians"].means, gaussians_ref_flat.means, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        ours_out["gaussians"].covariances, gaussians_ref_flat.covariances, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        ours_out["gaussians"].harmonics, gaussians_ref_flat.harmonics, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        ours_out["gaussians"].opacities, gaussians_ref_flat.opacities, atol=1e-6, rtol=1e-5
    )

    if device.type == "cuda":
        background = torch.zeros((images.shape[0] * images.shape[1], 3), dtype=torch.float32, device=device)
        means_bv = rearrange(
            gaussians_ref_flat.means[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.means.shape[1:]
            ),
            "b v g xyz -> (b v) g xyz",
        )
        covariances_bv = rearrange(
            gaussians_ref_flat.covariances[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.covariances.shape[1:]
            ),
            "b v g i j -> (b v) g i j",
        )
        harmonics_bv = rearrange(
            gaussians_ref_flat.harmonics[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.harmonics.shape[1:]
            ),
            "b v g c d_sh -> (b v) g c d_sh",
        )
        opacities_bv = rearrange(
            gaussians_ref_flat.opacities[:, None].expand(
                images.shape[0], images.shape[1], *gaussians_ref_flat.opacities.shape[1:]
            ),
            "b v g -> (b v) g",
        )
        ref_render = ref_render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (height, width),
            background,
            means_bv,
            covariances_bv,
            harmonics_bv,
            opacities_bv,
            scale_invariant=False,
            use_sh=True,
            vggt_meta=False,
        )
        ours_render = render_mono_voxel_lite(
            ours_out["gaussians"],
            extrinsics,
            intrinsics,
            near,
            far,
            (height, width),
        )
        assert torch.allclose(
            ours_render, ref_render.reshape(images.shape[0], images.shape[1], 3, height, width), atol=1e-5, rtol=1e-4
        )
