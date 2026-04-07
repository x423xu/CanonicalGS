from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from einops import rearrange

from canonicalgs.data import EpisodeBuilder, build_re10k_sample_tensor_episodes
from canonicalgs.model.mono_voxel_lite import (
    MonoVoxelLiteConfig,
    MonoVoxelLiteModel,
    OpacityMappingCfg,
    render_mono_voxel_lite,
)
from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import GaussianAdapterCfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--reference-repo", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming"))
    parser.add_argument(
        "--wandb-config",
        type=Path,
        default=Path(
            "/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/"
            "wandb/run-20260309_102424-awdebv94/files/config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/"
            "checkpoints/epoch_9-step_300000.ckpt"
        ),
    )
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--validation-holdout-stride", type=int, default=20)
    parser.add_argument("--validation-holdout-offset", type=int, default=0)
    parser.add_argument("--evaluation-holdout-stride", type=int, default=100)
    parser.add_argument("--evaluation-holdout-offset", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/mono_voxel_lite_canonical_contract_eval.json"),
    )
    return parser.parse_args()


def _flatten_wandb_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
        else:
            flat[key] = value
    return flat


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).square().mean(dim=(-3, -2, -1))
    return -10.0 * mse.clamp_min(1e-8).log10()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _build_batch(
    episode: dict[str, Any],
    context_size: int,
    near: float,
    far: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    context_indices = episode["context_indices"][context_size]
    target_indices = episode["target_indices"]

    context_images = episode["images"][context_indices].unsqueeze(0).to(device)
    context_intrinsics = episode["intrinsics"][context_indices].unsqueeze(0).to(device)
    context_extrinsics = episode["extrinsics"][context_indices].unsqueeze(0).to(device)
    target_images = episode["images"][target_indices].unsqueeze(0).to(device)
    target_intrinsics = episode["intrinsics"][target_indices].unsqueeze(0).to(device)
    target_extrinsics = episode["extrinsics"][target_indices].unsqueeze(0).to(device)

    context_near = torch.full((1, context_indices.numel()), near, dtype=torch.float32, device=device)
    context_far = torch.full((1, context_indices.numel()), far, dtype=torch.float32, device=device)
    target_near = torch.full((1, target_indices.numel()), near, dtype=torch.float32, device=device)
    target_far = torch.full((1, target_indices.numel()), far, dtype=torch.float32, device=device)

    context = {
        "image": context_images,
        "intrinsics": context_intrinsics,
        "extrinsics": context_extrinsics,
        "near": context_near,
        "far": context_far,
    }
    target = {
        "image": target_images,
        "intrinsics": target_intrinsics,
        "extrinsics": target_extrinsics,
        "near": target_near,
        "far": target_far,
    }
    meta = {
        "scene_id": episode["scene_id"],
        "clip_id": episode["clip_id"],
        "context_frame_ids": [episode["frame_ids"][index] for index in context_indices.tolist()],
        "target_frame_ids": [episode["frame_ids"][index] for index in target_indices.tolist()],
    }
    return context, target, meta


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    repo_root = args.reference_repo.resolve()
    sys.path.insert(0, str(repo_root))

    from src.config import TrainControllerCfg
    from src.model.encoder import get_encoder
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg as RefGaussianAdapterCfg
    from src.model.encoder.mono_model import MonoModelCfg, OpacityMappingCfg as RefOpacityMappingCfg
    from src.model.types import Gaussians as RefFlatGaussians

    with args.wandb_config.open("r", encoding="utf-8") as handle:
        cfg_dict = _flatten_wandb_config(yaml.safe_load(handle))

    train_controller_cfg = TrainControllerCfg(**cfg_dict["train_controller"])
    encoder_cfg = MonoModelCfg(
        name=cfg_dict["model"]["encoder"]["name"],
        d_feature=cfg_dict["model"]["encoder"]["d_feature"],
        num_depth_candidates=cfg_dict["model"]["encoder"]["num_depth_candidates"],
        num_surfaces=cfg_dict["model"]["encoder"]["num_surfaces"],
        gaussian_adapter=RefGaussianAdapterCfg(**cfg_dict["model"]["encoder"]["gaussian_adapter"]),
        opacity_mapping=RefOpacityMappingCfg(**cfg_dict["model"]["encoder"]["opacity_mapping"]),
        gaussians_per_pixel=cfg_dict["model"]["encoder"]["gaussians_per_pixel"],
        unimatch_weights_path=cfg_dict["model"]["encoder"]["unimatch_weights_path"],
        downscale_factor=cfg_dict["model"]["encoder"]["downscale_factor"],
        shim_patch_size=cfg_dict["model"]["encoder"]["shim_patch_size"],
        multiview_trans_attn_split=cfg_dict["model"]["encoder"]["multiview_trans_attn_split"],
        costvolume_unet_feat_dim=cfg_dict["model"]["encoder"]["costvolume_unet_feat_dim"],
        costvolume_unet_channel_mult=list(cfg_dict["model"]["encoder"]["costvolume_unet_channel_mult"]),
        costvolume_unet_attn_res=list(cfg_dict["model"]["encoder"]["costvolume_unet_attn_res"]),
        depth_unet_feat_dim=cfg_dict["model"]["encoder"]["depth_unet_feat_dim"],
        depth_unet_attn_res=list(cfg_dict["model"]["encoder"]["depth_unet_attn_res"]),
        depth_unet_channel_mult=list(cfg_dict["model"]["encoder"]["depth_unet_channel_mult"]),
        wo_depth_refine=cfg_dict["model"]["encoder"]["wo_depth_refine"],
        wo_cost_volume=cfg_dict["model"]["encoder"]["wo_cost_volume"],
        wo_backbone_cross_attn=cfg_dict["model"]["encoder"]["wo_backbone_cross_attn"],
        wo_cost_volume_refine=cfg_dict["model"]["encoder"]["wo_cost_volume_refine"],
        use_epipolar_trans=cfg_dict["model"]["encoder"]["use_epipolar_trans"],
        monodepth_vit_type=cfg_dict["model"]["encoder"]["monodepth_vit_type"],
        enable_voxelization=cfg_dict["model"]["encoder"]["enable_voxelization"],
        use_plucker_embedding=cfg_dict["model"]["encoder"]["use_plucker_embedding"],
        voxel_feature_dim=cfg_dict["model"]["encoder"]["voxel_feature_dim"],
        gaussians_per_cell=cfg_dict["model"]["encoder"]["gaussians_per_cell"],
        down_strides=list(cfg_dict["model"]["encoder"]["down_strides"]),
        cell_scale=cfg_dict["model"]["encoder"]["cell_scale"],
        cube_merge_type=cfg_dict["model"]["encoder"]["cube_merge_type"],
        voxelization_downsample_factor=cfg_dict["model"]["encoder"]["voxelization_downsample_factor"],
        voxel_conf_threshold=cfg_dict["model"]["encoder"]["voxel_conf_threshold"],
        expanded_voxel_topk=cfg_dict["model"]["encoder"].get("expanded_voxel_topk", 50000),
        expanded_voxel_bounds_filter=cfg_dict["model"]["encoder"].get("expanded_voxel_bounds_filter", False),
        world_cell_scale_3views=cfg_dict["model"]["encoder"].get("world_cell_scale_3views", 1.0),
        world_cell_scale_4views=cfg_dict["model"]["encoder"].get("world_cell_scale_4views", 0.75),
        profile_voxelization=cfg_dict["model"]["encoder"]["profile_voxelization"],
        voxel_compute_2d_branch=cfg_dict["model"]["encoder"]["voxel_compute_2d_branch"],
        voxel_low_vram_arch=cfg_dict["model"]["encoder"]["voxel_low_vram_arch"],
        voxel_train_depth_predictor=cfg_dict["model"]["encoder"]["voxel_train_depth_predictor"],
        return_depth=cfg_dict["model"]["encoder"]["return_depth"],
    )
    ref_encoder, _ = get_encoder(
        encoder_cfg,
        gs_cube=train_controller_cfg.gs_cube,
        vggt_meta=train_controller_cfg.vggt_meta,
        knn_down=train_controller_cfg.knn_down,
        gaussian_merge=train_controller_cfg.gaussian_merge,
        depth_distillation=train_controller_cfg.depth_distillation,
        train_controller_cfg=train_controller_cfg,
    )
    ref_encoder = ref_encoder.to(device).eval()

    ours = MonoVoxelLiteModel(
        MonoVoxelLiteConfig(
            d_feature=encoder_cfg.d_feature,
            num_depth_candidates=encoder_cfg.num_depth_candidates,
            num_surfaces=encoder_cfg.num_surfaces,
            gaussians_per_pixel=encoder_cfg.gaussians_per_pixel,
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=encoder_cfg.gaussian_adapter.gaussian_scale_min,
                gaussian_scale_max=encoder_cfg.gaussian_adapter.gaussian_scale_max,
                sh_degree=encoder_cfg.gaussian_adapter.sh_degree,
            ),
            opacity_mapping=OpacityMappingCfg(
                initial=encoder_cfg.opacity_mapping.initial,
                final=encoder_cfg.opacity_mapping.final,
                warm_up=encoder_cfg.opacity_mapping.warm_up,
            ),
            downscale_factor=encoder_cfg.downscale_factor,
            multiview_trans_attn_split=encoder_cfg.multiview_trans_attn_split,
            costvolume_unet_feat_dim=encoder_cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=list(encoder_cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=list(encoder_cfg.costvolume_unet_attn_res),
            depth_unet_feat_dim=encoder_cfg.depth_unet_feat_dim,
            depth_unet_attn_res=list(encoder_cfg.depth_unet_attn_res),
            depth_unet_channel_mult=list(encoder_cfg.depth_unet_channel_mult),
        )
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    ref_missing, ref_unexpected = ref_encoder.load_state_dict(encoder_state, strict=False)
    our_missing, our_unexpected = ours.load_active_ffgs_checkpoint(str(args.checkpoint))

    builder = EpisodeBuilder(
        context_sizes=(2, 3, 4, 5, 6),
        target_views=6,
        min_frames_per_episode=12,
        subsample_to=12,
        seed=111123,
    )
    episodes = build_re10k_sample_tensor_episodes(
        [str(args.root)],
        args.split,
        builder,
        args.samples,
        (args.image_height, args.image_width),
        args.validation_holdout_stride,
        args.validation_holdout_offset,
        args.evaluation_holdout_stride,
        args.evaluation_holdout_offset,
    )

    ref_psnrs: list[float] = []
    our_psnrs: list[float] = []
    psnr_deltas: list[float] = []
    render_diffs: list[float] = []
    depth_diffs: list[float] = []
    density_diffs: list[float] = []
    raw_gaussian_diffs: list[float] = []
    gaussian_mean_diffs: list[float] = []
    gaussian_cov_diffs: list[float] = []
    gaussian_harmonic_diffs: list[float] = []
    gaussian_opacity_diffs: list[float] = []
    sample_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index, episode in enumerate(episodes):
            context, target, meta = _build_batch(episode, 2, args.near, args.far, device)
            context_h, context_w = context["image"].shape[-2:]

            ref_depths_raw, ref_densities, ref_raw_gaussians = ref_encoder.depth_predictor(
                context["image"],
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=ref_encoder.cfg.gaussians_per_pixel,
                deterministic=True,
            )
            ref_gaussians_nested = ref_encoder._build_2d_gaussians(
                context,
                ref_depths_raw,
                ref_densities,
                ref_raw_gaussians,
                ref_encoder.cfg.gaussians_per_pixel,
                context_h,
                context_w,
                device,
                0,
            )
            ref_gaussians = RefFlatGaussians(
                rearrange(ref_gaussians_nested.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(ref_gaussians_nested.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(ref_gaussians_nested.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(ref_gaussians_nested.opacities, "b v r srf spp -> b (v r srf spp)"),
            )
            ref_render = render_mono_voxel_lite(
                ref_gaussians,
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                target["image"].shape[-2:],
            )

            our_output = ours(
                context["image"],
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                global_step=0,
                deterministic=True,
            )
            our_render = render_mono_voxel_lite(
                our_output["gaussians"],
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                target["image"].shape[-2:],
            )

            ref_depths = rearrange(
                ref_depths_raw,
                "b v (h w) srf spp -> b v h w srf spp",
                h=context_h,
                w=context_w,
            ).squeeze(-1).squeeze(-1)
            our_depths = rearrange(
                our_output["depths"],
                "b v (h w) srf spp -> b v h w srf spp",
                h=context_h,
                w=context_w,
            ).squeeze(-1).squeeze(-1)

            gt = rearrange(target["image"], "b v c h w -> (b v) c h w")
            ref_rgb = rearrange(ref_render, "b v c h w -> (b v) c h w")
            our_rgb = rearrange(our_render, "b v c h w -> (b v) c h w")
            ref_psnr = float(_psnr(ref_rgb, gt).mean().item())
            our_psnr = float(_psnr(our_rgb, gt).mean().item())

            ref_psnrs.append(ref_psnr)
            our_psnrs.append(our_psnr)
            psnr_deltas.append(abs(ref_psnr - our_psnr))
            render_diffs.append(_max_abs_diff(ref_render, our_render))
            depth_diffs.append(_max_abs_diff(ref_depths, our_depths))
            density_diffs.append(_max_abs_diff(ref_densities, our_output["densities"]))
            raw_gaussian_diffs.append(_max_abs_diff(ref_raw_gaussians, our_output["raw_gaussians"]))
            gaussian_mean_diffs.append(_max_abs_diff(ref_gaussians.means, our_output["gaussians"].means))
            gaussian_cov_diffs.append(_max_abs_diff(ref_gaussians.covariances, our_output["gaussians"].covariances))
            gaussian_harmonic_diffs.append(_max_abs_diff(ref_gaussians.harmonics, our_output["gaussians"].harmonics))
            gaussian_opacity_diffs.append(_max_abs_diff(ref_gaussians.opacities, our_output["gaussians"].opacities))

            if sample_index < 5:
                sample_summaries.append(
                    {
                        "sample_index": sample_index,
                        "scene_id": meta["scene_id"],
                        "clip_id": meta["clip_id"],
                        "context_frame_ids": meta["context_frame_ids"],
                        "target_frame_ids": meta["target_frame_ids"],
                        "reference_psnr": ref_psnr,
                        "canonical_psnr": our_psnr,
                        "render_max_abs_diff": render_diffs[-1],
                    }
                )

    report = {
        "split": args.split,
        "samples_evaluated": len(episodes),
        "image_shape": [args.image_height, args.image_width],
        "near": args.near,
        "far": args.far,
        "validation_holdout_stride": args.validation_holdout_stride,
        "validation_holdout_offset": args.validation_holdout_offset,
        "evaluation_holdout_stride": args.evaluation_holdout_stride,
        "evaluation_holdout_offset": args.evaluation_holdout_offset,
        "reference_mean_psnr": sum(ref_psnrs) / len(ref_psnrs),
        "canonical_mean_psnr": sum(our_psnrs) / len(our_psnrs),
        "max_psnr_delta": max(psnr_deltas),
        "max_render_abs_diff": max(render_diffs),
        "max_depth_abs_diff": max(depth_diffs),
        "max_density_abs_diff": max(density_diffs),
        "max_raw_gaussian_abs_diff": max(raw_gaussian_diffs),
        "max_gaussian_mean_abs_diff": max(gaussian_mean_diffs),
        "max_gaussian_cov_abs_diff": max(gaussian_cov_diffs),
        "max_gaussian_harmonic_abs_diff": max(gaussian_harmonic_diffs),
        "max_gaussian_opacity_abs_diff": max(gaussian_opacity_diffs),
        "reference_missing_keys": ref_missing,
        "reference_unexpected_keys": ref_unexpected,
        "canonical_missing_keys": our_missing,
        "canonical_unexpected_keys": our_unexpected,
        "sample_summaries": sample_summaries,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
