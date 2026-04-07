from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from einops import rearrange


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/data0/xxy/code/Active-FFGS-streaming"),
    )
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
    parser.add_argument(
        "--evaluation-index",
        type=Path,
        default=Path("/data0/xxy/code/Active-FFGS-streaming/assets/evaluation_index_re10k.json"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/mono_voxel_lite_compare_100.json"),
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


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    return batch


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    repo_root = args.reference_repo.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"reference repo not found: {repo_root}")
    sys.path.insert(0, str(repo_root))

    from src.config import TrainControllerCfg
    from src.dataset.data_module import DataLoaderCfg, DataLoaderStageCfg, DataModule, get_data_shim
    from src.dataset.dataset_re10k import DatasetRE10kCfg
    from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluationCfg
    from src.evaluation.metrics import compute_psnr
    from src.model.decoder import get_decoder
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
    from src.model.encoder import get_encoder
    from src.model.encoder.common.gaussian_adapter import (
        GaussianAdapterCfg as RefGaussianAdapterCfg,
    )
    from src.model.encoder.mono_model import MonoModelCfg, OpacityMappingCfg as RefOpacityMappingCfg
    from src.model.types import Gaussians as RefFlatGaussians

    from canonicalgs.model.mono_voxel_lite import (
        MonoVoxelLiteConfig,
        MonoVoxelLiteModel,
        OpacityMappingCfg,
        render_mono_voxel_lite,
    )
    from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import (
        GaussianAdapterCfg,
    )

    with args.wandb_config.open("r", encoding="utf-8") as f:
        wandb_cfg = yaml.safe_load(f)
    cfg_dict = _flatten_wandb_config(wandb_cfg)
    cfg_dict["mode"] = "test"
    cfg_dict["wandb"]["mode"] = "disabled"
    cfg_dict["dataset"]["view_sampler"] = {
        "name": "evaluation",
        "index_path": str(args.evaluation_index),
        "num_context_views": 2,
    }
    cfg_dict["data_loader"]["test"]["num_workers"] = 0
    cfg_dict["data_loader"]["test"]["persistent_workers"] = False
    cfg_dict["data_loader"]["test"]["batch_size"] = 1
    cfg_dict["dataset"]["test_len"] = args.samples
    train_controller_cfg = TrainControllerCfg(**cfg_dict["train_controller"])
    dataset_cfg = DatasetRE10kCfg(
        name=cfg_dict["dataset"]["name"],
        roots=[Path(root) for root in cfg_dict["dataset"]["roots"]],
        baseline_epsilon=cfg_dict["dataset"]["baseline_epsilon"],
        max_fov=cfg_dict["dataset"]["max_fov"],
        make_baseline_1=cfg_dict["dataset"]["make_baseline_1"],
        augment=cfg_dict["dataset"]["augment"],
        test_len=cfg_dict["dataset"]["test_len"],
        test_chunk_interval=cfg_dict["dataset"]["test_chunk_interval"],
        skip_bad_shape=cfg_dict["dataset"]["skip_bad_shape"],
        near=cfg_dict["dataset"]["near"],
        far=cfg_dict["dataset"]["far"],
        baseline_scale_bounds=cfg_dict["dataset"]["baseline_scale_bounds"],
        shuffle_val=cfg_dict["dataset"]["shuffle_val"],
        train_times_per_scene=cfg_dict["dataset"]["train_times_per_scene"],
        highres=cfg_dict["dataset"]["highres"],
        use_index_to_load_chunk=cfg_dict["dataset"]["use_index_to_load_chunk"],
        min_views=cfg_dict["dataset"]["min_views"],
        max_views=cfg_dict["dataset"]["max_views"],
        image_shape=list(cfg_dict["dataset"]["image_shape"]),
        background_color=list(cfg_dict["dataset"]["background_color"]),
        cameras_are_circular=cfg_dict["dataset"]["cameras_are_circular"],
        overfit_to_scene=cfg_dict["dataset"].get("overfit_to_scene"),
        view_sampler=ViewSamplerEvaluationCfg(
            name="evaluation",
            index_path=args.evaluation_index,
            num_context_views=2,
        ),
    )
    data_loader_cfg = DataLoaderCfg(
        train=DataLoaderStageCfg(**cfg_dict["data_loader"]["train"]),
        test=DataLoaderStageCfg(**cfg_dict["data_loader"]["test"]),
        val=DataLoaderStageCfg(**cfg_dict["data_loader"]["val"]),
    )
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
    decoder_cfg = DecoderSplattingCUDACfg(name=cfg_dict["model"]["decoder"]["name"])

    ref_encoder, _ = get_encoder(
        encoder_cfg,
        gs_cube=train_controller_cfg.gs_cube,
        vggt_meta=train_controller_cfg.vggt_meta,
        knn_down=train_controller_cfg.knn_down,
        gaussian_merge=train_controller_cfg.gaussian_merge,
        depth_distillation=train_controller_cfg.depth_distillation,
        train_controller_cfg=train_controller_cfg,
    )
    ref_decoder = get_decoder(decoder_cfg, dataset_cfg)
    ref_encoder = ref_encoder.to(device).eval()
    ref_decoder = ref_decoder.to(device).eval()

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

    data_module = DataModule(
        dataset_cfg,
        data_loader_cfg,
        step_tracker=None,
        global_rank=0,
        train_controller_cfg=train_controller_cfg,
    )
    data_shim = get_data_shim(ref_encoder)
    loader = data_module.test_dataloader()

    ref_psnrs: list[float] = []
    our_psnrs: list[float] = []
    render_psnr_deltas: list[float] = []
    render_max_abs_diffs: list[float] = []
    render_mean_abs_diffs: list[float] = []
    gaussian_mean_diffs: list[float] = []
    gaussian_cov_diffs: list[float] = []
    gaussian_harmonic_diffs: list[float] = []
    gaussian_opacity_diffs: list[float] = []
    depth_diffs: list[float] = []
    density_diffs: list[float] = []
    raw_gaussian_diffs: list[float] = []
    sample_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if sample_index >= args.samples:
                break

            batch = data_shim(batch)
            batch = _move_to_device(batch, device)
            context = batch["context"]
            target = batch["target"]

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
            ref_depths = rearrange(
                ref_depths_raw,
                "b v (h w) srf spp -> b v h w srf spp",
                h=context_h,
                w=context_w,
            ).squeeze(-1).squeeze(-1)
            ref_render = ref_decoder(
                ref_gaussians,
                target["extrinsics"],
                target["intrinsics"],
                target["near"],
                target["far"],
                target["image"].shape[-2:],
                scale_invariant=False,
                vggt_meta=train_controller_cfg.vggt_meta,
            ).color

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

            gt = rearrange(target["image"], "b v c h w -> (b v) c h w")
            ref_rgb = rearrange(ref_render, "b v c h w -> (b v) c h w")
            our_rgb = rearrange(our_render, "b v c h w -> (b v) c h w")
            ref_psnr = compute_psnr(gt, ref_rgb).mean().item()
            our_psnr = compute_psnr(gt, our_rgb).mean().item()

            our_depths = rearrange(
                our_output["depths"],
                "b v (h w) srf spp -> b v h w srf spp",
                h=context_h,
                w=context_w,
            ).squeeze(-1).squeeze(-1)

            ref_psnrs.append(ref_psnr)
            our_psnrs.append(our_psnr)
            render_psnr_deltas.append(abs(ref_psnr - our_psnr))
            render_max_abs_diffs.append(_max_abs_diff(ref_render, our_render))
            render_mean_abs_diffs.append(_mean_abs_diff(ref_render, our_render))
            gaussian_mean_diffs.append(_max_abs_diff(ref_gaussians.means, our_output["gaussians"].means))
            gaussian_cov_diffs.append(_max_abs_diff(ref_gaussians.covariances, our_output["gaussians"].covariances))
            gaussian_harmonic_diffs.append(_max_abs_diff(ref_gaussians.harmonics, our_output["gaussians"].harmonics))
            gaussian_opacity_diffs.append(_max_abs_diff(ref_gaussians.opacities, our_output["gaussians"].opacities))
            density_diffs.append(_max_abs_diff(ref_densities, our_output["densities"]))
            raw_gaussian_diffs.append(_max_abs_diff(ref_raw_gaussians, our_output["raw_gaussians"]))
            if ref_depths is not None:
                depth_diffs.append(_max_abs_diff(ref_depths, our_depths))

            sample_summaries.append(
                {
                    "sample_index": sample_index,
                    "scene": batch["scene"][0],
                    "ref_psnr": ref_psnr,
                    "canonical_psnr": our_psnr,
                    "psnr_delta": abs(ref_psnr - our_psnr),
                    "render_max_abs_diff": render_max_abs_diffs[-1],
                    "gaussian_mean_max_abs_diff": gaussian_mean_diffs[-1],
                    "density_max_abs_diff": density_diffs[-1],
                    "raw_gaussian_max_abs_diff": raw_gaussian_diffs[-1],
                }
            )

    report = {
        "samples_evaluated": len(sample_summaries),
        "reference_checkpoint": str(args.checkpoint),
        "reference_repo": str(repo_root),
        "reference_missing_keys": ref_missing,
        "reference_unexpected_keys": ref_unexpected,
        "canonical_missing_keys": our_missing,
        "canonical_unexpected_keys": our_unexpected,
        "reference_mean_psnr": sum(ref_psnrs) / len(ref_psnrs),
        "canonical_mean_psnr": sum(our_psnrs) / len(our_psnrs),
        "max_psnr_delta": max(render_psnr_deltas),
        "mean_psnr_delta": sum(render_psnr_deltas) / len(render_psnr_deltas),
        "max_render_abs_diff": max(render_max_abs_diffs),
        "mean_render_abs_diff": sum(render_mean_abs_diffs) / len(render_mean_abs_diffs),
        "max_gaussian_mean_abs_diff": max(gaussian_mean_diffs),
        "max_gaussian_cov_abs_diff": max(gaussian_cov_diffs),
        "max_gaussian_harmonic_abs_diff": max(gaussian_harmonic_diffs),
        "max_gaussian_opacity_abs_diff": max(gaussian_opacity_diffs),
        "max_density_abs_diff": max(density_diffs),
        "max_raw_gaussian_abs_diff": max(raw_gaussian_diffs),
        "max_depth_abs_diff": max(depth_diffs) if depth_diffs else None,
        "sample_summaries": sample_summaries,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
