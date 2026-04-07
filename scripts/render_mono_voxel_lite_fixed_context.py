from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from einops import rearrange
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from canonicalgs.data.re10k import _convert_poses, _load_scene_item
from canonicalgs.model.mono_voxel_lite import MonoVoxelLiteConfig, MonoVoxelLiteModel, OpacityMappingCfg
from canonicalgs.reference_ffgs.model.decoder.cuda_splatting import render_cuda, render_depth_cuda
from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import GaussianAdapterCfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument(
        "--context-input",
        type=int,
        nargs="+",
        default=[1, 89, 139, 179, 278],
        help="Ordered context frame indices used by MonoVoxelLite.",
    )
    parser.add_argument("--target-frame", type=int, default=139)
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
            "/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/"
            "epoch_8-step_280000.ckpt"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--image-height", type=int, default=294)
    parser.add_argument("--image-width", type=int, default=518)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data0/xxy/code/CanonicalGS/outputs/"
            "651a7f83ed093001/mono_voxel_lite_ctx_1_89_139_179_278_target_139"
        ),
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


def _to_u8_rgb(image: torch.Tensor) -> Image.Image:
    tensor = image.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _depth_to_u16(depth: torch.Tensor, near: float, far: float) -> Image.Image:
    tensor = depth.detach().cpu().clamp(min=near, max=far)
    norm = ((tensor - near) / max(far - near, 1e-6)).clamp(0.0, 1.0)
    arr = (norm * 65535.0).to(torch.uint16).numpy()
    return Image.fromarray(arr, mode="I;16")


def _load_scene_tensor_episode(root: Path, split: str, scene_id: str, image_shape: tuple[int, int]) -> dict[str, Any]:
    index_path = root / split / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if scene_id not in index:
        raise KeyError(f"scene {scene_id} not found in {split} index")

    chunk_name = index[scene_id]
    chunk_path = root / split / chunk_name
    item = _load_scene_item(chunk_path, scene_id)
    if item is None:
        raise RuntimeError(f"scene {scene_id} not found in chunk {chunk_path}")

    poses = item["cameras"]
    if not isinstance(poses, torch.Tensor):
        poses = torch.as_tensor(poses, dtype=torch.float32)
    extrinsics, intrinsics = _convert_poses(poses)

    h, w = image_shape
    images: list[torch.Tensor] = []
    for encoded in item["images"]:
        image = Image.open(BytesIO(encoded.numpy().tobytes())).convert("RGB")
        image = image.resize((w, h), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        images.append(tensor)

    return {
        "scene_id": scene_id,
        "split": split,
        "chunk": chunk_name,
        "images": torch.stack(images, dim=0),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "frame_ids": [f"{scene_id}:{i:04d}" for i in range(len(images))],
    }


def _build_model(args: argparse.Namespace, device: torch.device) -> MonoVoxelLiteModel:
    repo_root = args.reference_repo.resolve()
    sys.path.insert(0, str(repo_root))

    from src.config import TrainControllerCfg
    from src.model.encoder import get_encoder
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg as RefGaussianAdapterCfg
    from src.model.encoder.mono_model import MonoModelCfg, OpacityMappingCfg as RefOpacityMappingCfg

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

    get_encoder(
        encoder_cfg,
        gs_cube=train_controller_cfg.gs_cube,
        vggt_meta=train_controller_cfg.vggt_meta,
        knn_down=train_controller_cfg.knn_down,
        gaussian_merge=train_controller_cfg.gaussian_merge,
        depth_distillation=train_controller_cfg.depth_distillation,
        train_controller_cfg=train_controller_cfg,
    )

    model = MonoVoxelLiteModel(
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
    ).to(device)
    model.eval()
    return model


def _reshape_gaussian_grid(flat: torch.Tensor, context_size: int, image_height: int, image_width: int) -> torch.Tensor:
    return flat.reshape(context_size, image_height, image_width, *flat.shape[2:])


def _covariance_to_principal_scales(covariances: torch.Tensor) -> torch.Tensor:
    eigvals = torch.linalg.eigvalsh(covariances.float())
    return eigvals.clamp_min(0.0).sqrt()


def _configure_model_num_views(model: MonoVoxelLiteModel, num_views: int) -> None:
    model.depth_predictor.corr_refine_net.cross_view.num_views = int(num_views)
    model.depth_predictor.refine_unet.cross_view.num_views = int(num_views)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (args.image_height, args.image_width))
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]
    frame_ids = scene["frame_ids"]
    max_frame = int(images.shape[0] - 1)

    context_idx = [int(i) for i in args.context_input]
    target_idx = [int(args.target_frame)]
    if len(context_idx) == 0:
        raise ValueError("context-input must not be empty")
    if len(set(context_idx)) != len(context_idx):
        raise ValueError(f"context-input must contain unique frame indices, got {context_idx}")
    if any(i < 0 or i > max_frame for i in context_idx):
        raise ValueError(f"context-input contains an out-of-range frame index; valid range is [0, {max_frame}]")
    if target_idx[0] < 0 or target_idx[0] > max_frame:
        raise ValueError(f"target-frame must be in [0, {max_frame}]")

    model = _build_model(args, device)
    model.load_active_ffgs_checkpoint(str(args.checkpoint.resolve()))
    _configure_model_num_views(model, len(context_idx))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "image": images[context_idx].unsqueeze(0).to(device),
        "intrinsics": intrinsics[context_idx].unsqueeze(0).to(device),
        "extrinsics": extrinsics[context_idx].unsqueeze(0).to(device),
        "near": torch.full((1, len(context_idx)), args.near, dtype=torch.float32, device=device),
        "far": torch.full((1, len(context_idx)), args.far, dtype=torch.float32, device=device),
    }
    target = {
        "image": images[target_idx].unsqueeze(0).to(device),
        "intrinsics": intrinsics[target_idx].unsqueeze(0).to(device),
        "extrinsics": extrinsics[target_idx].unsqueeze(0).to(device),
        "near": torch.full((1, 1), args.near, dtype=torch.float32, device=device),
        "far": torch.full((1, 1), args.far, dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        out = model(
            context["image"],
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            global_step=0,
            deterministic=True,
        )

        b, views, _, _ = target["extrinsics"].shape
        means = rearrange(
            out["gaussians"].means[:, None].expand(b, views, *out["gaussians"].means.shape[1:]),
            "b v g xyz -> (b v) g xyz",
        )
        covariances = rearrange(
            out["gaussians"].covariances[:, None].expand(b, views, *out["gaussians"].covariances.shape[1:]),
            "b v g i j -> (b v) g i j",
        )
        harmonics = rearrange(
            out["gaussians"].harmonics[:, None].expand(b, views, *out["gaussians"].harmonics.shape[1:]),
            "b v g c d_sh -> (b v) g c d_sh",
        )
        opacities = rearrange(
            out["gaussians"].opacities[:, None].expand(b, views, *out["gaussians"].opacities.shape[1:]),
            "b v g -> (b v) g",
        )

        rgb_flat = render_cuda(
            rearrange(target["extrinsics"], "b v i j -> (b v) i j"),
            rearrange(target["intrinsics"], "b v i j -> (b v) i j"),
            rearrange(target["near"], "b v -> (b v)"),
            rearrange(target["far"], "b v -> (b v)"),
            (args.image_height, args.image_width),
            torch.zeros((b * views, 3), dtype=torch.float32, device=device),
            means,
            covariances,
            harmonics,
            opacities,
            scale_invariant=False,
            use_sh=True,
            vggt_meta=True,
        )
        pred_rgb = rgb_flat.reshape(b, views, 3, args.image_height, args.image_width)

        depth_flat = render_depth_cuda(
            rearrange(target["extrinsics"], "b v i j -> (b v) i j"),
            rearrange(target["intrinsics"], "b v i j -> (b v) i j"),
            rearrange(target["near"], "b v -> (b v)"),
            rearrange(target["far"], "b v -> (b v)"),
            (args.image_height, args.image_width),
            means,
            covariances,
            opacities,
            scale_invariant=False,
            mode="depth",
            vggt_meta=True,
        )
        pred_depth = depth_flat.reshape(b, views, args.image_height, args.image_width)

        context_size = len(context_idx)
        gaussian_means = out["gaussians"].means.detach().cpu().reshape(context_size, args.image_height, args.image_width, 3)
        gaussian_covariances = out["gaussians"].covariances.detach().cpu().reshape(
            context_size, args.image_height, args.image_width, 3, 3
        )
        gaussian_opacities = out["gaussians"].opacities.detach().cpu().reshape(
            context_size, args.image_height, args.image_width
        )
        principal_scales = _covariance_to_principal_scales(gaussian_covariances)
        average_scales = principal_scales.mean(dim=-1)

    pred_rgb_path = args.output_dir / f"target_frame_{target_idx[0]:04d}_pred_rgb.png"
    gt_rgb_path = args.output_dir / f"target_frame_{target_idx[0]:04d}_gt_rgb.png"
    pred_depth_path = args.output_dir / f"target_frame_{target_idx[0]:04d}_pred_depth_u16.png"
    gaussian_export_path = args.output_dir / "gaussian_export.pt"
    manifest_path = args.output_dir / "manifest.json"

    _to_u8_rgb(pred_rgb[0, 0]).save(pred_rgb_path)
    _to_u8_rgb(target["image"][0, 0]).save(gt_rgb_path)
    _depth_to_u16(pred_depth[0, 0], args.near, args.far).save(pred_depth_path)
    torch.save(pred_depth[0, 0].detach().cpu(), args.output_dir / f"target_frame_{target_idx[0]:04d}_pred_depth.pt")

    export_payload = {
        "scene_id": args.scene_id,
        "split": args.split,
        "chunk": scene["chunk"],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "image_shape": [args.image_height, args.image_width],
        "near": args.near,
        "far": args.far,
        "context_indices": context_idx,
        "target_index": target_idx[0],
        "context_frame_ids": [frame_ids[i] for i in context_idx],
        "target_frame_id": frame_ids[target_idx[0]],
        "gaussian_means": gaussian_means,
        "gaussian_covariances": gaussian_covariances,
        "gaussian_opacities": gaussian_opacities,
        "gaussian_principal_scales": principal_scales,
        "gaussian_average_scales": average_scales,
    }
    torch.save(export_payload, gaussian_export_path)

    manifest = {
        "scene_id": args.scene_id,
        "split": args.split,
        "chunk": scene["chunk"],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "image_shape": [args.image_height, args.image_width],
        "near": args.near,
        "far": args.far,
        "context_indices": context_idx,
        "target_index": target_idx[0],
        "pred_rgb_path": str(pred_rgb_path),
        "gt_rgb_path": str(gt_rgb_path),
        "pred_depth_path": str(pred_depth_path),
        "gaussian_export_path": str(gaussian_export_path),
        "num_gaussians": int(out["gaussians"].means.shape[1]),
        "average_scale_mean": float(average_scales.mean().item()),
        "average_scale_min": float(average_scales.min().item()),
        "average_scale_max": float(average_scales.max().item()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

