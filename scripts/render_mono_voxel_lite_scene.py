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
from canonicalgs.model.mono_voxel_lite import (
    MonoVoxelLiteConfig,
    MonoVoxelLiteModel,
    OpacityMappingCfg,
    render_mono_voxel_lite,
)
from canonicalgs.reference_ffgs.model.decoder.cuda_splatting import render_cuda, render_depth_cuda
from canonicalgs.reference_ffgs.model.encoder.common.gaussian_adapter import GaussianAdapterCfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--scene-id", type=str, required=True)
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
        "--checkpoint-dir",
        type=Path,
        default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths. If omitted, all *.ckpt under --checkpoint-dir are used.",
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--max-gap-per-context", type=int, nargs="+", default=[50, 60, 70, 80, 90])
    parser.add_argument("--target-views", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/651a7f83ed093001/render"),
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


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        checkpoints = [p.resolve() for p in args.checkpoint]
    else:
        checkpoints = sorted(args.checkpoint_dir.glob("*.ckpt"))
    missing = [str(p) for p in checkpoints if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint files: {missing}")
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints found; set --checkpoint or provide *.ckpt in {args.checkpoint_dir}"
        )
    return checkpoints


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if count > length:
        raise ValueError(f"cannot sample {count} from length {length}")
    if count == 1:
        return [length // 2]
    positions = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    dedup: list[int] = []
    seen: set[int] = set()
    for idx in positions:
        if idx not in seen:
            dedup.append(idx)
            seen.add(idx)
    cursor = 0
    while len(dedup) < count and cursor < length:
        if cursor not in seen:
            dedup.append(cursor)
            seen.add(cursor)
        cursor += 1
    return sorted(dedup[:count])


def _context_indices_with_max_span(total_frames: int, context_size: int, max_span: int) -> list[int]:
    if context_size > total_frames:
        raise ValueError(f"context_size {context_size} exceeds total_frames {total_frames}")
    span = min(max_span, total_frames - 1)
    span = max(span, context_size - 1)
    start = (total_frames - 1 - span) // 2
    local = _evenly_spaced_indices(span + 1, context_size)
    return [start + i for i in local]


def _to_u8_rgb(image: torch.Tensor) -> Image.Image:
    tensor = image.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _depth_to_u16(depth: torch.Tensor, near: float, far: float) -> Image.Image:
    tensor = depth.detach().cpu().clamp(min=near, max=far)
    norm = ((tensor - near) / max(far - near, 1e-6)).clamp(0.0, 1.0)
    arr = (norm * 65535.0).to(torch.uint16).numpy()
    return Image.fromarray(arr, mode="I;16")


def _load_scene_tensor_episode(
    root: Path,
    split: str,
    scene_id: str,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
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


def main() -> None:
    args = _parse_args()
    if len(args.max_gap_per_context) != len(args.context_sizes):
        raise ValueError("max-gap-per-context must match context-sizes length")
    max_gap_by_context = {int(c): int(g) for c, g in zip(args.context_sizes, args.max_gap_per_context)}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints = _resolve_checkpoints(args)
    scene = _load_scene_tensor_episode(
        args.root,
        args.split,
        args.scene_id,
        (args.image_height, args.image_width),
    )

    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]
    frame_ids = scene["frame_ids"]
    total_frames = images.shape[0]

    model = _build_model(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "scene_id": args.scene_id,
        "split": args.split,
        "chunk": scene["chunk"],
        "device": str(device),
        "image_shape": [args.image_height, args.image_width],
        "near": args.near,
        "far": args.far,
        "target_views": args.target_views,
        "context_sizes": args.context_sizes,
        "max_gap_per_context": max_gap_by_context,
        "num_frames": int(total_frames),
        "checkpoints": [],
    }

    with torch.no_grad():
        for checkpoint_path in checkpoints:
            model.load_active_ffgs_checkpoint(str(checkpoint_path))
            ckpt_tag = checkpoint_path.stem
            ckpt_dir = args.output_dir / ckpt_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_meta: dict[str, Any] = {
                "checkpoint": str(checkpoint_path),
                "contexts": [],
            }

            for context_size in args.context_sizes:
                if context_size + args.target_views > total_frames:
                    raise ValueError(
                        f"scene has {total_frames} frames, but needs at least {context_size + args.target_views}"
                    )

                max_gap = max_gap_by_context[context_size]
                context_idx = _context_indices_with_max_span(total_frames, context_size, max_gap)
                context_set = set(context_idx)
                in_span = [i for i in range(context_idx[0], context_idx[-1] + 1) if i not in context_set]
                if len(in_span) < args.target_views:
                    raise ValueError(
                        f"insufficient in-span target frames for context_size={context_size}, span={max_gap}"
                    )
                target_rel = _evenly_spaced_indices(len(in_span), args.target_views)
                target_idx = [in_span[i] for i in target_rel]

                context = {
                    "image": images[context_idx].unsqueeze(0).to(device),
                    "intrinsics": intrinsics[context_idx].unsqueeze(0).to(device),
                    "extrinsics": extrinsics[context_idx].unsqueeze(0).to(device),
                    "near": torch.full((1, context_size), args.near, dtype=torch.float32, device=device),
                    "far": torch.full((1, context_size), args.far, dtype=torch.float32, device=device),
                }
                target = {
                    "image": images[target_idx].unsqueeze(0).to(device),
                    "intrinsics": intrinsics[target_idx].unsqueeze(0).to(device),
                    "extrinsics": extrinsics[target_idx].unsqueeze(0).to(device),
                    "near": torch.full((1, args.target_views), args.near, dtype=torch.float32, device=device),
                    "far": torch.full((1, args.target_views), args.far, dtype=torch.float32, device=device),
                }

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
                    out["gaussians"].covariances[:, None].expand(
                        b, views, *out["gaussians"].covariances.shape[1:]
                    ),
                    "b v g i j -> (b v) g i j",
                )
                harmonics = rearrange(
                    out["gaussians"].harmonics[:, None].expand(
                        b, views, *out["gaussians"].harmonics.shape[1:]
                    ),
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

                ctx_dir = ckpt_dir / f"ctx_{context_size}"
                ctx_dir.mkdir(parents=True, exist_ok=True)

                for view_i in range(args.target_views):
                    frame_idx = target_idx[view_i]
                    frame_tag = f"t{view_i:02d}_f{frame_idx:04d}"

                    _to_u8_rgb(pred_rgb[0, view_i]).save(ctx_dir / f"{frame_tag}_pred_rgb.png")
                    _to_u8_rgb(target["image"][0, view_i]).save(ctx_dir / f"{frame_tag}_gt_rgb.png")
                    _depth_to_u16(pred_depth[0, view_i], args.near, args.far).save(
                        ctx_dir / f"{frame_tag}_pred_depth_u16.png"
                    )
                    torch.save(pred_depth[0, view_i].detach().cpu(), ctx_dir / f"{frame_tag}_pred_depth.pt")

                ckpt_meta["contexts"].append(
                    {
                        "context_size": context_size,
                        "context_indices": context_idx,
                        "target_indices": target_idx,
                        "context_frame_ids": [frame_ids[i] for i in context_idx],
                        "target_frame_ids": [frame_ids[i] for i in target_idx],
                        "output_dir": str(ctx_dir),
                    }
                )

            (ckpt_dir / "manifest.json").write_text(json.dumps(ckpt_meta, indent=2), encoding="utf-8")
            manifest["checkpoints"].append(ckpt_meta)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "checkpoints": len(checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
