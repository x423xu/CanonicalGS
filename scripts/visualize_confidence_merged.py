from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import cm
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_mono_voxel_lite_scene import _load_scene_tensor_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--render-dir", type=Path, default=Path("outputs/651a7f83ed093001/render"))
    parser.add_argument("--checkpoint-tag", type=str, default="epoch_8-step_280000")
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--depth-max", type=float, default=120.0)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render/multicontext_confidence"))
    return parser.parse_args()


def _intrinsics_to_pixels(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    k_px = k.clone()
    k_px[0, :] *= w
    k_px[1, :] *= h
    return k_px


def _depth_to_world_points(
    depth: torch.Tensor,
    image: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    k_px = _intrinsics_to_pixels(intrinsic, h, w)
    fx = float(k_px[0, 0].item())
    fy = float(k_px[1, 1].item())
    cx = float(k_px[0, 2].item())
    cy = float(k_px[1, 2].item())

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    z = depth.float()
    valid = torch.isfinite(z) & (z > 1e-6) & (z < depth_max)
    x = (xx - cx) * z / max(fx, 1e-6)
    y = (yy - cy) * z / max(fy, 1e-6)

    cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
    cam = cam[valid]
    world = (extrinsic_c2w @ cam.T).T[:, :3]

    color = image.permute(1, 2, 0)[valid]
    return world.cpu().numpy(), color.cpu().numpy()


def _project_accumulate_confidence(
    points_world: np.ndarray,
    confidence: np.ndarray,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate confidence values at each pixel."""
    k_px = _intrinsics_to_pixels(intrinsic, h, w).cpu().numpy()
    world_to_cam = torch.linalg.inv(extrinsic_c2w).cpu().numpy()

    n = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    conf = confidence[valid]

    pix = (k_px @ cam.T).T
    u = pix[:, 0] / np.maximum(pix[:, 2], 1e-6)
    v = pix[:, 1] / np.maximum(pix[:, 2], 1e-6)

    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inb]
    v = v[inb]
    conf = conf[inb]

    conf_sum = np.zeros((h, w), dtype=np.float32)
    conf_count = np.zeros((h, w), dtype=np.int32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    ui = np.clip(ui, 0, w - 1)
    vi = np.clip(vi, 0, h - 1)

    np.add.at(conf_sum, (vi, ui), conf)
    np.add.at(conf_count, (vi, ui), 1)

    mask = conf_count > 0
    conf_avg = np.zeros((h, w), dtype=np.float32)
    conf_avg[mask] = conf_sum[mask] / conf_count[mask]

    return conf_avg, mask


def _interpolate_missing_regions(conf_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill missing regions using nearest neighbor interpolation."""
    h, w = conf_img.shape
    filled = conf_img.copy()
    missing_mask = ~mask

    if missing_mask.sum() == 0:
        return filled

    dist, idx = distance_transform_edt(missing_mask, return_indices=True)
    filled[missing_mask] = conf_img[idx[0, missing_mask], idx[1, missing_mask]]
    
    return filled


def _voxel_uncertainty(points: np.ndarray, grid_size: int) -> dict[str, Any]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    norm = (points - mins) / span
    idx = np.floor(norm * (grid_size - 1)).astype(np.int32)
    idx = np.clip(idx, 0, grid_size - 1)

    flat = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)

    sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    sums2 = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inv, points)
    np.add.at(sums2, inv, points * points)

    means = sums / counts[:, None]
    var = np.maximum(sums2 / counts[:, None] - means * means, 0.0)
    unc = var.sum(axis=1)

    unc_norm = unc / max(float(np.percentile(unc, 95)), 1e-12)
    conf = 1.0 / (1.0 + unc_norm)
    
    # Normalize confidence to [0, 1] range
    conf_min = conf.min()
    conf_max = conf.max()
    if conf_max > conf_min:
        conf = (conf - conf_min) / (conf_max - conf_min)
    else:
        conf = np.ones_like(conf)

    return {
        "confidence": conf,
        "inv": inv,
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.render_dir / "manifest.json").read_text(encoding="utf-8"))
    ckpt = None
    for item in manifest["checkpoints"]:
        if Path(item["checkpoint"]).stem == args.checkpoint_tag:
            ckpt = item
            break
    if ckpt is None:
        raise RuntimeError(f"checkpoint tag not found: {args.checkpoint_tag}")

    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (256, 256))
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    ckpt_dir = args.render_dir / args.checkpoint_tag
    
    # Collect confidence views for all context sizes
    all_views = {}
    for context_size in args.context_sizes:
        print(f"Processing context_size={context_size}...")
        
        ctx_item = None
        for c in ckpt["contexts"]:
            if int(c["context_size"]) == context_size:
                ctx_item = c
                break
        if ctx_item is None:
            print(f"  ✗ context size {context_size} not found")
            continue

        target_indices = list(ctx_item["target_indices"])
        novel_pos = 1
        novel_frame = int(target_indices[novel_pos])

        world_points = []
        for view_i, frame_idx in enumerate(target_indices):
            depth_path = ckpt_dir / f"ctx_{context_size}" / f"t{view_i:02d}_f{int(frame_idx):04d}_pred_depth.pt"
            if not depth_path.exists():
                print(f"  ✗ missing depth: {depth_path}")
                break
            depth = torch.load(depth_path, map_location="cpu")
            img = images[frame_idx].cpu()
            pts, _ = _depth_to_world_points(
                depth=depth,
                image=img,
                intrinsic=intrinsics[frame_idx].cpu(),
                extrinsic_c2w=extrinsics[frame_idx].cpu(),
                depth_max=args.depth_max,
            )
            world_points.append(pts)
        
        if len(world_points) < len(target_indices):
            print(f"  ✗ incomplete point data")
            continue

        pts_all = np.concatenate(world_points, axis=0)
        
        # Compute voxel confidence
        vox = _voxel_uncertainty(pts_all, args.grid_size)
        inv_idx = vox["inv"]
        conf = vox["confidence"].astype(np.float32)
        
        # Map points to voxel confidence
        point_confidence = conf[inv_idx].astype(np.float32)
        
        # Project and accumulate confidence
        conf_img_raw, conf_mask = _project_accumulate_confidence(
            points_world=pts_all,
            confidence=point_confidence,
            intrinsic=intrinsics[novel_frame].cpu(),
            extrinsic_c2w=extrinsics[novel_frame].cpu(),
            h=images.shape[-2],
            w=images.shape[-1],
        )
        
        # Interpolate missing regions
        conf_img_filled = _interpolate_missing_regions(conf_img_raw, conf_mask)
        
        # Normalize
        conf_filled_min = conf_img_filled.min()
        conf_filled_max = conf_img_filled.max()
        if conf_filled_max > conf_filled_min:
            conf_img_filled = (conf_img_filled - conf_filled_min) / (conf_filled_max - conf_filled_min)
        
        # Convert to RGB
        cmap_fn = cm.get_cmap("coolwarm")
        conf_img_rgb = cmap_fn(conf_img_filled)[:, :, :3].astype(np.float32)
        
        # Get GT image
        gt_img = images[novel_frame].permute(1, 2, 0).cpu().numpy()
        
        # Store views
        all_views[context_size] = {
            "gt": gt_img,
            "conf_raw": conf_img_raw,
            "conf_mask": conf_mask,
            "conf_filled": conf_img_filled,
            "conf_rgb": conf_img_rgb,
            "cmap_fn": cmap_fn,
        }
        print(f"  ✓ processed")

    # Create merged figure (5 rows × 4 columns)
    fig, axs = plt.subplots(len(args.context_sizes), 4, figsize=(20, 5*len(args.context_sizes)))
    
    # Add column titles
    col_titles = ["GT View", "Confidence (Raw)", "Confidence (Filled)", "Blended (50% GT + 50% Conf)"]
    for col_idx, title in enumerate(col_titles):
        axs[0, col_idx].text(0.5, 1.08, title, transform=axs[0, col_idx].transAxes, 
                             ha="center", va="bottom", fontsize=14, weight="bold")
    
    # Fill in the subplots
    for row_idx, context_size in enumerate(args.context_sizes):
        if context_size not in all_views:
            continue
        
        v = all_views[context_size]
        
        # Row label (context size)
        axs[row_idx, 0].text(-0.35, 0.5, f"Ctx {context_size}", transform=axs[row_idx, 0].transAxes,
                             ha="center", va="center", fontsize=12, weight="bold", rotation=90)
        
        # GT view
        axs[row_idx, 0].imshow(np.clip(v["gt"], 0.0, 1.0))
        axs[row_idx, 0].axis("off")
        
        # Confidence (raw, before fill)
        conf_img_before = np.zeros((v["gt"].shape[0], v["gt"].shape[1], 3), dtype=np.float32)
        conf_img_before[v["conf_mask"]] = v["cmap_fn"](v["conf_raw"][v["conf_mask"]])[:, :3]
        axs[row_idx, 1].imshow(np.clip(conf_img_before, 0.0, 1.0))
        axs[row_idx, 1].axis("off")
        
        # Confidence (filled)
        axs[row_idx, 2].imshow(np.clip(v["conf_rgb"], 0.0, 1.0))
        axs[row_idx, 2].axis("off")
        
        # Blended
        blend = 0.5 * np.clip(v["gt"], 0.0, 1.0) + 0.5 * np.clip(v["conf_rgb"], 0.0, 1.0)
        axs[row_idx, 3].imshow(blend)
        axs[row_idx, 3].axis("off")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", pad=0.02, fraction=0.02)
    cbar.set_label("Confidence (0=Low, 1=High)", fontsize=12)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.96, left=0.08, right=0.99, hspace=0.15)
    fig.savefig(args.out_dir / "merged_confidence_all_contexts.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ saved merged_confidence_all_contexts.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
