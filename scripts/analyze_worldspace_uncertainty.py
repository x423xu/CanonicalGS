from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--checkpoint-tag", type=str, default="epoch_9-step_300000")
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--depth-max", type=float, default=120.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render/worldspace_analysis_ctx10"))
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


def _project_points_to_camera(
    points_world: np.ndarray,
    colors: np.ndarray,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_px = _intrinsics_to_pixels(intrinsic, h, w).cpu().numpy()
    world_to_cam = torch.linalg.inv(extrinsic_c2w).cpu().numpy()

    n = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    colors = colors[valid]

    pix = (k_px @ cam.T).T
    u = pix[:, 0] / np.maximum(pix[:, 2], 1e-6)
    v = pix[:, 1] / np.maximum(pix[:, 2], 1e-6)

    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inb]
    v = v[inb]
    z = cam[inb, 2]
    colors = colors[inb]

    img = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    ui = np.clip(ui, 0, w - 1)
    vi = np.clip(vi, 0, h - 1)

    order = np.argsort(z)
    for idx in order:
        x = ui[idx]
        y = vi[idx]
        if z[idx] < zbuf[y, x]:
            zbuf[y, x] = z[idx]
            img[y, x] = colors[idx]

    mask = np.isfinite(zbuf)
    return img, mask, zbuf


def _project_accumulate_confidence(
    points_world: np.ndarray,
    confidence: np.ndarray,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate confidence values at each pixel (sum + count) instead of overwriting."""
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

    # Accumulate confidence: sum + count
    conf_sum = np.zeros((h, w), dtype=np.float32)
    conf_count = np.zeros((h, w), dtype=np.int32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    ui = np.clip(ui, 0, w - 1)
    vi = np.clip(vi, 0, h - 1)

    np.add.at(conf_sum, (vi, ui), conf)
    np.add.at(conf_count, (vi, ui), 1)

    # Average confidence per pixel
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

    # Compute distance to nearest known pixel
    dist, idx = distance_transform_edt(missing_mask, return_indices=True)
    
    # Fill missing pixels with nearest neighbor values
    filled[missing_mask] = conf_img[idx[0, missing_mask], idx[1, missing_mask]]
    
    return filled


def _psnr_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 8:
        return float("nan")
    diff = pred[mask] - gt[mask]
    mse = float(np.mean(diff * diff))
    return float(-10.0 * np.log10(max(mse, 1e-12)))


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
    
    # Normalize confidence to [0, 1] range for better visualization
    conf_min = conf.min()
    conf_max = conf.max()
    if conf_max > conf_min:
        conf = (conf - conf_min) / (conf_max - conf_min)
    else:
        conf = np.ones_like(conf)

    gx = uniq // (grid_size * grid_size)
    gy = (uniq // grid_size) % grid_size
    gz = uniq % grid_size

    return {
        "mins": mins,
        "maxs": maxs,
        "grid_xyz": np.stack([gx, gy, gz], axis=1),
        "counts": counts,
        "uncertainty": unc,
        "confidence": conf,
        "grid_size": grid_size,
        "inv": inv,  # Return inverse mapping for point-to-voxel lookup
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

    ctx_item = None
    for c in ckpt["contexts"]:
        if int(c["context_size"]) == args.context_size:
            ctx_item = c
            break
    if ctx_item is None:
        raise RuntimeError(f"context size not found: {args.context_size}")

    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (256, 256))
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    ckpt_dir = args.render_dir / args.checkpoint_tag / f"ctx_{args.context_size}"
    target_indices = list(ctx_item["target_indices"])

    world_points = []
    world_colors = []
    per_view_counts = []
    for view_i, frame_idx in enumerate(target_indices):
        depth_path = ckpt_dir / f"t{view_i:02d}_f{int(frame_idx):04d}_pred_depth.pt"
        if not depth_path.exists():
            raise FileNotFoundError(f"missing depth file: {depth_path}")
        depth = torch.load(depth_path, map_location="cpu")
        img = images[frame_idx].cpu()
        pts, col = _depth_to_world_points(
            depth=depth,
            image=img,
            intrinsic=intrinsics[frame_idx].cpu(),
            extrinsic_c2w=extrinsics[frame_idx].cpu(),
            depth_max=args.depth_max,
        )
        world_points.append(pts)
        world_colors.append(col)
        per_view_counts.append(int(pts.shape[0]))

    pts_all = np.concatenate(world_points, axis=0)
    col_all = np.concatenate(world_colors, axis=0)

    if len(target_indices) < 3:
        raise RuntimeError("need at least 3 target views for projection check")
    novel_pos = 1
    novel_frame = int(target_indices[novel_pos])

    src_pts = []
    src_col = []
    for i in range(len(target_indices)):
        if i == novel_pos:
            continue
        src_pts.append(world_points[i])
        src_col.append(world_colors[i])
    src_pts = np.concatenate(src_pts, axis=0)
    src_col = np.concatenate(src_col, axis=0)

    novel_img, novel_mask, _ = _project_points_to_camera(
        points_world=src_pts,
        colors=src_col,
        intrinsic=intrinsics[novel_frame].cpu(),
        extrinsic_c2w=extrinsics[novel_frame].cpu(),
        h=images.shape[-2],
        w=images.shape[-1],
    )
    gt_img = images[novel_frame].permute(1, 2, 0).cpu().numpy()
    psnr_novel = _psnr_masked(novel_img, gt_img, novel_mask)
    coverage = float(novel_mask.mean())

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.clip(novel_img, 0.0, 1.0))
    axs[0].set_title(f"Reprojected novel view\ncoverage={coverage:.3f}")
    axs[1].imshow(np.clip(gt_img, 0.0, 1.0))
    axs[1].set_title("GT novel image")
    vis_mask = np.zeros_like(gt_img)
    vis_mask[..., 1] = novel_mask.astype(np.float32)
    axs[2].imshow(vis_mask)
    axs[2].set_title(f"Valid mask\nPSNR(masked)={psnr_novel:.2f} dB")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "step1_projection_check.png", dpi=220)
    plt.close(fig)

    # Step 3: Voxel uncertainty directly on all points (no outlier removal)
    vox = _voxel_uncertainty(pts_all, args.grid_size)
    gxyz = vox["grid_xyz"].astype(np.float32)
    conf = vox["confidence"].astype(np.float32)
    inv_idx = vox["inv"]  # Voxel index for each point

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    sample_v = np.random.default_rng(2).choice(len(gxyz), size=min(25000, len(gxyz)), replace=False)
    p = ax.scatter(gxyz[sample_v, 0], gxyz[sample_v, 1], gxyz[sample_v, 2], c=conf[sample_v], s=3, cmap="coolwarm", alpha=0.6)
    fig.colorbar(p, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("World-space voxel confidence (all points)")

    grid = np.full((args.grid_size, args.grid_size, args.grid_size), np.nan, dtype=np.float32)
    grid[gxyz[:, 0].astype(int), gxyz[:, 1].astype(int), gxyz[:, 2].astype(int)] = conf

    mid = args.grid_size // 2
    im = fig.add_subplot(2, 2, 2)
    q = im.imshow(grid[:, :, mid].T, origin="lower", cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(q, ax=im, fraction=0.046, pad=0.04)
    im.set_title("Confidence slice XY")

    im = fig.add_subplot(2, 2, 3)
    q = im.imshow(grid[:, mid, :].T, origin="lower", cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(q, ax=im, fraction=0.046, pad=0.04)
    im.set_title("Confidence slice XZ")

    im = fig.add_subplot(2, 2, 4)
    q = im.imshow(grid[mid, :, :].T, origin="lower", cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(q, ax=im, fraction=0.046, pad=0.04)
    im.set_title("Confidence slice YZ")

    fig.tight_layout()
    fig.savefig(args.out_dir / "step3_worldspace_confidence.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        args.out_dir / "voxel_confidence_volume.npz",
        grid_xyz=gxyz,
        confidence=conf,
        uncertainty=vox["uncertainty"].astype(np.float32),
        counts=vox["counts"].astype(np.int32),
        mins=vox["mins"].astype(np.float32),
        maxs=vox["maxs"].astype(np.float32),
        grid_size=np.array([args.grid_size], dtype=np.int32),
    )

    # STEP 4: Render point cloud confidence accumulated (sum + count, then average)
    # Map each point to its voxel's confidence value
    point_confidence = conf[inv_idx].astype(np.float32)
    
    # Project points and accumulate confidence instead of overwriting
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
    
    # Normalize accumulated confidence to [0, 1] range
    conf_filled_min = conf_img_filled.min()
    conf_filled_max = conf_img_filled.max()
    if conf_filled_max > conf_filled_min:
        conf_img_filled = (conf_img_filled - conf_filled_min) / (conf_filled_max - conf_filled_min)
    
    # Normalize accumulated confidence to [0, 1] range
    conf_filled_min = conf_img_filled.min()
    conf_filled_max = conf_img_filled.max()
    if conf_filled_max > conf_filled_min:
        conf_img_filled = (conf_img_filled - conf_filled_min) / (conf_filled_max - conf_filled_min)
    
    # Convert to RGB using turbo colormap
    cmap_fn = cm.get_cmap("coolwarm")
    conf_img_rgb = cmap_fn(conf_img_filled)[:, :, :3].astype(np.float32)
    
    # Compositing: show GT, confidence raw, confidence filled
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    axs[0].imshow(np.clip(gt_img, 0.0, 1.0))
    axs[0].set_title("GT novel view")
    axs[0].axis("off")
    
    # Raw accumulated confidence (with holes)
    conf_img_before_fill = np.zeros((images.shape[-2], images.shape[-1], 3), dtype=np.float32)
    conf_img_before_fill[conf_mask] = cmap_fn(conf_img_raw[conf_mask])[:, :3]
    axs[1].imshow(np.clip(conf_img_before_fill, 0.0, 1.0))
    axs[1].set_title("Accumulated Confidence (before fill)")
    axs[1].axis("off")
    
    axs[2].imshow(np.clip(conf_img_rgb, 0.0, 1.0))
    axs[2].set_title("Accumulated Confidence (filled)")
    axs[2].axis("off")
    
    # Blended view
    blend = 0.5 * np.clip(gt_img, 0.0, 1.0) + 0.5 * np.clip(conf_img_rgb, 0.0, 1.0)
    axs[3].imshow(blend)
    axs[3].set_title("Blended (50% GT + 50% Conf)")
    axs[3].axis("off")
    
    fig.tight_layout()
    fig.savefig(args.out_dir / "step4_confidence_rendered_view.png", dpi=220)
    plt.close(fig)

    report = {
        "scene_id": args.scene_id,
        "checkpoint": args.checkpoint_tag,
        "context_size": args.context_size,
        "target_indices": target_indices,
        "step1": {
            "per_view_world_points": per_view_counts,
            "novel_frame": novel_frame,
            "projection_coverage": coverage,
            "masked_psnr": psnr_novel,
        },
        "step3": {
            "total_points": int(len(pts_all)),
            "grid_size": args.grid_size,
            "occupied_voxels": int(len(gxyz)),
            "confidence_mean": float(np.mean(conf)),
            "confidence_p10": float(np.percentile(conf, 10)),
            "confidence_p50": float(np.percentile(conf, 50)),
            "confidence_p90": float(np.percentile(conf, 90)),
            "note": "Computed from ALL points without outlier removal",
        },
        "step4": {
            "accumulation_method": "sum + count (average per pixel)",
            "interpolation_method": "nearest neighbor fill for missing regions",
            "colormap": "coolwarm",
            "pixels_with_coverage": int(conf_mask.sum()),
            "total_pixels": int(conf_mask.size),
            "coverage_ratio": float(conf_mask.sum() / conf_mask.size),
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
