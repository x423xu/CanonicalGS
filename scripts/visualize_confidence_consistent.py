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
    parser.add_argument("--checkpoint-tag", type=str, default="epoch_8-step_280000")
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--depth-max", type=float, default=80.0)
    parser.add_argument("--novel-frame", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render/multicontext_confidence"))
    parser.add_argument("--max-search-grid", type=int, default=131072)
    parser.add_argument("--grid-scale", type=float, default=1.0)
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

    ui = np.clip(np.round(u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, h - 1)

    np.add.at(conf_sum, (vi, ui), conf)
    np.add.at(conf_count, (vi, ui), 1)

    mask = conf_count > 0
    conf_avg = np.zeros((h, w), dtype=np.float32)
    conf_avg[mask] = conf_sum[mask] / conf_count[mask]
    return conf_avg, mask


def _interpolate_missing_regions(conf_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = conf_img.copy()
    missing_mask = ~mask
    if not np.any(missing_mask):
        return filled

    _, idx = distance_transform_edt(missing_mask, return_indices=True)
    filled[missing_mask] = conf_img[idx[0, missing_mask], idx[1, missing_mask]]
    return filled


def _compute_voxel_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    rel = (points - origin[None, :]) / max(voxel_size, 1e-12)
    return np.floor(rel).astype(np.int64)


def _max_occupancy(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> int:
    idx = _compute_voxel_indices(points, origin, voxel_size)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    return int(counts.max())


def _derive_ctx2_voxel_size(points_ctx2: np.ndarray, max_grid: int) -> tuple[float, np.ndarray, int, int]:
    origin = points_ctx2.min(axis=0)
    span = np.maximum(points_ctx2.max(axis=0) - origin, 1e-9)

    start_grid = max(4, int(np.ceil(np.cbrt(points_ctx2.shape[0]))))
    grid = start_grid

    # Search finer grids until max occupancy <= 1 or limit reached.
    while grid <= max_grid:
        voxel_size = float(np.max(span) / grid)
        occ = _max_occupancy(points_ctx2, origin, voxel_size)
        if occ <= 1:
            return voxel_size, origin, grid, occ
        grid *= 2

    # If exact <=1 is not reachable due duplicate points, return finest tested.
    grid = max_grid
    voxel_size = float(np.max(span) / grid)
    occ = _max_occupancy(points_ctx2, origin, voxel_size)
    return voxel_size, origin, grid, occ


def _voxel_uncertainty_fixed(
    points: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> dict[str, Any]:
    idx = _compute_voxel_indices(points, origin, voxel_size)
    uniq, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)

    sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    sums2 = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inv, points)
    np.add.at(sums2, inv, points * points)

    means = sums / counts[:, None]
    var = np.maximum(sums2 / counts[:, None] - means * means, 0.0)
    unc = var.sum(axis=1)

    denom = max(float(np.percentile(unc, 95)), 1e-12)
    unc_norm = unc / denom
    conf = 1.0 / (1.0 + unc_norm)

    conf_min = float(conf.min())
    conf_max = float(conf.max())
    if conf_max > conf_min:
        conf = (conf - conf_min) / (conf_max - conf_min)
    else:
        conf = np.ones_like(conf)

    return {"confidence": conf.astype(np.float32), "inv": inv, "counts": counts}


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

    # Build ctx2 points once and derive a physical voxel size from it.
    ctx2_item = None
    for c in ckpt["contexts"]:
        if int(c["context_size"]) == 2:
            ctx2_item = c
            break
    if ctx2_item is None:
        raise RuntimeError("context size 2 not found")

    ctx2_targets = list(ctx2_item["target_indices"])
    pts_ctx2_list = []
    for view_i, frame_idx in enumerate(ctx2_targets):
        depth_path = ckpt_dir / "ctx_2" / f"t{view_i:02d}_f{int(frame_idx):04d}_pred_depth.pt"
        if not depth_path.exists():
            raise RuntimeError(f"missing depth for ctx2: {depth_path}")
        depth = torch.load(depth_path, map_location="cpu")
        pts, _ = _depth_to_world_points(
            depth=depth,
            image=images[frame_idx].cpu(),
            intrinsic=intrinsics[frame_idx].cpu(),
            extrinsic_c2w=extrinsics[frame_idx].cpu(),
            depth_max=args.depth_max,
        )
        pts_ctx2_list.append(pts)
    pts_ctx2 = np.concatenate(pts_ctx2_list, axis=0)

    voxel_size, voxel_origin, ref_grid, ref_occ = _derive_ctx2_voxel_size(pts_ctx2, args.max_search_grid)

    if args.grid_scale <= 0:
        raise ValueError("grid-scale must be > 0")
    if args.grid_scale != 1.0:
        ref_grid = int(round(ref_grid * args.grid_scale))
        voxel_size = voxel_size / float(args.grid_scale)
        ref_occ = _max_occupancy(pts_ctx2, voxel_origin, voxel_size)

    print("=" * 72)
    print("Fixed voxel size derived from ctx2")
    print("=" * 72)
    print(f"ctx2 points: {pts_ctx2.shape[0]}")
    print(f"reference search grid: {ref_grid}")
    print(f"voxel size (world units): {voxel_size:.8f}")
    print(f"ctx2 max points per voxel: {ref_occ}")

    all_views: dict[int, dict[str, Any]] = {}
    summary_lines = [
        "Fixed Voxel Size Configuration",
        "==============================",
        "Reference context: 2",
        f"ctx2 points: {pts_ctx2.shape[0]}",
        f"reference grid used in search: {ref_grid}",
        f"voxel size (world units): {voxel_size:.8f}",
        f"ctx2 max points per voxel: {ref_occ}",
        "",
        "Per-context occupancy stats:",
    ]

    for context_size in args.context_sizes:
        print(f"\nProcessing context_size={context_size}...")
        ctx_item = None
        for c in ckpt["contexts"]:
            if int(c["context_size"]) == context_size:
                ctx_item = c
                break
        if ctx_item is None:
            print(f"  missing context {context_size}")
            continue

        target_indices = list(ctx_item["target_indices"])
        if args.novel_frame is not None:
            novel_frame = int(args.novel_frame)
        elif len(target_indices) >= 2:
            novel_frame = int(target_indices[1])
        else:
            novel_frame = int(target_indices[0])

        pts_list = []
        for view_i, frame_idx in enumerate(target_indices):
            depth_path = ckpt_dir / f"ctx_{context_size}" / f"t{view_i:02d}_f{int(frame_idx):04d}_pred_depth.pt"
            if not depth_path.exists():
                print(f"  missing depth: {depth_path}")
                pts_list = []
                break

            depth = torch.load(depth_path, map_location="cpu")
            pts, _ = _depth_to_world_points(
                depth=depth,
                image=images[frame_idx].cpu(),
                intrinsic=intrinsics[frame_idx].cpu(),
                extrinsic_c2w=extrinsics[frame_idx].cpu(),
                depth_max=args.depth_max,
            )
            pts_list.append(pts)

        if not pts_list:
            continue

        pts_all = np.concatenate(pts_list, axis=0)
        vox = _voxel_uncertainty_fixed(pts_all, voxel_size=voxel_size, origin=voxel_origin)
        point_confidence = vox["confidence"][vox["inv"]]

        occ_counts = vox["counts"]
        max_occ = int(occ_counts.max())
        p99_occ = float(np.percentile(occ_counts, 99))
        print(f"  points: {pts_all.shape[0]}")
        print(f"  occupied voxels: {occ_counts.shape[0]}")
        print(f"  max points/voxel: {max_occ}")
        print(f"  p99 points/voxel: {p99_occ:.2f}")

        summary_lines.append(
            f"ctx{context_size}: points={pts_all.shape[0]}, occupied_voxels={occ_counts.shape[0]}, max_occ={max_occ}, p99_occ={p99_occ:.2f}"
        )

        conf_raw, conf_mask = _project_accumulate_confidence(
            points_world=pts_all,
            confidence=point_confidence,
            intrinsic=intrinsics[novel_frame].cpu(),
            extrinsic_c2w=extrinsics[novel_frame].cpu(),
            h=images.shape[-2],
            w=images.shape[-1],
        )

        conf_filled = _interpolate_missing_regions(conf_raw, conf_mask)
        cf_min = float(conf_filled.min())
        cf_max = float(conf_filled.max())
        if cf_max > cf_min:
            conf_filled = (conf_filled - cf_min) / (cf_max - cf_min)

        cmap_fn = cm.get_cmap("coolwarm")
        conf_rgb = cmap_fn(conf_filled)[:, :, :3].astype(np.float32)
        gt = images[novel_frame].permute(1, 2, 0).cpu().numpy()

        all_views[context_size] = {
            "gt": gt,
            "conf_raw": conf_raw,
            "conf_mask": conf_mask,
            "conf_rgb": conf_rgb,
            "cmap_fn": cmap_fn,
        }

    fig, axs = plt.subplots(len(args.context_sizes), 4, figsize=(20, 5 * len(args.context_sizes)))
    if len(args.context_sizes) == 1:
        axs = np.expand_dims(axs, axis=0)
    col_titles = ["GT View", "Confidence (Raw)", "Confidence (Filled)", "Blended (50% GT + 50% Conf)"]

    for c_idx, title in enumerate(col_titles):
        axs[0, c_idx].text(
            0.5,
            1.08,
            title,
            transform=axs[0, c_idx].transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )

    for r_idx, context_size in enumerate(args.context_sizes):
        if context_size not in all_views:
            for c_idx in range(4):
                axs[r_idx, c_idx].axis("off")
            continue

        v = all_views[context_size]
        axs[r_idx, 0].text(
            -0.35,
            0.5,
            f"Ctx {context_size}",
            transform=axs[r_idx, 0].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            rotation=90,
        )

        axs[r_idx, 0].imshow(np.clip(v["gt"], 0.0, 1.0))
        axs[r_idx, 0].axis("off")

        conf_before = np.zeros((v["gt"].shape[0], v["gt"].shape[1], 3), dtype=np.float32)
        conf_before[v["conf_mask"]] = v["cmap_fn"](v["conf_raw"][v["conf_mask"]])[:, :3]
        axs[r_idx, 1].imshow(np.clip(conf_before, 0.0, 1.0))
        axs[r_idx, 1].axis("off")

        axs[r_idx, 2].imshow(np.clip(v["conf_rgb"], 0.0, 1.0))
        axs[r_idx, 2].axis("off")

        blend = 0.5 * np.clip(v["gt"], 0.0, 1.0) + 0.5 * np.clip(v["conf_rgb"], 0.0, 1.0)
        axs[r_idx, 3].imshow(blend)
        axs[r_idx, 3].axis("off")

    sm = cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", pad=0.02, fraction=0.02)
    cbar.set_label("Confidence (0=Low, 1=High)", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(top=0.96, left=0.08, right=0.99, hspace=0.15)

    out_png = args.out_dir / "merged_confidence_all_contexts.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    info_file = args.out_dir / "grid_size_info.txt"
    info_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_png}")
    print(f"Saved: {info_file}")


if __name__ == "__main__":
    main()
