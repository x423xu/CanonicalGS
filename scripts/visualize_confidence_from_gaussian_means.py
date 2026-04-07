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
from render_mono_voxel_lite_scene import _build_model, _load_scene_tensor_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/epoch_8-step_280000.ckpt"))
    parser.add_argument("--reference-repo", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming"))
    parser.add_argument("--wandb-config", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/wandb/run-20260309_102424-awdebv94/files/config.yaml"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--novel-frame", type=int, default=139)
    parser.add_argument("--max-search-grid", type=int, default=131072)
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139/means_confidence"))
    return parser.parse_args()


def _intrinsics_to_pixels(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    k_px = k.clone()
    k_px[0, :] *= w
    k_px[1, :] *= h
    return k_px


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
    while grid <= max_grid:
        voxel_size = float(np.max(span) / grid)
        occ = _max_occupancy(points_ctx2, origin, voxel_size)
        if occ <= 1:
            return voxel_size, origin, grid, occ
        grid *= 2
    grid = max_grid
    voxel_size = float(np.max(span) / grid)
    occ = _max_occupancy(points_ctx2, origin, voxel_size)
    return voxel_size, origin, grid, occ


def _voxel_uncertainty_fixed(points: np.ndarray, voxel_size: float, origin: np.ndarray) -> dict[str, Any]:
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
    conf = 1.0 / (1.0 + (unc / denom))

    cmin, cmax = float(conf.min()), float(conf.max())
    if cmax > cmin:
        conf = (conf - cmin) / (cmax - cmin)
    else:
        conf = np.ones_like(conf)

    return {"confidence": conf.astype(np.float32), "inv": inv, "counts": counts}


def _clamp_indices(indices: list[int], max_idx: int) -> list[int]:
    return [min(int(i), max_idx) for i in indices]


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (args.image_height, args.image_width))
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]
    max_frame = int(images.shape[0] - 1)

    context_map = {
        2: [114, 164],
        4: [89, 114, 164, 189],
        6: [64, 89, 114, 164, 189, 214],
        8: [39, 64, 89, 114, 164, 189, 214, 239],
        10: [14, 39, 64, 89, 114, 164, 189, 214, 239, 264],
    }
    context_sizes = [2, 4, 6, 8, 10]
    novel_frame = min(int(args.novel_frame), max_frame)

    model = _build_model(args, device)
    model.load_active_ffgs_checkpoint(str(args.checkpoint.resolve()))
    model.eval()

    means_by_ctx: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for cs in context_sizes:
            context_idx = _clamp_indices(context_map[cs], max_frame)
            context = {
                "image": images[context_idx].unsqueeze(0).to(device),
                "intrinsics": intrinsics[context_idx].unsqueeze(0).to(device),
                "extrinsics": extrinsics[context_idx].unsqueeze(0).to(device),
                "near": torch.full((1, cs), args.near, dtype=torch.float32, device=device),
                "far": torch.full((1, cs), args.far, dtype=torch.float32, device=device),
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
            means_by_ctx[cs] = out["gaussians"].means[0].detach().cpu().numpy().astype(np.float32)
            print(f"ctx{cs}: means={means_by_ctx[cs].shape[0]}")

    pts_ctx2 = means_by_ctx[2]
    voxel_size, voxel_origin, ref_grid, ref_occ = _derive_ctx2_voxel_size(pts_ctx2, args.max_search_grid)

    if args.grid_scale <= 0:
        raise ValueError("grid-scale must be > 0")
    if args.grid_scale != 1.0:
        ref_grid = int(round(ref_grid * args.grid_scale))
        voxel_size = voxel_size / float(args.grid_scale)
        ref_occ = _max_occupancy(pts_ctx2, voxel_origin, voxel_size)

    print("=" * 72)
    print("Gaussian-means voxel calibration (ctx2 reference)")
    print("=" * 72)
    print(f"ctx2 means: {pts_ctx2.shape[0]}")
    print(f"reference search grid: {ref_grid}")
    print(f"voxel size (world units): {voxel_size:.8f}")
    print(f"ctx2 max points per voxel: {ref_occ}")

    all_views: dict[int, dict[str, Any]] = {}
    summary_lines = [
        "Gaussian Means Voxel Configuration",
        "==================================",
        "Reference context: 2",
        f"ctx2 means: {pts_ctx2.shape[0]}",
        f"reference grid used in search: {ref_grid}",
        f"voxel size (world units): {voxel_size:.8f}",
        f"ctx2 max points per voxel: {ref_occ}",
        "",
        "Per-context occupancy stats:",
    ]

    for cs in context_sizes:
        pts = means_by_ctx[cs]
        vox = _voxel_uncertainty_fixed(pts, voxel_size=voxel_size, origin=voxel_origin)
        point_conf = vox["confidence"][vox["inv"]]

        occ_counts = vox["counts"]
        max_occ = int(occ_counts.max())
        p99_occ = float(np.percentile(occ_counts, 99))
        print(f"ctx{cs}: means={pts.shape[0]}, occupied_voxels={occ_counts.shape[0]}, max_occ={max_occ}, p99={p99_occ:.2f}")
        summary_lines.append(
            f"ctx{cs}: points={pts.shape[0]}, occupied_voxels={occ_counts.shape[0]}, max_occ={max_occ}, p99_occ={p99_occ:.2f}"
        )

        conf_raw, conf_mask = _project_accumulate_confidence(
            points_world=pts,
            confidence=point_conf,
            intrinsic=intrinsics[novel_frame].cpu(),
            extrinsic_c2w=extrinsics[novel_frame].cpu(),
            h=images.shape[-2],
            w=images.shape[-1],
        )
        conf_filled = _interpolate_missing_regions(conf_raw, conf_mask)
        fmin, fmax = float(conf_filled.min()), float(conf_filled.max())
        if fmax > fmin:
            conf_filled = (conf_filled - fmin) / (fmax - fmin)

        cmap_fn = cm.get_cmap("coolwarm")
        conf_rgb = cmap_fn(conf_filled)[:, :, :3].astype(np.float32)
        gt = images[novel_frame].permute(1, 2, 0).cpu().numpy()

        all_views[cs] = {
            "gt": gt,
            "conf_raw": conf_raw,
            "conf_mask": conf_mask,
            "conf_rgb": conf_rgb,
            "cmap_fn": cmap_fn,
        }

    fig, axs = plt.subplots(len(context_sizes), 4, figsize=(20, 5 * len(context_sizes)))
    col_titles = ["GT View", "Means Confidence (Raw)", "Means Confidence (Filled)", "Blended (50% GT + 50% Conf)"]
    for c_idx, title in enumerate(col_titles):
        axs[0, c_idx].text(0.5, 1.08, title, transform=axs[0, c_idx].transAxes, ha="center", va="bottom", fontsize=14, weight="bold")

    for r_idx, cs in enumerate(context_sizes):
        v = all_views[cs]
        axs[r_idx, 0].text(-0.35, 0.5, f"Ctx {cs}", transform=axs[r_idx, 0].transAxes, ha="center", va="center", fontsize=12, weight="bold", rotation=90)
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

    out_png = args.out_dir / "merged_confidence_all_contexts_gaussian_means.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    info_file = args.out_dir / "grid_size_info_gaussian_means.txt"
    info_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"saved: {out_png}")
    print(f"saved: {info_file}")


if __name__ == "__main__":
    main()
