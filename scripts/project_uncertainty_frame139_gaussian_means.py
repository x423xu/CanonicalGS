from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    p.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--checkpoint", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/epoch_8-step_280000.ckpt"))
    p.add_argument("--reference-repo", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming"))
    p.add_argument("--wandb-config", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/wandb/run-20260309_102424-awdebv94/files/config.yaml"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--image-height", type=int, default=256)
    p.add_argument("--image-width", type=int, default=256)
    p.add_argument("--near", type=float, default=0.5)
    p.add_argument("--far", type=float, default=100.0)
    p.add_argument("--novel-frame", type=int, default=139)

    p.add_argument("--reference-grid", type=int, default=525)
    p.add_argument("--reference-voxel-size", type=float, default=0.02037695)
    p.add_argument("--grid", type=int, default=1500)
    p.add_argument("--point-gain", type=float, default=5.0)

    p.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139/uncertainty_projection_grid1500"))
    return p.parse_args()


def _intrinsics_to_pixels(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    k_px = k.clone()
    k_px[0, :] *= w
    k_px[1, :] *= h
    return k_px


def _compute_voxel_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    rel = (points - origin[None, :]) / max(voxel_size, 1e-12)
    return np.floor(rel).astype(np.int64)


def _voxel_uncertainty(points: np.ndarray, voxel_size: float, origin: np.ndarray) -> dict[str, np.ndarray]:
    idx = _compute_voxel_indices(points, origin, voxel_size)
    uniq, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)

    sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    sums2 = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inv, points)
    np.add.at(sums2, inv, points * points)

    means = sums / counts[:, None]
    var = np.maximum(sums2 / counts[:, None] - means * means, 0.0)
    unc = var.sum(axis=1).astype(np.float32)
    return {"uncertainty": unc, "inv": inv, "counts": counts}


def _project_accumulate(points_world: np.ndarray, values: np.ndarray, intrinsic: torch.Tensor, extrinsic_c2w: torch.Tensor, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    k_px = _intrinsics_to_pixels(intrinsic, h, w).cpu().numpy()
    world_to_cam = torch.linalg.inv(extrinsic_c2w).cpu().numpy()

    n = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ pts_h.T).T[:, :3]

    z = cam[:, 2]
    valid = z > 1e-6
    cam = cam[valid]
    val = values[valid]

    pix = (k_px @ cam.T).T
    u = pix[:, 0] / np.maximum(pix[:, 2], 1e-6)
    v = pix[:, 1] / np.maximum(pix[:, 2], 1e-6)

    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inb]
    v = v[inb]
    val = val[inb]

    ui = np.clip(np.round(u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, h - 1)

    # If multiple contributors map to one pixel, keep the maximum uncertainty.
    out = np.full((h, w), -np.inf, dtype=np.float32)
    np.maximum.at(out, (vi, ui), val.astype(np.float32))
    mask = np.isfinite(out)
    out[~mask] = 0.0
    return out, mask


def _fill_missing(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = img.copy()
    missing = ~mask
    if not np.any(missing):
        return filled
    _, idx = distance_transform_edt(missing, return_indices=True)
    filled[missing] = img[idx[0, missing], idx[1, missing]]
    return filled


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    voxel_size = float(args.reference_voxel_size * (args.reference_grid / float(args.grid)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (args.image_height, args.image_width))
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    max_frame = int(images.shape[0] - 1)
    novel_frame = min(int(args.novel_frame), max_frame)

    context_map = {
        2: [114, 164],
        4: [89, 114, 164, 189],
        6: [64, 89, 114, 164, 189, 214],
        8: [39, 64, 89, 114, 164, 189, 214, 239],
        10: [14, 39, 64, 89, 114, 164, 189, 214, 239, 264],
    }
    context_sizes = [2, 4, 6, 8, 10]

    model = _build_model(args, device)
    model.load_active_ffgs_checkpoint(str(args.checkpoint.resolve()))
    model.eval()

    means_by_ctx: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for cs in context_sizes:
            idx = [min(int(i), max_frame) for i in context_map[cs]]
            cimg = images[idx].unsqueeze(0).to(device)
            cin = intrinsics[idx].unsqueeze(0).to(device)
            cex = extrinsics[idx].unsqueeze(0).to(device)
            cn = torch.full((1, len(idx)), args.near, dtype=torch.float32, device=device)
            cf = torch.full((1, len(idx)), args.far, dtype=torch.float32, device=device)

            out = model(cimg, cin, cex, cn, cf, global_step=0, deterministic=True)
            means_by_ctx[cs] = out["gaussians"].means[0].detach().cpu().numpy().astype(np.float32)

    origin = means_by_ctx[2].min(axis=0)

    all_occ_values = []
    inv_by_ctx = {}
    occ_by_ctx = {}
    for cs in context_sizes:
        vox = _voxel_uncertainty(means_by_ctx[cs], voxel_size, origin)
        inv_by_ctx[cs] = vox["inv"]
        occ_by_ctx[cs] = vox["counts"]
        all_occ_values.append(vox["counts"].astype(np.float32))
    all_occ_cat = np.concatenate(all_occ_values)
    denom = max(float(np.percentile(all_occ_cat, 99.0)), 1e-12)

    gt = images[novel_frame].permute(1, 2, 0).cpu().numpy()
    h, w = gt.shape[:2]
    projected = {}
    summary = [
        "Projected Uncertainty (Gaussian Means)",
        "====================================",
        f"novel_frame: {novel_frame}",
        f"grid: {args.grid}",
        f"reference_grid: {args.reference_grid}",
        f"reference_voxel_size: {args.reference_voxel_size:.8f}",
        f"voxel_size_used: {voxel_size:.8f}",
        f"global_points_per_voxel_p99: {denom:.8f}",
        "point_conf_remap: (count - 1) / (p99_count - 1)",
        f"point_gain: {args.point_gain:.6f}",
        "point_sigmoid: sigmoid((point_confidence - 0.5) * point_gain)",
        "",
    ]

    for cs in context_sizes:
        pts = means_by_ctx[cs]
        remap_denom = max(denom - 1.0, 1e-12)
        base_conf = np.clip((occ_by_ctx[cs][inv_by_ctx[cs]].astype(np.float32) - 1.0) / remap_denom, 0.0, 1.0)
        point_unc = (1.0 / (1.0 + np.exp(-((base_conf - 0.5) * float(args.point_gain))))).astype(np.float32)
        unc_raw, unc_mask = _project_accumulate(
            points_world=pts,
            values=point_unc,
            intrinsic=intrinsics[novel_frame].cpu(),
            extrinsic_c2w=extrinsics[novel_frame].cpu(),
            h=h,
            w=w,
        )
        unc_filled = _fill_missing(unc_raw, unc_mask).astype(np.float32)
        projected[cs] = {"unc": unc_filled, "mask": unc_mask}

        conf_mean = float(base_conf.mean())
        conf_std = float(base_conf.std())
        proj_mean = float(unc_filled.mean())
        proj_std = float(unc_filled.std())
        summary.append(
            f"ctx{cs}: points={pts.shape[0]}, occupied_voxels={occ_by_ctx[cs].shape[0]}, max_occ={int(occ_by_ctx[cs].max())}, p99_occ={np.percentile(occ_by_ctx[cs],99):.2f}, conf_mean={conf_mean:.6f}, conf_std={conf_std:.6f}, proj_mean={proj_mean:.6f}, proj_std={proj_std:.6f}"
        )

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.ravel()

    axs[0].imshow(np.clip(gt, 0.0, 1.0))
    axs[0].set_title("GT Frame 139")
    axs[0].axis("off")

    cmap = cm.get_cmap("coolwarm")
    for i, cs in enumerate(context_sizes, start=1):
        img = projected[cs]["unc"]
        axs[i].imshow(cmap(np.clip(img, 0.0, 1.0))[:, :, :3])
        axs[i].set_title(f"ctx{cs} uncertainty")
        axs[i].axis("off")

    sm = cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.tolist(), orientation="horizontal", pad=0.03, fraction=0.03)
    cbar.set_label(f"sigmoid((point confidence - 0.5) * {args.point_gain:.3f}) (0=low, 1=high)")

    fig.suptitle(
        f"Projected Uncertainty to Frame {novel_frame} (grid={args.grid}, voxel_size={voxel_size:.8f})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    out_png = args.out_dir / f"projected_uncertainty_frame{novel_frame}_grid{args.grid}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_txt = args.out_dir / f"projected_uncertainty_frame{novel_frame}_grid{args.grid}.txt"
    out_txt.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"saved: {out_png}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
