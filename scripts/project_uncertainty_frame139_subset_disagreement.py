from __future__ import annotations

import argparse
import math
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

    p.add_argument("--point-gain", type=float, default=1.0)
    p.add_argument("--subset-samples", type=int, default=8)
    p.add_argument("--subset-seed", type=int, default=0)

    p.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139/uncertainty_subset_disagreement_grid1500"))
    return p.parse_args()


def _intrinsics_to_pixels(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    k_px = k.clone()
    k_px[0, :] *= w
    k_px[1, :] *= h
    return k_px


def _compute_voxel_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    rel = (points - origin[None, :]) / max(voxel_size, 1e-12)
    return np.floor(rel).astype(np.int64)


def _voxel_counts(points: np.ndarray, voxel_size: float, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = _compute_voxel_indices(points, origin, voxel_size)
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    return inv, counts


def _project_accumulate_max(points_world: np.ndarray, values: np.ndarray, intrinsic: torch.Tensor, extrinsic_c2w: torch.Tensor, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
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


def _sample_unique_subsets(pool: list[int], k: int, n_samples: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if k >= len(pool):
        return [tuple(sorted(pool))]

    max_combos = math.comb(len(pool), k)
    target = min(n_samples, max_combos)
    picked: set[tuple[int, ...]] = set()

    while len(picked) < target:
        subset = tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))
        picked.add(subset)

    return sorted(picked)


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

    pool = sorted(set(context_map[10]))
    rng = np.random.default_rng(args.subset_seed)
    subsets_by_ctx = {cs: _sample_unique_subsets(pool, cs, args.subset_samples, rng) for cs in context_sizes}

    model = _build_model(args, device)
    model.load_active_ffgs_checkpoint(str(args.checkpoint.resolve()))
    model.eval()

    means_cache: dict[tuple[int, ...], np.ndarray] = {}
    with torch.no_grad():
        for subsets in subsets_by_ctx.values():
            for subset in subsets:
                if subset in means_cache:
                    continue
                idx = [min(int(i), max_frame) for i in subset]
                cimg = images[idx].unsqueeze(0).to(device)
                cin = intrinsics[idx].unsqueeze(0).to(device)
                cex = extrinsics[idx].unsqueeze(0).to(device)
                cn = torch.full((1, len(idx)), args.near, dtype=torch.float32, device=device)
                cf = torch.full((1, len(idx)), args.far, dtype=torch.float32, device=device)
                out = model(cimg, cin, cex, cn, cf, global_step=0, deterministic=True)
                means_cache[subset] = out["gaussians"].means[0].detach().cpu().numpy().astype(np.float32)

    all_points = np.concatenate(list(means_cache.values()), axis=0)
    origin = all_points.min(axis=0)

    inv_by_subset: dict[tuple[int, ...], np.ndarray] = {}
    cnt_by_subset: dict[tuple[int, ...], np.ndarray] = {}
    all_counts: list[np.ndarray] = []
    for subset, pts in means_cache.items():
        inv, cnt = _voxel_counts(pts, voxel_size, origin)
        inv_by_subset[subset] = inv
        cnt_by_subset[subset] = cnt
        all_counts.append(cnt.astype(np.float32))
    denom = max(float(np.percentile(np.concatenate(all_counts), 99.0)), 1e-12)

    gt = images[novel_frame].permute(1, 2, 0).cpu().numpy()
    h, w = gt.shape[:2]

    var_maps: dict[int, np.ndarray] = {}
    summary = [
        "Projected Uncertainty (Subset Disagreement)",
        "==========================================",
        f"novel_frame: {novel_frame}",
        f"grid: {args.grid}",
        f"reference_grid: {args.reference_grid}",
        f"reference_voxel_size: {args.reference_voxel_size:.8f}",
        f"voxel_size_used: {voxel_size:.8f}",
        f"global_points_per_voxel_p99: {denom:.8f}",
        "point_conf_remap: (count - 1) / (p99_count - 1)",
        f"point_gain: {args.point_gain:.6f}",
        "point_sigmoid: sigmoid((point_confidence - 0.5) * point_gain)",
        f"subset_samples_target: {args.subset_samples}",
        f"subset_seed: {args.subset_seed}",
        "",
    ]

    remap_denom = max(denom - 1.0, 1e-12)
    for cs in context_sizes:
        subset_maps = []
        for subset in subsets_by_ctx[cs]:
            pts = means_cache[subset]
            inv = inv_by_subset[subset]
            cnt = cnt_by_subset[subset]

            base_conf = np.clip((cnt[inv].astype(np.float32) - 1.0) / remap_denom, 0.0, 1.0)
            point_unc = (1.0 / (1.0 + np.exp(-((base_conf - 0.5) * float(args.point_gain))))).astype(np.float32)

            unc_raw, unc_mask = _project_accumulate_max(
                points_world=pts,
                values=point_unc,
                intrinsic=intrinsics[novel_frame].cpu(),
                extrinsic_c2w=extrinsics[novel_frame].cpu(),
                h=h,
                w=w,
            )
            unc_filled = _fill_missing(unc_raw, unc_mask).astype(np.float32)
            subset_maps.append(unc_filled)

        stack = np.stack(subset_maps, axis=0)
        var_map = stack.var(axis=0, dtype=np.float32)
        var_maps[cs] = var_map

        summary.append(
            f"ctx{cs}: subsets={len(subsets_by_ctx[cs])}, var_mean={float(var_map.mean()):.6f}, var_std={float(var_map.std()):.6f}, var_p99={float(np.percentile(var_map, 99.0)):.6f}"
        )

    all_var = np.concatenate([var_maps[cs].reshape(-1) for cs in context_sizes]).astype(np.float32)
    var_scale = max(float(np.percentile(all_var, 99.0)), 1e-12)
    summary.append("")
    summary.append(f"variance_vis_scale_p99: {var_scale:.8f}")

    vis_maps = {cs: np.clip(var_maps[cs] / var_scale, 0.0, 1.0) for cs in context_sizes}

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs = axs.ravel()

    axs[0].imshow(np.clip(gt, 0.0, 1.0))
    axs[0].set_title("GT Frame 139")
    axs[0].axis("off")

    cmap = cm.get_cmap("coolwarm")
    for i, cs in enumerate(context_sizes, start=1):
        axs[i].imshow(cmap(vis_maps[cs])[:, :, :3])
        axs[i].set_title(f"ctx{cs} disagreement")
        axs[i].axis("off")

    sm = cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs.tolist(), orientation="horizontal", pad=0.03, fraction=0.03)
    cbar.set_label("Subset disagreement variance (normalized by global p99)")

    fig.suptitle(
        f"Projected Uncertainty by Context Subset Disagreement to Frame {novel_frame} (grid={args.grid})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    out_png = args.out_dir / f"projected_uncertainty_subset_disagreement_frame{novel_frame}_grid{args.grid}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_txt = args.out_dir / f"projected_uncertainty_subset_disagreement_frame{novel_frame}_grid{args.grid}.txt"
    out_txt.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"saved: {out_png}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
