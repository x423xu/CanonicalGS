from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_mono_voxel_lite_scene import _build_model, _load_scene_tensor_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--render-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139"))
    parser.add_argument("--checkpoint-tag", type=str, default="epoch_8-step_280000")
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10])

    parser.add_argument("--point-source", type=str, default="gaussian-means", choices=["gaussian-means", "target-depth"])
    parser.add_argument("--depth-max", type=float, default=80.0)

    # Keep exact calibration convention from previous run:
    # grid=525 -> voxel_size=0.02037695 (world units)
    parser.add_argument("--reference-grid", type=int, default=525)
    parser.add_argument("--reference-voxel-size", type=float, default=0.02037695)

    parser.add_argument("--grid-min", type=int, default=200)
    parser.add_argument("--grid-max", type=int, default=2000)
    parser.add_argument("--grid-step", type=int, default=100)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/checkpoints/epoch_8-step_280000.ckpt"))
    parser.add_argument("--reference-repo", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming"))
    parser.add_argument("--wandb-config", type=Path, default=Path("/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/wandb/run-20260309_102424-awdebv94/files/config.yaml"))

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/651a7f83ed093001/render_custom_novel139/voxel_histograms_range_200_2000_gaussian_means"),
    )
    return parser.parse_args()


def _intrinsics_to_pixels(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    k_px = k.clone()
    k_px[0, :] *= w
    k_px[1, :] *= h
    return k_px


def _depth_to_world_points(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
    depth_max: float,
) -> np.ndarray:
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
    return world.cpu().numpy().astype(np.float32)


def _compute_voxel_counts(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    rel = (points - origin[None, :]) / max(voxel_size, 1e-12)
    idx = np.floor(rel).astype(np.int64)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    return counts.astype(np.int32)


def _load_points_from_target_depth(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    ckpt_manifest: dict,
    ckpt_dir: Path,
) -> dict[int, np.ndarray]:
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    points_by_ctx: dict[int, np.ndarray] = {}
    for cs in args.context_sizes:
        ctx_item = next((c for c in ckpt_manifest["contexts"] if int(c["context_size"]) == cs), None)
        if ctx_item is None:
            raise RuntimeError(f"context {cs} not found in manifest")

        target_indices = list(ctx_item["target_indices"])
        pts_list = []
        for view_i, frame_idx in enumerate(target_indices):
            depth_path = ckpt_dir / f"ctx_{cs}" / f"t{view_i:02d}_f{int(frame_idx):04d}_pred_depth.pt"
            if not depth_path.exists():
                raise FileNotFoundError(f"missing depth: {depth_path}")

            depth = torch.load(depth_path, map_location="cpu")
            pts = _depth_to_world_points(
                depth=depth,
                intrinsic=intrinsics[frame_idx].cpu(),
                extrinsic_c2w=extrinsics[frame_idx].cpu(),
                depth_max=args.depth_max,
            )
            pts_list.append(pts)

        points_by_ctx[cs] = np.concatenate(pts_list, axis=0)
    return points_by_ctx


def _load_points_from_gaussian_means(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    ckpt_manifest: dict,
) -> dict[int, np.ndarray]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(args, device)
    model.load_active_ffgs_checkpoint(str(args.checkpoint.resolve()))
    model.eval()

    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    points_by_ctx: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for cs in args.context_sizes:
            ctx_item = next((c for c in ckpt_manifest["contexts"] if int(c["context_size"]) == cs), None)
            if ctx_item is None:
                raise RuntimeError(f"context {cs} not found in manifest")

            context_idx = [int(i) for i in ctx_item["context_indices"]]
            cimg = images[context_idx].unsqueeze(0).to(device)
            cin = intrinsics[context_idx].unsqueeze(0).to(device)
            cex = extrinsics[context_idx].unsqueeze(0).to(device)
            cn = torch.full((1, len(context_idx)), args.near, dtype=torch.float32, device=device)
            cf = torch.full((1, len(context_idx)), args.far, dtype=torch.float32, device=device)

            out = model(cimg, cin, cex, cn, cf, global_step=0, deterministic=True)
            points_by_ctx[cs] = out["gaussians"].means[0].detach().cpu().numpy().astype(np.float32)

    return points_by_ctx


def _plot_histogram_for_grid(
    counts_by_ctx: dict[int, np.ndarray],
    context_sizes: list[int],
    grid: int,
    voxel_size: float,
    out_png: Path,
    point_source: str,
) -> None:
    colors = {2: "#1f77b4", 4: "#ff7f0e", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd"}
    max_count = max(int(c.max()) for c in counts_by_ctx.values())
    bins = np.arange(1, max_count + 2) - 0.5

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 1.0], hspace=0.35, wspace=0.25)
    ax_combined = fig.add_subplot(gs[0, :])
    axes_ctx = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_axes([0.36, 0.04, 0.28, 0.20]),
    ]

    for i, cs in enumerate(context_sizes):
        counts = counts_by_ctx[cs]
        occ, freq = np.unique(counts, return_counts=True)
        color = colors[cs]

        ax_combined.step(occ, freq, where="mid", linewidth=2.4, color=color, label=f"ctx{cs}")
        ax_combined.scatter(occ, freq, s=14, color=color, alpha=0.8)

        ax_ctx = axes_ctx[i]
        ax_ctx.hist(counts, bins=bins, histtype="stepfilled", linewidth=1.5, alpha=0.35, edgecolor=color, color=color)
        ax_ctx.set_yscale("log")
        ax_ctx.set_title(f"ctx{cs} (points={counts.sum()}, voxels={counts.shape[0]}, max={int(counts.max())})", fontsize=9)
        ax_ctx.set_xlabel("Points", fontsize=9)
        ax_ctx.set_ylabel("Occur", fontsize=9)
        ax_ctx.grid(alpha=0.2, linestyle="--")

    ax_combined.set_yscale("log")
    ax_combined.set_xlabel("Number of points per voxel")
    ax_combined.set_ylabel("Occurrence")
    ax_combined.set_title(f"Voxel Histogram ({point_source}, grid={grid}, voxel_size={voxel_size:.8f})")
    ax_combined.legend(loc="upper right", fontsize=9)
    ax_combined.grid(alpha=0.25, linestyle="--")

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (args.image_height, args.image_width))

    manifest_path = args.render_dir / args.checkpoint_tag / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    ckpt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ckpt_dir = args.render_dir / args.checkpoint_tag

    if args.point_source == "target-depth":
        points_by_ctx = _load_points_from_target_depth(args, scene, ckpt_manifest, ckpt_dir)
    else:
        points_by_ctx = _load_points_from_gaussian_means(args, scene, ckpt_manifest)

    # Keep origin convention anchored to ctx2 as in previous confidence pipeline style.
    origin = points_by_ctx[2].min(axis=0)

    summary_lines = [
        "Voxel Count Histogram Summary",
        "===========================",
        f"point_source: {args.point_source}",
        f"reference_grid: {args.reference_grid}",
        f"reference_voxel_size: {args.reference_voxel_size:.8f}",
        f"grid_range: {args.grid_min} to {args.grid_max} step {args.grid_step}",
        "",
    ]

    for grid in range(args.grid_min, args.grid_max + 1, args.grid_step):
        voxel_size = float(args.reference_voxel_size * (args.reference_grid / float(max(grid, 1))))
        counts_by_ctx: dict[int, np.ndarray] = {}
        for cs in args.context_sizes:
            counts_by_ctx[cs] = _compute_voxel_counts(points_by_ctx[cs], origin, voxel_size)

        out_png = args.out_dir / f"voxel_count_histogram_grid{grid}.png"
        _plot_histogram_for_grid(counts_by_ctx, args.context_sizes, grid, voxel_size, out_png, args.point_source)
        print(f"saved: {out_png}")

        summary_lines.append(f"grid={grid}, voxel_size={voxel_size:.8f}")
        for cs in args.context_sizes:
            counts = counts_by_ctx[cs]
            summary_lines.append(
                f"  ctx{cs}: points={points_by_ctx[cs].shape[0]}, occupied_voxels={counts.shape[0]}, max_occ={int(counts.max())}, p99_occ={np.percentile(counts,99):.2f}, mean_occ={counts.mean():.3f}"
            )
        summary_lines.append("")

    out_txt = args.out_dir / f"voxel_count_summary_{args.grid_min}_{args.grid_max}.txt"
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
