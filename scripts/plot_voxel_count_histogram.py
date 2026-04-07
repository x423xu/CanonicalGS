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
from render_mono_voxel_lite_scene import _load_scene_tensor_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--render-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139"))
    parser.add_argument("--checkpoint-tag", type=str, default="epoch_8-step_280000")
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--depth-max", type=float, default=80.0)
    parser.add_argument("--reference-grid", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/651a7f83ed093001/render_custom_novel139/voxel_histograms"))
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


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (256, 256))
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    manifest_path = args.render_dir / args.checkpoint_tag / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    ckpt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ckpt_dir = args.render_dir / args.checkpoint_tag

    points_by_ctx: dict[int, np.ndarray] = {}
    for cs in args.context_sizes:
        ctx_item = None
        for c in ckpt_manifest["contexts"]:
            if int(c["context_size"]) == cs:
                ctx_item = c
                break
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

    pts_ctx2 = points_by_ctx[2]
    origin = pts_ctx2.min(axis=0)
    span = np.maximum(pts_ctx2.max(axis=0) - origin, 1e-9)
    voxel_size = float(np.max(span) / max(args.reference_grid, 1))

    counts_by_ctx: dict[int, np.ndarray] = {}
    max_count = 1
    for cs in args.context_sizes:
        counts = _compute_voxel_counts(points_by_ctx[cs], origin, voxel_size)
        counts_by_ctx[cs] = counts
        max_count = max(max_count, int(counts.max()))

    bins = np.arange(1, max_count + 2) - 0.5
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        2: "#1f77b4",
        4: "#ff7f0e",
        6: "#2ca02c",
        8: "#d62728",
        10: "#9467bd",
    }

    for cs in args.context_sizes:
        counts = counts_by_ctx[cs]
        ax.hist(
            counts,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=colors.get(cs, None),
            label=f"ctx{cs} (voxels={counts.shape[0]}, max={int(counts.max())}, p99={np.percentile(counts,99):.0f})",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Points per occupied voxel")
    ax.set_ylabel("Number of occupied voxels (log scale)")
    ax.set_title(f"Voxel Count Histogram Across Contexts (reference grid={args.reference_grid})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")

    out_png = args.out_dir / f"voxel_count_histogram_grid{args.reference_grid}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    lines = [
        "Voxel Count Histogram Summary",
        "===========================",
        f"reference_grid: {args.reference_grid}",
        f"voxel_size: {voxel_size:.8f}",
        "",
    ]
    for cs in args.context_sizes:
        counts = counts_by_ctx[cs]
        lines.append(
            f"ctx{cs}: points={points_by_ctx[cs].shape[0]}, occupied_voxels={counts.shape[0]}, max_occ={int(counts.max())}, p99_occ={np.percentile(counts,99):.2f}, mean_occ={counts.mean():.3f}"
        )

    out_txt = args.out_dir / f"voxel_count_histogram_grid{args.reference_grid}.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {out_png}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
