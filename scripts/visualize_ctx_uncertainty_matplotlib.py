from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
from render_mono_voxel_lite_scene import _build_model, _load_scene_tensor_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="651a7f83ed093001")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--render-dir", type=Path, default=Path("outputs/651a7f83ed093001/render"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near", type=float, default=0.5)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--ctx-a", type=int, default=2)
    parser.add_argument("--ctx-b", type=int, default=10)
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=Path("/data0/xxy/code/Active-FFGS-streaming"),
    )
    parser.add_argument(
        "--wandb-config",
        type=Path,
        default=Path(
            "/data0/xxy/code/Active-FFGS-streaming/checkpoints/mono_voxel_lite/"
            "wandb/run-20260309_102424-awdebv94/files/config.yaml"
        ),
    )
    return parser.parse_args()


def _to_cpu_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _robust_norm(v: np.ndarray) -> np.ndarray:
    lo = np.percentile(v, 1)
    hi = np.percentile(v, 99)
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _project_mean_field(
    means_xyz: np.ndarray,
    values: np.ndarray,
    axes: tuple[int, int],
    bins: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = axes
    x = means_xyz[:, a]
    y = means_xyz[:, b]
    vx, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=values)
    cnt, _, _ = np.histogram2d(x, y, bins=bins)
    field = vx / np.maximum(cnt, 1e-6)
    field[cnt < 1] = np.nan
    return field.T, xedges, yedges


def _compute_gaussian_proxies(
    model: Any,
    scene: dict[str, Any],
    context_indices: list[int],
    near: float,
    far: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    images = scene["images"]
    intrinsics = scene["intrinsics"]
    extrinsics = scene["extrinsics"]

    cimg = images[context_indices].unsqueeze(0).to(device)
    cin = intrinsics[context_indices].unsqueeze(0).to(device)
    cex = extrinsics[context_indices].unsqueeze(0).to(device)
    cn = torch.full((1, len(context_indices)), near, dtype=torch.float32, device=device)
    cf = torch.full((1, len(context_indices)), far, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(cimg, cin, cex, cn, cf, global_step=0, deterministic=True)

    means = out["gaussians"].means[0]
    cov = out["gaussians"].covariances[0]
    opa = out["gaussians"].opacities[0].clamp_min(1e-8)

    trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    sigma = torch.sqrt(trace / 3.0)

    eye = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov_stable = cov + 1e-6 * eye
    sign, logabsdet = torch.linalg.slogdet(cov_stable)
    logdet = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, np.nan))

    uncertainty = sigma * (1.0 - opa)
    confidence = opa / (sigma + 1e-6)

    valid = torch.isfinite(logdet) & torch.isfinite(confidence) & torch.isfinite(uncertainty)

    return {
        "means": _to_cpu_np(means[valid]),
        "opacity": _to_cpu_np(opa[valid]),
        "sigma": _to_cpu_np(sigma[valid]),
        "logdet": _to_cpu_np(logdet[valid]),
        "uncertainty": _to_cpu_np(uncertainty[valid]),
        "confidence": _to_cpu_np(confidence[valid]),
    }


def _subsample(bundle: dict[str, np.ndarray], max_points: int, seed: int = 0) -> dict[str, np.ndarray]:
    n = bundle["means"].shape[0]
    if n <= max_points:
        return bundle
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return {k: v[idx] for k, v in bundle.items()}


def _plot_checkpoint(
    ckpt_name: str,
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    out_dir: Path,
    ctx_a: int,
    ctx_b: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    a = _subsample(a, 60000, seed=42)
    b = _subsample(b, 60000, seed=43)

    conf_a = a["confidence"]
    conf_b = b["confidence"]
    unc_a = a["uncertainty"]
    unc_b = b["uncertainty"]

    conf_a_n = _robust_norm(conf_a)
    conf_b_n = _robust_norm(conf_b)
    unc_a_n = _robust_norm(unc_a)
    unc_b_n = _robust_norm(unc_b)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    m = a["means"]
    p = ax1.scatter(m[:, 0], m[:, 1], m[:, 2], c=conf_a_n, s=1, cmap="viridis", alpha=0.5)
    fig.colorbar(p, ax=ax1, fraction=0.03, pad=0.01)
    ax1.set_title(f"{ckpt_name} ctx_{ctx_a}: confidence proxy")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    m = b["means"]
    p = ax2.scatter(m[:, 0], m[:, 1], m[:, 2], c=conf_b_n, s=1, cmap="viridis", alpha=0.5)
    fig.colorbar(p, ax=ax2, fraction=0.03, pad=0.01)
    ax2.set_title(f"{ckpt_name} ctx_{ctx_b}: confidence proxy")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_3d_scatter.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fields = [
        (_project_mean_field(a["means"], conf_a_n, (0, 1))[0], f"ctx_{ctx_a} conf XY"),
        (_project_mean_field(b["means"], conf_b_n, (0, 1))[0], f"ctx_{ctx_b} conf XY"),
        (_project_mean_field(b["means"], conf_b_n, (0, 1))[0] - _project_mean_field(a["means"], conf_a_n, (0, 1))[0], "delta conf XY"),
        (_project_mean_field(a["means"], unc_a_n, (0, 2))[0], f"ctx_{ctx_a} unc XZ"),
        (_project_mean_field(b["means"], unc_b_n, (0, 2))[0], f"ctx_{ctx_b} unc XZ"),
        (_project_mean_field(b["means"], unc_b_n, (0, 2))[0] - _project_mean_field(a["means"], unc_a_n, (0, 2))[0], "delta unc XZ"),
    ]
    for ax, (field, title) in zip(axes.flat, fields):
        im = ax.imshow(field, origin="lower", cmap="magma")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "projection_maps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(conf_a, bins=120, alpha=0.6, label=f"ctx_{ctx_a}")
    axes[0, 0].hist(conf_b, bins=120, alpha=0.6, label=f"ctx_{ctx_b}")
    axes[0, 0].set_title("confidence histogram")
    axes[0, 0].legend()

    axes[0, 1].hist(unc_a, bins=120, alpha=0.6, label=f"ctx_{ctx_a}")
    axes[0, 1].hist(unc_b, bins=120, alpha=0.6, label=f"ctx_{ctx_b}")
    axes[0, 1].set_title("uncertainty histogram")
    axes[0, 1].legend()

    x = np.sort(conf_a)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1, 0].plot(x, y, label=f"ctx_{ctx_a}")
    x = np.sort(conf_b)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1, 0].plot(x, y, label=f"ctx_{ctx_b}")
    axes[1, 0].set_title("confidence CDF")
    axes[1, 0].legend()

    x = np.sort(unc_a)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1, 1].plot(x, y, label=f"ctx_{ctx_a}")
    x = np.sort(unc_b)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1, 1].plot(x, y, label=f"ctx_{ctx_b}")
    axes[1, 1].set_title("uncertainty CDF")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "distribution_compare.png", dpi=220)
    plt.close(fig)

    s_conf_a = _summarize(conf_a)
    s_conf_b = _summarize(conf_b)
    s_unc_a = _summarize(unc_a)
    s_unc_b = _summarize(unc_b)

    report = {
        "checkpoint": ckpt_name,
        "ctx_a": ctx_a,
        "ctx_b": ctx_b,
        "confidence": {
            f"ctx_{ctx_a}": s_conf_a,
            f"ctx_{ctx_b}": s_conf_b,
            "p95_ratio_b_over_a": float(s_conf_b["p95"] / max(s_conf_a["p95"], 1e-12)),
            "mean_ratio_b_over_a": float(s_conf_b["mean"] / max(s_conf_a["mean"], 1e-12)),
        },
        "uncertainty": {
            f"ctx_{ctx_a}": s_unc_a,
            f"ctx_{ctx_b}": s_unc_b,
            "p95_ratio_b_over_a": float(s_unc_b["p95"] / max(s_unc_a["p95"], 1e-12)),
            "mean_ratio_b_over_a": float(s_unc_b["mean"] / max(s_unc_a["mean"], 1e-12)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    manifest = json.loads((args.render_dir / "manifest.json").read_text(encoding="utf-8"))
    scene = _load_scene_tensor_episode(args.root, args.split, args.scene_id, (args.image_height, args.image_width))

    model = _build_model(args, device)

    out_root = args.render_dir / "uncertainty_viz"
    out_root.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict[str, Any]] = []

    for ckpt in manifest["checkpoints"]:
        ckpt_path = Path(ckpt["checkpoint"])
        ckpt_name = ckpt_path.stem
        model.load_active_ffgs_checkpoint(str(ckpt_path))

        ctx_map = {int(c["context_size"]): c for c in ckpt["contexts"]}
        if args.ctx_a not in ctx_map or args.ctx_b not in ctx_map:
            continue

        bundle_a = _compute_gaussian_proxies(
            model=model,
            scene=scene,
            context_indices=list(ctx_map[args.ctx_a]["context_indices"]),
            near=args.near,
            far=args.far,
            device=device,
        )
        bundle_b = _compute_gaussian_proxies(
            model=model,
            scene=scene,
            context_indices=list(ctx_map[args.ctx_b]["context_indices"]),
            near=args.near,
            far=args.far,
            device=device,
        )

        ckpt_out = out_root / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ckpt_out / f"ctx_{args.ctx_a}_proxy.npz", **bundle_a)
        np.savez_compressed(ckpt_out / f"ctx_{args.ctx_b}_proxy.npz", **bundle_b)

        report = _plot_checkpoint(ckpt_name, bundle_a, bundle_b, ckpt_out, args.ctx_a, args.ctx_b)
        all_reports.append(report)

    (out_root / "summary_all.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_root), "checkpoints": len(all_reports)}, indent=2))


if __name__ == "__main__":
    main()
