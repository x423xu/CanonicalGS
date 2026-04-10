from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import torch.nn.functional as F

from .render_metrics import render_target_views


def save_evaluation_visualizations(
    output_root: Path,
    episode: dict[str, Any],
    outputs: dict[int, dict[str, Any]],
    phase: str,
    step: int | None = None,
    max_points_per_cloud: int = 12000,
) -> dict[int, dict[str, Path]]:
    context_sizes = sorted(outputs)
    scene_id = _sanitize_name(str(episode["scene_id"]))
    step_label = f"step_{step:06d}" if step is not None else "step_latest"
    phase_root = output_root / phase / step_label / scene_id
    phase_root.mkdir(parents=True, exist_ok=True)

    saved: dict[int, dict[str, Path]] = {}
    for context_size in context_sizes:
        output = outputs[context_size]
        context_dir = phase_root / f"ctx_{context_size:02d}"
        context_dir.mkdir(parents=True, exist_ok=True)

        context_indices = output["context_indices"].detach().cpu()
        images = episode["images"][context_indices].detach().cpu()
        intrinsics = episode["intrinsics"][context_indices].detach().cpu()
        extrinsics = episode["extrinsics"][context_indices].detach().cpu()

        depth = output["depth"].detach().cpu()
        positional_certainty = output["positional_certainty"].detach().cpu()
        appearance_certainty = output["appearance_certainty"].detach().cpu()
        certainty = output["combined_certainty"].detach().cpu()
        reference_extrinsic = extrinsics[0]

        depth_path = context_dir / "predicted_depth.png"
        positional_path = context_dir / "positional_certainty.png"
        appearance_path = context_dir / "appearance_certainty.png"
        certainty_path = context_dir / "combined_certainty.png"
        certainty_histogram_path = context_dir / "certainty_histograms.png"
        camera_path = context_dir / "camera_poses.png"
        per_view_cloud_path = context_dir / "per_view_point_clouds.png"
        aggregated_cloud_path = context_dir / "aggregated_point_cloud.png"
        rendered_output_path = context_dir / "rendered_outputs.png"

        _save_image_grid(depth, depth_path, title="Predicted Depth", color_mode=False, cmap="viridis")
        _save_image_grid(
            positional_certainty,
            positional_path,
            title="Positional Certainty",
            color_mode=False,
            cmap="plasma",
        )
        _save_image_grid(
            appearance_certainty,
            appearance_path,
            title="Appearance Certainty",
            color_mode=False,
            cmap="plasma",
        )
        _save_image_grid(
            certainty,
            certainty_path,
            title="Combined Certainty",
            color_mode=False,
            cmap="plasma",
        )
        _save_certainty_histograms(
            positional_certainty=positional_certainty,
            appearance_certainty=appearance_certainty,
            combined_certainty=certainty,
            output_path=certainty_histogram_path,
        )
        per_view_clouds = _build_per_view_clouds(
            images=images,
            depth=depth,
            certainty=certainty,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            reference_extrinsic=reference_extrinsic,
            max_points_per_cloud=max_points_per_cloud,
        )
        _save_per_view_point_clouds(per_view_clouds, per_view_cloud_path)

        aggregated_points = output["gaussians"].means.detach().cpu()
        aggregated_colors = output["gaussians"].appearance.detach().cpu()
        _save_aggregated_point_cloud_with_cameras(
            aggregated_points=aggregated_points,
            aggregated_colors=aggregated_colors,
            extrinsics=extrinsics,
            reference_extrinsic=reference_extrinsic,
            output_path=aggregated_cloud_path,
            max_points=max_points_per_cloud,
        )
        _save_camera_overlay_figure(
            aggregated_points=aggregated_points,
            aggregated_colors=aggregated_colors,
            extrinsics=extrinsics,
            reference_extrinsic=reference_extrinsic,
            output_path=camera_path,
            max_points=max_points_per_cloud,
        )
        _save_render_output_grid(
            episode=episode,
            output=output,
            output_path=rendered_output_path,
        )

        raw_fields = {
            "context_indices": context_indices.numpy(),
            "intrinsics": intrinsics.numpy(),
            "extrinsics": extrinsics.numpy(),
            "depth": depth.numpy(),
            "positional_certainty": positional_certainty.numpy(),
            "appearance_certainty": appearance_certainty.numpy(),
            "combined_certainty": certainty.numpy(),
        }
        np.savez_compressed(context_dir / "raw_fields.npz", **raw_fields)
        np.savez_compressed(
            context_dir / "aggregated_point_cloud.npz",
            points=aggregated_points.numpy(),
            colors=aggregated_colors.numpy(),
        )
        for view_index, cloud in enumerate(per_view_clouds):
            np.savez_compressed(
                context_dir / f"per_view_point_cloud_{view_index:02d}.npz",
                points=cloud["points"],
                colors=cloud["colors"],
                certainty=cloud["certainty"],
            )

        context_saved = {
            "predicted_depth": depth_path,
            "positional_certainty": positional_path,
            "appearance_certainty": appearance_path,
            "combined_certainty": certainty_path,
            "certainty_histograms": certainty_histogram_path,
            "camera_poses": camera_path,
            "per_view_point_clouds": per_view_cloud_path,
            "aggregated_point_cloud": aggregated_cloud_path,
            "rendered_outputs": rendered_output_path,
        }
        saved[context_size] = context_saved

    manifest = {
        "phase": phase,
        "step": step,
        "scene_id": str(episode["scene_id"]),
        "contexts": {
            str(context_size): {
                name: str(path)
                for name, path in paths.items()
            }
            for context_size, paths in saved.items()
        },
    }
    (phase_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return saved


def build_wandb_visual_log(
    phase: str,
    saved_paths: dict[int, dict[str, Path]],
) -> dict[str, wandb.Image]:
    if not saved_paths:
        return {}

    largest_context = max(saved_paths)
    context_paths = saved_paths[largest_context]
    return {
        f"{phase}_visuals/{name}_ctx_{largest_context}": wandb.Image(str(path))
        for name, path in context_paths.items()
    }


def _save_image_grid(
    images: torch.Tensor,
    output_path: Path,
    title: str,
    color_mode: bool,
    cmap: str | None = None,
) -> None:
    count = max(1, int(images.shape[0]))
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    global_display_lo = float("nan")
    global_display_hi = float("nan")
    if not color_mode:
        global_display_lo, global_display_hi = _display_range(images.detach().cpu().numpy())
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for index, ax in enumerate(axes_arr.flat):
        if index >= count:
            ax.axis("off")
            continue
        image = images[index]
        if color_mode:
            array = image.permute(1, 2, 0).numpy()
            ax.imshow(np.clip(array, 0.0, 1.0))
        else:
            raw = image.squeeze(0).numpy()
            array, display_lo, display_hi = _normalize_field(raw)
            im = ax.imshow(array, cmap=cmap or "viridis")
            finite = raw[np.isfinite(raw)]
            if finite.size:
                ax.set_title(
                    "view "
                    f"{index}\nraw=[{float(finite.min()):.4f}, {float(finite.max()):.4f}] "
                    f"disp=[{display_lo:.4f}, {display_hi:.4f}]"
                )
            else:
                ax.set_title(f"view {index}\nraw=[nan, nan] disp=[nan, nan]")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            continue
        ax.set_title(f"view {index}")
        ax.axis("off")
    if color_mode:
        fig.suptitle(title)
    else:
        finite = images.detach().cpu().numpy()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            fig.suptitle(
                f"{title} | global raw=[{float(finite.min()):.4f}, {float(finite.max()):.4f}] "
                f"display=[{global_display_lo:.4f}, {global_display_hi:.4f}]"
            )
        else:
            fig.suptitle(f"{title} | global raw=[nan, nan] display=[nan, nan]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_certainty_histograms(
    positional_certainty: torch.Tensor,
    appearance_certainty: torch.Tensor,
    combined_certainty: torch.Tensor,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    histogram_specs = [
        ("Positional Certainty", positional_certainty, "tab:blue"),
        ("Appearance Certainty", appearance_certainty, "tab:orange"),
        ("Combined Certainty", combined_certainty, "tab:green"),
    ]
    for axis, (title, values, color) in zip(axes, histogram_specs, strict=True):
        flat = values.detach().cpu().numpy().reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            axis.text(0.5, 0.5, "empty", ha="center", va="center")
            axis.set_title(title)
            axis.set_xlim(0.0, 1.0)
            continue
        axis.hist(np.clip(flat, 0.0, 1.0), bins=40, range=(0.0, 1.0), color=color, alpha=0.85)
        axis.set_title(
            f"{title}\nmin={float(flat.min()):.4f} max={float(flat.max()):.4f} mean={float(flat.mean()):.4f}"
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("certainty")
        axis.set_ylabel("count")
    figure.suptitle("Certainty Histograms")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _build_per_view_clouds(
    images: torch.Tensor,
    depth: torch.Tensor,
    certainty: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
    max_points_per_cloud: int,
) -> list[dict[str, np.ndarray]]:
    clouds: list[dict[str, np.ndarray]] = []
    for image, depth_map, certainty_map, intrinsic, extrinsic in zip(
        images,
        depth,
        certainty,
        intrinsics,
        extrinsics,
        strict=True,
    ):
        points, colors, certainty_values = _backproject_view(
            image=image,
            depth_map=depth_map[0],
            certainty_map=certainty_map[0],
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        points = _transform_points_to_reference(points, reference_extrinsic)
        if points.shape[0] > max_points_per_cloud:
            keep = np.linspace(0, points.shape[0] - 1, max_points_per_cloud, dtype=np.int64)
            points = points[keep]
            colors = colors[keep]
            certainty_values = certainty_values[keep]
        clouds.append(
            {
                "points": points,
                "colors": colors,
                "certainty": certainty_values,
            }
        )
    return clouds


def _backproject_view(
    image: torch.Tensor,
    depth_map: torch.Tensor,
    certainty_map: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, height, width = image.shape
    depth_height, depth_width = depth_map.shape
    if height != depth_height or width != depth_width:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(depth_height, depth_width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        height, width = depth_height, depth_width
    device = depth_map.device
    dtype = depth_map.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    x_normalized = (grid_x + 0.5) / width
    y_normalized = (grid_y + 0.5) / height
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_camera = (x_normalized - cx) / fx * depth_map
    y_camera = (y_normalized - cy) / fy * depth_map
    camera_points = torch.stack((x_camera, y_camera, depth_map), dim=-1)
    world_points = torch.einsum("ij,hwj->hwi", extrinsic[:3, :3], camera_points)
    world_points = world_points + extrinsic[:3, 3]

    valid = torch.isfinite(depth_map) & (depth_map > 1e-6)
    points = world_points[valid].detach().cpu().numpy()
    colors = image.permute(1, 2, 0)[valid].detach().cpu().numpy()
    certainty = certainty_map[valid].detach().cpu().numpy()
    return points, colors, certainty


def _save_per_view_point_clouds(
    per_view_clouds: list[dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    count = max(1, len(per_view_clouds))
    cols = min(2, count)
    rows = int(np.ceil(count / cols))
    fig = plt.figure(figsize=(7 * cols, 5.5 * rows))
    for index, cloud in enumerate(per_view_clouds, start=1):
        ax = fig.add_subplot(rows, cols, index, projection="3d")
        if cloud["points"].size == 0:
            ax.text(0.5, 0.5, 0.5, "empty", ha="center", va="center")
        else:
            ax.scatter(
                cloud["points"][:, 0],
                cloud["points"][:, 1],
                cloud["points"][:, 2],
                c=np.clip(cloud["colors"], 0.0, 1.0),
                s=1.0,
                alpha=0.8,
            )
            _set_equal_axes(ax, cloud["points"])
        _apply_opencv_axes(ax)
        ax.set_title(f"Per-view Cloud {index - 1} (OpenCV ref)")
        ax.set_xlabel("x right")
        ax.set_ylabel("y down")
        ax.set_zlabel("z forward")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_aggregated_point_cloud_with_cameras(
    aggregated_points: torch.Tensor,
    aggregated_colors: torch.Tensor,
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    points = _transform_points_to_reference(aggregated_points.numpy(), reference_extrinsic)
    colors = aggregated_colors.numpy()
    points, colors = _filter_finite_rows(points, colors)
    if points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[keep]
        colors = colors[keep]

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    camera_centers = _camera_centers_in_reference(extrinsics, reference_extrinsic)
    frustum_scale = _estimate_frustum_scale(points, camera_centers)
    if points.size == 0:
        ax.text(0.0, 0.0, 0.0, "empty", ha="center", va="center")
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=np.clip(colors, 0.0, 1.0),
            s=1.4,
            alpha=0.9,
            linewidths=0.0,
        )
    _draw_camera_frustums_3d(ax, extrinsics, reference_extrinsic, frustum_scale)
    _configure_opencv_3d_view(
        ax,
        title="Aggregated Gaussian Cloud + Cameras (OpenCV view)",
        points=points,
        camera_centers=camera_centers,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_camera_overlay_figure(
    aggregated_points: torch.Tensor,
    aggregated_colors: torch.Tensor,
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    points = _transform_points_to_reference(aggregated_points.numpy(), reference_extrinsic)
    colors = aggregated_colors.numpy()
    points, colors = _filter_finite_rows(points, colors)
    if points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[keep]
        colors = colors[keep]

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    camera_centers = _camera_centers_in_reference(extrinsics, reference_extrinsic)
    frustum_scale = _estimate_frustum_scale(points, camera_centers)
    if points.size:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=np.clip(colors, 0.0, 1.0),
            s=1.4,
            alpha=0.9,
            linewidths=0.0,
        )
    _draw_camera_frustums_3d(ax, extrinsics, reference_extrinsic, frustum_scale)
    _configure_opencv_3d_view(
        ax,
        title="Camera Poses on Gaussian Cloud (OpenCV view)",
        points=points,
        camera_centers=camera_centers,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_render_output_grid(
    episode: dict[str, Any],
    output: dict[str, Any],
    output_path: Path,
    max_targets: int = 4,
) -> None:
    if "render_gaussians" not in output or "target_indices" not in output or "readout" not in output:
        context_indices = output.get("context_indices")
        context_images = None
        if context_indices is not None:
            context_images = episode["images"][context_indices].detach().cpu()
        rows = max(1, 0 if context_images is None else int(context_images.shape[0]))
        fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
        axes_arr = np.atleast_2d(axes)
        for row in range(rows):
            if context_images is not None and row < context_images.shape[0]:
                axes_arr[row, 0].imshow(np.clip(context_images[row].permute(1, 2, 0).numpy(), 0.0, 1.0))
                axes_arr[row, 0].set_title(f"context {row}")
            else:
                axes_arr[row, 0].axis("off")
            axes_arr[row, 1].text(0.5, 0.5, "render payload\nunavailable", ha="center", va="center")
            for col in range(4):
                axes_arr[row, col].axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    rendered, target_images, _, num_targets = render_target_views(
        episode=episode,
        output=output,
        max_targets=max_targets,
    )
    if num_targets == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no target renders", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    error = (rendered - target_images).abs().mean(dim=1, keepdim=True)
    context_images = episode["images"][output["context_indices"]].detach().cpu()
    rows = max(num_targets, int(context_images.shape[0]))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.5 * rows))
    axes_arr = np.atleast_2d(axes)
    for row in range(rows):
        if row < context_images.shape[0]:
            context = context_images[row].permute(1, 2, 0).numpy()
            axes_arr[row, 0].imshow(np.clip(context, 0.0, 1.0))
            axes_arr[row, 0].set_title(f"context {row}")
        else:
            axes_arr[row, 0].axis("off")
        if row < num_targets:
            gt = target_images[row].permute(1, 2, 0).detach().cpu().numpy()
            pred = rendered[row].permute(1, 2, 0).detach().cpu().numpy()
            err, _, _ = _normalize_field(error[row, 0].detach().cpu().numpy())
            axes_arr[row, 1].imshow(np.clip(gt, 0.0, 1.0))
            axes_arr[row, 1].set_title(f"target {row}")
            axes_arr[row, 2].imshow(np.clip(pred, 0.0, 1.0))
            axes_arr[row, 2].set_title(f"render {row}")
            axes_arr[row, 3].imshow(err, cmap="inferno")
            axes_arr[row, 3].set_title(f"|render-target| {row}")
        for col in range(4):
            axes_arr[row, col].axis("off")
    fig.suptitle("Context Inputs + Rendered Outputs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _display_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _normalize_field(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = _display_range(values)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(values, dtype=np.float32), lo, hi
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32), lo, hi


def _transform_points_to_reference(
    points: np.ndarray,
    reference_extrinsic: torch.Tensor,
) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 3)
    reference_world_to_camera = torch.linalg.inv(reference_extrinsic).detach().cpu().numpy()
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = (reference_world_to_camera @ points_h.T).T[:, :3]
    return transformed.astype(np.float32, copy=False)


def _camera_centers_in_reference(
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
) -> np.ndarray:
    centers = extrinsics[:, :3, 3].detach().cpu().numpy()
    transformed = _transform_points_to_reference(centers, reference_extrinsic)
    transformed, = _filter_finite_rows(transformed)
    return transformed


def _draw_camera_frustums_3d(
    ax: Any,
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
    frustum_scale: float,
) -> None:
    camera_centers = _camera_centers_in_reference(extrinsics, reference_extrinsic)
    for index, center in enumerate(camera_centers):
        color = _camera_color(index)
        frustum = _camera_frustum_in_reference(
            extrinsic=extrinsics[index],
            reference_extrinsic=reference_extrinsic,
            scale=frustum_scale,
        )
        ax.scatter(center[0], center[1], center[2], c=[color], s=32, zorder=4)
        for start, end in _frustum_edges(frustum):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=1.1,
                zorder=3,
            )
        ax.text(center[0], center[1], center[2], str(index), fontsize=8, color=color)


def _draw_projected_camera_frustums(
    ax: Any,
    extrinsics: torch.Tensor,
    reference_extrinsic: torch.Tensor,
) -> None:
    camera_centers = _camera_centers_in_reference(extrinsics, reference_extrinsic)
    frustum_scale = _estimate_frustum_scale(np.empty((0, 3), dtype=np.float32), camera_centers)
    for index, center in enumerate(camera_centers):
        color = _camera_color(index)
        frustum = _camera_frustum_in_reference(
            extrinsic=extrinsics[index],
            reference_extrinsic=reference_extrinsic,
            scale=frustum_scale,
        )
        center_uv, center_valid = _project_opencv(center[None, :])
        if center_valid.any():
            ax.scatter(center_uv[:, 0], center_uv[:, 1], c=[color], s=30, zorder=4)
            ax.text(center_uv[0, 0], center_uv[0, 1], str(index), fontsize=8, color=color)
        for start, end in _frustum_edges(frustum):
            segment = np.stack([start, end], axis=0)
            uv, valid = _project_opencv(segment)
            if valid.all():
                ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=1.2, zorder=3)


def _camera_frustum_in_reference(
    extrinsic: torch.Tensor,
    reference_extrinsic: torch.Tensor,
    scale: float,
) -> np.ndarray:
    frustum_camera = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.5 * scale, -0.35 * scale, scale],
            [0.5 * scale, -0.35 * scale, scale],
            [0.5 * scale, 0.35 * scale, scale],
            [-0.5 * scale, 0.35 * scale, scale],
        ],
        dtype=np.float32,
    )
    c2w = extrinsic.detach().cpu().numpy()
    frustum_h = np.concatenate([frustum_camera, np.ones((frustum_camera.shape[0], 1), dtype=np.float32)], axis=1)
    world = (c2w @ frustum_h.T).T[:, :3]
    return _transform_points_to_reference(world, reference_extrinsic)


def _camera_color(index: int) -> tuple[float, float, float, float]:
    return plt.get_cmap("tab10")(index % 10)


def _frustum_edges(frustum: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (frustum[0], frustum[1]),
        (frustum[0], frustum[2]),
        (frustum[0], frustum[3]),
        (frustum[0], frustum[4]),
        (frustum[1], frustum[2]),
        (frustum[2], frustum[3]),
        (frustum[3], frustum[4]),
        (frustum[4], frustum[1]),
    ]


def _estimate_frustum_scale(points: np.ndarray, camera_centers: np.ndarray) -> float:
    points, = _filter_finite_rows(points)
    camera_centers, = _filter_finite_rows(camera_centers)
    if points.size and camera_centers.size:
        all_points = np.concatenate([points, camera_centers], axis=0)
    elif points.size:
        all_points = points
    elif camera_centers.size:
        all_points = camera_centers
    else:
        return 0.1
    extent = float(np.max(all_points.max(axis=0) - all_points.min(axis=0)))
    return max(extent * 0.08, 0.03)


def _project_opencv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((valid.sum(), 2), dtype=np.float32)
    if valid.any():
        visible = points[valid]
        uv[:, 0] = visible[:, 0] / visible[:, 2]
        uv[:, 1] = visible[:, 1] / visible[:, 2]
    return uv, valid


def _style_opencv_projection(ax: Any, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axhline(0.0, color="lightgray", linewidth=0.8)
    ax.axvline(0.0, color="lightgray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x right")
    ax.set_ylabel("y down")


def _configure_opencv_3d_view(
    ax: Any,
    title: str,
    points: np.ndarray,
    camera_centers: np.ndarray,
) -> None:
    points, = _filter_finite_rows(points)
    camera_centers, = _filter_finite_rows(camera_centers)
    if points.size and camera_centers.size:
        axis_points = np.concatenate([points, camera_centers], axis=0)
    elif points.size:
        axis_points = points
    else:
        axis_points = camera_centers
    if axis_points.size:
        _set_equal_axes(ax, axis_points)
    ax.invert_yaxis()
    ax.view_init(elev=18, azim=-52)
    ax.set_title(title)
    ax.set_xlabel("x right")
    ax.set_ylabel("y down")
    ax.set_zlabel("z inward")


def _set_equal_axes(ax: Any, points: np.ndarray) -> None:
    points, = _filter_finite_rows(points)
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float((maxs - mins).max()) * 0.5, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _apply_opencv_axes(ax: Any) -> None:
    ax.invert_yaxis()


def _filter_finite_rows(points: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if points.size == 0:
        return (points, *arrays)
    valid = np.isfinite(points).all(axis=1)
    filtered = [points[valid]]
    for array in arrays:
        if array.shape[0] == valid.shape[0]:
            filtered.append(array[valid])
        else:
            filtered.append(array)
    return tuple(filtered)


def _sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
