from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from canonicalgs.config import RootConfig
from canonicalgs.data import build_re10k_asset_tensor_episodes, build_re10k_sample_tensor_episodes
from canonicalgs.data.episode_builder import EpisodeBuilder
from canonicalgs.logging import metric_name
from canonicalgs.model import CanonicalGsPipeline

from .device import move_to_device
from .eval_visualization import save_evaluation_visualizations
from .objectives import CanonicalLossComputer
from .render_metrics import compute_render_metrics


def evaluate_episodes(
    cfg: RootConfig,
    pipeline: CanonicalGsPipeline,
    episodes: list[dict],
    loss_computer: CanonicalLossComputer,
    eval_context_sizes: list[int],
    visualization_root: Path | None = None,
    phase: str = "eval",
    step: int | None = None,
    progress_desc: str = "eval",
) -> tuple[dict[str, object], dict[int, dict[str, Path]]]:
    pipeline.eval()
    aggregates: dict[str, list[float]] = {
        "total_loss": [],
        "render_loss": [],
        "monotone_loss": [],
    }
    context_aggregates = {
        context_size: {
            "render_mse": [],
            "render_psnr": [],
            "render_coverage": [],
            "gaussians": [],
            "active_cells": [],
            "mean_confidence": [],
            "mean_semantic_consistency": [],
            "mean_opacity": [],
        }
        for context_size in eval_context_sizes
    }

    saved_visuals: dict[int, dict[str, Path]] = {}
    device = next(pipeline.parameters()).device
    with torch.no_grad():
        progress = tqdm(
            episodes,
            total=len(episodes),
            desc=progress_desc,
            dynamic_ncols=True,
        )
        for episode in progress:
            episode_on_device = move_to_device(episode, device)
            outputs = pipeline.forward_prefixes(
                episode_on_device,
                context_sizes=eval_context_sizes,
                include_render_payload=visualization_root is not None,
            )
            losses = loss_computer(
                outputs,
                episode=episode_on_device,
                max_render_targets=cfg.train.render_eval_target_views,
            )
            aggregates["total_loss"].append(float(losses.total_loss.item()))
            aggregates["render_loss"].append(float(losses.render_loss.item()))
            aggregates["monotone_loss"].append(float(losses.monotone_loss.item()))
            progress.set_postfix(
                loss=f"{float(losses.total_loss.item()):.4f}",
                scene=str(episode["scene_id"])[-8:],
            )
            for context_size in eval_context_sizes:
                context_output = outputs[context_size]
                render_metrics = compute_render_metrics(
                    episode=episode_on_device,
                    output=context_output,
                    max_targets=cfg.train.render_eval_target_views,
                )
                context_summary = context_aggregates[context_size]
                context_summary["render_mse"].append(render_metrics.mse)
                context_summary["render_psnr"].append(render_metrics.psnr)
                context_summary["render_coverage"].append(render_metrics.coverage)
                context_summary["gaussians"].append(float(context_output["num_gaussians"]))
                context_summary["active_cells"].append(float(context_output["num_active_cells"]))
                context_summary["mean_confidence"].append(float(context_output["mean_confidence"]))
                context_summary["mean_semantic_consistency"].append(
                    float(context_output["mean_semantic_consistency"])
                )
                context_summary["mean_opacity"].append(float(context_output["mean_opacity"]))
            if visualization_root is not None and not saved_visuals:
                saved_visuals = save_evaluation_visualizations(
                    output_root=visualization_root,
                    episode=episode_on_device,
                    outputs=outputs,
                    phase=phase,
                    step=step,
                )
        progress.close()
    pipeline.train()

    summary: dict[str, object] = {"num_episodes": len(episodes)}
    for key, values in aggregates.items():
        summary[key] = mean(values)
    for context_size, values in context_aggregates.items():
        summary[f"ctx_{context_size}"] = {
            metric_name: mean(metric_values)
            for metric_name, metric_values in values.items()
        }
    return summary, saved_visuals


def build_eval_episodes(
    cfg: RootConfig,
    builder: EpisodeBuilder,
    eval_context_sizes: list[int],
    max_scenes: int,
    step: int | None = None,
) -> list[dict]:
    max_context_size = max(int(size) for size in eval_context_sizes)
    asset_episodes = build_re10k_asset_tensor_episodes(
        roots=cfg.dataset.roots,
        split=cfg.dataset.split,
        max_context_size=max_context_size,
        eval_context_sizes=eval_context_sizes,
        image_shape=tuple(cfg.dataset.image_shape),
        max_scenes=max_scenes,
    )
    if asset_episodes:
        return asset_episodes
    return build_re10k_sample_tensor_episodes(
        cfg.dataset.roots,
        cfg.dataset.split,
        builder,
        max_scenes,
        tuple(cfg.dataset.image_shape),
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        cfg.dataset.fixed_scene_count,
        cfg.dataset.fixed_scene_seed,
        global_step=step,
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def flatten_phase_summary(phase: str, summary: dict[str, object]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, value in summary.items():
        if key.startswith("ctx_"):
            context_size = int(key.split("_", maxsplit=1)[1])
            context_metrics = value
            if not isinstance(context_metrics, dict):
                continue
            for metric_key, metric_value in context_metrics.items():
                flattened[metric_name(phase, metric_key, context_size)] = float(metric_value)
            continue
        if key == "num_episodes":
            flattened[f"{phase}/num_episodes"] = int(value)
        else:
            flattened[f"{phase}/{key}"] = float(value)
    return flattened
