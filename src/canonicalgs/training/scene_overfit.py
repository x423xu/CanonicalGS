from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import wandb

from canonicalgs.config import RootConfig
from canonicalgs.data import EpisodeBuilder, build_re10k_sample_tensor_episodes
from canonicalgs.logging import build_wandb_kwargs, metric_name
from canonicalgs.model import CanonicalGsPipeline

from .device import move_to_device, resolve_runtime_device
from .eval_visualization import build_wandb_visual_log, save_evaluation_visualizations
from .objectives import CanonicalLossComputer
from .render_metrics import compute_render_metrics, compute_render_stats


def run_scene_overfit_training(cfg: RootConfig) -> None:
    builder_context_sizes = _stage_builder_context_sizes(cfg)
    train_context_sizes = _sorted_unique(cfg.dataset.train_context_sizes)
    eval_context_sizes = _sorted_unique(cfg.dataset.eval_context_sizes)
    builder = EpisodeBuilder(
        context_sizes=tuple(builder_context_sizes),
        target_views=cfg.dataset.target_views,
        min_frames_per_episode=cfg.dataset.min_frames_per_episode,
        subsample_to=cfg.dataset.subsample_to,
        seed=cfg.dataset.seed,
    )
    episode_count = max(1, cfg.train.overfit_scene_index + 1)
    episodes = build_re10k_sample_tensor_episodes(
        cfg.dataset.roots,
        cfg.dataset.split,
        builder,
        episode_count,
        tuple(cfg.dataset.image_shape),
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        cfg.dataset.fixed_scene_count,
        cfg.dataset.fixed_scene_seed,
    )
    if len(episodes) <= cfg.train.overfit_scene_index:
        raise ValueError("not enough buildable RE10K episodes for requested overfit_scene_index")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_runtime_device(cfg.runtime.device)
    episode = move_to_device(episodes[cfg.train.overfit_scene_index], device)

    pipeline = CanonicalGsPipeline(cfg.model).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=cfg.train.bootstrap_lr)
    loss_computer = CanonicalLossComputer(cfg.objective)

    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(**build_wandb_kwargs(cfg.wandb, output_dir))

    initial_summary, initial_visuals = _evaluate_episode(
        cfg,
        pipeline,
        episode,
        loss_computer,
        eval_context_sizes,
        visualization_root=output_dir / "evaluation_visualizations",
        phase="overfit_initial",
        step=0,
    )
    print(_format_summary("overfit_initial", initial_summary))
    train_context_size = cfg.train.overfit_train_context_size
    if train_context_size not in train_context_sizes:
        raise ValueError("train.overfit_train_context_size must be one of dataset.train_context_sizes")

    for step in range(1, cfg.train.overfit_steps + 1):
        pipeline.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = pipeline.forward_prefixes(
            episode,
            context_sizes=train_context_sizes,
            include_render_payload=False,
        )
        losses = loss_computer(
            outputs,
            episode=episode,
            max_render_targets=cfg.train.render_train_target_views,
        )
        render_stats = compute_render_stats(
            episode=episode,
            output=outputs[train_context_size],
            max_targets=cfg.train.render_train_target_views,
        )
        total_loss = losses.total_loss.float()
        total_loss.backward()
        optimizer.step()

        if step % cfg.train.log_every == 0:
            summary = {
                "train/step": step,
                "train/total_loss": float(total_loss.item()),
                "train/render_loss": float(losses.render_loss.item()),
                "train/monotone_loss": float(losses.monotone_loss.item()),
                "train/render_mse": float(render_stats.mse.detach().item()),
                "train/render_psnr": float(render_stats.psnr.detach().item()),
                "train/render_coverage": float(render_stats.coverage.detach().item()),
                metric_name("train", "active_cells", train_context_size): outputs[train_context_size]["num_active_cells"],
                metric_name("train", "gaussians", train_context_size): outputs[train_context_size]["num_gaussians"],
                metric_name("train", "mean_confidence", train_context_size): outputs[train_context_size]["mean_confidence"],
                metric_name("train", "mean_semantic_consistency", train_context_size): outputs[train_context_size][
                    "mean_semantic_consistency"
                ],
                metric_name("train", "mean_opacity", train_context_size): outputs[train_context_size]["mean_opacity"],
            }
            print(
                "[CanonicalGS] "
                f"overfit_step={step} "
                f"total={summary['train/total_loss']:.6f} "
                f"render={summary['train/render_loss']:.6f} "
                f"mono={summary['train/monotone_loss']:.6f} "
                f"psnr={summary['train/render_psnr']:.3f} "
                f"gaussians_ctx_{train_context_size}={summary[metric_name('train', 'gaussians', train_context_size)]}"
            )
            if run is not None:
                wandb.log(summary, step=step)

    final_summary, final_visuals = _evaluate_episode(
        cfg,
        pipeline,
        episode,
        loss_computer,
        eval_context_sizes,
        visualization_root=output_dir / "evaluation_visualizations",
        phase="overfit_final",
        step=cfg.train.overfit_steps,
    )
    print(_format_summary("overfit_final", final_summary))

    summary_payload = {
        "scene_id": episode["scene_id"],
        "clip_id": episode["clip_id"],
        "steps": cfg.train.overfit_steps,
        "image_shape": cfg.dataset.image_shape,
        "initial": initial_summary,
        "final": final_summary,
    }
    summary_path = output_dir / "scene_overfit_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if cfg.train.save_checkpoint:
        checkpoint_path = output_dir / "scene_overfit_last.pt"
        torch.save(
            {
                "model": pipeline.state_dict(),
                "optimizer": optimizer.state_dict(),
                "summary": summary_payload,
                "cfg": {
                    "dataset": asdict(cfg.dataset),
                    "model": asdict(cfg.model),
                    "objective": asdict(cfg.objective),
                    "train": asdict(cfg.train),
                },
            },
            checkpoint_path,
        )

    if run is not None:
        run.log(
            {
                "overfit/final_total_loss": final_summary["total_loss"],
                f"overfit/final_gaussians_ctx_{max(eval_context_sizes)}": final_summary[
                    f"ctx_{max(eval_context_sizes)}"
                ]["gaussians"],
            },
            step=cfg.train.overfit_steps,
        )
        run.log(build_wandb_visual_log("overfit_initial", initial_visuals), step=0)
        run.log(build_wandb_visual_log("overfit_final", final_visuals), step=cfg.train.overfit_steps)
        run.finish()


def _evaluate_episode(
    cfg: RootConfig,
    pipeline: CanonicalGsPipeline,
    episode: dict,
    loss_computer: CanonicalLossComputer,
    eval_context_sizes: list[int],
    visualization_root: Path | None = None,
    phase: str = "eval",
    step: int | None = None,
) -> tuple[dict[str, object], dict[int, dict[str, Path]]]:
    pipeline.eval()
    with torch.no_grad():
        outputs = pipeline.forward_prefixes(
            episode,
            context_sizes=eval_context_sizes,
            include_render_payload=True,
        )
        losses = loss_computer(
            outputs,
            episode=episode,
            max_render_targets=cfg.train.render_eval_target_views,
        )
        saved_visuals = {}
        if visualization_root is not None:
            saved_visuals = save_evaluation_visualizations(
                output_root=visualization_root,
                episode=episode,
                outputs=outputs,
                phase=phase,
                step=step,
            )

    summary: dict[str, object] = {
        "render_loss": float(losses.render_loss.item()),
        "monotone_loss": float(losses.monotone_loss.item()),
        "total_loss": float(losses.total_loss.item()),
    }
    for context_size in eval_context_sizes:
        context_output = outputs[context_size]
        render_metrics = compute_render_metrics(
            episode=episode,
            output=context_output,
            max_targets=cfg.train.render_eval_target_views,
        )
        summary[f"ctx_{context_size}"] = {
            "active_cells": int(context_output["num_active_cells"]),
            "gaussians": int(context_output["num_gaussians"]),
            "mean_confidence": float(context_output["mean_confidence"]),
            "mean_opacity": float(context_output["mean_opacity"]),
            "render_mse": render_metrics.mse,
            "render_psnr": render_metrics.psnr,
            "render_coverage": render_metrics.coverage,
            "total_loss": float(losses.total_loss.item()),
        }
    return summary, saved_visuals


def _format_summary(prefix: str, summary: dict[str, object]) -> str:
    context_keys = sorted(key for key in summary if key.startswith("ctx_"))
    ctx = summary[context_keys[-1]]
    ctx_label = context_keys[-1]
    return (
        "[CanonicalGS] "
        f"{prefix} "
        f"total={summary['total_loss']:.6f} "
        f"render={summary['render_loss']:.6f} "
        f"mono={summary['monotone_loss']:.6f} "
        f"{ctx_label}_psnr={ctx['render_psnr']:.3f} "
        f"{ctx_label}_gaussians={ctx['gaussians']} "
        f"{ctx_label}_cells={ctx['active_cells']}"
    )


def _stage_builder_context_sizes(cfg: RootConfig) -> list[int]:
    return _sorted_unique(cfg.dataset.train_context_sizes, cfg.dataset.eval_context_sizes)


def _sorted_unique(*groups: list[int]) -> list[int]:
    return sorted({value for group in groups for value in group})
