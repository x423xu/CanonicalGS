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
from .objectives import CanonicalLossComputer
from .render_metrics import compute_render_metrics, compute_render_stats


def run_scene_overfit_training(cfg: RootConfig) -> None:
    builder = EpisodeBuilder(
        context_sizes=tuple(cfg.dataset.context_sizes),
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
        cfg.dataset.validation_holdout_stride,
        cfg.dataset.validation_holdout_offset,
        cfg.dataset.evaluation_holdout_stride,
        cfg.dataset.evaluation_holdout_offset,
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
    amp_enabled = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(**build_wandb_kwargs(cfg.wandb, output_dir))

    initial_summary = _evaluate_episode(cfg, pipeline, episode, loss_computer)
    print(_format_summary("overfit_initial", initial_summary))
    train_context_size = cfg.train.overfit_train_context_size
    if train_context_size not in cfg.dataset.context_sizes:
        raise ValueError("train.overfit_train_context_size must be one of dataset.context_sizes")

    for step in range(1, cfg.train.overfit_steps + 1):
        pipeline.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = {
                train_context_size: pipeline(
                    episode,
                    train_context_size,
                    include_render_payload=True,
                )
            }
            losses = loss_computer(outputs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
            render_stats = compute_render_stats(
                episode=episode,
                output=outputs[train_context_size],
                max_targets=cfg.train.render_train_target_views,
            )
            total_loss = losses.total_loss.float() + cfg.objective.lambda_rend * render_stats.mse
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.log_every == 0:
            summary = {
                "train/step": step,
                "train/total_loss": float(total_loss.item()),
                "train/structural_loss": float(losses.total_loss.item()),
                "train/convergence_loss": float(losses.convergence_loss.item()),
                "train/monotone_loss": float(losses.monotone_loss.item()),
                "train/null_loss": float(losses.null_loss.item()),
                "train/render_mse": float(render_stats.mse.detach().item()),
                "train/render_psnr": float(render_stats.psnr.detach().item()),
                "train/render_coverage": float(render_stats.coverage.detach().item()),
                metric_name("train", "active_cells", train_context_size): outputs[train_context_size]["num_active_cells"],
                metric_name("train", "gaussians", train_context_size): outputs[train_context_size]["num_gaussians"],
                metric_name("train", "mean_confidence", train_context_size): outputs[train_context_size]["mean_confidence"],
                metric_name("train", "mean_support", train_context_size): outputs[train_context_size]["mean_support"],
                metric_name("train", "mean_opacity", train_context_size): outputs[train_context_size]["mean_opacity"],
            }
            print(
                "[CanonicalGS] "
                f"overfit_step={step} "
                f"total={summary['train/total_loss']:.6f} "
                f"conv={summary['train/convergence_loss']:.6f} "
                f"mono={summary['train/monotone_loss']:.6f} "
                f"null={summary['train/null_loss']:.6f} "
                f"psnr={summary['train/render_psnr']:.3f} "
                f"gaussians_ctx_{train_context_size}={summary[metric_name('train', 'gaussians', train_context_size)]}"
            )
            if run is not None:
                wandb.log(summary, step=step)

    final_summary = _evaluate_episode(cfg, pipeline, episode, loss_computer)
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
                "overfit/final_gaussians_ctx_6": final_summary["ctx_6"]["gaussians"],
            },
            step=cfg.train.overfit_steps,
        )
        run.finish()


def _evaluate_episode(
    cfg: RootConfig,
    pipeline: CanonicalGsPipeline,
    episode: dict,
    loss_computer: CanonicalLossComputer,
) -> dict[str, object]:
    pipeline.eval()
    amp_enabled = cfg.train.amp and next(pipeline.parameters()).device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(
            device_type=next(pipeline.parameters()).device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = {
                context_size: pipeline(
                    episode,
                    context_size,
                    include_render_payload=True,
                )
                for context_size in cfg.dataset.context_sizes
            }
            losses = loss_computer(outputs)

    summary: dict[str, object] = {
        "structural_loss": float(losses.total_loss.item()),
        "convergence_loss": float(losses.convergence_loss.item()),
        "monotone_loss": float(losses.monotone_loss.item()),
        "null_loss": float(losses.null_loss.item()),
    }
    for context_size in cfg.dataset.context_sizes:
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
            "mean_support": float(context_output["mean_support"]),
            "mean_opacity": float(context_output["mean_opacity"]),
            "render_mse": render_metrics.mse,
            "render_psnr": render_metrics.psnr,
            "render_coverage": render_metrics.coverage,
            "total_loss": float(losses.total_loss.item() + cfg.objective.lambda_rend * render_metrics.mse),
        }
    largest_context = max(cfg.dataset.context_sizes)
    summary["total_loss"] = summary[f"ctx_{largest_context}"]["total_loss"]
    return summary


def _format_summary(prefix: str, summary: dict[str, object]) -> str:
    ctx6 = summary["ctx_6"]
    return (
        "[CanonicalGS] "
        f"{prefix} "
        f"total={summary['total_loss']:.6f} "
        f"conv={summary['convergence_loss']:.6f} "
        f"mono={summary['monotone_loss']:.6f} "
        f"null={summary['null_loss']:.6f} "
        f"ctx6_psnr={ctx6['render_psnr']:.3f} "
        f"ctx6_gaussians={ctx6['gaussians']} "
        f"ctx6_cells={ctx6['active_cells']}"
    )
