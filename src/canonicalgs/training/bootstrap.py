from __future__ import annotations

from pathlib import Path

import torch
import wandb

from canonicalgs.config import RootConfig
from canonicalgs.data import EpisodeBuilder, Re10kEpisodeDataset
from canonicalgs.logging import build_wandb_kwargs, metric_name
from canonicalgs.model import CanonicalGsPipeline

from .device import move_to_device, resolve_runtime_device
from .objectives import CanonicalLossComputer
from .render_metrics import compute_render_stats


def run_bootstrap_training(cfg: RootConfig) -> None:
    builder = EpisodeBuilder(
        context_sizes=tuple(cfg.dataset.context_sizes),
        target_views=cfg.dataset.target_views,
        min_frames_per_episode=cfg.dataset.min_frames_per_episode,
        subsample_to=cfg.dataset.subsample_to,
        seed=cfg.dataset.seed,
    )
    dataset = Re10kEpisodeDataset(
        roots=cfg.dataset.roots,
        split=cfg.dataset.split,
        episode_builder=builder,
        image_shape=tuple(cfg.dataset.image_shape),
        max_scenes=cfg.train.bootstrap_steps,
        validation_holdout_stride=cfg.dataset.validation_holdout_stride,
        validation_holdout_offset=cfg.dataset.validation_holdout_offset,
        evaluation_holdout_stride=cfg.dataset.evaluation_holdout_stride,
        evaluation_holdout_offset=cfg.dataset.evaluation_holdout_offset,
    )
    device = resolve_runtime_device(cfg.runtime.device)
    pipeline = CanonicalGsPipeline(cfg.model).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=cfg.train.bootstrap_lr)
    loss_computer = CanonicalLossComputer(cfg.objective)
    amp_enabled = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(**build_wandb_kwargs(cfg.wandb, Path(cfg.output_dir)))

    for step, episode in enumerate(dataset, start=1):
        episode = move_to_device(episode, device)
        largest_context = max(cfg.dataset.context_sizes)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = {
                context_size: pipeline(
                    episode,
                    context_size,
                    include_render_payload=(context_size == largest_context),
                )
                for context_size in cfg.dataset.context_sizes
            }
            losses = loss_computer(outputs)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
            render_stats = compute_render_stats(
                episode=episode,
                output=outputs[largest_context],
                max_targets=cfg.train.render_train_target_views,
            )
            total_loss = losses.total_loss.float() + cfg.objective.lambda_rend * render_stats.mse
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.log_every == 0:
            summary = {
                "train/total_loss": float(total_loss.item()),
                "train/structural_loss": float(losses.total_loss.item()),
                "train/convergence_loss": float(losses.convergence_loss.item()),
                "train/monotone_loss": float(losses.monotone_loss.item()),
                "train/null_loss": float(losses.null_loss.item()),
                "train/render_mse": float(render_stats.mse.detach().item()),
                "train/render_psnr": float(render_stats.psnr.detach().item()),
                "train/render_coverage": float(render_stats.coverage.detach().item()),
            }
            summary[metric_name("train", "active_cells", largest_context)] = outputs[
                largest_context
            ]["num_active_cells"]
            summary[metric_name("train", "gaussians", largest_context)] = outputs[
                largest_context
            ]["num_gaussians"]
            print(
                "[CanonicalGS] "
                f"train_step={step} "
                f"total={summary['train/total_loss']:.6f} "
                f"conv={summary['train/convergence_loss']:.6f} "
                f"mono={summary['train/monotone_loss']:.6f} "
                f"null={summary['train/null_loss']:.6f} "
                f"psnr={summary['train/render_psnr']:.3f}"
            )
            if run is not None:
                wandb.log(summary, step=step)

        if step >= cfg.train.bootstrap_steps:
            break

    if run is not None:
        run.finish()
