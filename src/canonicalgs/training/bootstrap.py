from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from canonicalgs.config import RootConfig
from canonicalgs.data import (
    EpisodeBuilder,
    Re10kEpisodeDataset,
    Re10kStepAwareEpisodeDataset,
    build_re10k_sample_tensor_episodes,
)
from canonicalgs.data.re10k import _iter_re10k_scene_entries
from canonicalgs.logging import build_wandb_kwargs, metric_name
from canonicalgs.model import CanonicalGsPipeline

from .device import move_to_device, resolve_runtime_device
from .eval_visualization import build_wandb_visual_log
from .evaluation import build_eval_episodes, evaluate_episodes, flatten_phase_summary
from .objectives import CanonicalLossComputer
from .render_metrics import compute_render_stats


def run_bootstrap_training(cfg: RootConfig) -> None:
    device = resolve_runtime_device(cfg.runtime.device)
    train_log_every = _effective_train_log_every(cfg)
    eval_every = _effective_bootstrap_eval_every(cfg)
    train_context_sizes = _active_context_sizes(cfg.dataset.train_context_sizes, cfg.objective.mono_on)
    eval_context_sizes = _active_context_sizes(cfg.dataset.eval_context_sizes, cfg.objective.mono_on)
    builder_context_sizes = _sorted_unique(train_context_sizes, eval_context_sizes)
    builder = EpisodeBuilder(
        context_sizes=tuple(builder_context_sizes),
        target_views=cfg.dataset.target_views,
        min_frames_per_episode=cfg.dataset.min_frames_per_episode,
        subsample_to=cfg.dataset.subsample_to,
        seed=cfg.dataset.seed,
        context_gap_min=cfg.dataset.context_gap_min,
        context_gap_max=cfg.dataset.context_gap_max,
        context_gap_warmup_steps=cfg.dataset.context_gap_warmup_steps,
    )
    train_episodes: list[dict] | None = None
    data_loader = None
    if _uses_step_aware_context_sampling(cfg):
        train_scene_entries = _iter_re10k_scene_entries(
            roots=cfg.dataset.roots,
            split=cfg.dataset.split,
            validation_holdout_stride=cfg.dataset.eval_holdout_stride,
            validation_holdout_offset=cfg.dataset.eval_holdout_offset,
            evaluation_holdout_stride=cfg.dataset.eval_holdout_stride,
            evaluation_holdout_offset=cfg.dataset.eval_holdout_offset,
            fixed_scene_count=cfg.dataset.fixed_scene_count,
            fixed_scene_seed=cfg.dataset.fixed_scene_seed,
        )
        train_scene_entries.sort(key=lambda entry: (str(entry[0]), entry[1], entry[3], entry[2]))
        if not train_scene_entries:
            raise ValueError("bootstrap training found no RE10K scene entries")
        step_dataset = Re10kStepAwareEpisodeDataset(
            scene_entries=train_scene_entries,
            episode_builder=builder,
            image_shape=tuple(cfg.dataset.image_shape),
            max_steps=cfg.train.bootstrap_steps,
        )
        data_loader = DataLoader(
            step_dataset,
            batch_size=None,
            num_workers=4,
            pin_memory=device.type == "cuda",
            prefetch_factor=2,
            persistent_workers=True,
        )
    elif _should_preload_train_subset(cfg):
        train_episodes = build_re10k_sample_tensor_episodes(
            cfg.dataset.roots,
            cfg.dataset.split,
            builder,
            cfg.dataset.fixed_scene_count,
            tuple(cfg.dataset.image_shape),
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.fixed_scene_count,
            cfg.dataset.fixed_scene_seed,
        )
        if not train_episodes:
            raise ValueError("bootstrap training found no tensor episodes for the fixed subset")
        dataset = None
    else:
        dataset = Re10kEpisodeDataset(
            roots=cfg.dataset.roots,
            split=cfg.dataset.split,
            episode_builder=builder,
            image_shape=tuple(cfg.dataset.image_shape),
            max_scenes=cfg.train.bootstrap_steps,
            validation_holdout_stride=cfg.dataset.eval_holdout_stride,
            validation_holdout_offset=cfg.dataset.eval_holdout_offset,
            evaluation_holdout_stride=cfg.dataset.eval_holdout_stride,
            evaluation_holdout_offset=cfg.dataset.eval_holdout_offset,
            fixed_scene_count=cfg.dataset.fixed_scene_count,
            fixed_scene_seed=cfg.dataset.fixed_scene_seed,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=4,
            pin_memory=device.type == "cuda",
            prefetch_factor=2,
            persistent_workers=True,
        )
    eval_episode_count = _resolve_bootstrap_eval_count(cfg)
    pipeline = CanonicalGsPipeline(cfg.model).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=cfg.train.bootstrap_lr)
    loss_computer = CanonicalLossComputer(cfg.objective)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_log_path = output_dir / "bootstrap_eval_log.jsonl"
    eval_log_path.write_text("", encoding="utf-8")
    train_step_log_path = output_dir / "train_step_log.jsonl" if cfg.train.debug else None
    if train_step_log_path is not None:
        train_step_log_path.write_text("", encoding="utf-8")

    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(**build_wandb_kwargs(cfg.wandb, output_dir))
        model_summary = _summarize_model_parameters(pipeline)
        wandb.config.update(model_summary, allow_val_change=True)
        for key, value in model_summary.items():
            run.summary[key] = value

    if cfg.train.debug and eval_every > 0:
        _run_bootstrap_eval(
            cfg=cfg,
            pipeline=pipeline,
            builder=builder,
            eval_context_sizes=eval_context_sizes,
            eval_episode_count=eval_episode_count,
            loss_computer=loss_computer,
            output_dir=output_dir,
            eval_log_path=eval_log_path,
            run=run,
            step=0,
        )

    dataset_iter = iter(data_loader) if train_episodes is None else None
    progress = tqdm(
        range(1, cfg.train.bootstrap_steps + 1),
        total=cfg.train.bootstrap_steps,
        desc="bootstrap_train",
        dynamic_ncols=True,
    )
    for step in progress:
        if train_episodes is not None:
            episode = dict(train_episodes[(step - 1) % len(train_episodes)])
        else:
            try:
                assert dataset_iter is not None
                episode = next(dataset_iter)
            except StopIteration:
                dataset_iter = iter(data_loader)
                episode = next(dataset_iter)

        episode = move_to_device(episode, device)
        largest_context = max(train_context_sizes)
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
        render_stats = None
        if cfg.train.debug or step % train_log_every == 0 or step == cfg.train.bootstrap_steps:
            render_stats = compute_render_stats(
                episode=episode,
                output=outputs[largest_context],
                max_targets=cfg.train.render_train_target_views,
            )
        losses.total_loss.float().backward()
        grad_summary = _summarize_gradients(pipeline) if cfg.train.debug else None
        optimizer.step()
        progress.set_postfix(
            scene=str(episode["scene_id"])[-8:],
            loss=f"{float(losses.total_loss.item()):.4f}",
            gauss=int(outputs[largest_context]["num_gaussians"]),
        )

        if train_step_log_path is not None and render_stats is not None and grad_summary is not None:
            step_log = _build_train_step_log(
                step=step,
                episode=episode,
                outputs=outputs,
                losses=losses,
                render_stats=render_stats,
                grad_summary=grad_summary,
            )
            with train_step_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(step_log) + "\n")
            if run is not None:
                wandb.log(_flatten_train_step_log(step_log), step=step)

        if step % train_log_every == 0 or step == cfg.train.bootstrap_steps:
            assert render_stats is not None
            summary = {
                "train/step": step,
                "train/total_loss": float(losses.total_loss.item()),
                "train/render_loss": float(losses.render_loss.item()),
                "train/monotone_loss": float(losses.monotone_loss.item()),
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
            summary[metric_name("train", "mean_semantic_consistency", largest_context)] = outputs[
                largest_context
            ]["mean_semantic_consistency"]
            print(
                "[CanonicalGS] "
                f"train_step={step} "
                f"total={summary['train/total_loss']:.6f} "
                f"render={summary['train/render_loss']:.6f} "
                f"mono={summary['train/monotone_loss']:.6f} "
                f"psnr={summary['train/render_psnr']:.3f} "
                f"gaussians={summary[metric_name('train', 'gaussians', largest_context)]}"
                )
            if run is not None and not cfg.train.debug:
                wandb.log(summary, step=step)

        if (
            eval_every > 0
            and (step % eval_every == 0 or step == cfg.train.bootstrap_steps)
        ):
            _run_bootstrap_eval(
                cfg=cfg,
                pipeline=pipeline,
                builder=builder,
                eval_context_sizes=eval_context_sizes,
                eval_episode_count=eval_episode_count,
                loss_computer=loss_computer,
                output_dir=output_dir,
                eval_log_path=eval_log_path,
                run=run,
                step=step,
            )
    progress.close()

    if run is not None:
        run.finish()


def _build_train_step_log(
    step: int,
    episode: dict,
    outputs: dict[int, dict],
    losses,
    render_stats,
    grad_summary: dict[str, float | int | bool | str | None],
) -> dict[str, object]:
    context_summaries: dict[str, object] = {}
    for context_size, output in sorted(outputs.items()):
        readout = output["readout"]
        gaussians = output["gaussians"]
        context_summaries[str(context_size)] = {
            "num_context_views": int(output["num_context_views"]),
            "num_active_cells": int(output["num_active_cells"]),
            "num_gaussians": int(output["num_gaussians"]),
            "mean_confidence": float(output["mean_confidence"]),
            "mean_semantic_consistency": float(output["mean_semantic_consistency"]),
            "mean_opacity": float(output["mean_opacity"]),
            "certainty_min": _tensor_stat(readout.canonical_certainty, "min"),
            "certainty_max": _tensor_stat(readout.canonical_certainty, "max"),
            "certainty_mean": _tensor_stat(readout.canonical_certainty, "mean"),
            "uncertainty_min": _tensor_stat(readout.uncertainty, "min"),
            "uncertainty_max": _tensor_stat(readout.uncertainty, "max"),
            "uncertainty_mean": _tensor_stat(readout.uncertainty, "mean"),
            "opacity_min": _tensor_stat(gaussians.opacities, "min"),
            "opacity_max": _tensor_stat(gaussians.opacities, "max"),
            "opacity_mean": _tensor_stat(gaussians.opacities, "mean"),
        }

    return {
        "step": step,
        "scene_id": str(episode["scene_id"]),
        "target_count": int(episode["target_indices"].numel()),
        "losses": {
            "total": float(losses.total_loss.item()),
            "render": float(losses.render_loss.item()),
            "monotone": float(losses.monotone_loss.item()),
        },
        "render": {
            "mse": float(render_stats.mse.detach().item()),
            "psnr": float(render_stats.psnr.detach().item()),
            "coverage": float(render_stats.coverage.detach().item()),
            "num_targets": int(render_stats.num_targets),
        },
        "gradients": grad_summary,
        "contexts": context_summaries,
    }


def _summarize_gradients(model: torch.nn.Module) -> dict[str, float | int | bool | str | None]:
    total_sq_norm = 0.0
    max_abs = 0.0
    finite = True
    param_count = 0
    first_nonfinite_name: str | None = None
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        detached = grad.detach()
        param_count += 1
        if not torch.isfinite(detached).all():
            finite = False
            if first_nonfinite_name is None:
                first_nonfinite_name = name
        grad_float = detached.float()
        total_sq_norm += float(grad_float.square().sum().item())
        max_abs = max(max_abs, float(grad_float.abs().max().item()))
    total_norm = math.sqrt(total_sq_norm) if total_sq_norm > 0.0 else 0.0
    return {
        "finite": finite,
        "parameter_count": param_count,
        "total_norm": total_norm,
        "max_abs": max_abs,
        "first_nonfinite_parameter": first_nonfinite_name,
    }


def _tensor_stat(values: torch.Tensor, stat: str) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return 0.0
    if stat == "min":
        return float(finite.min().item())
    if stat == "max":
        return float(finite.max().item())
    if stat == "mean":
        return float(finite.mean().item())
    raise ValueError(f"unsupported stat: {stat}")


def _flatten_train_step_log(step_log: dict[str, object]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {
        "train/step": int(step_log["step"]),
        "train/target_count": int(step_log["target_count"]),
    }
    losses = step_log.get("losses", {})
    if isinstance(losses, dict):
        for key, value in losses.items():
            flattened[f"train/{key}_loss"] = float(value)
    render = step_log.get("render", {})
    if isinstance(render, dict):
        for key, value in render.items():
            if key == "num_targets":
                flattened["train/render_num_targets"] = int(value)
            else:
                flattened[f"train/render_{key}"] = float(value)
    gradients = step_log.get("gradients", {})
    if isinstance(gradients, dict):
        for key, value in gradients.items():
            if key == "first_nonfinite_parameter" or value is None:
                continue
            if isinstance(value, bool):
                flattened[f"train/grad_{key}"] = int(value)
            elif isinstance(value, int):
                flattened[f"train/grad_{key}"] = value
            else:
                flattened[f"train/grad_{key}"] = float(value)
    contexts = step_log.get("contexts", {})
    if isinstance(contexts, dict):
        for context_size, context_metrics in contexts.items():
            if not isinstance(context_metrics, dict):
                continue
            prefix = f"train_ctx_{context_size}"
            for key, value in context_metrics.items():
                if isinstance(value, bool):
                    flattened[f"{prefix}/{key}"] = int(value)
                elif isinstance(value, int):
                    flattened[f"{prefix}/{key}"] = value
                else:
                    flattened[f"{prefix}/{key}"] = float(value)
    return flattened


def _stage_builder_context_sizes(cfg: RootConfig) -> list[int]:
    return _sorted_unique(cfg.dataset.train_context_sizes, cfg.dataset.eval_context_sizes)


def _sorted_unique(*groups: list[int]) -> list[int]:
    return sorted({value for group in groups for value in group})


def _active_context_sizes(context_sizes: list[int], mono_on: bool) -> list[int]:
    ordered = sorted(set(int(size) for size in context_sizes))
    if mono_on:
        return ordered
    return [ordered[-1]]


def _uses_step_aware_context_sampling(cfg: RootConfig) -> bool:
    return cfg.dataset.context_gap_warmup_steps > 0


def _should_preload_train_subset(cfg: RootConfig) -> bool:
    if cfg.dataset.fixed_scene_count is None:
        return False
    return cfg.dataset.fixed_scene_count <= 256


def _resolve_bootstrap_eval_count(cfg: RootConfig) -> int:
    fixed_count = cfg.dataset.fixed_scene_count
    if fixed_count is not None:
        return max(1, min(cfg.train.bootstrap_eval_num_scenes, fixed_count))
    return max(1, cfg.train.bootstrap_eval_num_scenes)


def _effective_train_log_every(cfg: RootConfig) -> int:
    if cfg.train.debug:
        return 1
    return max(1, cfg.train.log_every)


def _effective_bootstrap_eval_every(cfg: RootConfig) -> int:
    if cfg.train.bootstrap_eval_every > 0:
        return cfg.train.bootstrap_eval_every
    if cfg.train.debug:
        return 200
    return 0


def _summarize_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    frozen = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
    return {
        "model/params_total": int(total),
        "model/params_trainable": int(trainable),
        "model/params_frozen": int(frozen),
    }


def _evaluate_bootstrap_subset(
    cfg: RootConfig,
    pipeline: CanonicalGsPipeline,
    episodes: list[dict],
    loss_computer: CanonicalLossComputer,
    eval_context_sizes: list[int],
    visualization_root: Path | None = None,
    phase: str = "eval",
    step: int | None = None,
) -> tuple[dict[str, object], dict[int, dict[str, Path]]]:
    return evaluate_episodes(
        cfg=cfg,
        pipeline=pipeline,
        episodes=episodes,
        loss_computer=loss_computer,
        eval_context_sizes=eval_context_sizes,
        visualization_root=visualization_root,
        phase=phase,
        step=step,
        progress_desc="bootstrap_eval",
    )


def _run_bootstrap_eval(
    cfg: RootConfig,
    pipeline: CanonicalGsPipeline,
    builder: EpisodeBuilder,
    eval_context_sizes: list[int],
    eval_episode_count: int,
    loss_computer: CanonicalLossComputer,
    output_dir: Path,
    eval_log_path: Path,
    run,
    step: int,
) -> None:
    eval_episodes = build_eval_episodes(
        cfg=cfg,
        builder=builder,
        eval_context_sizes=eval_context_sizes,
        max_scenes=eval_episode_count,
        step=step if _uses_step_aware_context_sampling(cfg) else None,
    )
    if not eval_episodes:
        raise ValueError("bootstrap evaluation found no tensor episodes")
    eval_summary, eval_visuals = _evaluate_bootstrap_subset(
        cfg=cfg,
        pipeline=pipeline,
        episodes=eval_episodes,
        loss_computer=loss_computer,
        eval_context_sizes=eval_context_sizes,
        visualization_root=output_dir / "evaluation_visualizations",
        phase="eval",
        step=step,
    )
    with eval_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": step, "summary": eval_summary}) + "\n")
    largest_eval_context = max(eval_context_sizes)
    eval_metrics = eval_summary[f"ctx_{largest_eval_context}"]
    print(
        "[CanonicalGS] "
        f"bootstrap_eval_step={step} "
        f"episodes={eval_summary['num_episodes']} "
        f"ctx_{largest_eval_context}_psnr={eval_metrics['render_psnr']:.3f} "
        f"ctx_{largest_eval_context}_mse={eval_metrics['render_mse']:.6f} "
        f"ctx_{largest_eval_context}_gaussians={eval_metrics['gaussians']:.1f}"
    )
    if run is not None:
        wandb.log(flatten_phase_summary("eval", eval_summary), step=step)
        wandb.log(build_wandb_visual_log("eval", eval_visuals), step=step)
