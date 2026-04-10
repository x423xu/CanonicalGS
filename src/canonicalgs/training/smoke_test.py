from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb
from tqdm.auto import tqdm

from canonicalgs.config import RootConfig
from canonicalgs.data import (
    EpisodeBuilder,
    build_re10k_sample_tensor_episodes,
    resolve_re10k_scene_keys,
)
from canonicalgs.logging import build_wandb_kwargs, metric_name
from canonicalgs.model import CanonicalGsPipeline

from .device import move_to_device, resolve_runtime_device
from .eval_visualization import build_wandb_visual_log
from .evaluation import build_eval_episodes, evaluate_episodes, flatten_phase_summary
from .objectives import CanonicalLossComputer


def run_subset_smoke_test(cfg: RootConfig) -> None:
    subset_count = cfg.dataset.fixed_scene_count or 100
    if subset_count < 1:
        raise ValueError("dataset.fixed_scene_count must be positive for smoke testing")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train_step_log.jsonl"
    eval_log_path = output_dir / "periodic_eval_log.jsonl"

    builder_context_sizes = _sorted_unique(cfg.dataset.train_context_sizes, cfg.dataset.eval_context_sizes)
    train_context_sizes = _sorted_unique(cfg.dataset.train_context_sizes)
    eval_context_sizes = _sorted_unique(cfg.dataset.eval_context_sizes)
    builder = EpisodeBuilder(
        context_sizes=tuple(builder_context_sizes),
        target_views=cfg.dataset.target_views,
        min_frames_per_episode=cfg.dataset.min_frames_per_episode,
        subsample_to=cfg.dataset.subsample_to,
        seed=cfg.dataset.seed,
    )

    scene_keys = resolve_re10k_scene_keys(
        roots=cfg.dataset.roots,
        split=cfg.dataset.split,
        validation_holdout_stride=cfg.dataset.eval_holdout_stride,
        validation_holdout_offset=cfg.dataset.eval_holdout_offset,
        evaluation_holdout_stride=cfg.dataset.eval_holdout_stride,
        evaluation_holdout_offset=cfg.dataset.eval_holdout_offset,
        fixed_scene_count=subset_count,
        fixed_scene_seed=cfg.dataset.fixed_scene_seed,
    )
    if not scene_keys:
        raise ValueError("no buildable scenes found for the requested smoke-test subset")
    subset_count = len(scene_keys)
    smoke_steps = max(1, cfg.train.smoke_steps)
    smoke_eval_every = max(1, cfg.train.smoke_eval_every)
    (output_dir / "fixed_scene_ids.txt").write_text("\n".join(scene_keys) + "\n", encoding="utf-8")
    train_log_path.write_text("", encoding="utf-8")
    eval_log_path.write_text("", encoding="utf-8")

    device = resolve_runtime_device(cfg.runtime.device)
    pipeline = CanonicalGsPipeline(cfg.model).to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=cfg.train.bootstrap_lr)
    loss_computer = CanonicalLossComputer(cfg.objective)
    run = None
    if cfg.wandb.mode != "disabled":
        run = wandb.init(**build_wandb_kwargs(cfg.wandb, output_dir))

    train_episodes = build_re10k_sample_tensor_episodes(
        cfg.dataset.roots,
        cfg.dataset.split,
        builder,
        subset_count,
        tuple(cfg.dataset.image_shape),
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        cfg.dataset.eval_holdout_stride,
        cfg.dataset.eval_holdout_offset,
        subset_count,
        cfg.dataset.fixed_scene_seed,
    )
    eval_episodes = build_eval_episodes(
        cfg=cfg,
        builder=builder,
        eval_context_sizes=eval_context_sizes,
        max_scenes=subset_count,
    )
    if not train_episodes:
        raise ValueError("smoke-test training found no tensor episodes")
    if not eval_episodes:
        raise ValueError("smoke-test evaluation found no tensor episodes")

    train_log: list[dict[str, float | int | str]] = []
    periodic_evals: list[dict[str, object]] = []
    largest_train_context = max(train_context_sizes)
    progress = tqdm(
        range(1, smoke_steps + 1),
        total=smoke_steps,
        desc="smoke_train",
        dynamic_ncols=True,
    )
    for step in progress:
        episode = train_episodes[(step - 1) % len(train_episodes)]
        episode = move_to_device(episode, device)
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
        total_loss = losses.total_loss.float()
        total_loss.backward()
        optimizer.step()

        entry = {
            "step": step,
            "scene_id": str(episode["scene_id"]),
            "total_loss": float(total_loss.item()),
            "render_loss": float(losses.render_loss.item()),
            "monotone_loss": float(losses.monotone_loss.item()),
            "gaussians": int(outputs[largest_train_context]["num_gaussians"]),
            "active_cells": int(outputs[largest_train_context]["num_active_cells"]),
            "mean_confidence": float(outputs[largest_train_context]["mean_confidence"]),
            "mean_semantic_consistency": float(
                outputs[largest_train_context]["mean_semantic_consistency"]
            ),
            "mean_opacity": float(outputs[largest_train_context]["mean_opacity"]),
        }
        train_log.append(entry)
        with train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        progress.set_postfix(
            scene=entry["scene_id"][-8:],
            loss=f"{entry['total_loss']:.4f}",
            gauss=entry["gaussians"],
        )
        if run is not None:
            wandb.log(
                {
                    "train/step": step,
                    "train/scene_id": entry["scene_id"],
                    "train/total_loss": entry["total_loss"],
                    "train/render_loss": entry["render_loss"],
                    "train/monotone_loss": entry["monotone_loss"],
                    metric_name("train", "gaussians", largest_train_context): entry["gaussians"],
                    metric_name("train", "active_cells", largest_train_context): entry["active_cells"],
                },
                step=step,
            )
        if step % max(1, cfg.train.log_every) == 0 or step == smoke_steps:
            print(
                "[CanonicalGS] "
                f"smoke_train_step={step}/{smoke_steps} "
                f"scene={entry['scene_id']} "
                f"total={entry['total_loss']:.6f} "
                f"render={entry['render_loss']:.6f} "
                f"mono={entry['monotone_loss']:.6f} "
                f"gaussians={entry['gaussians']}"
            )
        if step % smoke_eval_every == 0 or step == smoke_steps:
            eval_summary, eval_visuals = _evaluate_subset(
                cfg,
                pipeline,
                eval_episodes,
                loss_computer,
                eval_context_sizes,
                output_dir / "evaluation_visualizations",
                "eval",
                step,
            )
            eval_entry = {"step": step, "summary": eval_summary}
            periodic_evals.append(eval_entry)
            with eval_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_entry) + "\n")
            largest_eval_context = max(eval_context_sizes)
            largest_metrics = eval_summary[f"ctx_{largest_eval_context}"]
            print(
                "[CanonicalGS] "
                f"smoke_eval_step={step}/{smoke_steps} "
                f"episodes={eval_summary['num_episodes']} "
                f"ctx_{largest_eval_context}_psnr={largest_metrics['render_psnr']:.3f} "
                f"ctx_{largest_eval_context}_mse={largest_metrics['render_mse']:.6f} "
                f"ctx_{largest_eval_context}_gaussians={largest_metrics['gaussians']:.1f}"
            )
            if run is not None:
                wandb.log(flatten_phase_summary("eval", eval_summary), step=step)
                wandb.log(build_wandb_visual_log("eval", eval_visuals), step=step)
    progress.close()

    if periodic_evals and periodic_evals[-1]["step"] == smoke_steps:
        eval_summary = periodic_evals[-1]["summary"]
    else:
        eval_summary, eval_visuals = _evaluate_subset(
            cfg,
            pipeline,
            eval_episodes,
            loss_computer,
            eval_context_sizes,
            output_dir / "evaluation_visualizations",
            "eval",
            smoke_steps,
        )
        periodic_evals.append({"step": smoke_steps, "summary": eval_summary})
        if run is not None:
            wandb.log(flatten_phase_summary("eval", eval_summary), step=smoke_steps)
            wandb.log(build_wandb_visual_log("eval", eval_visuals), step=smoke_steps)
    test_context_sizes = [
        context_size
        for context_size in _sorted_unique(cfg.dataset.test_context_sizes)
        if context_size in builder_context_sizes
    ]
    if not test_context_sizes:
        test_context_sizes = eval_context_sizes
    test_summary, _ = _evaluate_subset(
        cfg,
        pipeline,
        eval_episodes,
        loss_computer,
        test_context_sizes,
    )
    payload = {
        "subset_count": subset_count,
        "smoke_steps": smoke_steps,
        "scene_keys": scene_keys,
        "train_steps": len(train_log),
        "train_log_tail": train_log[-10:],
        "periodic_eval": periodic_evals,
        "eval": eval_summary,
        "test": test_summary,
    }
    (output_dir / "smoke_test_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if cfg.train.save_checkpoint:
        torch.save({"model": pipeline.state_dict(), "summary": payload}, output_dir / "smoke_test_last.pt")

    largest_eval_context = max(eval_context_sizes)
    largest_metrics = eval_summary[f"ctx_{largest_eval_context}"]
    if run is not None:
        wandb.log(flatten_phase_summary("test", test_summary), step=smoke_steps)
        run.finish()
    print(
        "[CanonicalGS] "
        f"smoke_eval episodes={eval_summary['num_episodes']} "
        f"ctx_{largest_eval_context}_psnr={largest_metrics['render_psnr']:.3f} "
        f"ctx_{largest_eval_context}_mse={largest_metrics['render_mse']:.6f} "
        f"ctx_{largest_eval_context}_gaussians={largest_metrics['gaussians']:.1f}"
    )


def _evaluate_subset(
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
        progress_desc="smoke_eval",
    )


def _sorted_unique(*groups: list[int]) -> list[int]:
    return sorted({value for group in groups for value in group})
