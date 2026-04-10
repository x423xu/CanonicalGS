from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from canonicalgs.config import load_typed_root_config
from canonicalgs.data import (
    EpisodeBuilder,
    build_re10k_sample_episodes,
    build_re10k_sample_tensor_episodes,
    inspect_re10k_roots,
    load_manifest_records,
)
from canonicalgs.model import CanonicalGsPipeline
from canonicalgs.model.renderer import render_gaussian_views
from canonicalgs.training import (
    CanonicalLossComputer,
    move_to_device,
    resolve_runtime_device,
    run_bootstrap_training,
    run_scene_overfit_training,
    run_subset_smoke_test,
)


def _print_startup_banner(output_dir: Path, cfg_mode: str) -> None:
    print(f"[CanonicalGS] mode={cfg_mode}")
    print(f"[CanonicalGS] output_dir={output_dir}")


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    cfg = load_typed_root_config(cfg_dict)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_startup_banner(output_dir, cfg.mode)

    if cfg.mode == "inspect_dataset":
        builder = EpisodeBuilder(
            context_sizes=tuple(cfg.dataset.context_sizes),
            target_views=cfg.dataset.target_views,
            min_frames_per_episode=cfg.dataset.min_frames_per_episode,
            subsample_to=cfg.dataset.subsample_to,
            seed=cfg.dataset.seed,
        )

        if cfg.dataset.roots:
            summaries = inspect_re10k_roots(cfg.dataset.roots)
            for summary in summaries:
                print(f"[CanonicalGS] dataset_root={summary['root']} exists={summary['exists']}")
                for split, split_summary in summary.get("splits", {}).items():
                    print(
                        "[CanonicalGS] "
                        f"split={split} scenes={split_summary['scene_count']} "
                        f"chunks={split_summary['chunk_count']} "
                        f"sample_scene={split_summary['sample_scene']} "
                        f"sample_chunk={split_summary['sample_chunk']}"
                    )
                    if "sample_frame_count" in split_summary:
                        print(
                            "[CanonicalGS] "
                            f"split={split} sample_item_keys={split_summary['sample_item_keys']} "
                            f"sample_frame_count={split_summary['sample_frame_count']}"
                        )
                for split, split_summary in summary.get("resolved_splits", {}).items():
                    print(
                        "[CanonicalGS] "
                        f"resolved_split={split} raw_split={split_summary['raw_split']} "
                        f"scenes={split_summary['scene_count']}"
                    )

            if cfg.dataset.name == "re10k":
                episodes = build_re10k_sample_episodes(
                    cfg.dataset.roots,
                    cfg.dataset.split,
                    builder,
                    cfg.dataset.inspect_num_scenes,
                    cfg.dataset.eval_holdout_stride,
                    cfg.dataset.eval_holdout_offset,
                    cfg.dataset.eval_holdout_stride,
                    cfg.dataset.eval_holdout_offset,
                    cfg.dataset.fixed_scene_count,
                    cfg.dataset.fixed_scene_seed,
                )
                print(
                    "[CanonicalGS] "
                    f"re10k_sample_episodes={len(episodes)} "
                    f"split={cfg.dataset.split}"
                )
                if episodes:
                    sample = episodes[0]
                    print(
                        "[CanonicalGS] "
                        f"sample_episode scene={sample.scene_id} "
                        f"frame_pool={len(sample.frame_pool)} "
                        f"targets={len(sample.target_set)} "
                        f"c6={len(sample.context_sets[max(sample.context_sets)])}"
                    )

                tensor_episodes = build_re10k_sample_tensor_episodes(
                    cfg.dataset.roots,
                    cfg.dataset.split,
                    builder,
                    min(cfg.dataset.inspect_num_scenes, 1),
                    tuple(cfg.dataset.image_shape),
                    cfg.dataset.eval_holdout_stride,
                    cfg.dataset.eval_holdout_offset,
                    cfg.dataset.eval_holdout_stride,
                    cfg.dataset.eval_holdout_offset,
                    cfg.dataset.fixed_scene_count,
                    cfg.dataset.fixed_scene_seed,
                )
                if tensor_episodes:
                    sample_tensors = tensor_episodes[0]
                    print(
                        "[CanonicalGS] "
                        f"tensor_episode images={tuple(sample_tensors['images'].shape)} "
                        f"extrinsics={tuple(sample_tensors['extrinsics'].shape)} "
                        f"intrinsics={tuple(sample_tensors['intrinsics'].shape)}"
                    )
                    print(
                        "[CanonicalGS] "
                        f"tensor_episode targets={tuple(sample_tensors['target_indices'].shape)} "
                        f"context_keys={sorted(sample_tensors['context_indices'])}"
                    )

        if cfg.dataset.manifest_path is None:
            print("[CanonicalGS] dataset.manifest_path is not set; skipping manifest inspection.")
            return

        grouped_records = load_manifest_records(cfg.dataset.manifest_path)
        episodes = builder.build_many(grouped_records.values())
        print(f"[CanonicalGS] loaded clips={len(grouped_records)}")
        print(f"[CanonicalGS] buildable episodes={len(episodes)}")
        if episodes:
            sample = episodes[0]
            print(
                "[CanonicalGS] sample episode "
                f"scene={sample.scene_id} clip={sample.clip_id} "
                f"frame_pool={len(sample.frame_pool)} "
                f"targets={len(sample.target_set)} "
                f"contexts={sorted(sample.context_sets)}"
            )
        return

    if cfg.mode == "inspect_forward":
        builder = EpisodeBuilder(
            context_sizes=tuple(cfg.dataset.context_sizes),
            target_views=cfg.dataset.target_views,
            min_frames_per_episode=cfg.dataset.min_frames_per_episode,
            subsample_to=cfg.dataset.subsample_to,
            seed=cfg.dataset.seed,
        )
        tensor_episodes = build_re10k_sample_tensor_episodes(
            cfg.dataset.roots,
            cfg.dataset.split,
            builder,
            1,
            tuple(cfg.dataset.image_shape),
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.fixed_scene_count,
            cfg.dataset.fixed_scene_seed,
        )
        if not tensor_episodes:
            raise ValueError("no buildable RE10K tensor episodes found")

        device = resolve_runtime_device(cfg.runtime.device)
        pipeline = CanonicalGsPipeline(cfg.model).to(device)
        sample_context = min(cfg.dataset.context_sizes)
        result = pipeline(move_to_device(tensor_episodes[0], device), sample_context)
        print(
            "[CanonicalGS] "
            f"forward_context={result['context_size']} "
            f"active_cells={result['num_active_cells']} "
            f"gaussians={result['num_gaussians']} "
            f"mean_confidence={result['mean_confidence']:.6f} "
            f"mean_semantic_consistency={result['mean_semantic_consistency']:.6f} "
            f"mean_opacity={result['mean_opacity']:.6f}"
        )
        print(
            "[CanonicalGS] "
            f"target_count={int(result['target_indices'].shape[0])} "
            f"state_indices_shape={tuple(result['state'].indices.shape)} "
            f"gaussian_means_shape={tuple(result['gaussians'].means.shape)}"
        )
        return

    if cfg.mode == "inspect_diagnostics":
        builder = EpisodeBuilder(
            context_sizes=tuple(cfg.dataset.context_sizes),
            target_views=cfg.dataset.target_views,
            min_frames_per_episode=cfg.dataset.min_frames_per_episode,
            subsample_to=cfg.dataset.subsample_to,
            seed=cfg.dataset.seed,
        )
        tensor_episodes = build_re10k_sample_tensor_episodes(
            cfg.dataset.roots,
            cfg.dataset.split,
            builder,
            1,
            tuple(cfg.dataset.image_shape),
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.fixed_scene_count,
            cfg.dataset.fixed_scene_seed,
        )
        if not tensor_episodes:
            raise ValueError("no buildable RE10K tensor episodes found")

        device = resolve_runtime_device(cfg.runtime.device)
        pipeline = CanonicalGsPipeline(cfg.model).to(device)
        episode = move_to_device(tensor_episodes[0], device)
        sample_context = min(cfg.dataset.context_sizes)
        result = pipeline(episode, sample_context, include_render_payload=True)
        readout = result["readout"]
        state = result["state"]
        depth = result["depth"]
        confidence = result["view_confidence"]
        print(
            "[CanonicalGS] "
            f"diagnostic_context={sample_context} "
            f"depth_shape={tuple(depth.shape)} "
            f"depth_min={depth.min().item():.6f} "
            f"depth_max={depth.max().item():.6f} "
            f"depth_mean={depth.mean().item():.6f}"
        )
        print(
            "[CanonicalGS] "
            f"confidence_min={confidence.min().item():.6f} "
            f"confidence_max={confidence.max().item():.6f} "
            f"confidence_mean={confidence.mean().item():.6f}"
        )
        print(
            "[CanonicalGS] "
            f"surface_sum={state.surface_evidence.sum().item():.6f} "
            f"free_sum={state.free_evidence.sum().item():.6f} "
            f"canonical_certainty_mean={readout.canonical_certainty.mean().item():.6f} "
            f"semantic_consistency_mean={readout.semantic_consistency.mean().item():.6f} "
            f"uncertainty_mean={readout.uncertainty.mean().item():.6f}"
        )
        print(
            "[CanonicalGS] "
            f"active_cells={result['num_active_cells']} "
            f"gaussians={result['num_gaussians']} "
            f"mean_opacity={result['mean_opacity']:.6f}"
        )
        return

    if cfg.mode == "inspect_render_sanity":
        builder = EpisodeBuilder(
            context_sizes=tuple(cfg.dataset.context_sizes),
            target_views=cfg.dataset.target_views,
            min_frames_per_episode=cfg.dataset.min_frames_per_episode,
            subsample_to=cfg.dataset.subsample_to,
            seed=cfg.dataset.seed,
        )
        tensor_episodes = build_re10k_sample_tensor_episodes(
            cfg.dataset.roots,
            cfg.dataset.split,
            builder,
            1,
            tuple(cfg.dataset.image_shape),
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.fixed_scene_count,
            cfg.dataset.fixed_scene_seed,
        )
        if not tensor_episodes:
            raise ValueError("no buildable RE10K tensor episodes found")

        device = resolve_runtime_device(cfg.runtime.device)
        pipeline = CanonicalGsPipeline(cfg.model).to(device)
        episode = move_to_device(tensor_episodes[0], device)
        context_size = min(cfg.dataset.context_sizes)
        result = pipeline(episode, context_size, include_render_payload=True)

        context_index = result["context_indices"][:1]
        target_index = result["target_indices"][:1]
        context_render, context_cov = render_gaussian_views(
            result["render_gaussians"],
            episode["extrinsics"][context_index],
            episode["intrinsics"][context_index],
            tuple(episode["images"][context_index].shape[-2:]),
        )
        target_render, target_cov = render_gaussian_views(
            result["render_gaussians"],
            episode["extrinsics"][target_index],
            episode["intrinsics"][target_index],
            tuple(episode["images"][target_index].shape[-2:]),
        )
        context_gt = episode["images"][context_index]
        target_gt = episode["images"][target_index]
        context_mse = (context_render - context_gt).square().mean()
        target_mse = (target_render - target_gt).square().mean()
        context_psnr = -10.0 * context_mse.clamp_min(1e-8).log10()
        target_psnr = -10.0 * target_mse.clamp_min(1e-8).log10()
        means = result["render_gaussians"].means
        opacities = result["render_gaussians"].opacities
        support = result["render_gaussians"].support
        confidence = result["render_gaussians"].confidence
        context_extrinsic = episode["extrinsics"][context_index][0]
        context_intrinsic = episode["intrinsics"][context_index][0]
        means_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)
        world_to_camera = torch.linalg.inv(context_extrinsic)
        camera_points = (world_to_camera @ means_h.T).T[:, :3]
        z = camera_points[:, 2]
        intrinsics_px = context_intrinsic.clone()
        intrinsics_px[0, :] *= context_gt.shape[-1]
        intrinsics_px[1, :] *= context_gt.shape[-2]
        projected = camera_points @ intrinsics_px.T
        xy = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        in_bounds = (
            (z > 1e-4)
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] <= context_gt.shape[-1] - 1)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] <= context_gt.shape[-2] - 1)
        )
        print(
            "[CanonicalGS] "
            f"render_sanity_context_psnr={context_psnr.item():.3f} "
            f"render_sanity_context_coverage={context_cov.mean().item():.6f}"
        )
        print(
            "[CanonicalGS] "
            f"render_sanity_target_psnr={target_psnr.item():.3f} "
            f"render_sanity_target_coverage={target_cov.mean().item():.6f} "
            f"render_gaussians={result['render_gaussians'].means.shape[0]} "
            f"hard_gaussians={result['num_gaussians']}"
        )
        print(
            "[CanonicalGS] "
            f"render_sanity_opacity_mean={opacities.mean().item():.6f} "
            f"render_sanity_opacity_max={opacities.max().item():.6f} "
            f"render_sanity_support_mean={support.mean().item():.6f} "
            f"render_sanity_conf_mean={confidence.mean().item():.6f} "
            f"render_sanity_z_min={z.min().item():.6f} "
            f"render_sanity_z_max={z.max().item():.6f} "
            f"render_sanity_in_bounds={in_bounds.float().mean().item():.6f}"
        )
        return

    if cfg.mode == "inspect_losses":
        builder = EpisodeBuilder(
            context_sizes=tuple(cfg.dataset.context_sizes),
            target_views=cfg.dataset.target_views,
            min_frames_per_episode=cfg.dataset.min_frames_per_episode,
            subsample_to=cfg.dataset.subsample_to,
            seed=cfg.dataset.seed,
        )
        tensor_episodes = build_re10k_sample_tensor_episodes(
            cfg.dataset.roots,
            cfg.dataset.split,
            builder,
            1,
            tuple(cfg.dataset.image_shape),
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.eval_holdout_stride,
            cfg.dataset.eval_holdout_offset,
            cfg.dataset.fixed_scene_count,
            cfg.dataset.fixed_scene_seed,
        )
        if not tensor_episodes:
            raise ValueError("no buildable RE10K tensor episodes found")

        device = resolve_runtime_device(cfg.runtime.device)
        pipeline = CanonicalGsPipeline(cfg.model).to(device)
        episode = move_to_device(tensor_episodes[0], device)
        outputs = {
            context_size: pipeline(episode, context_size)
            for context_size in cfg.dataset.context_sizes
        }
        losses = CanonicalLossComputer(cfg.objective)(outputs)
        print(
            "[CanonicalGS] "
            f"loss_total={losses.total_loss.item():.6f} "
            f"loss_render={losses.render_loss.item():.6f} "
            f"loss_mono={losses.monotone_loss.item():.6f} "
        )
        return

    if cfg.mode == "bootstrap_train":
        run_bootstrap_training(cfg)
        return

    if cfg.mode == "scene_overfit_train":
        run_scene_overfit_training(cfg)
        return

    if cfg.mode == "smoke_test_100scenes":
        run_subset_smoke_test(cfg)
        return

    raise ValueError(f"unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
