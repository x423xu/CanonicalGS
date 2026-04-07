from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


TRAIN_COMMAND = r"""CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -p /data0/xxy/conda_envs/depthsplat --no-capture-output python -m src.main \
    +experiment=re10k \
    mode=train \
    model/encoder=mono_model \
    dataset.roots=[/data0/xxy/data/re10k] \
    data_loader.train.batch_size=2 \
    dataset.min_views=2 \
    dataset.max_views=2 \
    model.encoder.profile_voxelization=false \
    train.video_viz_interval_steps=1000 \
    trainer.max_steps=300000 \
    output_dir=checkpoints/mono_voxel_lite \
    dataset.test_chunk_interval=10 \
    dataset.image_shape=[256,256]"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--eval-index",
        type=Path,
        default=Path("/data0/xxy/code/Active-FFGS-streaming/assets/evaluation_index_re10k.json"),
    )
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mono_voxel_lite_dataset_trace.json"),
    )
    return parser.parse_args()


def _flatten_wandb_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
        else:
            flat[key] = value
    return flat


def _normalize_cfg_from_wandb(cfg_dict: dict[str, Any], eval_index: Path) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg_dict))
    cfg["mode"] = "train"
    cfg["wandb"]["mode"] = "disabled"
    cfg["dataset"]["view_sampler"] = {
        "name": "bounded",
        "num_target_views": 4,
        "num_context_views": 2,
        "min_distance_between_context_views": 45,
        "max_distance_between_context_views": 135,
        "min_distance_to_context_views": 0,
        "warm_up_steps": 30000,
        "initial_min_distance_between_context_views": 25,
        "initial_max_distance_between_context_views": 45,
    }
    cfg["dataset"]["roots"] = ["/data0/xxy/data/re10k"]
    cfg["dataset"]["image_shape"] = [256, 256]
    cfg["dataset"]["min_views"] = 2
    cfg["dataset"]["max_views"] = 2
    cfg["dataset"]["test_chunk_interval"] = 10
    cfg["data_loader"]["train"]["batch_size"] = 2
    cfg["train"]["video_viz_interval_steps"] = 1000
    cfg["trainer"]["max_steps"] = 300000
    cfg["output_dir"] = "checkpoints/mono_voxel_lite"
    cfg["model"]["encoder"]["profile_voxelization"] = False
    cfg["eval_dataset"] = {
        **cfg["dataset"],
        "view_sampler": {
            "name": "evaluation",
            "index_path": str(eval_index),
            "num_context_views": 2,
        },
    }
    return cfg


def _overlap(context_indices: list[int], target_indices: list[int]) -> list[int]:
    context_set = set(context_indices)
    return sorted(context_set.intersection(target_indices))


def _serialize_batch(batch: dict[str, Any], step: int | None = None, tracker_step: int | None = None) -> dict[str, Any]:
    context_index = batch["context"]["index"].tolist()
    target_index = batch["target"]["index"].tolist()
    context_image_shape = list(batch["context"]["image"].shape)
    target_image_shape = list(batch["target"]["image"].shape)
    items = []
    batch_size = len(batch["scene"])
    for i in range(batch_size):
        ctx = context_index[i]
        tgt = target_index[i]
        items.append(
            {
                "scene": batch["scene"][i],
                "context_indices": ctx,
                "target_indices": tgt,
                "context_target_overlap": _overlap(ctx, tgt),
                "num_context": len(ctx),
                "num_target": len(tgt),
                "near_mean": float(batch["context"]["near"][i].float().mean().item()),
                "far_mean": float(batch["context"]["far"][i].float().mean().item()),
            }
        )
    result = {
        "batch_size": batch_size,
        "context_image_shape": context_image_shape,
        "target_image_shape": target_image_shape,
        "items": items,
    }
    if step is not None:
        result["step"] = step
    if tracker_step is not None:
        result["sampler_global_step"] = tracker_step
    return result


def main() -> None:
    args = _parse_args()
    repo_root = args.reference_repo.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"reference repo not found: {repo_root}")
    sys.path.insert(0, str(repo_root))

    from src.dataset.data_module import DataLoaderCfg, DataLoaderStageCfg, DataModule, get_data_shim
    from src.dataset.dataset_re10k import DatasetRE10kCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
    from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluationCfg
    from src.misc.step_tracker import StepTracker
    from src.model.encoder import get_encoder
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg as RefGaussianAdapterCfg
    from src.model.encoder.mono_model import MonoModelCfg, OpacityMappingCfg as RefOpacityMappingCfg
    from src.config import TrainControllerCfg

    with args.wandb_config.open("r", encoding="utf-8") as f:
        wandb_cfg = yaml.safe_load(f)
    cfg_dict = _normalize_cfg_from_wandb(_flatten_wandb_config(wandb_cfg), args.eval_index)

    train_controller_cfg = TrainControllerCfg(**cfg_dict["train_controller"])
    bounded_sampler = ViewSamplerBoundedCfg(**cfg_dict["dataset"]["view_sampler"])
    dataset_cfg = DatasetRE10kCfg(
        name=cfg_dict["dataset"]["name"],
        roots=[Path(root) for root in cfg_dict["dataset"]["roots"]],
        baseline_epsilon=cfg_dict["dataset"]["baseline_epsilon"],
        max_fov=cfg_dict["dataset"]["max_fov"],
        make_baseline_1=cfg_dict["dataset"]["make_baseline_1"],
        augment=cfg_dict["dataset"]["augment"],
        test_len=cfg_dict["dataset"]["test_len"],
        test_chunk_interval=cfg_dict["dataset"]["test_chunk_interval"],
        skip_bad_shape=cfg_dict["dataset"]["skip_bad_shape"],
        near=cfg_dict["dataset"]["near"],
        far=cfg_dict["dataset"]["far"],
        baseline_scale_bounds=cfg_dict["dataset"]["baseline_scale_bounds"],
        shuffle_val=cfg_dict["dataset"]["shuffle_val"],
        train_times_per_scene=cfg_dict["dataset"]["train_times_per_scene"],
        highres=cfg_dict["dataset"]["highres"],
        use_index_to_load_chunk=cfg_dict["dataset"]["use_index_to_load_chunk"],
        min_views=cfg_dict["dataset"]["min_views"],
        max_views=cfg_dict["dataset"]["max_views"],
        image_shape=list(cfg_dict["dataset"]["image_shape"]),
        background_color=list(cfg_dict["dataset"]["background_color"]),
        cameras_are_circular=cfg_dict["dataset"]["cameras_are_circular"],
        overfit_to_scene=cfg_dict["dataset"].get("overfit_to_scene"),
        view_sampler=bounded_sampler,
    )
    eval_dataset_cfg = DatasetRE10kCfg(
        name=cfg_dict["eval_dataset"]["name"],
        roots=[Path(root) for root in cfg_dict["eval_dataset"]["roots"]],
        baseline_epsilon=cfg_dict["eval_dataset"]["baseline_epsilon"],
        max_fov=cfg_dict["eval_dataset"]["max_fov"],
        make_baseline_1=cfg_dict["eval_dataset"]["make_baseline_1"],
        augment=cfg_dict["eval_dataset"]["augment"],
        test_len=cfg_dict["eval_dataset"]["test_len"],
        test_chunk_interval=cfg_dict["eval_dataset"]["test_chunk_interval"],
        skip_bad_shape=cfg_dict["eval_dataset"]["skip_bad_shape"],
        near=cfg_dict["eval_dataset"]["near"],
        far=cfg_dict["eval_dataset"]["far"],
        baseline_scale_bounds=cfg_dict["eval_dataset"]["baseline_scale_bounds"],
        shuffle_val=cfg_dict["eval_dataset"]["shuffle_val"],
        train_times_per_scene=cfg_dict["eval_dataset"]["train_times_per_scene"],
        highres=cfg_dict["eval_dataset"]["highres"],
        use_index_to_load_chunk=cfg_dict["eval_dataset"]["use_index_to_load_chunk"],
        min_views=cfg_dict["eval_dataset"]["min_views"],
        max_views=cfg_dict["eval_dataset"]["max_views"],
        image_shape=list(cfg_dict["eval_dataset"]["image_shape"]),
        background_color=list(cfg_dict["eval_dataset"]["background_color"]),
        cameras_are_circular=cfg_dict["eval_dataset"]["cameras_are_circular"],
        overfit_to_scene=cfg_dict["eval_dataset"].get("overfit_to_scene"),
        view_sampler=ViewSamplerEvaluationCfg(
            name=cfg_dict["eval_dataset"]["view_sampler"]["name"],
            index_path=Path(cfg_dict["eval_dataset"]["view_sampler"]["index_path"]),
            num_context_views=cfg_dict["eval_dataset"]["view_sampler"]["num_context_views"],
        ),
    )

    data_loader_cfg = DataLoaderCfg(
        train=DataLoaderStageCfg(**cfg_dict["data_loader"]["train"]),
        test=DataLoaderStageCfg(**cfg_dict["data_loader"]["test"]),
        val=DataLoaderStageCfg(**cfg_dict["data_loader"]["val"]),
    )
    trace_loader_cfg = DataLoaderCfg(
        train=DataLoaderStageCfg(
            batch_size=data_loader_cfg.train.batch_size,
            num_workers=data_loader_cfg.train.num_workers,
            persistent_workers=data_loader_cfg.train.persistent_workers,
            seed=data_loader_cfg.train.seed,
        ),
        test=DataLoaderStageCfg(
            batch_size=data_loader_cfg.test.batch_size,
            num_workers=data_loader_cfg.test.num_workers,
            persistent_workers=data_loader_cfg.test.persistent_workers,
            seed=data_loader_cfg.test.seed,
        ),
        val=DataLoaderStageCfg(
            batch_size=data_loader_cfg.val.batch_size,
            num_workers=data_loader_cfg.val.num_workers,
            persistent_workers=data_loader_cfg.val.persistent_workers,
            seed=data_loader_cfg.val.seed,
        ),
    )

    encoder_cfg = MonoModelCfg(
        name=cfg_dict["model"]["encoder"]["name"],
        d_feature=cfg_dict["model"]["encoder"]["d_feature"],
        num_depth_candidates=cfg_dict["model"]["encoder"]["num_depth_candidates"],
        num_surfaces=cfg_dict["model"]["encoder"]["num_surfaces"],
        gaussian_adapter=RefGaussianAdapterCfg(**cfg_dict["model"]["encoder"]["gaussian_adapter"]),
        opacity_mapping=RefOpacityMappingCfg(**cfg_dict["model"]["encoder"]["opacity_mapping"]),
        gaussians_per_pixel=cfg_dict["model"]["encoder"]["gaussians_per_pixel"],
        unimatch_weights_path=cfg_dict["model"]["encoder"]["unimatch_weights_path"],
        downscale_factor=cfg_dict["model"]["encoder"]["downscale_factor"],
        shim_patch_size=cfg_dict["model"]["encoder"]["shim_patch_size"],
        multiview_trans_attn_split=cfg_dict["model"]["encoder"]["multiview_trans_attn_split"],
        costvolume_unet_feat_dim=cfg_dict["model"]["encoder"]["costvolume_unet_feat_dim"],
        costvolume_unet_channel_mult=list(cfg_dict["model"]["encoder"]["costvolume_unet_channel_mult"]),
        costvolume_unet_attn_res=list(cfg_dict["model"]["encoder"]["costvolume_unet_attn_res"]),
        depth_unet_feat_dim=cfg_dict["model"]["encoder"]["depth_unet_feat_dim"],
        depth_unet_attn_res=list(cfg_dict["model"]["encoder"]["depth_unet_attn_res"]),
        depth_unet_channel_mult=list(cfg_dict["model"]["encoder"]["depth_unet_channel_mult"]),
        wo_depth_refine=cfg_dict["model"]["encoder"]["wo_depth_refine"],
        wo_cost_volume=cfg_dict["model"]["encoder"]["wo_cost_volume"],
        wo_backbone_cross_attn=cfg_dict["model"]["encoder"]["wo_backbone_cross_attn"],
        wo_cost_volume_refine=cfg_dict["model"]["encoder"]["wo_cost_volume_refine"],
        use_epipolar_trans=cfg_dict["model"]["encoder"]["use_epipolar_trans"],
        monodepth_vit_type=cfg_dict["model"]["encoder"]["monodepth_vit_type"],
        enable_voxelization=cfg_dict["model"]["encoder"]["enable_voxelization"],
        use_plucker_embedding=cfg_dict["model"]["encoder"]["use_plucker_embedding"],
        voxel_feature_dim=cfg_dict["model"]["encoder"]["voxel_feature_dim"],
        gaussians_per_cell=cfg_dict["model"]["encoder"]["gaussians_per_cell"],
        down_strides=list(cfg_dict["model"]["encoder"]["down_strides"]),
        cell_scale=cfg_dict["model"]["encoder"]["cell_scale"],
        cube_merge_type=cfg_dict["model"]["encoder"]["cube_merge_type"],
        voxelization_downsample_factor=cfg_dict["model"]["encoder"]["voxelization_downsample_factor"],
        voxel_conf_threshold=cfg_dict["model"]["encoder"]["voxel_conf_threshold"],
        expanded_voxel_topk=cfg_dict["model"]["encoder"].get("expanded_voxel_topk", 50000),
        expanded_voxel_bounds_filter=cfg_dict["model"]["encoder"].get("expanded_voxel_bounds_filter", False),
        world_cell_scale_3views=cfg_dict["model"]["encoder"].get("world_cell_scale_3views", 1.0),
        world_cell_scale_4views=cfg_dict["model"]["encoder"].get("world_cell_scale_4views", 0.75),
        profile_voxelization=cfg_dict["model"]["encoder"]["profile_voxelization"],
        voxel_compute_2d_branch=cfg_dict["model"]["encoder"]["voxel_compute_2d_branch"],
        voxel_low_vram_arch=cfg_dict["model"]["encoder"]["voxel_low_vram_arch"],
        voxel_train_depth_predictor=cfg_dict["model"]["encoder"]["voxel_train_depth_predictor"],
        return_depth=cfg_dict["model"]["encoder"]["return_depth"],
    )
    encoder, _ = get_encoder(
        encoder_cfg,
        gs_cube=train_controller_cfg.gs_cube,
        vggt_meta=train_controller_cfg.vggt_meta,
        knn_down=train_controller_cfg.knn_down,
        gaussian_merge=train_controller_cfg.gaussian_merge,
        depth_distillation=train_controller_cfg.depth_distillation,
        train_controller_cfg=train_controller_cfg,
    )
    data_shim = get_data_shim(encoder)

    step_tracker = StepTracker()
    data_module = DataModule(
        dataset_cfg,
        trace_loader_cfg,
        step_tracker=step_tracker,
        global_rank=0,
        train_controller_cfg=train_controller_cfg,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    eval_loader = data_module.test_dataloader(dataset_cfg=eval_dataset_cfg)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset.dataset
    test_dataset = test_loader.dataset
    eval_dataset = eval_loader.dataset

    report: dict[str, Any] = {
        "training_command": TRAIN_COMMAND,
        "resolved_config": cfg_dict,
        "stage_summary": {
            "train": {
                "stage": "train",
                "data_stage": train_dataset.data_stage,
                "dataset_length": len(train_dataset),
                "dataloader_batch_size": trace_loader_cfg.train.batch_size,
                "num_workers": trace_loader_cfg.train.num_workers,
                "view_sampler": cfg_dict["dataset"]["view_sampler"],
            },
            "val": {
                "stage": "val",
                "data_stage": val_dataset.data_stage,
                "underlying_dataset_length": len(val_dataset),
                "wrapper_length": len(val_loader.dataset),
                "dataloader_batch_size": trace_loader_cfg.val.batch_size,
                "num_workers": trace_loader_cfg.val.num_workers,
                "view_sampler": cfg_dict["dataset"]["view_sampler"],
                "note": "validation uses ValidationWrapper(length=1) over dataset stage=test",
            },
            "eval": {
                "stage": "eval_full_testsets",
                "data_stage": eval_dataset.data_stage,
                "dataset_length": len(eval_dataset),
                "dataloader_batch_size": trace_loader_cfg.test.batch_size,
                "num_workers": trace_loader_cfg.test.num_workers,
                "view_sampler": cfg_dict["eval_dataset"]["view_sampler"],
                "note": "this is the train-time full evaluation loader built from eval_data_cfg",
            },
            "test": {
                "stage": "test",
                "data_stage": test_dataset.data_stage,
                "dataset_length": len(test_dataset),
                "dataloader_batch_size": trace_loader_cfg.test.batch_size,
                "num_workers": trace_loader_cfg.test.num_workers,
                "view_sampler": cfg_dict["dataset"]["view_sampler"],
            },
        },
        "train_trace": [],
        "val_events": [],
        "eval_preview": [],
        "test_preview": [],
    }

    step_tracker.set_step(0)

    # Sanity validation before training.
    sanity_val_batch = data_shim(next(iter(val_loader)))
    report["val_events"].append(
        {
            "event": "sanity_val_before_training",
            "trainer_global_step": 0,
            "sampler_global_step": 0,
            **_serialize_batch(sanity_val_batch, step=0, tracker_step=0),
        }
    )

    train_iter = iter(train_loader)
    for step in range(args.train_steps):
        batch = next(train_iter)
        batch = data_shim(batch)
        tracker_step = step_tracker.get_step()
        report["train_trace"].append(_serialize_batch(batch, step=step, tracker_step=tracker_step))
        step_tracker.set_step(step)

        if step + 1 == 50:
            val_batch = data_shim(next(iter(val_loader)))
            report["val_events"].append(
                {
                    "event": "validation_after_50_steps",
                    "trainer_global_step": 50,
                    "sampler_global_step": step_tracker.get_step(),
                    **_serialize_batch(val_batch, step=50, tracker_step=step_tracker.get_step()),
                }
            )
        if step + 1 == 100:
            val_batch = data_shim(next(iter(val_loader)))
            report["val_events"].append(
                {
                    "event": "validation_after_100_steps",
                    "trainer_global_step": 100,
                    "sampler_global_step": step_tracker.get_step(),
                    **_serialize_batch(val_batch, step=100, tracker_step=step_tracker.get_step()),
                }
            )

    eval_iter = iter(eval_loader)
    for index in range(args.preview_count):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break
        batch = data_shim(batch)
        report["eval_preview"].append(_serialize_batch(batch, step=index, tracker_step=step_tracker.get_step()))

    test_iter = iter(test_loader)
    for index in range(args.preview_count):
        try:
            batch = next(test_iter)
        except StopIteration:
            break
        batch = data_shim(batch)
        report["test_preview"].append(_serialize_batch(batch, step=index, tracker_step=step_tracker.get_step()))

    def summarize_overlap(entries: list[dict[str, Any]]) -> dict[str, Any]:
        item_count = 0
        overlap_count = 0
        unique_scenes = set()
        for entry in entries:
            for item in entry["items"]:
                item_count += 1
                unique_scenes.add(item["scene"])
                if item["context_target_overlap"]:
                    overlap_count += 1
        return {
            "items": item_count,
            "unique_scenes": len(unique_scenes),
            "items_with_context_target_overlap": overlap_count,
            "overlap_fraction": 0.0 if item_count == 0 else overlap_count / item_count,
        }

    report["overlap_summary"] = {
        "train": summarize_overlap(report["train_trace"]),
        "val": summarize_overlap(report["val_events"]),
        "eval": summarize_overlap(report["eval_preview"]),
        "test": summarize_overlap(report["test_preview"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["stage_summary"], indent=2))
    print(json.dumps(report["overlap_summary"], indent=2))
    print(f"wrote trace to {args.output}")


if __name__ == "__main__":
    main()
