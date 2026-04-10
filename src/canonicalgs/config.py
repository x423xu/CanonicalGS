from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class WandbConfig:
    project: str = "canonicalgs"
    entity: str = ""
    mode: str = "online"
    id: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetConfig:
    name: str = "re10k"
    roots: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    split: str = "train"
    context_sizes: list[int] = field(default_factory=lambda: [2, 3, 4])
    train_context_sizes: list[int] = field(default_factory=lambda: [2, 3, 4])
    eval_context_sizes: list[int] = field(default_factory=lambda: [2, 4])
    test_context_sizes: list[int] = field(default_factory=lambda: [2, 4])
    eval_holdout_stride: int = 100
    eval_holdout_offset: int = 0
    fixed_scene_count: int | None = None
    fixed_scene_seed: int = 111123
    inspect_num_scenes: int = 4
    image_shape: list[int] = field(default_factory=lambda: [180, 320])
    seed: int = 111123
    min_frames_per_episode: int = 12
    target_views: int = 6
    subsample_to: int = 12
    context_gap_min: int = 45
    context_gap_max: int = 145
    context_gap_warmup_steps: int = 5000


@dataclass(slots=True)
class ModelConfig:
    name: str = "canonicalgs_mono_multiview"
    merge_views: bool = True
    voxel_size: float = 0.005
    max_active_voxels: int = 300000
    feature_dim: int = 128
    appearance_dim: int = 32
    gaussians_per_cell: int = 1
    encoder_downsample: int = 8
    dpt_output_stride: int = 4
    dinov2_model_name: str = "dinov2_vits14"
    dinov2_pretrained: bool = True
    freeze_dinov2: bool = False
    allow_dinov2_fallback: bool = False
    num_depth_bins: int = 128
    min_depth: float = 0.1
    max_depth: float = 20.0
    min_depth_uncertainty: float = 0.05
    positional_certainty_tau: float = 0.5
    max_positional_uncertainty: float = 5.0
    cost_volume_temperature: float = 0.35
    cost_volume_visibility_beta: float = 0.2
    free_space_ratio: float = 0.0
    free_space_steps: int = 0
    free_space_margin_multiplier: float = 1.5
    surface_weight: float = 1.0
    free_weight: float = 0.5
    appearance_uncertainty_bias: float = 0.05
    appearance_uncertainty_init: float = 0.05
    gaussian_scale_min: float = 0.02
    gaussian_scale_max: float = 1.0
    decoder_hidden_dim: int = 128


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cuda:7"
    conda_env: str = "depthsplat"
    remote_host: str = "malab"
    remote_root: str = "/data0/xxy/code/CanonicalGS"


@dataclass(slots=True)
class ObjectiveConfig:
    lambda_rend: float = 1.0
    lambda_mono: float = 0.05
    mono_on: bool = True


@dataclass(slots=True)
class TrainConfig:
    bootstrap_steps: int = 2
    bootstrap_eval_every: int = 0
    bootstrap_eval_num_scenes: int = 100
    bootstrap_lr: float = 1e-4
    debug: bool = False
    log_every: int = 1
    smoke_steps: int = 500
    smoke_eval_every: int = 50
    overfit_steps: int = 100
    overfit_scene_index: int = 0
    overfit_train_context_size: int = 4
    render_train_target_views: int = 1
    render_eval_target_views: int = 2
    save_checkpoint: bool = True
    amp: bool = False


@dataclass(slots=True)
class RootConfig:
    mode: str = "inspect_dataset"
    seed: int = 111123
    output_dir: str = "outputs/canonicalgs"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _to_plain_dict(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def load_typed_root_config(cfg: DictConfig | Mapping[str, Any]) -> RootConfig:
    raw = _to_plain_dict(cfg)
    dataset_raw = dict(raw.get("dataset", {}))
    model_raw = dict(raw.get("model", {}))
    model_raw.pop("surface_band_offsets", None)
    if "eval_holdout_stride" not in dataset_raw:
        dataset_raw["eval_holdout_stride"] = dataset_raw.get(
            "evaluation_holdout_stride",
            dataset_raw.get("validation_holdout_stride", 100),
        )
    if "eval_holdout_offset" not in dataset_raw:
        dataset_raw["eval_holdout_offset"] = dataset_raw.get(
            "evaluation_holdout_offset",
            dataset_raw.get("validation_holdout_offset", 0),
        )
    if "max_positional_uncertainty" not in model_raw:
        model_raw["max_positional_uncertainty"] = model_raw.get(
            "max_relative_depth_uncertainty",
            5.0,
        )
    return RootConfig(
        mode=raw.get("mode", "inspect_dataset"),
        seed=raw.get("seed", 111123),
        output_dir=raw.get("output_dir", "outputs/canonicalgs"),
        wandb=WandbConfig(**raw.get("wandb", {})),
        dataset=DatasetConfig(**dataset_raw),
        model=ModelConfig(**model_raw),
        objective=ObjectiveConfig(**raw.get("objective", {})),
        train=TrainConfig(**raw.get("train", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
