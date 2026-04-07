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
    validation_holdout_stride: int = 20
    validation_holdout_offset: int = 0
    evaluation_holdout_stride: int = 100
    evaluation_holdout_offset: int = 0
    inspect_num_scenes: int = 4
    image_shape: list[int] = field(default_factory=lambda: [180, 320])
    seed: int = 111123
    min_frames_per_episode: int = 12
    context_sizes: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    target_views: int = 6
    subsample_to: int = 12


@dataclass(slots=True)
class ModelConfig:
    name: str = "canonicalgs_mono_multiview"
    voxel_size: float = 0.12
    max_active_voxels: int = 300000
    feature_dim: int = 128
    appearance_dim: int = 32
    gaussians_per_cell: int = 1
    encoder_downsample: int = 8
    dpt_output_stride: int = 4
    dinov2_model_name: str = "dinov2_vits14"
    dinov2_pretrained: bool = True
    freeze_dinov2: bool = False
    num_depth_bins: int = 128
    min_depth: float = 0.01
    max_depth: float = 100.0
    min_depth_uncertainty: float = 0.05
    free_space_ratio: float = 0.5
    free_space_steps: int = 8
    free_space_margin_multiplier: float = 1.5
    surface_weight: float = 1.0
    free_weight: float = 0.5
    surface_band_offsets: list[float] = field(default_factory=lambda: [-1.5, -0.5, 0.5, 1.5])
    support_threshold: float = 0.1
    confidence_threshold: float = 0.05
    gaussian_scale_min: float = 0.02
    gaussian_scale_max: float = 1.0
    opacity_gain: float = 3.0


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cuda:7"
    conda_env: str = "depthsplat"
    remote_host: str = "malab"
    remote_root: str = "/data0/xxy/code/CanonicalGS"


@dataclass(slots=True)
class ObjectiveConfig:
    lambda_rend: float = 1.0
    lambda_conv: float = 0.2
    lambda_mono: float = 0.05
    lambda_null: float = 0.001
    teacher_support_threshold: float = 0.5
    teacher_confidence_threshold: float = 0.5
    low_confidence_threshold: float = 0.25


@dataclass(slots=True)
class TrainConfig:
    bootstrap_steps: int = 2
    bootstrap_lr: float = 1e-3
    log_every: int = 1
    overfit_steps: int = 100
    overfit_scene_index: int = 0
    overfit_train_context_size: int = 6
    render_train_target_views: int = 1
    render_eval_target_views: int = 2
    save_checkpoint: bool = True
    amp: bool = True


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
    return RootConfig(
        mode=raw.get("mode", "inspect_dataset"),
        seed=raw.get("seed", 111123),
        output_dir=raw.get("output_dir", "outputs/canonicalgs"),
        wandb=WandbConfig(**raw.get("wandb", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        objective=ObjectiveConfig(**raw.get("objective", {})),
        train=TrainConfig(**raw.get("train", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
