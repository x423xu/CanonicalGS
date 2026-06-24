#!/usr/bin/env python3
"""Run CanonicalGS RE10K evaluation.

The default path evaluates the transplanted RE10K 2-view CanonicalGS checkpoint on
100 scenes using the canonical len100 evaluation index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_RE10K_INDEX = Path("assets/re10k_2v_canonical_len100_indicies.json")
DEFAULT_RE10K_CHECKPOINT = Path(
    "checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/"
    "checkpoints_backups/epoch_10-step_340070.ckpt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CanonicalGS on RE10K.")
    parser.add_argument(
        "--re10k-root",
        default=os.environ.get("CANONICALGS_RE10K_ROOT", "datasets/re10k"),
        help="RE10K dataset root. Defaults to CANONICALGS_RE10K_ROOT or datasets/re10k.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_RE10K_CHECKPOINT),
        help="Checkpoint path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_RE10K_INDEX),
        help="Evaluation index path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation/re10k_2v")
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--num-context-views", type=int, default=2)
    parser.add_argument("--cell-scale", type=float, default=2.8)
    parser.add_argument("--cube-merge-type", default="max_conf")
    parser.add_argument("--render-chunk-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cuda-device", default=os.environ.get("CANONICALGS_CUDA_DEVICE", "9"))
    parser.add_argument("--omp-num-threads", default=os.environ.get("OMP_NUM_THREADS", "32"))
    parser.add_argument("--view-base", type=int, default=4)
    parser.add_argument("--chunk-num", type=int, default=2)
    parser.add_argument("--anchor-base", type=int, default=4)
    parser.add_argument("--iter-depth", action="store_true")
    parser.add_argument("--batch-forward", action="store_true")
    parser.add_argument("--anchor-features", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--save-gt-image", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-gaussian", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-input-images", action="store_true")
    parser.add_argument("--no-strict-load", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_overrides(args: argparse.Namespace, repo_root: Path) -> list[str]:
    checkpoint = _repo_path(repo_root, args.checkpoint)
    index_path = _repo_path(repo_root, args.index_path)
    output_dir = args.output_dir
    if args.cell_scale is not None:
        output_dir = f"{output_dir}_scale{args.cell_scale}"

    overrides = [
        "+experiment=re10k",
        f"dataset.roots=[{args.re10k_root}]",
        "data_loader.train.batch_size=2",
        "data_loader.test.batch_size=1",
        f"data_loader.test.num_workers={args.num_workers}",
        "data_loader.test.persistent_workers=false",
        "dataset.test_chunk_interval=1",
        f"dataset.test_len={args.num_scenes}",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={index_path}",
        f"dataset.view_sampler.num_context_views={args.num_context_views}",
        "mode=test",
        f"checkpointing.pretrained_model={checkpoint}",
        f"checkpointing.no_strict_load={_bool(args.no_strict_load)}",
        f"output_dir={output_dir}",
        "train_controller.base_model=true",
        "trainer.val_check_interval=0.25",
        "test.compute_scores=true",
        f"test.save_image={_bool(args.save_image)}",
        f"test.save_gt_image={_bool(args.save_gt_image)}",
        f"test.save_video={_bool(args.save_video)}",
        f"test.save_gaussian={_bool(args.save_gaussian)}",
        f"test.save_depth={_bool(args.save_depth)}",
        f"test.save_input_images={_bool(args.save_input_images)}",
        f"test.render_chunk_size={args.render_chunk_size}",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=4",
        "model.encoder.gaussians_per_cell=1",
        f"model.encoder.cell_scale={args.cell_scale}",
        "model.encoder.down_strides=[3,2]",
        "model.encoder.cube_encoder_type=small",
        f"model.encoder.cube_merge_type={args.cube_merge_type}",
    ]
    return overrides


def validate_inputs(args: argparse.Namespace, repo_root: Path) -> None:
    paths = {
        "checkpoint": _repo_path(repo_root, args.checkpoint),
        "index": _repo_path(repo_root, args.index_path),
        "re10k root": Path(args.re10k_root),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Missing evaluation input(s):\n  "
            f"{joined}\nSet CANONICALGS_RE10K_ROOT or pass --re10k-root for the dataset."
        )


def run_evaluation(args: argparse.Namespace, repo_root: Path) -> dict:
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    os.chdir(repo_root)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import hydra
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from colorama import Fore
    from hydra import compose, initialize_config_dir
    from jaxtyping import install_import_hook
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer

    with install_import_hook(("canonicalgs",), ("beartype", "beartype")):
        from canonicalgs.config import load_typed_root_config
        from canonicalgs.dataset.data_module import DataModule
        from canonicalgs.global_cfg import set_cfg
        from canonicalgs.loss import get_losses
        from canonicalgs.misc.step_tracker import StepTracker
        from canonicalgs.misc.wandb_tools import update_checkpoint_path
        from canonicalgs.model.decoder import get_decoder
        from canonicalgs.model.encoder import get_encoder
        from canonicalgs.model.model_wrapper import ModelWrapper

    def cyan(text: str) -> str:
        return f"{Fore.CYAN}{text}{Fore.RESET}"

    validate_inputs(args, repo_root)
    torch.manual_seed(42)
    np.random.seed(42)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    overrides = build_overrides(args, repo_root)
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
        cfg_dict = compose(config_name="main", overrides=overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg_dict))

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    print(cyan(f"Saving outputs to {cfg_dict.output_dir}."))
    print(cyan(f"Using checkpoint {cfg.checkpointing.pretrained_model}."))
    print(cyan(f"Using index {cfg.dataset.view_sampler.index_path}."))

    step_tracker = StepTracker()
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=False,
        devices=1,
        strategy="auto",
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_nodes=cfg.trainer.num_nodes,
        limit_test_batches=args.num_scenes,
        plugins=None,
        profiler=None,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(
        cfg.model.encoder,
        vggt_meta=cfg.train_controller.vggt_meta,
        knn_down=cfg.train_controller.knn_down,
        gaussian_merge=cfg.train_controller.gaussian_merge,
    )
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=None,
        train_controller_cfg=cfg.train_controller,
        iter_depth=args.iter_depth,
        view_base=args.view_base,
        batch_forward=args.batch_forward,
        anchor_features=args.anchor_features,
        chunk_num=args.chunk_num,
        anchor_base=args.anchor_base,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
        train_controller_cfg=cfg.train_controller,
    )

    strict_load = not cfg.checkpointing.no_strict_load
    if cfg.checkpointing.pretrained_model is not None:
        pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
        if "state_dict" in pretrained_model:
            pretrained_model = pretrained_model["state_dict"]
        model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
        print(cyan(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"))

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

    scores_path = Path(cfg_dict.output_dir) / "metrics" / "scores_all_avg.json"
    scores = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            scores = json.load(f)
        print(cyan(f"Saved metric summary: {scores_path}"))
        print(json.dumps(scores, indent=2))
    else:
        print(cyan(f"Metric summary was not found at {scores_path}."))
    return scores


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    return run_evaluation(args, repo_root)


if __name__ == "__main__":
    main()
