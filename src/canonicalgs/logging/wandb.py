from __future__ import annotations

from pathlib import Path

from canonicalgs.config import WandbConfig


def build_wandb_kwargs(cfg: WandbConfig, output_dir: str | Path) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "project": cfg.project,
        "entity": cfg.entity,
        "mode": cfg.mode,
        "dir": str(output_dir),
        "tags": cfg.tags,
    }
    if cfg.id is not None:
        kwargs["id"] = cfg.id
        kwargs["resume"] = "allow"
    return kwargs


def metric_name(group: str, name: str, context_size: int | None = None) -> str:
    if context_size is None:
        return f"{group}/{name}"
    return f"{group}/ctx_{context_size}/{name}"
