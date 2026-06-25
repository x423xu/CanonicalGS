#!/usr/bin/env python3
"""Rename legacy DepthSplat/GS-Cube checkpoint keys to CanonicalGS names."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, MutableMapping

KEY_RENAMES = (
    ("encoder.gs_cube_encoder.", "encoder.scene_field_encoder."),
    (".gs_cube_head.", ".gp_decoder_head."),
)


def rename_state_dict_key(key: str) -> str:
    for old, new in KEY_RENAMES:
        key = key.replace(old, new)
    return key


def rename_state_dict_keys(state_dict: Mapping[str, object]) -> OrderedDict[str, object]:
    renamed: OrderedDict[str, object] = OrderedDict()
    for key, value in state_dict.items():
        new_key = rename_state_dict_key(key)
        if new_key in renamed:
            raise ValueError(f"Checkpoint key collision while renaming: {key} -> {new_key}")
        renamed[new_key] = value
    return renamed


def rename_checkpoint(source: Path, destination: Path) -> None:
    import torch

    checkpoint = torch.load(source, map_location="cpu")
    if isinstance(checkpoint, MutableMapping) and "state_dict" in checkpoint:
        checkpoint["state_dict"] = rename_state_dict_keys(checkpoint["state_dict"])
    elif isinstance(checkpoint, Mapping):
        checkpoint = rename_state_dict_keys(checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rename_checkpoint(args.source, args.destination)


if __name__ == "__main__":
    main()
