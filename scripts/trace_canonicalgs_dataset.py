from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonicalgs.config import DatasetConfig
from canonicalgs.data import EpisodeBuilder, build_re10k_sample_episodes


def _summarize_split(dataset_cfg: DatasetConfig, split: str, limit: int) -> dict[str, object]:
    builder = EpisodeBuilder(
        context_sizes=tuple(dataset_cfg.context_sizes),
        target_views=dataset_cfg.target_views,
        min_frames_per_episode=dataset_cfg.min_frames_per_episode,
        subsample_to=dataset_cfg.subsample_to,
        seed=dataset_cfg.seed,
    )
    episodes = build_re10k_sample_episodes(
        dataset_cfg.roots,
        split,
        builder,
        limit,
        dataset_cfg.validation_holdout_stride,
        dataset_cfg.validation_holdout_offset,
    )

    overlap_count = 0
    duplicate_target_count = 0
    invalid_target_count = 0
    nested_failures = 0
    sample_episodes: list[dict[str, object]] = []
    for episode in episodes:
        target_ids = [frame.frame_id for frame in episode.target_set]
        target_id_set = set(target_ids)
        if len(target_id_set) != len(target_ids):
            duplicate_target_count += 1
        if len(target_ids) != dataset_cfg.target_views:
            invalid_target_count += 1

        previous_ids: set[str] = set()
        nested_ok = True
        max_context_ids: set[str] = set()
        for size in sorted(episode.context_sets):
            context_ids = {frame.frame_id for frame in episode.context_sets[size]}
            if context_ids & target_id_set:
                overlap_count += 1
                break
            if previous_ids and not previous_ids.issubset(context_ids):
                nested_ok = False
            previous_ids = context_ids
            max_context_ids = context_ids
        if not nested_ok:
            nested_failures += 1

        if len(sample_episodes) < 3:
            sample_episodes.append(
                {
                    "scene_id": episode.scene_id,
                    "clip_id": episode.clip_id,
                    "frame_pool_size": len(episode.frame_pool),
                    "context_sizes": {str(size): len(frames) for size, frames in episode.context_sets.items()},
                    "target_count": len(episode.target_set),
                    "target_context_overlap": len(max_context_ids & target_id_set),
                    "target_ids": target_ids,
                }
            )

    return {
        "split": split,
        "episodes_traced": len(episodes),
        "overlap_count": overlap_count,
        "duplicate_target_count": duplicate_target_count,
        "invalid_target_count": invalid_target_count,
        "nested_failures": nested_failures,
        "sample_episodes": sample_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--validation-holdout-stride", type=int, default=20)
    parser.add_argument("--validation-holdout-offset", type=int, default=0)
    parser.add_argument("--evaluation-holdout-stride", type=int, default=100)
    parser.add_argument("--evaluation-holdout-offset", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    args = parser.parse_args()

    dataset_cfg = DatasetConfig(
        roots=[args.root],
        image_shape=[args.image_height, args.image_width],
        validation_holdout_stride=args.validation_holdout_stride,
        validation_holdout_offset=args.validation_holdout_offset,
        evaluation_holdout_stride=args.evaluation_holdout_stride,
        evaluation_holdout_offset=args.evaluation_holdout_offset,
    )

    payload = {
        "dataset": {
            "roots": dataset_cfg.roots,
            "context_sizes": dataset_cfg.context_sizes,
            "target_views": dataset_cfg.target_views,
            "validation_holdout_stride": dataset_cfg.validation_holdout_stride,
            "validation_holdout_offset": dataset_cfg.validation_holdout_offset,
            "evaluation_holdout_stride": dataset_cfg.evaluation_holdout_stride,
            "evaluation_holdout_offset": dataset_cfg.evaluation_holdout_offset,
        },
        "splits": {
            split: _summarize_split(dataset_cfg, split, args.limit)
            for split in ("train", "val", "eval", "test")
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
