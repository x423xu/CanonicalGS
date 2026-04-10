from pathlib import Path

from canonicalgs.data.re10k import (
    _apply_fixed_scene_subset,
    _partition_index_items,
    derive_asset_context_sets,
)


def test_train_and_val_partitions_are_disjoint() -> None:
    index = {f"scene-{idx:03d}": f"chunk-{idx % 3}.pt" for idx in range(120)}

    train_entries = _partition_index_items(index, "train", validation_holdout_stride=10)
    val_entries = _partition_index_items(index, "val", validation_holdout_stride=10)

    train_keys = {scene_key for scene_key, _ in train_entries}
    val_keys = {scene_key for scene_key, _ in val_entries}

    assert train_keys
    assert val_keys
    assert train_keys.isdisjoint(val_keys)


def test_eval_partition_uses_its_own_holdout_rule() -> None:
    index = {f"scene-{idx:03d}": f"chunk-{idx % 2}.pt" for idx in range(60)}

    val_entries = _partition_index_items(index, "val", validation_holdout_stride=10)
    eval_entries = _partition_index_items(index, "eval", evaluation_holdout_stride=8)

    assert len(eval_entries) > 0
    assert len(eval_entries) < len(index)
    assert eval_entries != val_entries


def test_train_excludes_both_val_and_eval_holdouts() -> None:
    index = {f"scene-{idx:03d}": f"chunk-{idx % 2}.pt" for idx in range(200)}

    train_entries = _partition_index_items(
        index,
        "train",
        validation_holdout_stride=10,
        evaluation_holdout_stride=25,
    )
    val_entries = _partition_index_items(index, "val", validation_holdout_stride=10)
    eval_entries = _partition_index_items(index, "eval", evaluation_holdout_stride=25)

    train_keys = {scene_key for scene_key, _ in train_entries}
    val_keys = {scene_key for scene_key, _ in val_entries}
    eval_keys = {scene_key for scene_key, _ in eval_entries}

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(eval_keys)


def test_test_split_keeps_all_entries() -> None:
    index = {f"scene-{idx:03d}": f"chunk-{idx % 4}.pt" for idx in range(25)}

    test_entries = _partition_index_items(index, "test", validation_holdout_stride=12)

    assert test_entries == sorted(index.items())


def test_fixed_scene_subset_is_deterministic_for_same_seed() -> None:
    root = Path("/tmp/re10k")
    entries = [(root, "train", f"scene-{idx:03d}", f"chunk-{idx % 3}.pt") for idx in range(50)]

    first = _apply_fixed_scene_subset(entries, fixed_scene_count=10, fixed_scene_seed=11)
    second = _apply_fixed_scene_subset(entries, fixed_scene_count=10, fixed_scene_seed=11)
    third = _apply_fixed_scene_subset(entries, fixed_scene_count=10, fixed_scene_seed=12)

    assert [scene_key for _, _, scene_key, _ in first] == [
        scene_key for _, _, scene_key, _ in second
    ]
    assert [scene_key for _, _, scene_key, _ in first] != [
        scene_key for _, _, scene_key, _ in third
    ]


def test_fixed_scene_subset_rejects_non_positive_counts() -> None:
    root = Path("/tmp/re10k")
    entries = [(root, "train", "scene-000", "chunk-0.pt")]

    try:
        _apply_fixed_scene_subset(entries, fixed_scene_count=0, fixed_scene_seed=7)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected fixed scene subset validation failure")


def test_asset_context_sets_drop_middle_recursively_for_four_views() -> None:
    contexts = derive_asset_context_sets([0, 49, 74, 99], [2, 3, 4])

    assert contexts[4] == [0, 49, 74, 99]
    assert contexts[3] == [0, 49, 99]
    assert contexts[2] == [0, 99]


def test_asset_context_sets_work_for_max_contexts_four_to_eight() -> None:
    for max_context in (4, 5, 6, 7, 8):
        max_indices = list(range(max_context))
        requested = [2, max_context]
        if max_context >= 4:
            requested.append(max_context - 1)
        contexts = derive_asset_context_sets(max_indices, requested)

        assert contexts[max_context] == max_indices
        assert contexts[2] == [0, max_context - 1]
        for size, indices in contexts.items():
            assert len(indices) == size
            assert indices == sorted(indices)
