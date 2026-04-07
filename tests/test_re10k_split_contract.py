from canonicalgs.data.re10k import _partition_index_items


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
