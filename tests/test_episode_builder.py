from canonicalgs.data.episode_builder import EpisodeBuilder
from canonicalgs.data.types import CameraFrame, Episode


def make_frames(count: int, scene_id: str = "scene", clip_id: str = "clip") -> list[CameraFrame]:
    frames = []
    for index in range(count):
        frames.append(
            CameraFrame(
                scene_id=scene_id,
                clip_id=clip_id,
                frame_id=f"frame-{index:03d}",
                image_path=f"/tmp/frame-{index:03d}.png",
                intrinsics=[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                pose=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                timestamp=float(index),
            )
        )
    return frames


def test_build_episode_creates_nested_contexts_and_fixed_targets() -> None:
    builder = EpisodeBuilder()
    episode = builder.build_episode(make_frames(20))

    assert sorted(episode.context_sets) == [2, 3, 4, 5, 6]
    assert len(episode.target_set) == 6

    target_ids = {frame.frame_id for frame in episode.target_set}
    previous_ids: set[str] = set()
    for size in [2, 3, 4, 5, 6]:
        current_ids = {frame.frame_id for frame in episode.context_sets[size]}
        assert len(current_ids) == size
        assert not (current_ids & target_ids)
        assert previous_ids.issubset(current_ids)
        previous_ids = current_ids


def test_builder_is_deterministic_for_same_seed() -> None:
    frames = make_frames(40, clip_id="clip-a")
    first = EpisodeBuilder(seed=7).build_episode(frames)
    second = EpisodeBuilder(seed=7).build_episode(frames)

    assert [frame.frame_id for frame in first.frame_pool] == [
        frame.frame_id for frame in second.frame_pool
    ]
    assert [frame.frame_id for frame in first.target_set] == [
        frame.frame_id for frame in second.target_set
    ]


def test_builder_rejects_short_frame_pools() -> None:
    builder = EpisodeBuilder()
    try:
        builder.build_episode(make_frames(11))
    except ValueError as exc:
        assert "at least 12 frames" in str(exc)
    else:
        raise AssertionError("expected ValueError for short frame pool")


def test_build_many_skips_incomplete_clips_by_default() -> None:
    builder = EpisodeBuilder()
    episodes = builder.build_many([make_frames(20), make_frames(11)])
    assert len(episodes) == 1


def test_episode_validation_rejects_duplicate_targets() -> None:
    frames = make_frames(12)
    episode = Episode(
        scene_id="scene",
        clip_id="clip",
        frame_pool=tuple(frames),
        context_sets={
            2: tuple(frames[:2]),
            3: tuple(frames[:3]),
            4: tuple(frames[:4]),
            5: tuple(frames[:5]),
            6: tuple(frames[:6]),
        },
        target_set=(frames[6], frames[7], frames[8], frames[9], frames[10], frames[10]),
    )

    try:
        episode.validate()
    except ValueError as exc:
        assert "target_set contains duplicate frames" in str(exc)
    else:
        raise AssertionError("expected duplicate target validation failure")
