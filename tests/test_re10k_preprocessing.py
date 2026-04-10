from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from canonicalgs.data.re10k import (
    _convert_re10k_image_and_intrinsics,
    _update_normalized_intrinsics_for_crop,
)


def test_update_normalized_intrinsics_for_crop_updates_focal_and_principal_point() -> None:
    intrinsics = torch.tensor(
        [
            [0.8, 0.0, 0.4],
            [0.0, 0.6, 0.3],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    updated = _update_normalized_intrinsics_for_crop(
        intrinsics=intrinsics,
        input_shape=(300, 400),
        crop_top=30,
        crop_left=40,
        output_shape=(240, 320),
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.375],
            [0.0, 0.75, 0.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(updated, expected, atol=1e-6, rtol=0.0)


def test_convert_re10k_image_and_intrinsics_applies_resize_crop_and_patch_crop() -> None:
    array = np.linspace(0, 255, num=320 * 480 * 3, dtype=np.uint8).reshape(320, 480, 3)
    image = Image.fromarray(array, mode="RGB")
    intrinsics = torch.tensor(
        [
            [0.9, 0.0, 0.4],
            [0.0, 0.8, 0.6],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    image_tensor, updated_intrinsics = _convert_re10k_image_and_intrinsics(
        image=image,
        intrinsics=intrinsics,
        target_shape=(256, 256),
        patch_size=14,
    )

    assert tuple(image_tensor.shape) == (3, 252, 252)
    expected = torch.tensor(
        [
            [1.3714286, 0.0, 0.3476190],
            [0.0, 0.8126984, 0.6015873],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(updated_intrinsics, expected, atol=1e-5, rtol=0.0)
