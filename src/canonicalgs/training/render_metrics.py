from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from canonicalgs.model import render_gaussian_views


@dataclass(slots=True)
class RenderMetricBundle:
    mse: float
    psnr: float
    coverage: float
    num_targets: int


@dataclass(slots=True)
class RenderStats:
    mse: torch.Tensor
    psnr: torch.Tensor
    coverage: torch.Tensor
    num_targets: int


def render_target_views(
    episode: dict,
    output: dict,
    max_targets: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    target_indices = output["target_indices"]
    if max_targets is not None:
        target_indices = target_indices[:max_targets]

    if target_indices.numel() == 0:
        zero = output["readout"].canonical_certainty.new_zeros(())
        image_shape = tuple(episode["images"].shape[-2:])
        empty = output["readout"].canonical_certainty.new_zeros((0, 3, *image_shape))
        return empty, empty, zero, 0

    target_images = episode["images"][target_indices]
    target_extrinsics = episode["extrinsics"][target_indices]
    target_intrinsics = episode["intrinsics"][target_indices]
    device_type = target_images.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        rendered, coverage = render_gaussian_views(
            gaussians=output["render_gaussians"],
            extrinsics=target_extrinsics.float(),
            intrinsics=target_intrinsics.float(),
            image_shape=tuple(target_images.shape[-2:]),
        )
    return rendered.float(), target_images.float(), coverage.float().mean(), int(target_indices.numel())


def compute_render_stats(
    episode: dict,
    output: dict,
    max_targets: int | None = None,
) -> RenderStats:
    rendered, target_images, coverage, num_targets = render_target_views(
        episode,
        output,
        max_targets=max_targets,
    )
    if num_targets == 0:
        zero = output["readout"].canonical_certainty.new_zeros(())
        one = output["readout"].canonical_certainty.new_ones(())
        return RenderStats(mse=one, psnr=zero, coverage=zero, num_targets=0)

    mse = F.mse_loss(rendered, target_images)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
    return RenderStats(
        mse=mse,
        psnr=psnr,
        coverage=coverage,
        num_targets=num_targets,
    )


def compute_render_metrics(
    episode: dict,
    output: dict,
    max_targets: int | None = None,
) -> RenderMetricBundle:
    stats = compute_render_stats(episode, output, max_targets=max_targets)
    return RenderMetricBundle(
        mse=float(stats.mse.item()),
        psnr=float(stats.psnr.item()),
        coverage=float(stats.coverage.item()),
        num_targets=stats.num_targets,
    )
