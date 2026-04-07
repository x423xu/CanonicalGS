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


def compute_render_stats(
    episode: dict,
    output: dict,
    max_targets: int | None = None,
) -> RenderStats:
    target_indices = output["target_indices"]
    if max_targets is not None:
        target_indices = target_indices[:max_targets]

    if target_indices.numel() == 0:
        zero = output["readout"].support_probability.new_zeros(())
        one = output["readout"].support_probability.new_ones(())
        return RenderStats(mse=one, psnr=zero, coverage=zero, num_targets=0)

    target_images = episode["images"][target_indices]
    target_extrinsics = episode["extrinsics"][target_indices]
    target_intrinsics = episode["intrinsics"][target_indices]
    rendered, coverage = render_gaussian_views(
        gaussians=output["render_gaussians"],
        extrinsics=target_extrinsics,
        intrinsics=target_intrinsics,
        image_shape=tuple(target_images.shape[-2:]),
    )
    mse = F.mse_loss(rendered, target_images)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
    return RenderStats(
        mse=mse,
        psnr=psnr,
        coverage=coverage.mean(),
        num_targets=int(target_indices.numel()),
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
