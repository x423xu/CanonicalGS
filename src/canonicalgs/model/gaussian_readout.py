from __future__ import annotations

from dataclasses import dataclass

import torch

from canonicalgs.config import ModelConfig

from .readout import PosteriorReadout


@dataclass(slots=True)
class GaussianSet:
    indices: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    opacities: torch.Tensor
    appearance: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor


class LocalGaussianReadout:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def __call__(self, readout: PosteriorReadout, prune: bool = True) -> GaussianSet:
        if prune:
            support_mask = readout.support_probability >= self.cfg.support_threshold
            confidence_mask = readout.confidence >= self.cfg.confidence_threshold
            mask = support_mask & confidence_mask
        else:
            mask = torch.ones_like(readout.support_probability, dtype=torch.bool)

        indices = readout.indices[mask]
        means = readout.geometry_mean[mask]
        variances = readout.geometry_variance[mask].clamp_min(
            self.cfg.gaussian_scale_min**2
        )
        variances = variances.clamp_max(self.cfg.gaussian_scale_max**2)
        covariances = torch.diag_embed(variances)

        support = readout.support_probability[mask]
        confidence = readout.confidence[mask]
        opacities = support * (1.0 - torch.exp(-confidence * self.cfg.opacity_gain))
        appearance = readout.appearance_mean[mask].clamp(0.0, 1.0)

        return GaussianSet(
            indices=indices,
            means=means,
            covariances=covariances,
            opacities=opacities,
            appearance=appearance,
            support=support,
            confidence=confidence,
        )
