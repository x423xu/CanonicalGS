from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

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


class LocalGaussianReadout(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = max(cfg.decoder_hidden_dim, cfg.appearance_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.appearance_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)
        self.scale_head = nn.Linear(hidden_dim, 3)
        self.color_head = nn.Linear(hidden_dim, 3)

    def forward(self, readout: PosteriorReadout, prune: bool = True) -> GaussianSet:
        indices = readout.indices
        if indices.numel() == 0:
            return GaussianSet(
                indices=indices,
                means=readout.geometry_mean.new_zeros((0, 3)),
                covariances=readout.geometry_mean.new_zeros((0, 3, 3)),
                opacities=readout.canonical_certainty.new_zeros((0,)),
                appearance=readout.geometry_mean.new_zeros((0, 3)),
                support=readout.support_probability.new_zeros((0,)),
                confidence=readout.canonical_certainty.new_zeros((0,)),
            )

        canonical_features = readout.canonical_features
        hidden = self.decoder(canonical_features)

        means = readout.geometry_mean
        means = means + 0.5 * self.cfg.voxel_size * torch.tanh(self.offset_head(hidden))

        base_scales = readout.geometry_variance.sqrt().clamp_min(self.cfg.gaussian_scale_min)
        decoded_scales = self.cfg.gaussian_scale_min + (
            self.cfg.gaussian_scale_max - self.cfg.gaussian_scale_min
        ) * torch.sigmoid(self.scale_head(hidden))
        scales = torch.maximum(base_scales, decoded_scales).clamp_max(self.cfg.gaussian_scale_max)
        covariances = torch.diag_embed(scales.square())

        support = readout.support_probability
        confidence = readout.canonical_certainty
        if prune:
            gate = self._soft_gate(
                support,
                confidence,
                self.cfg.support_threshold,
                self.cfg.confidence_threshold,
            )
        else:
            gate = torch.ones_like(support)
        opacities = (1.0 - torch.exp(-confidence * self.cfg.opacity_gain)) * gate
        appearance = torch.sigmoid(self.color_head(hidden))

        return GaussianSet(
            indices=indices,
            means=means,
            covariances=covariances,
            opacities=opacities,
            appearance=appearance,
            support=support,
            confidence=confidence,
        )

    def _soft_gate(
        self,
        support: torch.Tensor,
        confidence: torch.Tensor,
        support_threshold: float,
        confidence_threshold: float,
    ) -> torch.Tensor:
        support_gate = torch.sigmoid((support - support_threshold) * self.cfg.gate_temperature)
        confidence_gate = torch.sigmoid(
            (confidence - confidence_threshold) * self.cfg.gate_temperature
        )
        return support_gate * confidence_gate
