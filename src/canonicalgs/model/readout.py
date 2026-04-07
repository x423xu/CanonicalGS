from __future__ import annotations

from dataclasses import dataclass

import torch

from .evidence import CanonicalState


@dataclass(slots=True)
class PosteriorReadout:
    indices: torch.Tensor
    support_probability: torch.Tensor
    free_probability: torch.Tensor
    unknown_probability: torch.Tensor
    confidence: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_variance: torch.Tensor
    appearance_mean: torch.Tensor
    uncertainty: torch.Tensor


class CanonicalReadout:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, state: CanonicalState) -> PosteriorReadout:
        confidence = state.confidence.clamp_min(self.eps)
        surface_evidence = state.surface_evidence.clamp_min(0.0)
        free_evidence = state.free_evidence.clamp_min(0.0)
        total_evidence = 1.0 + surface_evidence + free_evidence
        support_probability = surface_evidence / total_evidence
        free_probability = free_evidence / total_evidence
        unknown_probability = 1.0 / total_evidence

        geometry_mean = state.geo_moment_1 / confidence.unsqueeze(-1)
        second_moment = state.geo_moment_2 / confidence.unsqueeze(-1)
        geometry_variance = torch.clamp(second_moment - geometry_mean.square(), min=0.0)

        appearance_weight = state.app_weight.clamp_min(self.eps)
        appearance_mean = state.app_moment_1 / appearance_weight.unsqueeze(-1)

        uncertainty = unknown_probability + geometry_variance.mean(dim=-1)

        return PosteriorReadout(
            indices=state.indices,
            support_probability=support_probability,
            free_probability=free_probability,
            unknown_probability=unknown_probability,
            confidence=confidence,
            geometry_mean=geometry_mean,
            geometry_variance=geometry_variance,
            appearance_mean=appearance_mean,
            uncertainty=uncertainty,
        )
