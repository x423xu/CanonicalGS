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
    canonical_certainty: torch.Tensor
    positional_certainty: torch.Tensor
    appearance_certainty: torch.Tensor
    semantic_consistency: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_variance: torch.Tensor
    canonical_features: torch.Tensor
    uncertainty: torch.Tensor

    @property
    def confidence(self) -> torch.Tensor:
        return self.canonical_certainty

    @property
    def appearance_mean(self) -> torch.Tensor:
        return self.canonical_features


class CanonicalReadout:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, state: CanonicalState) -> PosteriorReadout:
        surface_evidence = state.surface_evidence.clamp_min(0.0)
        free_evidence = state.free_evidence.clamp_min(0.0)
        total_evidence = 1.0 + surface_evidence + free_evidence
        support_probability = surface_evidence / total_evidence
        free_probability = free_evidence / total_evidence
        unknown_probability = 1.0 / total_evidence

        canonical_certainty = state.canonical_certainty.clamp_min(0.0)
        geometry_uncertainty = state.geometry_variance.mean(dim=-1)
        certainty_uncertainty = 1.0 / (1.0 + canonical_certainty)
        appearance_uncertainty = 1.0 - state.semantic_consistency.clamp(0.0, 1.0)
        uncertainty = (
            unknown_probability
            + geometry_uncertainty
            + certainty_uncertainty
            + appearance_uncertainty
        )

        return PosteriorReadout(
            indices=state.indices,
            support_probability=support_probability,
            free_probability=free_probability,
            unknown_probability=unknown_probability,
            canonical_certainty=canonical_certainty,
            positional_certainty=state.positional_certainty.clamp_min(0.0),
            appearance_certainty=state.appearance_certainty.clamp_min(0.0),
            semantic_consistency=state.semantic_consistency.clamp(0.0, 1.0),
            geometry_mean=state.geometry_mean,
            geometry_variance=state.geometry_variance.clamp_min(self.eps),
            canonical_features=state.canonical_features,
            uncertainty=uncertainty,
        )
