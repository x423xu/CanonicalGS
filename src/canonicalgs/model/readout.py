from __future__ import annotations

from dataclasses import dataclass

import torch

from .evidence import CanonicalState


@dataclass(slots=True)
class PosteriorReadout:
    indices: torch.Tensor
    canonical_certainty: torch.Tensor
    positional_certainty: torch.Tensor
    appearance_certainty: torch.Tensor
    semantic_consistency: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_variance: torch.Tensor
    canonical_features: torch.Tensor
    canonical_color: torch.Tensor
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
        canonical_certainty = state.canonical_certainty.clamp_min(0.0)
        positional_certainty = state.positional_certainty.clamp(0.0, 1.0)
        appearance_certainty = state.appearance_certainty.clamp(0.0, 1.0)
        semantic_consistency = state.semantic_consistency.clamp(-1.0, 1.0)
        uncertainty = (
            state.geometry_variance.mean(dim=-1)
            + (1.0 - positional_certainty)
            + (1.0 - appearance_certainty)
        )

        return PosteriorReadout(
            indices=state.indices,
            canonical_certainty=canonical_certainty,
            positional_certainty=positional_certainty,
            appearance_certainty=appearance_certainty,
            semantic_consistency=semantic_consistency,
            geometry_mean=state.geometry_mean,
            geometry_variance=state.geometry_variance.clamp_min(self.eps),
            canonical_features=state.canonical_features,
            canonical_color=state.canonical_color.clamp(0.0, 1.0),
            uncertainty=uncertainty,
        )
