from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SparseEvidence:
    indices: torch.Tensor
    surface_evidence: torch.Tensor
    free_evidence: torch.Tensor
    canonical_weight: torch.Tensor
    positional_certainty: torch.Tensor
    appearance_certainty: torch.Tensor
    world_points: torch.Tensor
    semantic_features: torch.Tensor

    def validate(self) -> None:
        count = self.indices.shape[0]
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError("indices must have shape [N, 3]")
        for name in (
            "surface_evidence",
            "free_evidence",
            "canonical_weight",
            "positional_certainty",
            "appearance_certainty",
        ):
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
        if self.world_points.shape != (count, 3):
            raise ValueError("world_points must have shape [N, 3]")
        if self.semantic_features.ndim != 2 or self.semantic_features.shape[0] != count:
            raise ValueError("semantic_features must have shape [N, C]")


@dataclass(slots=True)
class CanonicalState:
    indices: torch.Tensor
    surface_evidence: torch.Tensor
    free_evidence: torch.Tensor
    canonical_certainty: torch.Tensor
    positional_certainty: torch.Tensor
    appearance_certainty: torch.Tensor
    semantic_consistency: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_variance: torch.Tensor
    canonical_features: torch.Tensor
    feature_weight: torch.Tensor

    @property
    def confidence(self) -> torch.Tensor:
        return self.canonical_certainty


class SparseEvidenceAccumulator:
    def __init__(
        self,
        max_active_voxels: int | None = None,
        eps: float = 1e-6,
        representative_temperature: float = 16.0,
    ) -> None:
        self.max_active_voxels = max_active_voxels
        self.eps = eps
        self.representative_temperature = representative_temperature

    def accumulate(self, evidence_list: list[SparseEvidence]) -> CanonicalState:
        if not evidence_list:
            raise ValueError("evidence_list must not be empty")
        for evidence in evidence_list:
            evidence.validate()

        indices = torch.cat([e.indices for e in evidence_list], dim=0)
        surface_evidence = torch.cat([e.surface_evidence for e in evidence_list], dim=0)
        free_evidence = torch.cat([e.free_evidence for e in evidence_list], dim=0)
        canonical_weight = torch.cat([e.canonical_weight for e in evidence_list], dim=0)
        positional_certainty = torch.cat([e.positional_certainty for e in evidence_list], dim=0)
        appearance_certainty = torch.cat([e.appearance_certainty for e in evidence_list], dim=0)
        world_points = torch.cat([e.world_points for e in evidence_list], dim=0)
        semantic_features = torch.cat([e.semantic_features for e in evidence_list], dim=0)

        unique_indices, inverse = torch.unique(indices, dim=0, sorted=True, return_inverse=True)
        cell_count = unique_indices.shape[0]

        normalized_features = F.normalize(semantic_features, dim=-1)
        representative_weight = self._group_softmax(
            canonical_weight,
            inverse,
            cell_count,
            temperature=self.representative_temperature,
        )
        representative_features = self._index_add_nd(
            normalized_features * representative_weight.unsqueeze(-1),
            inverse,
            cell_count,
        )
        representative_features = F.normalize(representative_features, dim=-1)
        semantic_alignment = (
            normalized_features * representative_features[inverse]
        ).sum(dim=-1).clamp_min(0.0)
        sample_weight = canonical_weight * semantic_alignment

        feature_weight = self._index_add_1d(sample_weight, inverse, cell_count)
        geometry_sum = self._index_add_nd(world_points * sample_weight.unsqueeze(-1), inverse, cell_count)
        geometry_sq_sum = self._index_add_nd(
            world_points.square() * sample_weight.unsqueeze(-1),
            inverse,
            cell_count,
        )
        feature_sum = self._index_add_nd(
            semantic_features * sample_weight.unsqueeze(-1),
            inverse,
            cell_count,
        )

        safe_weight = feature_weight.clamp_min(self.eps).unsqueeze(-1)
        geometry_mean = geometry_sum / safe_weight
        geometry_second = geometry_sq_sum / safe_weight
        geometry_variance = torch.clamp(geometry_second - geometry_mean.square(), min=0.0)
        canonical_features = feature_sum / safe_weight

        base_weight = self._index_add_1d(canonical_weight, inverse, cell_count).clamp_min(self.eps)
        state = CanonicalState(
            indices=unique_indices,
            surface_evidence=self._index_add_1d(surface_evidence, inverse, cell_count),
            free_evidence=self._index_add_1d(free_evidence, inverse, cell_count),
            canonical_certainty=self._index_add_1d(sample_weight, inverse, cell_count),
            positional_certainty=self._index_add_1d(
                positional_certainty * canonical_weight,
                inverse,
                cell_count,
            )
            / base_weight,
            appearance_certainty=self._index_add_1d(
                appearance_certainty * canonical_weight,
                inverse,
                cell_count,
            )
            / base_weight,
            semantic_consistency=self._index_add_1d(
                semantic_alignment * canonical_weight,
                inverse,
                cell_count,
            )
            / base_weight,
            geometry_mean=geometry_mean,
            geometry_variance=geometry_variance,
            canonical_features=canonical_features,
            feature_weight=feature_weight,
        )
        return self._prune_state(state)

    def _index_add_1d(
        self, values: torch.Tensor, inverse: torch.Tensor, cell_count: int
    ) -> torch.Tensor:
        output = torch.zeros(cell_count, dtype=values.dtype, device=values.device)
        output.index_add_(0, inverse, values)
        return output

    def _index_add_nd(
        self, values: torch.Tensor, inverse: torch.Tensor, cell_count: int
    ) -> torch.Tensor:
        output = torch.zeros(
            (cell_count, values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        output.index_add_(0, inverse, values)
        return output

    def _group_softmax(
        self,
        logits: torch.Tensor,
        inverse: torch.Tensor,
        cell_count: int,
        temperature: float,
    ) -> torch.Tensor:
        scaled = logits * temperature
        max_values = torch.full(
            (cell_count,),
            torch.finfo(logits.dtype).min,
            dtype=logits.dtype,
            device=logits.device,
        )
        max_values.index_reduce_(0, inverse, scaled, reduce="amax", include_self=True)
        stabilized = torch.exp(scaled - max_values[inverse])
        denom = self._index_add_1d(stabilized, inverse, cell_count).clamp_min(self.eps)
        return stabilized / denom[inverse]

    def _prune_state(self, state: CanonicalState) -> CanonicalState:
        return state
