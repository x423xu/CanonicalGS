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
    colors: torch.Tensor

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
        if self.colors.shape != (count, 3):
            raise ValueError("colors must have shape [N, 3]")


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
    canonical_color: torch.Tensor
    feature_weight: torch.Tensor

    @property
    def confidence(self) -> torch.Tensor:
        return self.canonical_certainty


class SparseEvidenceAccumulator:
    def __init__(
        self,
        max_active_voxels: int | None = None,
        merge_views: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self.max_active_voxels = max_active_voxels
        self.merge_views = merge_views
        self.eps = eps

    def accumulate(self, evidence_list: list[SparseEvidence]) -> CanonicalState:
        return self.accumulate_prefixes(evidence_list, [len(evidence_list)])[len(evidence_list)]

    def accumulate_prefixes(
        self,
        evidence_list: list[SparseEvidence],
        prefix_sizes: list[int],
    ) -> dict[int, CanonicalState]:
        if not evidence_list:
            raise ValueError("evidence_list must not be empty")
        for evidence in evidence_list:
            evidence.validate()
        requested_prefixes = sorted({int(size) for size in prefix_sizes})
        if not requested_prefixes:
            raise ValueError("prefix_sizes must not be empty")
        if requested_prefixes[0] < 1 or requested_prefixes[-1] > len(evidence_list):
            raise ValueError("prefix_sizes must lie within the available evidence length")

        outputs: dict[int, CanonicalState] = {}
        running = evidence_list[0]
        if 1 in requested_prefixes:
            outputs[1] = self._materialize_state(running)

        for prefix_size in range(2, requested_prefixes[-1] + 1):
            running = self._concat_evidence(running, evidence_list[prefix_size - 1])
            if prefix_size in requested_prefixes:
                outputs[prefix_size] = self._materialize_state(running)
        return outputs

    def _concat_evidence(
        self,
        left: SparseEvidence,
        right: SparseEvidence,
    ) -> SparseEvidence:
        return SparseEvidence(
            indices=torch.cat([left.indices, right.indices], dim=0),
            surface_evidence=torch.cat([left.surface_evidence, right.surface_evidence], dim=0),
            free_evidence=torch.cat([left.free_evidence, right.free_evidence], dim=0),
            canonical_weight=torch.cat([left.canonical_weight, right.canonical_weight], dim=0),
            positional_certainty=torch.cat(
                [left.positional_certainty, right.positional_certainty],
                dim=0,
            ),
            appearance_certainty=torch.cat(
                [left.appearance_certainty, right.appearance_certainty],
                dim=0,
            ),
            world_points=torch.cat([left.world_points, right.world_points], dim=0),
            semantic_features=torch.cat([left.semantic_features, right.semantic_features], dim=0),
            colors=torch.cat([left.colors, right.colors], dim=0),
        )

    def _materialize_state(self, evidence: SparseEvidence) -> CanonicalState:
        if not self.merge_views:
            return self._materialize_unmerged_state(evidence)

        indices = evidence.indices
        surface_evidence = evidence.surface_evidence
        free_evidence = evidence.free_evidence
        # In the current pipeline, canonical_weight is the per-point certainty C(u).
        certainty = evidence.canonical_weight
        positional_certainty = evidence.positional_certainty
        appearance_certainty = evidence.appearance_certainty
        world_points = evidence.world_points
        semantic_features = evidence.semantic_features
        colors = evidence.colors

        unique_indices, inverse = torch.unique(indices, dim=0, sorted=True, return_inverse=True)
        cell_count = unique_indices.shape[0]

        normalized_features = F.normalize(semantic_features, dim=-1, eps=self.eps)
        selected_positions = self._select_representative_positions(
            certainty,
            inverse,
            cell_count,
        )
        representative_features = normalized_features[selected_positions]
        positions = torch.arange(certainty.shape[0], device=certainty.device, dtype=torch.long)
        representative_mask = positions == selected_positions[inverse]
        semantic_alignment = (normalized_features * representative_features[inverse]).sum(dim=-1)
        semantic_alignment = torch.where(
            representative_mask,
            torch.ones_like(semantic_alignment),
            semantic_alignment,
        )
        sample_weight = certainty * semantic_alignment

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
        color_sum = self._index_add_nd(
            colors * sample_weight.unsqueeze(-1),
            inverse,
            cell_count,
        )

        safe_weight = feature_weight.clamp_min(self.eps).unsqueeze(-1)
        geometry_mean = geometry_sum / safe_weight
        geometry_second = geometry_sq_sum / safe_weight
        geometry_variance = torch.clamp(geometry_second - geometry_mean.square(), min=0.0)
        canonical_features = feature_sum / safe_weight
        canonical_color = color_sum / safe_weight

        base_weight = self._index_add_1d(certainty, inverse, cell_count).clamp_min(self.eps)
        state = CanonicalState(
            indices=unique_indices,
            surface_evidence=self._index_add_1d(surface_evidence, inverse, cell_count),
            free_evidence=self._index_add_1d(free_evidence, inverse, cell_count),
            canonical_certainty=self._index_add_1d(sample_weight, inverse, cell_count),
            positional_certainty=self._index_add_1d(
                positional_certainty * certainty,
                inverse,
                cell_count,
            )
            / base_weight,
            appearance_certainty=self._index_add_1d(
                appearance_certainty * certainty,
                inverse,
                cell_count,
            )
            / base_weight,
            semantic_consistency=self._index_add_1d(
                semantic_alignment * certainty,
                inverse,
                cell_count,
            )
            / base_weight,
            geometry_mean=geometry_mean,
            geometry_variance=geometry_variance,
            canonical_features=canonical_features,
            canonical_color=canonical_color.clamp(0.0, 1.0),
            feature_weight=feature_weight,
        )
        return self._prune_state(state)

    def _materialize_unmerged_state(self, evidence: SparseEvidence) -> CanonicalState:
        certainty = evidence.canonical_weight.clamp_min(0.0)
        count = certainty.shape[0]
        feature_dim = evidence.semantic_features.shape[1]
        semantic_consistency = torch.ones_like(certainty)
        zero_variance = torch.zeros_like(evidence.world_points)
        feature_weight = certainty.clamp_min(self.eps)

        return CanonicalState(
            indices=evidence.indices,
            surface_evidence=evidence.surface_evidence,
            free_evidence=evidence.free_evidence,
            canonical_certainty=certainty,
            positional_certainty=evidence.positional_certainty.clamp(0.0, 1.0),
            appearance_certainty=evidence.appearance_certainty.clamp(0.0, 1.0),
            semantic_consistency=semantic_consistency,
            geometry_mean=evidence.world_points,
            geometry_variance=zero_variance,
            canonical_features=evidence.semantic_features.reshape(count, feature_dim),
            canonical_color=evidence.colors.clamp(0.0, 1.0),
            feature_weight=feature_weight,
        )

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

    def _select_representative_positions(
        self,
        weights: torch.Tensor,
        inverse: torch.Tensor,
        cell_count: int,
    ) -> torch.Tensor:
        safe_weights = torch.where(
            torch.isfinite(weights),
            weights,
            torch.full_like(weights, torch.finfo(weights.dtype).min),
        )
        max_values = torch.full(
            (cell_count,),
            torch.finfo(safe_weights.dtype).min,
            dtype=safe_weights.dtype,
            device=safe_weights.device,
        )
        max_values.index_reduce_(0, inverse, safe_weights, reduce="amax", include_self=True)
        candidate_mask = safe_weights == max_values[inverse]
        positions = torch.arange(safe_weights.shape[0], device=safe_weights.device, dtype=torch.long)
        large = torch.full_like(positions, positions.shape[0])
        candidate_positions = torch.where(candidate_mask, positions, large)
        selected_positions = torch.full(
            (cell_count,),
            positions.shape[0],
            dtype=torch.long,
            device=safe_weights.device,
        )
        selected_positions.index_reduce_(
            0,
            inverse,
            candidate_positions,
            reduce="amin",
            include_self=True,
        )
        selected_positions = torch.clamp(selected_positions, max=positions.shape[0] - 1)
        return selected_positions

    def _prune_state(self, state: CanonicalState) -> CanonicalState:
        return state
