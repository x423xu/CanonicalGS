from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SparseEvidence:
    indices: torch.Tensor
    surface_evidence: torch.Tensor
    free_evidence: torch.Tensor
    confidence: torch.Tensor
    geo_moment_1: torch.Tensor
    geo_moment_2: torch.Tensor
    app_moment_1: torch.Tensor
    app_weight: torch.Tensor

    def validate(self) -> None:
        count = self.indices.shape[0]
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError("indices must have shape [N, 3]")
        for name in ("surface_evidence", "free_evidence", "confidence", "app_weight"):
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
        if self.geo_moment_1.shape[0] != count or self.geo_moment_2.shape != self.geo_moment_1.shape:
            raise ValueError("geometry moments must have shape [N, G]")
        if self.app_moment_1.shape[0] != count:
            raise ValueError("appearance moments must have shape [N, A]")


@dataclass(slots=True)
class CanonicalState:
    indices: torch.Tensor
    surface_evidence: torch.Tensor
    free_evidence: torch.Tensor
    confidence: torch.Tensor
    geo_moment_1: torch.Tensor
    geo_moment_2: torch.Tensor
    app_moment_1: torch.Tensor
    app_weight: torch.Tensor


class SparseEvidenceAccumulator:
    def __init__(self, max_active_voxels: int | None = None) -> None:
        self.max_active_voxels = max_active_voxels

    def accumulate(self, evidence_list: list[SparseEvidence]) -> CanonicalState:
        if not evidence_list:
            raise ValueError("evidence_list must not be empty")
        for evidence in evidence_list:
            evidence.validate()

        indices = torch.cat([e.indices for e in evidence_list], dim=0)
        surface_evidence = torch.cat([e.surface_evidence for e in evidence_list], dim=0)
        free_evidence = torch.cat([e.free_evidence for e in evidence_list], dim=0)
        confidence = torch.cat([e.confidence for e in evidence_list], dim=0)
        geo_moment_1 = torch.cat([e.geo_moment_1 for e in evidence_list], dim=0)
        geo_moment_2 = torch.cat([e.geo_moment_2 for e in evidence_list], dim=0)
        app_moment_1 = torch.cat([e.app_moment_1 for e in evidence_list], dim=0)
        app_weight = torch.cat([e.app_weight for e in evidence_list], dim=0)

        unique_indices, inverse = torch.unique(indices, dim=0, sorted=True, return_inverse=True)
        cell_count = unique_indices.shape[0]

        state = CanonicalState(
            indices=unique_indices,
            surface_evidence=self._index_add_1d(surface_evidence, inverse, cell_count),
            free_evidence=self._index_add_1d(free_evidence, inverse, cell_count),
            confidence=self._index_add_1d(confidence, inverse, cell_count),
            geo_moment_1=self._index_add_nd(geo_moment_1, inverse, cell_count),
            geo_moment_2=self._index_add_nd(geo_moment_2, inverse, cell_count),
            app_moment_1=self._index_add_nd(app_moment_1, inverse, cell_count),
            app_weight=self._index_add_1d(app_weight, inverse, cell_count),
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

    def _prune_state(self, state: CanonicalState) -> CanonicalState:
        if self.max_active_voxels is None or state.indices.shape[0] <= self.max_active_voxels:
            return state

        keep = torch.topk(
            state.confidence,
            k=self.max_active_voxels,
            largest=True,
            sorted=False,
        ).indices
        keep = keep.sort().values
        return CanonicalState(
            indices=state.indices[keep],
            surface_evidence=state.surface_evidence[keep],
            free_evidence=state.free_evidence[keep],
            confidence=state.confidence[keep],
            geo_moment_1=state.geo_moment_1[keep],
            geo_moment_2=state.geo_moment_2[keep],
            app_moment_1=state.app_moment_1[keep],
            app_weight=state.app_weight[keep],
        )
