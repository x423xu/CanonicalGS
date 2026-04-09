from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from canonicalgs.config import ModelConfig

from .evidence import SparseEvidenceAccumulator
from .evidence_writer import VoxelEvidenceWriter
from .gaussian_readout import LocalGaussianReadout
from .readout import CanonicalReadout
from .view_encoder import SymmetricMultiViewEncoder


class CanonicalGsPipeline(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = SymmetricMultiViewEncoder(
            feature_dim=cfg.feature_dim,
            appearance_dim=cfg.appearance_dim,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            min_depth_uncertainty=cfg.min_depth_uncertainty,
            num_depth_bins=cfg.num_depth_bins,
            cost_volume_temperature=cfg.cost_volume_temperature,
            cost_volume_visibility_beta=cfg.cost_volume_visibility_beta,
            dpt_output_stride=cfg.dpt_output_stride,
            dinov2_model_name=cfg.dinov2_model_name,
            dinov2_pretrained=cfg.dinov2_pretrained,
            freeze_dinov2=cfg.freeze_dinov2,
            allow_dinov2_fallback=cfg.allow_dinov2_fallback,
            appearance_uncertainty_bias=cfg.appearance_uncertainty_bias,
        )
        self.writer = VoxelEvidenceWriter(cfg)
        self.accumulator = SparseEvidenceAccumulator(
            max_active_voxels=cfg.max_active_voxels,
            representative_temperature=cfg.representative_temperature,
        )
        self.readout = CanonicalReadout()
        self.gaussian_readout = LocalGaussianReadout(cfg)

    def forward(
        self,
        episode: dict,
        context_size: int,
        include_render_payload: bool = False,
    ) -> dict:
        outputs = self.forward_prefixes(
            episode,
            context_sizes=[context_size],
            include_render_payload=include_render_payload,
        )
        return outputs[context_size]

    def forward_prefixes(
        self,
        episode: dict,
        context_sizes: Iterable[int] | None = None,
        include_render_payload: bool = False,
    ) -> dict[int, dict]:
        available_sizes = sorted(int(key) for key in episode["context_indices"].keys())
        if context_sizes is None:
            context_sizes = available_sizes
        prefix_sizes = sorted(int(size) for size in context_sizes)
        if not prefix_sizes:
            raise ValueError("context_sizes must not be empty")

        max_context = prefix_sizes[-1]
        max_indices = episode["context_indices"][max_context]
        self._validate_nested_prefixes(episode["context_indices"], max_indices, prefix_sizes)

        images = episode["images"][max_indices]
        extrinsics = episode["extrinsics"][max_indices]
        intrinsics = episode["intrinsics"][max_indices]
        encoded = self.encoder.encode_views(images)

        outputs: dict[int, dict] = {}
        for prefix_size in prefix_sizes:
            subset_encoded = encoded.subset(prefix_size)
            subset_extrinsics = extrinsics[:prefix_size]
            subset_intrinsics = intrinsics[:prefix_size]
            encoder_output = self.encoder.decode_views(
                subset_encoded,
                subset_intrinsics,
                subset_extrinsics,
            )
            evidence = self.writer.write(encoder_output, subset_extrinsics, subset_intrinsics)
            state = self.accumulator.accumulate(evidence)
            readout = self.readout(state)
            gaussians = self.gaussian_readout(readout, prune=True)
            render_gaussians = self.gaussian_readout(readout, prune=False)

            output = {
                "context_size": prefix_size,
                "context_indices": max_indices[:prefix_size],
                "state": state,
                "readout": readout,
                "gaussians": gaussians,
                "render_gaussians": render_gaussians,
                "target_indices": episode["target_indices"],
                "num_context_views": prefix_size,
                "num_active_cells": int(state.indices.shape[0]),
                "num_gaussians": int((gaussians.opacities > 1e-4).sum().item()),
                "mean_confidence": self._safe_mean(readout.canonical_certainty),
                "mean_support": self._safe_mean(readout.support_probability),
                "mean_semantic_consistency": self._safe_mean(readout.semantic_consistency),
                "mean_opacity": float(
                    self._safe_mean(gaussians.opacities) if gaussians.opacities.numel() else 0.0
                ),
            }
            if include_render_payload:
                output["depth"] = encoder_output.depth
                output["view_confidence"] = encoder_output.depth_confidence
                output["appearance_uncertainty"] = encoder_output.appearance_uncertainty
                output["canonical_view_certainty"] = encoder_output.combined_certainty
            outputs[prefix_size] = output
        return outputs

    def _validate_nested_prefixes(
        self,
        context_indices: dict,
        max_indices: torch.Tensor,
        prefix_sizes: list[int],
    ) -> None:
        for prefix_size in prefix_sizes:
            expected = max_indices[:prefix_size]
            actual = context_indices[prefix_size]
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(
                    "CanonicalGS prefix training expects nested context prefixes that match the "
                    "first k views of the maximal context ordering."
                )

    def _safe_mean(self, values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.mean().item())
