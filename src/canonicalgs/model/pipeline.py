from __future__ import annotations

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
            dpt_output_stride=cfg.dpt_output_stride,
            dinov2_model_name=cfg.dinov2_model_name,
            dinov2_pretrained=cfg.dinov2_pretrained,
            freeze_dinov2=cfg.freeze_dinov2,
        )
        self.writer = VoxelEvidenceWriter(cfg)
        self.accumulator = SparseEvidenceAccumulator(max_active_voxels=cfg.max_active_voxels)
        self.readout = CanonicalReadout()
        self.gaussian_readout = LocalGaussianReadout(cfg)

    def forward(
        self,
        episode: dict,
        context_size: int,
        include_render_payload: bool = False,
    ) -> dict:
        context_indices = episode["context_indices"][context_size]
        images = episode["images"][context_indices]
        extrinsics = episode["extrinsics"][context_indices]
        intrinsics = episode["intrinsics"][context_indices]

        encoder_output = self.encoder(images, intrinsics, extrinsics)
        evidence = self.writer.write(encoder_output, extrinsics, intrinsics)
        state = self.accumulator.accumulate(evidence)
        readout = self.readout(state)
        gaussians = self.gaussian_readout(readout, prune=True)
        render_gaussians = self.gaussian_readout(readout, prune=False)

        output = {
            "context_size": context_size,
            "context_indices": context_indices,
            "state": state,
            "readout": readout,
            "gaussians": gaussians,
            "render_gaussians": render_gaussians,
            "target_indices": episode["target_indices"],
            "num_context_views": int(context_indices.shape[0]),
            "num_active_cells": int(state.indices.shape[0]),
            "num_gaussians": int(gaussians.means.shape[0]),
            "mean_confidence": self._safe_mean(readout.confidence),
            "mean_support": self._safe_mean(readout.support_probability),
            "mean_opacity": float(
                self._safe_mean(gaussians.opacities) if gaussians.opacities.numel() else 0.0
            ),
        }
        if include_render_payload:
            output["depth"] = encoder_output.depth
            output["view_confidence"] = encoder_output.confidence
        return output

    def _safe_mean(self, values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.mean().item())
