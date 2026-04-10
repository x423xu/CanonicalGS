from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import torch
from torch import nn

from canonicalgs.config import ModelConfig

from .evidence import SparseEvidenceAccumulator
from .evidence_writer import VoxelEvidenceWriter
from .gaussian_readout import LocalGaussianReadout, PerViewMonoGaussianBuilder
from .readout import CanonicalReadout
from .view_encoder import SymmetricMultiViewEncoder, ViewEncoderOutput


class CanonicalGsPipeline(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = SymmetricMultiViewEncoder(
            feature_dim=cfg.feature_dim,
            appearance_dim=cfg.appearance_dim,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            min_depth_uncertainty=cfg.min_depth_uncertainty,
            positional_certainty_tau=cfg.positional_certainty_tau,
            max_positional_uncertainty=cfg.max_positional_uncertainty,
            num_depth_bins=cfg.num_depth_bins,
            cost_volume_temperature=cfg.cost_volume_temperature,
            cost_volume_visibility_beta=cfg.cost_volume_visibility_beta,
            dpt_output_stride=cfg.dpt_output_stride,
            dinov2_model_name=cfg.dinov2_model_name,
            dinov2_pretrained=cfg.dinov2_pretrained,
            freeze_dinov2=cfg.freeze_dinov2,
            allow_dinov2_fallback=cfg.allow_dinov2_fallback,
            appearance_uncertainty_bias=cfg.appearance_uncertainty_bias,
            appearance_uncertainty_init=cfg.appearance_uncertainty_init,
        )
        self.writer = VoxelEvidenceWriter(cfg)
        self.accumulator = SparseEvidenceAccumulator(
            max_active_voxels=cfg.max_active_voxels,
            merge_views=cfg.merge_views,
        )
        self.readout = CanonicalReadout()
        self.gaussian_readout = LocalGaussianReadout(cfg)
        self.per_view_gaussian_builder = PerViewMonoGaussianBuilder(cfg)
        self.merge_views = cfg.merge_views

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
        outputs, _ = self.forward_prefixes_timed(
            episode,
            context_sizes=context_sizes,
            include_render_payload=include_render_payload,
        )
        return outputs

    def forward_prefixes_timed(
        self,
        episode: dict,
        context_sizes: Iterable[int] | None = None,
        include_render_payload: bool = False,
    ) -> tuple[dict[int, dict], dict[str, float]]:
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
        timings: dict[str, float] = {}

        def time_block(name: str, fn):
            device = images.device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            result = fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[name] = timings.get(name, 0.0) + (perf_counter() - start) * 1000.0
            return result

        encoded = time_block("forward/encode_views_ms", lambda: self.encoder.encode_views(images))
        max_encoded = encoded.subset(max_context)
        max_extrinsics = extrinsics[:max_context]
        max_intrinsics = intrinsics[:max_context]
        encoder_output = time_block(
            "forward/decode_views_ms",
            lambda: self.encoder.decode_views(
                max_encoded,
                max_intrinsics,
                max_extrinsics,
            ),
        )
        outputs: dict[int, dict] = {}
        if not self.merge_views:
            for prefix_size in prefix_sizes:
                prefix_encoder = self._slice_encoder_output(encoder_output, prefix_size)
                prefix_intrinsics = max_intrinsics[:prefix_size]
                prefix_extrinsics = max_extrinsics[:prefix_size]
                readout, gaussians = time_block(
                    f"forward/gaussian_decode_ctx_{prefix_size}_ms",
                    lambda prefix_encoder=prefix_encoder, prefix_intrinsics=prefix_intrinsics, prefix_extrinsics=prefix_extrinsics: self.per_view_gaussian_builder(
                        prefix_encoder,
                        prefix_intrinsics,
                        prefix_extrinsics,
                    ),
                )
                active_mask = gaussians.opacities > 1e-4
                output = {
                    "context_size": prefix_size,
                    "context_indices": max_indices[:prefix_size],
                    "state": None,
                    "readout": readout,
                    "gaussians": gaussians,
                    "render_gaussians": gaussians,
                    "target_indices": episode["target_indices"],
                    "num_context_views": prefix_size,
                    "num_active_cells": int(readout.indices.shape[0]),
                    "num_gaussians": int(active_mask.sum().item()),
                    "mean_confidence": self._safe_mean(readout.canonical_certainty),
                    "mean_semantic_consistency": self._safe_mean(readout.semantic_consistency),
                    "mean_opacity": float(
                        self._safe_mean(gaussians.opacities) if gaussians.opacities.numel() else 0.0
                    ),
                }
                if include_render_payload:
                    output["depth"] = prefix_encoder.depth
                    output["positional_certainty"] = prefix_encoder.positional_certainty
                    output["view_confidence"] = prefix_encoder.depth_confidence
                    output["appearance_certainty"] = prefix_encoder.appearance_certainty
                    output["combined_certainty"] = prefix_encoder.combined_certainty
                outputs[prefix_size] = output
        else:
            evidence = time_block(
                "forward/write_evidence_ms",
                lambda: self.writer.write(encoder_output, max_extrinsics, max_intrinsics),
            )
            states = time_block(
                "forward/accumulate_prefixes_ms",
                lambda: self.accumulator.accumulate_prefixes(evidence, prefix_sizes),
            )

            for prefix_size in prefix_sizes:
                state = states[prefix_size]
                readout = time_block(
                    f"forward/readout_ctx_{prefix_size}_ms",
                    lambda state=state: self.readout(state),
                )
                gaussians = time_block(
                    f"forward/gaussian_decode_ctx_{prefix_size}_ms",
                    lambda readout=readout: self.gaussian_readout(readout),
                )
                active_mask = gaussians.opacities > 1e-4

                output = {
                    "context_size": prefix_size,
                    "context_indices": max_indices[:prefix_size],
                    "state": state,
                    "readout": readout,
                    "gaussians": gaussians,
                    "render_gaussians": gaussians,
                    "target_indices": episode["target_indices"],
                    "num_context_views": prefix_size,
                    "num_active_cells": int(state.indices.shape[0]),
                    "num_gaussians": int(active_mask.sum().item()),
                    "mean_confidence": self._safe_mean(readout.canonical_certainty),
                    "mean_semantic_consistency": self._safe_mean(readout.semantic_consistency),
                    "mean_opacity": float(
                        self._safe_mean(gaussians.opacities) if gaussians.opacities.numel() else 0.0
                    ),
                }
                if include_render_payload:
                    output["depth"] = encoder_output.depth[:prefix_size]
                    output["positional_certainty"] = encoder_output.positional_certainty[:prefix_size]
                    output["view_confidence"] = encoder_output.depth_confidence[:prefix_size]
                    output["appearance_certainty"] = encoder_output.appearance_certainty[:prefix_size]
                    output["combined_certainty"] = encoder_output.combined_certainty[:prefix_size]
                outputs[prefix_size] = output
        timings["forward/readout_total_ms"] = sum(
            value for key, value in timings.items() if key.startswith("forward/readout_ctx_")
        )
        timings["forward/gaussian_decode_total_ms"] = sum(
            value
            for key, value in timings.items()
            if key.startswith("forward/gaussian_decode_ctx_")
        )
        timings["forward/total_ms"] = sum(
            value
            for key, value in timings.items()
            if key.startswith("forward/")
            and not key.endswith("_total_ms")
            and key != "forward/total_ms"
        )
        return outputs, timings

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

    def _slice_encoder_output(
        self,
        encoder_output: ViewEncoderOutput,
        count: int,
    ) -> ViewEncoderOutput:
        return ViewEncoderOutput(
            input_rgb=encoder_output.input_rgb[:count],
            appearance_features=encoder_output.appearance_features[:count],
            geometry_features=encoder_output.geometry_features[:count],
            depth=encoder_output.depth[:count],
            density=encoder_output.density[:count],
            gaussian_raw_params=encoder_output.gaussian_raw_params[:count],
            positional_certainty=encoder_output.positional_certainty[:count],
            appearance_certainty=encoder_output.appearance_certainty[:count],
            combined_certainty=encoder_output.combined_certainty[:count],
            depth_confidence=encoder_output.depth_confidence[:count],
            confidence=encoder_output.confidence[:count],
        )

    def _safe_mean(self, values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 0.0
        return float(finite.mean().item())
