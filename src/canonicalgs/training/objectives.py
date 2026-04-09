from __future__ import annotations

from dataclasses import dataclass

import torch

from canonicalgs.config import ObjectiveConfig

from .render_metrics import compute_render_stats, render_target_views


@dataclass(slots=True)
class LossBundle:
    render_loss: torch.Tensor
    convergence_loss: torch.Tensor
    monotone_loss: torch.Tensor
    null_loss: torch.Tensor
    total_loss: torch.Tensor


class CanonicalLossComputer:
    def __init__(self, cfg: ObjectiveConfig) -> None:
        self.cfg = cfg
        self.lpips = self._build_lpips_if_available()

    def __call__(
        self,
        outputs: dict[int, dict],
        episode: dict | None = None,
        max_render_targets: int | None = None,
    ) -> LossBundle:
        context_sizes = sorted(outputs)
        largest_context = context_sizes[-1]
        largest_output = outputs[largest_context]
        device = largest_output["readout"].support_probability.device

        prefix_states = {context_size: output["state"] for context_size, output in outputs.items()}
        decoded = {context_size: output["render_gaussians"] for context_size, output in outputs.items()}
        renders = self._render_prefixes(outputs, episode, max_render_targets)
        render_loss = self._compute_main_render_loss(
            largest_output,
            renders.get(largest_context),
            episode=episode,
            max_render_targets=max_render_targets,
        )

        monotone_terms = []
        convergence_terms = []
        for lower_key, upper_key in zip(context_sizes[:-1], context_sizes[1:]):
            lower_output = outputs[lower_key]
            upper_output = outputs[upper_key]
            lower_render = renders.get(lower_key)
            upper_render = renders.get(upper_key)
            _lower_state = prefix_states[lower_key]
            _upper_state = prefix_states[upper_key]
            _lower_decoded = decoded[lower_key]
            _upper_decoded = decoded[upper_key]

            if lower_render is not None and upper_render is not None:
                monotone_terms.append(torch.relu(upper_render.mse - lower_render.mse))

            lower_readout = lower_output["readout"]
            upper_readout = upper_output["readout"]
            convergence_terms.append(
                torch.relu(
                    lower_readout.canonical_certainty.mean() - upper_readout.canonical_certainty.mean()
                )
                + torch.relu(
                    upper_readout.uncertainty.mean() - lower_readout.uncertainty.mean()
                )
                + torch.relu(
                    lower_readout.support_probability.mean() - upper_readout.support_probability.mean()
                )
            )

        low_cert_mask = largest_output["readout"].canonical_certainty < self.cfg.low_confidence_threshold
        if torch.any(low_cert_mask):
            null_loss = (
                largest_output["readout"].support_probability[low_cert_mask]
                * (1.0 + largest_output["readout"].semantic_consistency[low_cert_mask])
            ).mean()
        else:
            null_loss = torch.zeros((), device=device)

        convergence_loss = self._stack_mean(convergence_terms, device)
        monotone_loss = self._stack_mean(monotone_terms, device)
        total_loss = (
            self.cfg.lambda_rend * render_loss
            + self.cfg.lambda_conv * convergence_loss
            + self.cfg.lambda_mono * monotone_loss
            + self.cfg.lambda_null * null_loss
        )
        return LossBundle(
            render_loss=render_loss,
            convergence_loss=convergence_loss,
            monotone_loss=monotone_loss,
            null_loss=null_loss,
            total_loss=total_loss,
        )

    def _build_lpips_if_available(self):
        if self.cfg.lambda_lpips <= 0.0:
            return None
        try:
            import lpips  # type: ignore[import-not-found]
        except ImportError:
            return None
        return lpips.LPIPS(net="vgg")

    def _stack_mean(self, values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.zeros((), device=device)
        return torch.stack(values).mean()

    def _render_prefixes(
        self,
        outputs: dict[int, dict],
        episode: dict | None,
        max_render_targets: int | None,
    ) -> dict[int, object]:
        if episode is None:
            return {}
        return {
            context_size: compute_render_stats(
                episode=episode,
                output=output,
                max_targets=max_render_targets,
            )
            for context_size, output in outputs.items()
        }

    def _compute_main_render_loss(
        self,
        output: dict,
        render_stats,
        episode: dict | None,
        max_render_targets: int | None,
    ) -> torch.Tensor:
        device = output["readout"].support_probability.device
        if render_stats is None:
            return torch.zeros((), device=device)

        render_loss = render_stats.mse
        if self.lpips is None or episode is None or render_stats.num_targets == 0:
            return render_loss

        rendered, target_images, _, num_targets = render_target_views(
            episode,
            output,
            max_targets=max_render_targets,
        )
        if num_targets == 0:
            return render_loss
        lpips_module = self.lpips.to(device=device)
        lpips_value = lpips_module(rendered * 2.0 - 1.0, target_images * 2.0 - 1.0).mean()
        return render_loss + self.cfg.lambda_lpips * lpips_value
