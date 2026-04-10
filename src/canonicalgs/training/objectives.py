from __future__ import annotations

from dataclasses import dataclass

import torch

from canonicalgs.config import ObjectiveConfig

from .render_metrics import compute_render_stats


@dataclass(slots=True)
class LossBundle:
    render_loss: torch.Tensor
    monotone_loss: torch.Tensor
    total_loss: torch.Tensor


class CanonicalLossComputer:
    def __init__(self, cfg: ObjectiveConfig) -> None:
        self.cfg = cfg

    def __call__(
        self,
        outputs: dict[int, dict],
        episode: dict | None = None,
        max_render_targets: int | None = None,
    ) -> LossBundle:
        context_sizes = sorted(outputs)
        largest_context = context_sizes[-1]
        largest_output = outputs[largest_context]
        device = largest_output["readout"].canonical_certainty.device
        monotone_pair = self._select_monotone_pair(context_sizes) if self.cfg.mono_on else None
        render_contexts = {largest_context}
        if monotone_pair is not None:
            render_contexts.update(monotone_pair)

        renders = self._render_prefixes(outputs, episode, max_render_targets, sorted(render_contexts))
        render_loss = self._compute_main_render_loss(
            renders.get(largest_context),
            device=device,
        )

        monotone_terms = []
        if monotone_pair is not None:
            lower_key, upper_key = monotone_pair
            lower_render = renders.get(lower_key)
            upper_render = renders.get(upper_key)
            if lower_render is None or upper_render is None:
                monotone_terms = []
            else:
                monotone_terms.append(torch.relu(upper_render.mse - lower_render.mse))

        monotone_loss = self._stack_mean(monotone_terms, device)
        total_loss = self.cfg.lambda_rend * render_loss + self.cfg.lambda_mono * monotone_loss
        return LossBundle(
            render_loss=render_loss,
            monotone_loss=monotone_loss,
            total_loss=total_loss,
        )

    def _stack_mean(self, values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.zeros((), device=device)
        return torch.stack(values).mean()

    def _render_prefixes(
        self,
        outputs: dict[int, dict],
        episode: dict | None,
        max_render_targets: int | None,
        render_contexts: list[int],
    ) -> dict[int, object]:
        if episode is None:
            return {}
        return {
            context_size: compute_render_stats(
                episode=episode,
                output=outputs[context_size],
                max_targets=max_render_targets,
            )
            for context_size in render_contexts
        }

    def _compute_main_render_loss(
        self,
        render_stats,
        device: torch.device,
    ) -> torch.Tensor:
        if render_stats is None:
            return torch.zeros((), device=device)
        return render_stats.mse

    def _select_monotone_pair(self, context_sizes: list[int]) -> tuple[int, int] | None:
        if len(context_sizes) < 2:
            return None
        largest_context = context_sizes[-1]
        if 2 in context_sizes and largest_context != 2:
            return 2, largest_context
        return context_sizes[0], largest_context
