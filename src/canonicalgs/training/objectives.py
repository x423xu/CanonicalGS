from __future__ import annotations

from dataclasses import dataclass

import torch

from canonicalgs.config import ObjectiveConfig


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

    def __call__(self, outputs: dict[int, dict]) -> LossBundle:
        context_sizes = sorted(outputs)
        teacher_key = context_sizes[-1]
        teacher = outputs[teacher_key]["readout"]
        device = teacher.support_probability.device

        render_loss = torch.zeros((), device=device)
        convergence_terms = []
        monotone_terms = []

        teacher_lookup = self._readout_lookup(teacher)

        for context_size in context_sizes[:-1]:
            student = outputs[context_size]["readout"]
            overlap = self._shared_entries(student, teacher_lookup)
            if overlap is None:
                continue
            student_support, teacher_support, student_mean, teacher_mean, _, _, teacher_conf = overlap
            mask = (
                (teacher_support > self.cfg.teacher_support_threshold)
                & (teacher_conf > self.cfg.teacher_confidence_threshold)
            )
            if torch.any(mask):
                support_term = torch.abs(student_support[mask] - teacher_support[mask]).mean()
                geometry_term = torch.abs(student_mean[mask] - teacher_mean[mask]).mean()
                convergence_terms.append(support_term + geometry_term)

        for lower_key, upper_key in zip(context_sizes[:-1], context_sizes[1:]):
            lower = outputs[lower_key]["readout"]
            upper = outputs[upper_key]["readout"]
            overlap = self._shared_entries(lower, self._readout_lookup(upper))
            if overlap is None:
                continue
            _, teacher_support, _, _, lower_uncert, upper_uncert, upper_conf = overlap
            mask = (
                (teacher_support > self.cfg.teacher_support_threshold)
                & (upper_conf > self.cfg.teacher_confidence_threshold)
            )
            if torch.any(mask):
                monotone_terms.append(torch.relu(upper_uncert[mask] - lower_uncert[mask]).mean())

        low_conf_mask = teacher.confidence < self.cfg.low_confidence_threshold
        if torch.any(low_conf_mask):
            null_loss = teacher.support_probability[low_conf_mask].mean()
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

    def _stack_mean(self, values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.zeros((), device=device)
        return torch.stack(values).mean()

    def _readout_lookup(self, readout) -> dict[tuple[int, int, int], tuple[torch.Tensor, ...]]:
        lookup = {}
        for index, support, mean, uncert, conf in zip(
            readout.indices,
            readout.support_probability,
            readout.geometry_mean,
            readout.uncertainty,
            readout.confidence,
        ):
            lookup[tuple(int(v) for v in index.tolist())] = (support, mean, uncert, conf)
        return lookup

    def _shared_entries(self, readout, teacher_lookup):
        keys = [tuple(int(v) for v in index.tolist()) for index in readout.indices]
        keep_indices = [i for i, key in enumerate(keys) if key in teacher_lookup]
        if not keep_indices:
            return None

        keep = torch.tensor(keep_indices, dtype=torch.long, device=readout.indices.device)
        teacher_support = []
        teacher_mean = []
        teacher_uncert = []
        teacher_conf = []
        for i in keep_indices:
            support, mean, uncert, conf = teacher_lookup[keys[i]]
            teacher_support.append(support)
            teacher_mean.append(mean)
            teacher_uncert.append(uncert)
            teacher_conf.append(conf)

        return (
            readout.support_probability[keep],
            torch.stack(teacher_support).to(readout.indices.device),
            readout.geometry_mean[keep],
            torch.stack(teacher_mean).to(readout.indices.device),
            readout.uncertainty[keep],
            torch.stack(teacher_uncert).to(readout.indices.device),
            torch.stack(teacher_conf).to(readout.indices.device),
        )
