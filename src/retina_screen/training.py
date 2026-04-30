"""
training.py -- Masked multi-task loss computation and training utilities.

Owns: compute_masked_task_loss, KendallUncertaintyWeighting, train_one_step.

Must not contain: concrete adapter imports, native dataset parsing, evaluation
metrics, or paper/dashboard logic.

NaN + mask contract (regression tasks)
--------------------------------------
Missing regression targets are stored as NaN (MISSING_REGRESSION_PLACEHOLDER)
with mask=0 in data.py.  This module filters by mask BEFORE computing loss:

    valid = mask == 1.0
    loss  = mse_loss(pred[valid], target[valid])

Do NOT compute raw_loss * mask after the fact, because NaN * 0.0 = NaN.
This contract must also be honoured in Stage 5+ training code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from retina_screen.tasks import TASK_REGISTRY, TaskType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loss result
# ---------------------------------------------------------------------------


@dataclass
class TaskLossResult:
    """Per-task losses and valid observation counts returned by compute_masked_task_loss.

    ``valid_counts`` allows callers to distinguish all-missing tasks (count=0)
    from tasks with a legitimately near-zero loss (count>0).  Kendall weighting
    uses this to skip all-missing tasks rather than inspecting the loss value.
    """

    losses: dict[str, torch.Tensor]  # task_name → scalar loss (zero if no valid rows)
    valid_counts: dict[str, int]      # task_name → number of observed rows used


# ---------------------------------------------------------------------------
# Masked loss computation
# ---------------------------------------------------------------------------


def compute_masked_task_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_names: Sequence[str],
) -> TaskLossResult:
    """Compute per-task masked losses.

    Rules
    -----
    - Only rows where mask == 1 contribute to loss.
    - Missing binary/ordinal placeholders (-1) are excluded by the mask and
      never reach the loss function.
    - Regression NaN targets are excluded by the mask; loss is never computed
      on NaN values (see module docstring for the NaN contract).
    - If no valid rows for a task: returns a zero-scalar loss and valid_count=0.
      Zero from valid_count=0 is distinguishable from a legitimately zero loss
      via the valid_counts field.
    """
    losses: dict[str, torch.Tensor] = {}
    valid_counts: dict[str, int] = {}

    for tn in task_names:
        task = TASK_REGISTRY[tn]
        mask = masks[tn]
        target = targets[tn]
        pred = predictions[tn]

        device = pred.device if isinstance(pred, torch.Tensor) else torch.device("cpu")
        valid_idx = (mask == 1.0).bool()
        n_valid = int(valid_idx.sum().item())
        valid_counts[tn] = n_valid

        if n_valid == 0:
            losses[tn] = torch.tensor(0.0, device=device)
            continue

        if task.task_type == TaskType.BINARY:
            v_pred = pred[valid_idx]
            v_target = target[valid_idx]
            losses[tn] = F.binary_cross_entropy_with_logits(
                v_pred, v_target, reduction="mean"
            )

        elif task.task_type == TaskType.ORDINAL:
            v_pred = pred[valid_idx]               # (n_valid, num_classes)
            v_target = target[valid_idx].long()    # (n_valid,) class IDs
            losses[tn] = F.cross_entropy(v_pred, v_target, reduction="mean")

        elif task.task_type == TaskType.REGRESSION:
            # Hard rule: filter by mask FIRST, then compute loss.
            # Never compute loss on NaN targets.
            v_pred = pred[valid_idx]
            v_target = target[valid_idx]
            losses[tn] = F.mse_loss(v_pred, v_target, reduction="mean")

        else:
            raise ValueError(
                f"Unrecognised task_type={task.task_type!r} for task {tn!r}"
            )

    return TaskLossResult(losses=losses, valid_counts=valid_counts)


# ---------------------------------------------------------------------------
# Kendall uncertainty weighting
# ---------------------------------------------------------------------------


class KendallUncertaintyWeighting(nn.Module):
    """Learns per-task uncertainty (log_sigma) to weight multi-task losses.

    Formula per task (applied to all task types uniformly):
        s_t           = clamp(log_sigma_t, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        weighted_t    = exp(-2 * s_t) * loss_t + s_t

    Total loss = sum of weighted_t over tasks with valid_count > 0.

    Tasks with valid_count == 0 (all labels missing) are excluded from the
    sum so their log_sigma parameters do not receive fabricated gradients.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.
    """

    LOG_SIGMA_MIN: float = -2.0
    LOG_SIGMA_MAX: float = 2.0

    def __init__(self, task_names: Sequence[str]) -> None:
        super().__init__()
        self._task_names = list(task_names)
        self.log_sigmas = nn.ParameterDict(
            {tn: nn.Parameter(torch.zeros(1)) for tn in task_names}
        )

    def forward(self, task_loss_result: TaskLossResult) -> torch.Tensor:
        """Compute the uncertainty-weighted total loss.

        Skips tasks with valid_count == 0 to prevent gradient fabrication.
        """
        weighted: list[torch.Tensor] = []

        for tn in self._task_names:
            if task_loss_result.valid_counts.get(tn, 0) == 0:
                continue  # all-missing: skip, determined by count not loss value
            s = torch.clamp(
                self.log_sigmas[tn], self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX
            ).squeeze()
            task_loss = task_loss_result.losses[tn]
            weighted.append(torch.exp(-2.0 * s) * task_loss + s)

        if not weighted:
            # All tasks missing: return zero with a grad_fn so backward is safe.
            dummy = next(iter(self.log_sigmas.values()))
            return (dummy * 0.0).squeeze()

        return torch.stack(weighted).sum()

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    embedding: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_names: Sequence[str],
    loss_weighter: KendallUncertaintyWeighting | None = None,
    max_grad_norm: float = 1.0,
) -> dict:
    """Run one forward/backward/step cycle.

    Returns
    -------
    dict with keys: total_loss, per_task_losses, valid_counts, grad_norm.
    """
    model.train()
    optimizer.zero_grad()

    predictions = model(embedding)
    task_loss_result = compute_masked_task_loss(
        predictions, targets, masks, task_names
    )

    if loss_weighter is not None:
        total_loss = loss_weighter(task_loss_result)
    else:
        valid_losses = [
            task_loss_result.losses[tn]
            for tn in task_names
            if task_loss_result.valid_counts[tn] > 0
        ]
        if not valid_losses:
            device = next(iter(predictions.values())).device
            total_loss = torch.tensor(0.0, device=device)
        else:
            total_loss = torch.stack(valid_losses).mean()

    total_loss.backward()

    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
    )
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "per_task_losses": {
            tn: task_loss_result.losses[tn].item() for tn in task_names
        },
        "valid_counts": dict(task_loss_result.valid_counts),
        "grad_norm": grad_norm,
    }
