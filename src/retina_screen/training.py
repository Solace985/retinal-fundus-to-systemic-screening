"""
training.py -- Masked multi-task loss computation and training utilities.

Owns: compute_masked_task_loss, KendallUncertaintyWeighting, train_one_step,
      train_one_epoch, EarlyStopping, compute_class_weights,
      get_kendall_log_sigmas.

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
    class_weights: dict[str, torch.Tensor] | None = None,
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
    - class_weights: optional per-task weight tensors. Binary tasks use pos_weight
      (scalar tensor); ordinal tasks use per-class weight tensor. Regression tasks
      ignore class_weights. Default None = no reweighting (backward-compatible).
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
            pw = None
            if class_weights is not None and tn in class_weights:
                pw = class_weights[tn].to(device)
            losses[tn] = F.binary_cross_entropy_with_logits(
                v_pred, v_target, pos_weight=pw, reduction="mean"
            )

        elif task.task_type == TaskType.ORDINAL:
            v_pred = pred[valid_idx]               # (n_valid, num_classes)
            v_target = target[valid_idx].long()    # (n_valid,) class IDs
            cw = None
            if class_weights is not None and tn in class_weights:
                cw = class_weights[tn].to(device)
            losses[tn] = F.cross_entropy(v_pred, v_target, weight=cw, reduction="mean")

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
# Single training step (signature extended, backward-compatible via P1 contract)
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
    class_weights: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Run one forward/backward/step cycle on a mini-batch or full tensor.

    Parameters
    ----------
    class_weights : optional dict task_name -> weight tensor.
        Passed through to compute_masked_task_loss. Default None = no reweighting.
        Existing callers that omit this parameter are fully unaffected.

    Returns
    -------
    dict with keys: total_loss, per_task_losses, valid_counts, grad_norm.
    """
    model.train()
    optimizer.zero_grad()

    predictions = model(embedding)
    task_loss_result = compute_masked_task_loss(
        predictions, targets, masks, task_names, class_weights=class_weights
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


# ---------------------------------------------------------------------------
# Mini-batch epoch training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    embedding: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_names: Sequence[str],
    batch_size: int,
    weighter: KendallUncertaintyWeighting | None = None,
    max_grad_norm: float = 1.0,
    class_weights: dict[str, torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Train one full epoch using mini-batch gradient descent.

    Shuffles training indices with a seeded torch.Generator (reproducible for
    the same seed), splits into mini-batches of size batch_size, and calls
    train_one_step for each batch.

    The generator state advances naturally across epochs — do NOT reseed it
    between epochs; the caller must maintain the same generator object.

    Returns
    -------
    dict with keys:
        avg_total_loss      -- mean total loss across all mini-batch steps
        per_task_avg_loss   -- {task: mean loss across steps}
        avg_grad_norm       -- mean gradient norm across steps
        n_optimizer_steps   -- number of optimizer steps (> 1 when n > batch_size)
        n_samples           -- total training samples processed
    """
    n = embedding.shape[0]
    idx = torch.randperm(n, generator=generator)

    total_loss_sum = 0.0
    per_task_loss_sum: dict[str, float] = {tn: 0.0 for tn in task_names}
    grad_norm_sum = 0.0
    n_steps = 0

    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batch_idx = idx[start:end]

        batch_emb = embedding[batch_idx]
        batch_targets = {tn: targets[tn][batch_idx] for tn in task_names}
        batch_masks = {tn: masks[tn][batch_idx] for tn in task_names}

        result = train_one_step(
            model, optimizer, batch_emb, batch_targets, batch_masks,
            task_names, loss_weighter=weighter,
            max_grad_norm=max_grad_norm, class_weights=class_weights,
        )
        total_loss_sum += result["total_loss"]
        for tn in task_names:
            per_task_loss_sum[tn] += result["per_task_losses"].get(tn, 0.0)
        grad_norm_sum += result["grad_norm"]
        n_steps += 1
        start = end

    n_steps = max(n_steps, 1)
    return {
        "avg_total_loss": total_loss_sum / n_steps,
        "per_task_avg_loss": {tn: per_task_loss_sum[tn] / n_steps for tn in task_names},
        "avg_grad_norm": grad_norm_sum / n_steps,
        "n_optimizer_steps": n_steps,
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Early stopping based on a monitored validation metric.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without improvement before stopping.
    mode : str
        "max" to maximize (default, for AUROC). "min" to minimize (for loss).
    """

    def __init__(self, patience: int, mode: str = "max") -> None:
        self._patience = patience
        self._mode = mode
        self._best_metric: float | None = None
        self._best_epoch: int = -1
        self._epochs_without_improvement: int = 0
        self._total_epochs: int = 0

    def step(self, metric: float | None) -> bool:
        """Update stopping state. Returns True if training should stop.

        If metric is None (e.g., all binary val metrics returned NA), the epoch
        counts toward patience but does not reset the counter.
        """
        epoch = self._total_epochs
        self._total_epochs += 1

        if metric is None:
            self._epochs_without_improvement += 1
        else:
            improved = (
                self._best_metric is None
                or (self._mode == "max" and metric > self._best_metric)
                or (self._mode == "min" and metric < self._best_metric)
            )
            if improved:
                self._best_metric = metric
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

        return self._epochs_without_improvement >= self._patience

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def best_metric(self) -> float | None:
        return self._best_metric

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement


# ---------------------------------------------------------------------------
# Class weight computation (config-controlled; disabled for standard run)
# ---------------------------------------------------------------------------


def compute_class_weights(
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_names: Sequence[str],
    max_weight: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Compute capped inverse-frequency class weights from training labels only.

    Must be called with train targets/masks only — never with val/test/reliability
    data. Missing labels (mask == 0) are excluded from weight computation so
    masked rows never influence the weights.

    Binary tasks:  pos_weight tensor (shape [1]) = min(n_neg / n_pos, max_weight).
    Ordinal tasks: per-class weight tensor (shape [n_classes]) via inverse-frequency.
    Regression tasks: skipped (not applicable).

    Returns a dict mapping task_name -> weight tensor. Tasks with no valid
    labels or single-class support are omitted (loss function uses no weighting).
    """
    weights: dict[str, torch.Tensor] = {}

    for tn in task_names:
        task = TASK_REGISTRY[tn]
        mask = masks[tn]
        target = targets[tn]
        valid_idx = (mask == 1.0).bool()
        n_valid = int(valid_idx.sum().item())

        if n_valid == 0:
            logger.debug("compute_class_weights: task=%r has no valid labels; skipping.", tn)
            continue

        v_target = target[valid_idx]

        if task.task_type == TaskType.BINARY:
            n_pos = float((v_target == 1).sum().item())
            n_neg = float((v_target == 0).sum().item())
            if n_pos == 0 or n_neg == 0:
                logger.warning(
                    "compute_class_weights: task=%r is single-class in training data "
                    "(n_pos=%d, n_neg=%d); skipping pos_weight.",
                    tn, int(n_pos), int(n_neg),
                )
                continue
            pos_weight = min(n_neg / n_pos, max_weight)
            weights[tn] = torch.tensor([pos_weight], dtype=torch.float32)
            logger.info(
                "compute_class_weights: task=%r binary pos_weight=%.3f "
                "(n_pos=%d, n_neg=%d, cap=%.1f)",
                tn, pos_weight, int(n_pos), int(n_neg), max_weight,
            )

        elif task.task_type == TaskType.ORDINAL:
            classes = v_target.long()
            n_classes = int(classes.max().item()) + 1
            class_w = torch.zeros(n_classes, dtype=torch.float32)
            for c in range(n_classes):
                n_c = float((classes == c).sum().item())
                class_w[c] = min(n_valid / (n_classes * n_c), max_weight) if n_c > 0 else float(max_weight)
            weights[tn] = class_w
            logger.info(
                "compute_class_weights: task=%r ordinal weights=%s",
                tn, [round(float(w), 3) for w in class_w.tolist()],
            )

        # Regression: no class weighting applicable

    return weights


# ---------------------------------------------------------------------------
# Kendall log-sigma logging helper
# ---------------------------------------------------------------------------


def get_kendall_log_sigmas(weighter: KendallUncertaintyWeighting) -> dict[str, float]:
    """Return current log_sigma value for each task as a plain float."""
    return {
        tn: float(weighter.log_sigmas[tn].item())
        for tn in weighter.task_names
    }
