"""
evaluation.py -- Evaluation metrics with sparse subgroup safety.

Owns: binary/regression/ordinal metric computation, MetricResult dataclass,
sparse subgroup NA handling, evaluate_predictions, evaluate_subgroups.

Must not contain: model training, optimizer, concrete adapter imports, real
dataset parsing, or paper figure rendering.

Deferred for later stages: bootstrap confidence intervals, Brier score, ECE,
PR-AUC, F1/recall/specificity, quadratic weighted kappa (QWK).

Sparse subgroup safety contract
--------------------------------
Subgroups that are too small or single-class return MetricResult with
status=NA and a reason string.  roc_auc_score is NEVER called on
single-class data.

Thresholds:
- Binary AUROC:   min_n=30, min_pos=5, min_neg=5
- Regression MAE: min_n=5
- Ordinal accuracy: min_n=5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from retina_screen.tasks import TASK_REGISTRY, TaskType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status and result types
# ---------------------------------------------------------------------------


class MetricStatus(str, Enum):
    """Outcome of a metric computation attempt."""

    OK = "ok"
    NA = "na"


@dataclass
class MetricResult:
    """Result of one metric computation for a task / subgroup.

    Attributes
    ----------
    metric_name:
        E.g. "auroc", "mae", "accuracy".
    value:
        Computed value when status=OK; None when status=NA.
    status:
        OK or NA.
    reason:
        Empty string when OK; describes why the metric is NA otherwise.
        Values: "sparse_subgroup", "insufficient_class_counts",
        "single_class_subgroup", "unsupported_task_type".
    n:
        Number of valid (observed, non-NaN) rows in this computation.
    positives:
        Count of positive labels for binary tasks; -1 otherwise.
    negatives:
        Count of negative labels for binary tasks; -1 otherwise.
    """

    metric_name: str
    value: float | None
    status: MetricStatus
    reason: str
    n: int
    positives: int
    negatives: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _na(name: str, reason: str, n: int, pos: int = -1, neg: int = -1) -> MetricResult:
    return MetricResult(name, None, MetricStatus.NA, reason, n, pos, neg)


def _ok(name: str, value: float, n: int, pos: int = -1, neg: int = -1) -> MetricResult:
    return MetricResult(name, value, MetricStatus.OK, "", n, pos, neg)


# ---------------------------------------------------------------------------
# Per-type metric computation
# ---------------------------------------------------------------------------


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_n: int = 30,
    min_pos: int = 5,
    min_neg: int = 5,
) -> list[MetricResult]:
    """Compute binary classification metrics with sparse-subgroup safety.

    Returns NA with reason if the subgroup is too small or single-class.
    roc_auc_score is never called on single-class data.
    """
    n = len(y_true)
    pos = int(y_true.sum())
    neg = n - pos

    if n < min_n:
        return [_na("auroc", "sparse_subgroup", n, pos, neg)]
    # Single-class check before pos/neg counts: a single-class group is a
    # distinct failure mode from merely having few positives or negatives.
    if len(np.unique(y_true)) < 2:
        return [_na("auroc", "single_class_subgroup", n, pos, neg)]
    if pos < min_pos or neg < min_neg:
        return [_na("auroc", "insufficient_class_counts", n, pos, neg)]

    auc = float(roc_auc_score(y_true, y_score))
    return [_ok("auroc", auc, n, pos, neg)]


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_n: int = 5,
) -> list[MetricResult]:
    """Compute regression metrics. NaN targets must be excluded before calling."""
    n = len(y_true)
    if n < min_n:
        return [_na("mae", "sparse_subgroup", n)]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return [_ok("mae", mae, n)]


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    task_name: str,
    min_n: int = 5,
) -> list[MetricResult]:
    """Compute ordinal classification accuracy for Stage 5.

    Uses accuracy instead of QWK to avoid class-diversity requirements on small
    dummy test sets.  QWK can be added in a later stage once evaluation is stable.
    y_pred_logits: (n, num_classes) array of raw logits.
    """
    n = len(y_true)
    if n < min_n:
        return [_na("accuracy", "sparse_subgroup", n)]
    y_pred_class = np.argmax(y_pred_logits, axis=-1)
    acc = float(accuracy_score(y_true.astype(int), y_pred_class))
    return [_ok("accuracy", acc, n)]


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    targets: dict[str, list[float]],
    masks: dict[str, list[float]],
    task_names: Sequence[str],
    min_n_binary: int = 30,
    min_n_other: int = 5,
) -> dict[str, list[MetricResult]]:
    """Evaluate model predictions across a set of tasks.

    Parameters
    ----------
    predictions:
        task_name → numpy array. Binary/regression: (n,); ordinal: (n, num_classes).
    targets:
        task_name → list of float target values (MISSING_CLASS_PLACEHOLDER for missing).
    masks:
        task_name → list of 0.0/1.0 (0 = missing/excluded).
    task_names:
        Tasks to evaluate. All must be in TASK_REGISTRY.

    Returns
    -------
    dict[task_name, list[MetricResult]]
    """
    results: dict[str, list[MetricResult]] = {}

    for tn in task_names:
        task = TASK_REGISTRY[tn]
        mask_arr = np.array(masks[tn], dtype=np.float32)
        target_arr = np.array(targets[tn], dtype=np.float64)
        pred_arr = predictions[tn]

        # Select valid rows: mask==1 and not NaN
        valid = (mask_arr == 1.0) & ~np.isnan(target_arr)
        y_true = target_arr[valid]
        y_pred = pred_arr[valid] if pred_arr.ndim == 1 else pred_arr[valid]
        n_valid = int(valid.sum())

        if n_valid == 0:
            results[tn] = [_na(task.primary_metric.value, "sparse_subgroup", 0)]
            continue

        if task.task_type == TaskType.BINARY:
            # Sigmoid to convert logits to probabilities (pure numpy, no torch dep).
            y_score = 1.0 / (1.0 + np.exp(-y_pred.astype(np.float64)))
            results[tn] = compute_binary_metrics(
                y_true, y_score, min_n=min_n_binary
            )

        elif task.task_type == TaskType.REGRESSION:
            results[tn] = compute_regression_metrics(y_true, y_pred, min_n=min_n_other)

        elif task.task_type == TaskType.ORDINAL:
            results[tn] = compute_ordinal_metrics(
                y_true, y_pred, tn, min_n=min_n_other
            )

        else:
            logger.warning("Unknown task_type for task %r; skipping.", tn)
            results[tn] = [_na("unknown", "unsupported_task_type", n_valid)]

    return results


# ---------------------------------------------------------------------------
# Subgroup evaluation
# ---------------------------------------------------------------------------


def evaluate_subgroups(
    predictions: dict[str, np.ndarray],
    targets: dict[str, list[float]],
    masks: dict[str, list[float]],
    task_names: Sequence[str],
    subgroup_labels: np.ndarray,
    min_n_binary: int = 30,
    min_n_other: int = 5,
) -> dict[str, dict[str, list[MetricResult]]]:
    """Evaluate predictions per subgroup value.

    Sparse subgroups return NA with reason instead of crashing.
    """
    results: dict[str, dict[str, list[MetricResult]]] = {}
    subgroup_arr = np.asarray(subgroup_labels, dtype=object)
    unique_groups = _unique_observed_values(subgroup_arr)

    for group_val in unique_groups:
        group_str = str(group_val)
        idx = np.where(subgroup_arr == group_val)[0]

        sub_preds = {
            tn: (predictions[tn][idx] if predictions[tn].ndim == 1 else predictions[tn][idx])
            for tn in task_names
        }
        sub_targets = {tn: [targets[tn][i] for i in idx] for tn in task_names}
        sub_masks = {tn: [masks[tn][i] for i in idx] for tn in task_names}

        results[group_str] = evaluate_predictions(
            sub_preds, sub_targets, sub_masks, task_names,
            min_n_binary=min_n_binary, min_n_other=min_n_other,
        )

    return results


def _is_missing_value(value: object) -> bool:
    """Return True for None or numeric NaN without treating strings as missing."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def pd_isna(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask for missing subgroup labels.

    Handles numeric arrays, object arrays with mixed None/NaN values, and
    categorical string labels without coercing strings to float.
    """
    values = np.asarray(arr)
    if np.issubdtype(values.dtype, np.number):
        return np.isnan(values)
    return np.array([_is_missing_value(v) for v in values], dtype=bool)


def _unique_observed_values(arr: np.ndarray) -> list[object]:
    """Return non-missing subgroup values in first-seen order."""
    missing = pd_isna(arr)
    observed: list[object] = []
    for value, is_missing in zip(arr, missing):
        if is_missing:
            continue
        if not any(value == existing for existing in observed):
            observed.append(value)
    return observed
