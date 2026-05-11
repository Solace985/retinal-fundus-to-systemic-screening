"""
tests/test_evaluation_metrics.py -- Tests for Stage 8D-2A evaluation metric additions.

Covers:
  - compute_ordinal_metrics: accuracy + macro_f1 + balanced_accuracy + per_class_support
  - Majority-class imbalance exposure (high accuracy, low macro_f1/balanced_accuracy)
  - Single-class ordinal input safety (NA for macro_f1/balanced_accuracy, no crash)
  - Per-class support counts correctness
  - Sparse subgroup behaviour unchanged (n < min_n → single NA)
  - Existing binary AUROC unaffected
  - MetricResult.per_class_support defaults to None
"""

from __future__ import annotations

import numpy as np
import pytest

from retina_screen.evaluation import (
    MetricResult,
    MetricStatus,
    compute_binary_metrics,
    compute_ordinal_metrics,
)


# ---------------------------------------------------------------------------
# MetricResult field
# ---------------------------------------------------------------------------


def test_metric_result_per_class_support_defaults_to_none() -> None:
    r = MetricResult("accuracy", 0.9, MetricStatus.OK, "", 100, -1, -1)
    assert r.per_class_support is None


def test_metric_result_per_class_support_can_be_set() -> None:
    r = MetricResult("accuracy", 0.9, MetricStatus.OK, "", 100, -1, -1, {0: 90, 1: 10})
    assert r.per_class_support == {0: 90, 1: 10}


# ---------------------------------------------------------------------------
# compute_ordinal_metrics — normal multi-class input
# ---------------------------------------------------------------------------


def _make_logits_from_classes(y_pred_classes: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert predicted class indices to one-hot logit array."""
    logits = np.zeros((len(y_pred_classes), num_classes), dtype=np.float32)
    logits[np.arange(len(y_pred_classes)), y_pred_classes] = 10.0
    return logits


def test_ordinal_returns_three_results_with_multiple_classes() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred = _make_logits_from_classes(y_true, num_classes=3)
    results = compute_ordinal_metrics(y_true, y_pred, "dr_grade")
    assert len(results) == 3
    names = [r.metric_name for r in results]
    assert "accuracy" in names
    assert "macro_f1" in names
    assert "balanced_accuracy" in names


def test_ordinal_accuracy_correct() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = _make_logits_from_classes(y_true, num_classes=3)
    results = compute_ordinal_metrics(y_true, y_pred, "dr_grade")
    acc = next(r for r in results if r.metric_name == "accuracy")
    assert acc.status == MetricStatus.OK
    assert abs(acc.value - 1.0) < 1e-6


def test_ordinal_all_metrics_ok_for_balanced_input() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = _make_logits_from_classes(y_true, num_classes=3)
    results = compute_ordinal_metrics(y_true, y_pred, "dr_grade")
    for r in results:
        assert r.status == MetricStatus.OK, f"{r.metric_name} unexpectedly NA: {r.reason}"


# ---------------------------------------------------------------------------
# Imbalance exposure: high accuracy but low macro_f1 / balanced_accuracy
# ---------------------------------------------------------------------------


def test_ordinal_imbalanced_high_acc_low_macro_f1() -> None:
    """Majority-class model: high accuracy masks poor performance on minority class."""
    # 90 class-0, 5 class-1, 5 class-2 — model always predicts class-0.
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    y_pred_classes = np.zeros(100, dtype=int)   # always predict class 0
    logits = _make_logits_from_classes(y_pred_classes, num_classes=3)
    results = compute_ordinal_metrics(y_true, logits, "dr_grade")
    acc = next(r for r in results if r.metric_name == "accuracy")
    macro_f1 = next(r for r in results if r.metric_name == "macro_f1")
    bal_acc = next(r for r in results if r.metric_name == "balanced_accuracy")
    assert acc.value > 0.85, "Accuracy should be high for majority-class model"
    assert macro_f1.value < 0.5, f"macro_f1 should reveal weakness: {macro_f1.value}"
    assert bal_acc.value < 0.5, f"balanced_accuracy should reveal weakness: {bal_acc.value}"


# ---------------------------------------------------------------------------
# Single-class input safety
# ---------------------------------------------------------------------------


def test_ordinal_single_class_does_not_crash() -> None:
    """All y_true the same class — must not crash, macro_f1/balanced_acc must be NA."""
    y_true = np.zeros(20, dtype=float)
    logits = _make_logits_from_classes(np.zeros(20, dtype=int), num_classes=3)
    results = compute_ordinal_metrics(y_true, logits, "dr_grade")
    assert len(results) >= 1, "Must return at least one result"
    names = {r.metric_name for r in results}
    assert "accuracy" in names
    for r in results:
        if r.metric_name in ("macro_f1", "balanced_accuracy"):
            assert r.status == MetricStatus.NA
            assert r.reason == "single_class_ordinal"


def test_ordinal_single_class_accuracy_still_ok() -> None:
    y_true = np.zeros(20, dtype=float)
    logits = _make_logits_from_classes(np.zeros(20, dtype=int), num_classes=3)
    results = compute_ordinal_metrics(y_true, logits, "dr_grade")
    acc = next((r for r in results if r.metric_name == "accuracy"), None)
    assert acc is not None
    assert acc.status == MetricStatus.OK


# ---------------------------------------------------------------------------
# Per-class support
# ---------------------------------------------------------------------------


def test_ordinal_per_class_support_correct() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 2, 0, 1, 2, 2])
    y_pred = _make_logits_from_classes(y_true, num_classes=3)
    results = compute_ordinal_metrics(y_true, y_pred, "dr_grade")
    acc = next(r for r in results if r.metric_name == "accuracy")
    assert acc.per_class_support is not None
    assert acc.per_class_support[0] == 4  # four class-0
    assert acc.per_class_support[1] == 3  # three class-1
    assert acc.per_class_support[2] == 3  # three class-2


def test_ordinal_per_class_support_none_for_na_metrics() -> None:
    """NA metrics (macro_f1, balanced_accuracy for single-class) must have per_class_support=None."""
    y_true = np.zeros(20, dtype=float)
    logits = _make_logits_from_classes(np.zeros(20, dtype=int), num_classes=3)
    results = compute_ordinal_metrics(y_true, logits, "dr_grade")
    for r in results:
        if r.metric_name in ("macro_f1", "balanced_accuracy"):
            assert r.per_class_support is None


# ---------------------------------------------------------------------------
# Sparse subgroup unchanged
# ---------------------------------------------------------------------------


def test_ordinal_sparse_subgroup_returns_one_na() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 0.0])  # n=4 < min_n=5
    logits = _make_logits_from_classes(np.array([0, 1, 2, 0]), num_classes=3)
    results = compute_ordinal_metrics(y_true, logits, "dr_grade")
    assert len(results) == 1
    assert results[0].status == MetricStatus.NA
    assert results[0].reason == "sparse_subgroup"


# ---------------------------------------------------------------------------
# Binary AUROC unaffected
# ---------------------------------------------------------------------------


def test_binary_auroc_unaffected_by_changes() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
                       0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                       0, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=float)
    y_score = np.linspace(0, 1, 30)
    results = compute_binary_metrics(y_true, y_score)
    assert len(results) == 1
    assert results[0].metric_name == "auroc"
    assert results[0].status == MetricStatus.OK
    assert 0.0 <= results[0].value <= 1.0


def test_binary_per_class_support_is_none() -> None:
    y_true = np.array([0] * 15 + [1] * 15, dtype=float)
    y_score = np.linspace(0, 1, 30)
    results = compute_binary_metrics(y_true, y_score)
    assert results[0].per_class_support is None
