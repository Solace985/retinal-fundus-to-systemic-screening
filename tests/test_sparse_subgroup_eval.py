from __future__ import annotations
import math
import numpy as np
import pytest
from retina_screen.evaluation import (
    MetricResult, MetricStatus,
    compute_binary_metrics, compute_ordinal_metrics, compute_regression_metrics,
    evaluate_predictions, evaluate_subgroups,
)
from retina_screen.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Binary sparse-subgroup safety
# ---------------------------------------------------------------------------

def test_n_less_than_30_returns_na():
    y_true = np.array([0, 1, 0, 1, 0] * 5)   # n=25 < 30
    y_score = np.random.default_rng(0).random(len(y_true))
    results = compute_binary_metrics(y_true, y_score, min_n=30)
    assert len(results) == 1
    r = results[0]
    assert r.status == MetricStatus.NA
    assert r.reason == "sparse_subgroup"

def test_positives_less_than_5_returns_na():
    y_true = np.zeros(40)
    y_true[:3] = 1    # only 3 positives
    y_score = np.random.default_rng(1).random(40)
    results = compute_binary_metrics(y_true, y_score, min_n=30, min_pos=5)
    assert results[0].status == MetricStatus.NA
    assert results[0].reason == "insufficient_class_counts"

def test_negatives_less_than_5_returns_na():
    y_true = np.ones(40)
    y_true[:3] = 0    # only 3 negatives
    y_score = np.random.default_rng(2).random(40)
    results = compute_binary_metrics(y_true, y_score, min_n=30, min_neg=5)
    assert results[0].status == MetricStatus.NA
    assert results[0].reason == "insufficient_class_counts"

def test_single_class_returns_na_no_auc_crash():
    y_true = np.ones(50)   # all positives, single class
    y_score = np.random.default_rng(3).random(50)
    results = compute_binary_metrics(y_true, y_score)
    assert results[0].status == MetricStatus.NA
    assert results[0].reason == "single_class_subgroup"

def test_valid_binary_computes_ok():
    rng = np.random.default_rng(4)
    y_true = np.array([0] * 25 + [1] * 25)   # balanced, n=50
    y_score = rng.random(50)
    results = compute_binary_metrics(y_true, y_score, min_n=30, min_pos=5, min_neg=5)
    assert len(results) == 1
    r = results[0]
    assert r.status == MetricStatus.OK
    assert r.value is not None
    assert 0.0 <= r.value <= 1.0

def test_na_result_has_reason():
    y_true = np.zeros(5)
    y_score = np.zeros(5)
    results = compute_binary_metrics(y_true, y_score)
    assert results[0].status == MetricStatus.NA
    assert isinstance(results[0].reason, str)
    assert len(results[0].reason) > 0

def test_metric_result_has_n_pos_neg():
    y_true = np.zeros(5)
    y_score = np.zeros(5)
    results = compute_binary_metrics(y_true, y_score)
    r = results[0]
    assert isinstance(r.n, int)
    assert isinstance(r.positives, int)
    assert isinstance(r.negatives, int)

# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def test_regression_small_n_returns_na():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    results = compute_regression_metrics(y_true, y_pred, min_n=5)
    assert results[0].status == MetricStatus.NA
    assert results[0].reason == "sparse_subgroup"

def test_regression_valid_computes_ok():
    rng = np.random.default_rng(5)
    y_true = rng.random(20)
    y_pred = y_true + rng.normal(0, 0.1, 20)
    results = compute_regression_metrics(y_true, y_pred, min_n=5)
    assert results[0].status == MetricStatus.OK
    assert results[0].value is not None
    assert results[0].value >= 0.0

def test_regression_ignores_nan_targets():
    # NaN targets must be excluded before calling compute_regression_metrics.
    # evaluate_predictions does this; we test that valid rows compute correctly.
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    results = compute_regression_metrics(y_true, y_pred, min_n=5)
    assert results[0].status == MetricStatus.OK
    assert not math.isnan(results[0].value)

# ---------------------------------------------------------------------------
# Ordinal metrics
# ---------------------------------------------------------------------------

def test_ordinal_small_n_returns_na():
    y_true = np.array([0, 1, 2])
    logits = np.zeros((3, 5))
    results = compute_ordinal_metrics(y_true, logits, "dr_grade", min_n=5)
    assert results[0].status == MetricStatus.NA

def test_ordinal_valid_computes_ok():
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 5, size=20)
    logits = rng.random((20, 5))
    results = compute_ordinal_metrics(y_true, logits, "dr_grade", min_n=5)
    assert results[0].status == MetricStatus.OK
    assert 0.0 <= results[0].value <= 1.0

# ---------------------------------------------------------------------------
# evaluate_predictions — overall metrics (not subgroups)
# ---------------------------------------------------------------------------

def test_evaluate_predictions_nonempty():
    rng = np.random.default_rng(7)
    n = 60
    # dr_grade: always observed, ordinal
    dr_true = list(rng.integers(0, 5, n).astype(float))
    dr_mask = [1.0] * n
    dr_pred = rng.random((n, 5))
    results = evaluate_predictions(
        predictions={"dr_grade": dr_pred},
        targets={"dr_grade": dr_true},
        masks={"dr_grade": dr_mask},
        task_names=["dr_grade"],
        min_n_other=5,
    )
    assert "dr_grade" in results
    assert len(results["dr_grade"]) > 0

def test_evaluate_predictions_sparse_subgroup_does_not_crash():
    rng = np.random.default_rng(8)
    n = 3   # tiny — well below any threshold
    results = evaluate_predictions(
        predictions={"glaucoma": rng.random(n)},
        targets={"glaucoma": list(rng.integers(0, 2, n).astype(float))},
        masks={"glaucoma": [1.0] * n},
        task_names=["glaucoma"],
        min_n_binary=30,
    )
    assert "glaucoma" in results
    assert results["glaucoma"][0].status == MetricStatus.NA


# ---------------------------------------------------------------------------
# evaluate_subgroups
# ---------------------------------------------------------------------------


def _evaluate_binary_subgroups(subgroup_labels: np.ndarray):
    n = len(subgroup_labels)
    return evaluate_subgroups(
        predictions={"glaucoma": np.linspace(-1.0, 1.0, n)},
        targets={"glaucoma": [float(i % 2) for i in range(n)]},
        masks={"glaucoma": [1.0] * n},
        task_names=["glaucoma"],
        subgroup_labels=subgroup_labels,
        min_n_binary=30,
    )


def test_evaluate_subgroups_skips_numeric_nan_labels():
    labels = np.array([1.0, np.nan, 1.0, 2.0, np.nan, 2.0])

    results = _evaluate_binary_subgroups(labels)

    assert set(results) == {"1.0", "2.0"}
    assert "nan" not in results


def test_evaluate_subgroups_skips_none_labels():
    labels = np.array(["a", None, "a", "b", None, "b"], dtype=object)

    results = _evaluate_binary_subgroups(labels)

    assert set(results) == {"a", "b"}
    assert "None" not in results


def test_evaluate_subgroups_returns_na_reason_for_sparse_cells():
    labels = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)

    results = _evaluate_binary_subgroups(labels)

    for group_result in results.values():
        metric = group_result["glaucoma"][0]
        assert metric.status == MetricStatus.NA
        assert metric.reason == "sparse_subgroup"


def test_evaluate_subgroups_does_not_crash_on_mixed_labels():
    labels = np.array(["a", 1, np.nan, None, "a", 1], dtype=object)

    results = _evaluate_binary_subgroups(labels)

    assert set(results) == {"a", "1"}


def test_evaluate_subgroups_returns_structured_metric_results():
    labels = np.array(["a", "a", np.nan, None, "b", "b"], dtype=object)

    results = _evaluate_binary_subgroups(labels)
    metric = results["a"]["glaucoma"][0]

    assert isinstance(metric, MetricResult)
    assert metric.status == MetricStatus.NA
    assert metric.value is None
    assert metric.reason
