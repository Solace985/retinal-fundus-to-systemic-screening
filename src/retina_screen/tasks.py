"""
tasks.py -- Task registry: definitions, types, and lookup helpers.

Owns: task type definitions, TaskDefinition dataclass, task registry dict,
task lookup helpers, and task metadata needed by model/training/evaluation.

Must not contain: native dataset column names, dataset-specific conditionals,
training loop code, or model head implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from types import MappingProxyType

from retina_screen.schema import CanonicalSample, Sex


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """High-level category of a prediction task."""

    BINARY = "binary"
    ORDINAL = "ordinal"
    REGRESSION = "regression"


class LossType(str, Enum):
    """Loss function identifier used by training.py."""

    BCE = "bce"
    CE = "ce"
    MSE = "mse"


class MetricType(str, Enum):
    """Primary evaluation metric identifier used by evaluation.py."""

    AUROC = "auroc"
    QUADRATIC_KAPPA = "quadratic_kappa"
    MAE = "mae"
    ACCURACY = "accuracy"


class LabelQuality(str, Enum):
    """Conservative assessment of label quality for a task.

    Adapters and configs may annotate scenario-specific quality on top of this
    conservative default.
    """

    HIGH = "high"
    PROXY = "proxy"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# TaskDefinition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable descriptor for a single prediction task.

    name:
        Unique task identifier; must match its key in TASK_REGISTRY.
    task_type:
        Category of prediction (binary / ordinal / regression).
    target_column:
        The field name in CanonicalSample that this task predicts.
    loss:
        Loss function to use during training.
    primary_metric:
        Primary evaluation metric for this task.
    label_quality:
        Conservative default label-quality assessment.
    allowed_as_headline:
        Whether this task may appear in headline paper results.
        claim_mode.yaml and dataset scenario still gate actual reporting.
    num_classes:
        Required for ORDINAL tasks; None for BINARY and REGRESSION.
    target_encoding:
        Optional mapping from canonical non-numeric target values to tensor labels.
        Values absent from the mapping should be treated as missing/masked downstream.
    """

    name: str
    task_type: TaskType
    target_column: str
    loss: LossType
    primary_metric: MetricType
    label_quality: LabelQuality
    allowed_as_headline: bool
    num_classes: Optional[int] = None
    target_encoding: Mapping[object, float] | None = None


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, TaskDefinition] = {
    # --- DR grading (ordinal 0-4) ---
    "dr_grade": TaskDefinition(
        name="dr_grade",
        task_type=TaskType.ORDINAL,
        target_column="dr_grade",
        loss=LossType.CE,
        primary_metric=MetricType.QUADRATIC_KAPPA,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
        num_classes=5,
    ),
    # --- Ocular conditions ---
    "glaucoma": TaskDefinition(
        name="glaucoma",
        task_type=TaskType.BINARY,
        target_column="glaucoma",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "cataract": TaskDefinition(
        name="cataract",
        task_type=TaskType.BINARY,
        target_column="cataract",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "amd": TaskDefinition(
        name="amd",
        task_type=TaskType.BINARY,
        target_column="amd",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "pathological_myopia": TaskDefinition(
        name="pathological_myopia",
        task_type=TaskType.BINARY,
        target_column="pathological_myopia",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "hypertensive_retinopathy": TaskDefinition(
        name="hypertensive_retinopathy",
        task_type=TaskType.BINARY,
        target_column="hypertensive_retinopathy",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "drusen": TaskDefinition(
        name="drusen",
        task_type=TaskType.BINARY,
        target_column="drusen",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=True,
    ),
    "other_ocular": TaskDefinition(
        name="other_ocular",
        task_type=TaskType.BINARY,
        target_column="other_ocular",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=False,
    ),
    # --- Systemic conditions (proxy by default; adapters/configs refine) ---
    "diabetes": TaskDefinition(
        name="diabetes",
        task_type=TaskType.BINARY,
        target_column="diabetes",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.PROXY,
        allowed_as_headline=False,
    ),
    "hypertension": TaskDefinition(
        name="hypertension",
        task_type=TaskType.BINARY,
        target_column="hypertension",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.PROXY,
        allowed_as_headline=False,
    ),
    "smoking": TaskDefinition(
        name="smoking",
        task_type=TaskType.BINARY,
        target_column="smoking",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.UNKNOWN,
        allowed_as_headline=False,
    ),
    "obesity": TaskDefinition(
        name="obesity",
        task_type=TaskType.BINARY,
        target_column="obesity",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.UNKNOWN,
        allowed_as_headline=False,
    ),
    "insulin_use": TaskDefinition(
        name="insulin_use",
        task_type=TaskType.BINARY,
        target_column="insulin_use",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.UNKNOWN,
        allowed_as_headline=False,
    ),
    "cardiovascular_composite": TaskDefinition(
        name="cardiovascular_composite",
        task_type=TaskType.REGRESSION,
        target_column="cardiovascular_composite",
        loss=LossType.MSE,
        primary_metric=MetricType.MAE,
        label_quality=LabelQuality.PROXY,
        allowed_as_headline=False,
    ),
    # --- Regression targets ---
    "retinal_age": TaskDefinition(
        name="retinal_age",
        task_type=TaskType.REGRESSION,
        target_column="retinal_age",
        loss=LossType.MSE,
        primary_metric=MetricType.MAE,
        label_quality=LabelQuality.UNKNOWN,
        # Can be headline with BRSET; secondary under ODIR-only.
        # claim_mode.yaml gates actual headline reporting per scenario.
        allowed_as_headline=True,
    ),
    "diabetes_duration_years": TaskDefinition(
        name="diabetes_duration_years",
        task_type=TaskType.REGRESSION,
        target_column="diabetes_duration_years",
        loss=LossType.MSE,
        primary_metric=MetricType.MAE,
        label_quality=LabelQuality.UNKNOWN,
        allowed_as_headline=False,
    ),
    # --- Fairness probe (not a clinical target) ---
    "sex": TaskDefinition(
        name="sex",
        task_type=TaskType.BINARY,
        target_column="sex",
        loss=LossType.BCE,
        primary_metric=MetricType.AUROC,
        label_quality=LabelQuality.HIGH,
        allowed_as_headline=False,
        target_encoding=MappingProxyType({Sex.FEMALE: 0.0, Sex.MALE: 1.0}),
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_task(name: str) -> TaskDefinition:
    """Return the TaskDefinition for *name*; raises KeyError if not found."""
    try:
        return TASK_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Task {name!r} not found in TASK_REGISTRY. "
            f"Available tasks: {sorted(TASK_REGISTRY)}"
        ) from None


def get_tasks_by_type(task_type: TaskType) -> list[TaskDefinition]:
    """Return all registered tasks of the given *task_type*."""
    return [t for t in TASK_REGISTRY.values() if t.task_type == task_type]


def get_headline_tasks() -> list[TaskDefinition]:
    """Return tasks that are eligible to appear in headline paper results."""
    return [t for t in TASK_REGISTRY.values() if t.allowed_as_headline]


# ---------------------------------------------------------------------------
# Registry self-validation (runs at import time to catch schema drift)
# ---------------------------------------------------------------------------


def _validate_registry() -> None:
    """Verify every task target_column exists in CanonicalSample."""
    schema_fields = set(CanonicalSample.model_fields.keys())
    errors: list[str] = []
    for name, task in TASK_REGISTRY.items():
        if name != task.name:
            errors.append(f"Task key {name!r} != task.name {task.name!r}")
        if task.target_column not in schema_fields:
            errors.append(
                f"Task {name!r}: target_column {task.target_column!r} "
                f"not in CanonicalSample"
            )
        if task.task_type == TaskType.ORDINAL and task.num_classes is None:
            errors.append(f"Ordinal task {name!r} is missing num_classes")
        if task.target_encoding is not None:
            non_numeric_values = [
                value
                for value in task.target_encoding.values()
                if not isinstance(value, (int, float))
            ]
            if non_numeric_values:
                errors.append(
                    f"Task {name!r}: target_encoding values must be numeric, "
                    f"got {non_numeric_values!r}"
                )
    if errors:
        raise RuntimeError(
            "TASK_REGISTRY validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


_validate_registry()
