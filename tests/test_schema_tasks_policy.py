"""
tests/test_schema_tasks_policy.py -- Canonical schema, task registry, and policy contracts.
Stage 2 / Batch 2 gate test.
"""

from __future__ import annotations
import pytest
from pydantic import ValidationError
from retina_screen.schema import (
    CANONICAL_LABEL_FIELDS, CANONICAL_METADATA_FIELDS, CanonicalSample,
    EyeLaterality, Sex, validate_sample,
)
from retina_screen.tasks import (
    TASK_REGISTRY, LabelQuality, LossType, MetricType, TaskDefinition, TaskType, get_task,
)

_REQ = dict(sample_id="s1", patient_id="p1", dataset_source="dummy", image_path="/fake/img.jpg")

def make(**kw):
    return CanonicalSample(**{**_REQ, **kw})


# --- Required fields ---

def test_sample_requires_sample_id():
    with pytest.raises(ValidationError):
        CanonicalSample(patient_id="p1", dataset_source="ds", image_path="/x")

def test_sample_requires_patient_id():
    with pytest.raises(ValidationError):
        CanonicalSample(sample_id="s1", dataset_source="ds", image_path="/x")

def test_sample_requires_dataset_source():
    with pytest.raises(ValidationError):
        CanonicalSample(sample_id="s1", patient_id="p1", image_path="/x")

def test_sample_requires_image_path():
    with pytest.raises(ValidationError):
        CanonicalSample(sample_id="s1", patient_id="p1", dataset_source="ds")

def test_minimal_valid_sample():
    s = make()
    assert s.sample_id == "s1"
    assert s.patient_id == "p1"


@pytest.mark.parametrize("field", ("sample_id", "patient_id", "dataset_source", "image_path"))
def test_required_identifier_fields_reject_blank_strings(field):
    bad = {**_REQ, field: "   "}
    with pytest.raises(ValidationError):
        CanonicalSample(**bad)


# --- Missing values default to None, not 0 ---

def test_binary_labels_default_to_none():
    s = make()
    for f in ("glaucoma","cataract","amd","pathological_myopia","hypertensive_retinopathy",
               "drusen","other_ocular","diabetes","hypertension","smoking","obesity",
               "insulin_use"):
        assert getattr(s, f) is None, f"{f} should default to None, not 0"

def test_regression_labels_default_to_none():
    s = make()
    assert s.cardiovascular_composite is None
    assert s.retinal_age is None
    assert s.diabetes_duration_years is None
    assert s.dr_grade is None

def test_metadata_defaults_to_none():
    s = make()
    assert s.age_years is None
    assert s.sex is None
    assert s.camera_type is None


# --- Binary label validation ---

def test_binary_label_0_valid():
    assert make(diabetes=0).diabetes == 0

def test_binary_label_1_valid():
    assert make(glaucoma=1).glaucoma == 1

def test_binary_label_2_invalid():
    with pytest.raises(ValidationError):
        make(diabetes=2)

def test_binary_label_minus1_invalid():
    with pytest.raises(ValidationError):
        make(glaucoma=-1)


def test_cardiovascular_composite_accepts_probability_range():
    assert make(cardiovascular_composite=0.0).cardiovascular_composite == 0.0
    assert make(cardiovascular_composite=0.5).cardiovascular_composite == 0.5
    assert make(cardiovascular_composite=1.0).cardiovascular_composite == 1.0


@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_cardiovascular_composite_rejects_out_of_range_values(value):
    with pytest.raises(ValidationError):
        make(cardiovascular_composite=value)


# --- DR grade validation ---

def test_dr_grade_0_valid():
    assert make(dr_grade=0).dr_grade == 0

def test_dr_grade_4_valid():
    assert make(dr_grade=4).dr_grade == 4

def test_dr_grade_5_invalid():
    with pytest.raises(ValidationError):
        make(dr_grade=5)

def test_dr_grade_minus1_invalid():
    with pytest.raises(ValidationError):
        make(dr_grade=-1)


# --- Enum fields ---

def test_sex_enum_accepted():
    assert make(sex=Sex.FEMALE).sex == Sex.FEMALE

def test_sex_string_accepted():
    assert make(sex="male").sex == Sex.MALE

def test_invalid_sex_rejected():
    with pytest.raises(ValidationError):
        make(sex="other")

def test_eye_laterality_enum():
    assert make(eye_laterality=EyeLaterality.LEFT).eye_laterality == EyeLaterality.LEFT


# --- Extra fields rejected ---

def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        CanonicalSample(**_REQ, odir_native_column="x")


# --- validate_sample helper ---

def test_validate_sample_returns_instance():
    assert isinstance(validate_sample(_REQ), CanonicalSample)

def test_validate_sample_raises_on_bad_data():
    with pytest.raises(ValidationError):
        validate_sample({**_REQ, "diabetes": 99})


# --- Task registry: target columns in schema ---

def test_all_task_target_columns_in_schema():
    schema_fields = set(CanonicalSample.model_fields.keys())
    for name, task in TASK_REGISTRY.items():
        assert task.target_column in schema_fields, (
            f"Task {name!r}: target_column {task.target_column!r} not in CanonicalSample"
        )


# --- Task registry: required attributes ---

def test_all_tasks_have_required_attributes():
    for name, task in TASK_REGISTRY.items():
        assert isinstance(task, TaskDefinition)
        assert task.name == name
        assert isinstance(task.task_type, TaskType)
        assert task.target_column
        assert task.loss is not None
        assert isinstance(task.primary_metric, MetricType)
        assert isinstance(task.label_quality, LabelQuality)
        assert isinstance(task.allowed_as_headline, bool)
        assert hasattr(task, "target_encoding")


# --- Ordinal tasks have num_classes ---

def test_ordinal_tasks_have_num_classes():
    for name, task in TASK_REGISTRY.items():
        if task.task_type == TaskType.ORDINAL:
            assert task.num_classes is not None, f"Ordinal task {name!r} missing num_classes"
            assert task.num_classes > 1

def test_binary_tasks_no_num_classes():
    for name, task in TASK_REGISTRY.items():
        if task.task_type == TaskType.BINARY:
            assert task.num_classes is None, f"Binary task {name!r} should not have num_classes"

def test_dr_grade_is_ordinal_5_classes():
    dr = TASK_REGISTRY["dr_grade"]
    assert dr.task_type == TaskType.ORDINAL
    assert dr.num_classes == 5


# --- Lookup helpers ---

def test_get_task_returns_definition():
    assert get_task("glaucoma").name == "glaucoma"

def test_get_task_raises_key_error_for_unknown():
    with pytest.raises(KeyError):
        get_task("nonexistent_task_xyz_abc")

def test_task_registry_key_error_direct():
    with pytest.raises(KeyError):
        _ = TASK_REGISTRY["definitely_not_a_task"]


# --- Proxy / headline flags ---

def test_diabetes_is_proxy():
    assert TASK_REGISTRY["diabetes"].label_quality == LabelQuality.PROXY

def test_hypertension_is_proxy():
    assert TASK_REGISTRY["hypertension"].label_quality == LabelQuality.PROXY

def test_sex_task_not_headline():
    assert TASK_REGISTRY["sex"].allowed_as_headline is False

def test_cardiovascular_composite_not_headline():
    assert TASK_REGISTRY["cardiovascular_composite"].allowed_as_headline is False

def test_cardiovascular_composite_is_regression_task():
    task = TASK_REGISTRY["cardiovascular_composite"]
    assert task.task_type == TaskType.REGRESSION
    assert task.loss == LossType.MSE
    assert task.primary_metric == MetricType.MAE

def test_sex_task_has_explicit_target_encoding():
    encoding = TASK_REGISTRY["sex"].target_encoding
    assert encoding is not None
    assert encoding[Sex.FEMALE] == 0.0
    assert encoding[Sex.MALE] == 1.0
    assert Sex.UNKNOWN not in encoding


# --- Field-set constants ---

def test_label_fields_not_empty():
    assert len(CANONICAL_LABEL_FIELDS) > 0

def test_metadata_fields_not_empty():
    assert len(CANONICAL_METADATA_FIELDS) > 0

def test_label_and_metadata_no_overlap():
    overlap = CANONICAL_LABEL_FIELDS & CANONICAL_METADATA_FIELDS
    assert len(overlap) == 0, f"Label/metadata fields overlap: {overlap}"
