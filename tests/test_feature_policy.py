"""
tests/test_feature_policy.py -- Verify FeaturePolicy leakage-prevention contracts.
"""

from __future__ import annotations
import pytest
from retina_screen.feature_policy import FeaturePolicy, ModelInputMode
from retina_screen.schema import CANONICAL_METADATA_FIELDS


@pytest.fixture
def policy() -> FeaturePolicy:
    return FeaturePolicy()


def test_image_only_returns_empty_set(policy):
    result = policy.allowed_fields("glaucoma", ModelInputMode.IMAGE_ONLY)
    assert result == frozenset(), f"image_only must allow no metadata, got {result}"


def test_image_only_string_mode(policy):
    assert policy.allowed_fields("glaucoma", "image_only") == frozenset()


def test_image_only_with_explicit_allow_still_returns_empty(policy):
    result = policy.allowed_fields("glaucoma", "image_only", explicit_allow=frozenset({"dataset_source"}))
    assert result == frozenset()


def test_image_only_with_invalid_explicit_allow_raises(policy):
    with pytest.raises(ValueError, match="unrecognised metadata fields"):
        policy.allowed_fields(
            "glaucoma",
            "image_only",
            explicit_allow=frozenset({"totally_fake_column_xyz"}),
        )


def test_retinal_age_blocks_age_years_image_plus_metadata(policy):
    result = policy.allowed_fields("retinal_age", "image_plus_metadata")
    assert "age_years" not in result, "age_years must be blocked for retinal_age (leakage)"


def test_retinal_age_blocks_age_years_clinical_deployment(policy):
    assert "age_years" not in policy.allowed_fields("retinal_age", "clinical_deployment")


def test_retinal_age_blocks_age_years_fairness_ablation(policy):
    assert "age_years" not in policy.allowed_fields("retinal_age", "fairness_ablation")


def test_sex_task_blocks_sex_field_image_plus_metadata(policy):
    result = policy.allowed_fields("sex", "image_plus_metadata")
    assert "sex" not in result, "sex field must be blocked for sex prediction task"


def test_sex_task_blocks_sex_field_clinical_deployment(policy):
    assert "sex" not in policy.allowed_fields("sex", "clinical_deployment")


def test_dataset_source_blocked_by_default(policy):
    result = policy.allowed_fields("glaucoma", "image_plus_metadata")
    assert "dataset_source" not in result, "dataset_source must be blocked by default"


def test_camera_type_blocked_by_default(policy):
    result = policy.allowed_fields("glaucoma", "image_plus_metadata")
    assert "camera_type" not in result, "camera_type must be blocked by default"


def test_dataset_source_blocked_in_clinical_deployment(policy):
    assert "dataset_source" not in policy.allowed_fields("glaucoma", "clinical_deployment")


def test_camera_type_blocked_in_fairness_ablation(policy):
    assert "camera_type" not in policy.allowed_fields("glaucoma", "fairness_ablation")


def test_dataset_source_allowed_with_explicit_permission(policy):
    result = policy.allowed_fields("glaucoma", "image_plus_metadata", explicit_allow=frozenset({"dataset_source"}))
    assert "dataset_source" in result


def test_camera_type_allowed_with_explicit_permission(policy):
    result = policy.allowed_fields("glaucoma", "image_plus_metadata", explicit_allow=frozenset({"camera_type"}))
    assert "camera_type" in result


def test_explicit_allow_does_not_bypass_task_leakage_block(policy):
    result = policy.allowed_fields("retinal_age", "image_plus_metadata", explicit_allow=frozenset({"age_years"}))
    assert "age_years" not in result, "Explicit allow must not bypass age_years->retinal_age block"


def test_unknown_task_raises_value_error(policy):
    with pytest.raises(ValueError, match="Unknown task"):
        policy.allowed_fields("nonexistent_task_xyz", "image_plus_metadata")


def test_unknown_mode_raises_value_error(policy):
    with pytest.raises(ValueError, match="Unknown mode"):
        policy.allowed_fields("glaucoma", "nonexistent_mode_xyz")


def test_unrecognised_field_in_explicit_allow_raises(policy):
    with pytest.raises(ValueError):
        policy.allowed_fields("glaucoma", "image_plus_metadata", explicit_allow=frozenset({"totally_fake_column_xyz"}))


def test_label_field_in_explicit_allow_raises(policy):
    with pytest.raises(ValueError):
        policy.allowed_fields("glaucoma", "image_plus_metadata", explicit_allow=frozenset({"diabetes"}))


def test_image_plus_metadata_includes_age_years_for_non_retinal_age(policy):
    assert "age_years" in policy.allowed_fields("glaucoma", "image_plus_metadata")


def test_image_plus_metadata_includes_eye_laterality(policy):
    assert "eye_laterality" in policy.allowed_fields("glaucoma", "image_plus_metadata")


def test_is_field_allowed_true(policy):
    assert policy.is_field_allowed("age_years", "glaucoma", "image_plus_metadata") is True


def test_is_field_allowed_false_restricted(policy):
    assert policy.is_field_allowed("dataset_source", "glaucoma", "image_plus_metadata") is False


def test_is_field_allowed_false_task_block(policy):
    assert policy.is_field_allowed("age_years", "retinal_age", "image_plus_metadata") is False


def test_restricted_fields_contains_dataset_source():
    assert "dataset_source" in FeaturePolicy.RESTRICTED_FIELDS


def test_restricted_fields_contains_camera_type():
    assert "camera_type" in FeaturePolicy.RESTRICTED_FIELDS
