from __future__ import annotations
import math
import pytest
from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.data import (
    MISSING_CLASS_PLACEHOLDER, MISSING_REGRESSION_PLACEHOLDER,
    build_metadata_features, build_task_targets_and_masks, encode_task_target,
)
from retina_screen.feature_policy import FeaturePolicy, ModelInputMode
from retina_screen.schema import CanonicalSample, Sex
from retina_screen.tasks import TASK_REGISTRY


def _make(sample_id="s", **kwargs):
    base = dict(
        sample_id=sample_id, patient_id="P001",
        dataset_source="test", image_path=f"test://{sample_id}",
    )
    base.update(kwargs)
    return CanonicalSample(**base)


@pytest.fixture(scope="module")
def policy():
    return FeaturePolicy()


@pytest.fixture(scope="module")
def dummy_manifest():
    return DummyAdapter(n_patients=20).build_manifest()


# ---------------------------------------------------------------------------
# Missing binary label
# ---------------------------------------------------------------------------

def test_missing_binary_label_produces_mask_zero():
    enc = encode_task_target(_make(glaucoma=None), "glaucoma")
    assert enc.mask == 0.0


def test_missing_binary_label_not_class_zero():
    enc = encode_task_target(_make(glaucoma=None), "glaucoma")
    assert enc.value != 0.0, "Missing label was encoded as 0.0 (the negative class)"


def test_missing_binary_placeholder_is_minus_one():
    enc = encode_task_target(_make(diabetes=None), "diabetes")
    assert enc.value == MISSING_CLASS_PLACEHOLDER == -1.0


# ---------------------------------------------------------------------------
# Observed binary labels
# ---------------------------------------------------------------------------

def test_observed_binary_zero_target_zero_mask_one():
    enc = encode_task_target(_make(glaucoma=0), "glaucoma")
    assert enc.value == 0.0 and enc.mask == 1.0


def test_observed_binary_one_target_one_mask_one():
    enc = encode_task_target(_make(cataract=1), "cataract")
    assert enc.value == 1.0 and enc.mask == 1.0


# ---------------------------------------------------------------------------
# Ordinal (dr_grade 0-4)
# ---------------------------------------------------------------------------

def test_missing_ordinal_mask_zero():
    enc = encode_task_target(_make(dr_grade=None), "dr_grade")
    assert enc.mask == 0.0


def test_missing_ordinal_placeholder_not_valid_class():
    enc = encode_task_target(_make(dr_grade=None), "dr_grade")
    assert enc.value not in range(5)


def test_observed_ordinal_mask_one():
    enc = encode_task_target(_make(dr_grade=2), "dr_grade")
    assert enc.value == 2.0 and enc.mask == 1.0


# ---------------------------------------------------------------------------
# Regression (retinal_age)
# ---------------------------------------------------------------------------

def test_missing_regression_mask_zero_nan():
    enc = encode_task_target(_make(retinal_age=None), "retinal_age")
    assert enc.mask == 0.0
    assert math.isnan(enc.value), f"Expected NaN placeholder, got {enc.value!r}"


def test_observed_regression_mask_one():
    enc = encode_task_target(_make(retinal_age=45.0), "retinal_age")
    assert enc.mask == 1.0 and enc.value == 45.0


# ---------------------------------------------------------------------------
# cardiovascular_composite — continuous float regression
# ---------------------------------------------------------------------------

def test_cardiovascular_missing_mask_zero():
    enc = encode_task_target(_make(cardiovascular_composite=None), "cardiovascular_composite")
    assert enc.mask == 0.0


def test_cardiovascular_observed_treated_as_regression():
    enc = encode_task_target(_make(cardiovascular_composite=0.75), "cardiovascular_composite")
    assert enc.mask == 1.0 and enc.value == 0.75


# ---------------------------------------------------------------------------
# Sex — uses TaskDefinition.target_encoding, not hardcoded logic
# ---------------------------------------------------------------------------

def test_sex_female_uses_target_encoding():
    enc = encode_task_target(_make(sex=Sex.FEMALE), "sex")
    expected = TASK_REGISTRY["sex"].target_encoding[Sex.FEMALE]
    assert enc.value == expected and enc.mask == 1.0


def test_sex_male_uses_target_encoding():
    enc = encode_task_target(_make(sex=Sex.MALE), "sex")
    expected = TASK_REGISTRY["sex"].target_encoding[Sex.MALE]
    assert enc.value == expected and enc.mask == 1.0


def test_sex_unknown_unmapped_mask_zero():
    enc = encode_task_target(_make(sex=Sex.UNKNOWN), "sex")
    assert enc.mask == 0.0, "Sex.UNKNOWN is not in target_encoding; must produce mask=0"


def test_sex_none_mask_zero():
    enc = encode_task_target(_make(sex=None), "sex")
    assert enc.mask == 0.0


# ---------------------------------------------------------------------------
# Batch targets / masks
# ---------------------------------------------------------------------------

def test_batch_masks_length_matches_samples(dummy_manifest):
    batch = build_task_targets_and_masks(dummy_manifest, ["glaucoma", "dr_grade"])
    for tn in ("glaucoma", "dr_grade"):
        assert len(batch.masks[tn]) == len(dummy_manifest)
        assert len(batch.targets[tn]) == len(dummy_manifest)


def test_batch_mask_values_are_zero_or_one(dummy_manifest):
    batch = build_task_targets_and_masks(dummy_manifest, ["glaucoma"])
    for m in batch.masks["glaucoma"]:
        assert m in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Masked loss contract
# ---------------------------------------------------------------------------

def test_masked_loss_zero_for_missing():
    enc = encode_task_target(_make(glaucoma=None), "glaucoma")
    assert enc.mask == 0.0
    assert 1.0 * enc.mask == 0.0, "raw_loss * mask must be 0 when label is missing"


# ---------------------------------------------------------------------------
# FeaturePolicy applied before metadata exposure
# ---------------------------------------------------------------------------

def test_image_only_returns_no_metadata(policy, dummy_manifest):
    s = dummy_manifest[0]
    meta = build_metadata_features(s, "glaucoma", policy, "image_only")
    assert meta.allowed_fields == frozenset()
    assert meta.values == {}
    assert meta.observation_mask == {}


def test_image_plus_metadata_returns_allowed_fields(policy, dummy_manifest):
    s = dummy_manifest[0]
    meta = build_metadata_features(s, "glaucoma", policy, "image_plus_metadata")
    assert len(meta.allowed_fields) > 0


def test_age_years_blocked_for_retinal_age(policy):
    s = _make(age_years=45.0, retinal_age=None)
    meta = build_metadata_features(s, "retinal_age", policy, "image_plus_metadata")
    assert "age_years" not in meta.allowed_fields


def test_sex_field_blocked_for_sex_task(policy):
    s = _make(sex=Sex.FEMALE)
    meta = build_metadata_features(s, "sex", policy, "image_plus_metadata")
    assert "sex" not in meta.allowed_fields


def test_dataset_source_blocked_by_default(policy, dummy_manifest):
    s = dummy_manifest[0]
    meta = build_metadata_features(s, "glaucoma", policy, "image_plus_metadata")
    assert "dataset_source" not in meta.allowed_fields


def test_camera_type_blocked_by_default(policy, dummy_manifest):
    s = dummy_manifest[0]
    meta = build_metadata_features(s, "glaucoma", policy, "image_plus_metadata")
    assert "camera_type" not in meta.allowed_fields


# ---------------------------------------------------------------------------
# observation_mask is independent of FeaturePolicy
# ---------------------------------------------------------------------------

def test_observation_mask_zero_for_allowed_but_none_field(policy):
    s = _make(age_years=None)
    meta = build_metadata_features(s, "glaucoma", policy, "image_plus_metadata")
    if "age_years" in meta.allowed_fields:
        assert meta.observation_mask["age_years"] == 0.0, (
            "Allowed but None field must have observation_mask=0.0"
        )


def test_observation_mask_one_for_observed_allowed_field(policy):
    s = _make(age_years=40.0)
    meta = build_metadata_features(s, "glaucoma", policy, "image_plus_metadata")
    if "age_years" in meta.allowed_fields:
        assert meta.observation_mask["age_years"] == 1.0


# ---------------------------------------------------------------------------
# Unknown task fails closed
# ---------------------------------------------------------------------------

def test_unknown_task_encode_raises_key_error():
    with pytest.raises(KeyError):
        encode_task_target(_make(), "totally_unknown_task_xyz_000")


def test_unknown_task_in_metadata_raises(policy):
    with pytest.raises((KeyError, ValueError)):
        build_metadata_features(_make(), "totally_unknown_task_xyz_000", policy, "image_plus_metadata")
