from __future__ import annotations
from collections import defaultdict
import pytest
from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.schema import CanonicalSample
from retina_screen.splitting import (
    DEFAULT_SPLIT_RATIOS, SplitName,
    assert_no_patient_overlap, split_patients,
)

@pytest.fixture(scope="module")
def dummy_manifest():
    return DummyAdapter(n_patients=20).build_manifest()

@pytest.fixture(scope="module")
def split(dummy_manifest):
    return split_patients(dummy_manifest, seed=42)

def _tiny(n):
    return [
        CanonicalSample(
            sample_id=f"s{i:04d}", patient_id=f"P{i:04d}",
            dataset_source="test", image_path=f"test://s{i:04d}",
        )
        for i in range(n)
    ]

def test_split_names_are_exactly_four(split):
    assert set(split.keys()) == {"train", "val", "reliability", "test"}

def test_every_sample_assigned_exactly_once(split, dummy_manifest):
    all_ids = {s.sample_id for s in dummy_manifest}
    assigned = [sid for sids in split.values() for sid in sids]
    assert sorted(assigned) == sorted(all_ids)

def test_all_splits_nonempty(split):
    for name, sids in split.items():
        assert len(sids) > 0, f"Split {name!r} is empty"

def test_patient_level_grouping(split, dummy_manifest):
    sid_to_pid = {s.sample_id: s.patient_id for s in dummy_manifest}
    pid_to_split = {}
    for split_name, sids in split.items():
        for sid in sids:
            pid = sid_to_pid[sid]
            if pid in pid_to_split:
                assert pid_to_split[pid] == split_name, (
                    f"Patient {pid!r} is split: {pid_to_split[pid]!r} and {split_name!r}"
                )
            pid_to_split[pid] = split_name

def test_no_patient_overlap_across_all_six_pairs(split, dummy_manifest):
    assert_no_patient_overlap(split, dummy_manifest)

def test_ratios_approximately_correct(split, dummy_manifest):
    total = len(dummy_manifest)
    for split_name, sids in split.items():
        ratio = len(sids) / total
        expected = DEFAULT_SPLIT_RATIOS[split_name]
        assert abs(ratio - expected) < 0.15, (
            f"Split {split_name!r}: got {ratio:.3f}, expected ~{expected:.2f}"
        )

def test_deterministic_with_same_seed(dummy_manifest):
    s1 = split_patients(dummy_manifest, seed=42)
    s2 = split_patients(dummy_manifest, seed=42)
    assert s1 == s2

def test_different_seed_produces_different_assignment(dummy_manifest):
    s1 = split_patients(dummy_manifest, seed=42)
    s2 = split_patients(dummy_manifest, seed=1337)
    assert set(s1["train"]) != set(s2["train"]), (
        "Seeds 42 and 1337 with 20 patients must differ"
    )

def test_invalid_ratios_raise_value_error(dummy_manifest):
    bad = {"train": 0.50, "val": 0.20, "reliability": 0.20, "test": 0.20}
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_patients(dummy_manifest, ratios=bad)

def test_missing_reliability_ratio_key_raises(dummy_manifest):
    bad = {"train": 0.60, "val": 0.15, "test": 0.25}
    with pytest.raises(ValueError, match="missing.*reliability"):
        split_patients(dummy_manifest, ratios=bad)

def test_extra_ratio_key_raises(dummy_manifest):
    bad = {
        "train": 0.55,
        "val": 0.15,
        "reliability": 0.15,
        "test": 0.10,
        "junk": 0.05,
    }
    with pytest.raises(ValueError, match="unsupported.*junk"):
        split_patients(dummy_manifest, ratios=bad)

def test_blank_patient_id_raises_value_error():
    # Schema now validates patient_id is non-blank, so we use model_construct
    # to bypass schema validation and test split_patients' own guard.
    manifest = _tiny(5)
    broken = CanonicalSample.model_construct(
        sample_id="bad", patient_id="   ",
        dataset_source="test", image_path="test://bad",
    )
    with pytest.raises(ValueError, match="blank"):
        split_patients([*manifest, broken])

def test_too_few_patients_raises_value_error():
    with pytest.raises(ValueError):
        split_patients(_tiny(4))

def test_invalid_group_on_field_raises(dummy_manifest):
    with pytest.raises(ValueError, match="group_on"):
        split_patients(dummy_manifest, group_on="nonexistent_field_xyz")

def test_group_on_sample_id_is_rejected(dummy_manifest):
    with pytest.raises(ValueError, match="patient-level.*group_on"):
        split_patients(dummy_manifest, group_on="sample_id")

def test_group_on_image_path_is_rejected(dummy_manifest):
    with pytest.raises(ValueError, match="patient-level.*group_on"):
        split_patients(dummy_manifest, group_on="image_path")

def test_invalid_stratify_on_field_raises(dummy_manifest):
    with pytest.raises(ValueError, match="stratify_on"):
        split_patients(dummy_manifest, stratify_on="nonexistent_field_xyz")

def test_custom_ratios_are_patient_level():
    manifest = _tiny(20)
    custom = {"train": 0.70, "val": 0.10, "reliability": 0.10, "test": 0.10}
    sp = split_patients(manifest, ratios=custom)
    assert set(sp) == {"train", "val", "reliability", "test"}
    sid_to_pid = {s.sample_id: s.patient_id for s in manifest}
    pid_to_split = {}
    for split_name, sids in sp.items():
        for sid in sids:
            pid = sid_to_pid[sid]
            if pid in pid_to_split:
                assert pid_to_split[pid] == split_name
            pid_to_split[pid] = split_name

def test_splitname_enum_ratio_keys_are_supported():
    manifest = _tiny(20)
    custom = {
        SplitName.TRAIN: 0.70,
        SplitName.VAL: 0.10,
        SplitName.RELIABILITY: 0.10,
        SplitName.TEST: 0.10,
    }
    sp = split_patients(manifest, ratios=custom)
    assert set(sp) == {"train", "val", "reliability", "test"}

def test_stratify_on_does_not_break_patient_grouping(dummy_manifest):
    sp = split_patients(dummy_manifest, stratify_on="sex", seed=42)
    sid_to_pid = {s.sample_id: s.patient_id for s in dummy_manifest}
    pid_to_split = {}
    for split_name, sids in sp.items():
        for sid in sids:
            pid = sid_to_pid[sid]
            if pid in pid_to_split:
                assert pid_to_split[pid] == split_name, (
                    f"Stratification broke patient grouping for {pid!r}"
                )
            pid_to_split[pid] = split_name
