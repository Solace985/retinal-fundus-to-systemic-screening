from __future__ import annotations
import json
import pytest
from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.schema import CanonicalSample
from retina_screen.splitting import (
    SplitName, SplitAudit,
    assert_no_patient_overlap, audit_split, split_patients, write_split,
)

@pytest.fixture(scope="module")
def dummy_manifest():
    return DummyAdapter(n_patients=20).build_manifest()

@pytest.fixture(scope="module")
def clean_split(dummy_manifest):
    return split_patients(dummy_manifest, seed=42)

# ---------- sample/patient counts ----------

def test_audit_has_all_four_split_names(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    for sn in SplitName:
        assert sn.value in audit.sample_counts

def test_audit_sample_counts_correct(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    for split_name, sids in clean_split.items():
        assert audit.sample_counts[split_name] == len(sids)

def test_audit_patient_counts_positive(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    for split_name in clean_split:
        assert audit.patient_counts[split_name] > 0

def test_audit_total_sample_count(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    assert audit.total_samples == len(dummy_manifest)

def test_audit_total_patient_count(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    pids = {s.patient_id for s in dummy_manifest}
    assert audit.total_patients == len(pids)

# ---------- valid clean split ----------

def test_audit_valid_when_no_overlap(dummy_manifest, clean_split):
    audit = audit_split(clean_split, dummy_manifest)
    assert audit.valid, f"Audit failed on clean split: {audit.message}"
    assert audit.overlap_pairs == {}

# ---------- overlap detection ----------

def _inject_patient_overlap(split_dict, victim_split, target_split, dummy_manifest, n=1):
    """Return a new split_dict with n patients duplicated across two splits."""
    import copy
    from collections import defaultdict
    sid_to_pid = {s.sample_id: s.patient_id for s in dummy_manifest}
    new = {k: list(v) for k, v in split_dict.items()}
    moved = []
    for sid in split_dict.get(victim_split, []):
        pid = sid_to_pid.get(sid, "")
        if pid not in moved:
            new[target_split] = list(new.get(target_split, [])) + [sid]
            moved.append(pid)
            if len(moved) >= n:
                break
    return new

@pytest.mark.parametrize("pair", [
    ("train", "val"),
    ("train", "reliability"),
    ("train", "test"),
    ("val", "reliability"),
    ("val", "test"),
    ("reliability", "test"),
])
def test_audit_detects_overlap_in_each_pair(pair, dummy_manifest, clean_split):
    a, b = pair
    overlap_dict = _inject_patient_overlap(clean_split, a, b, dummy_manifest)
    audit = audit_split(overlap_dict, dummy_manifest)
    assert not audit.valid
    pair_key = f"{a}/{b}"
    assert pair_key in audit.overlap_pairs, (
        f"audit_split did not detect overlap in {pair_key!r}"
    )
    assert len(audit.overlap_pairs[pair_key]) > 0

def test_overlap_report_includes_patient_ids(dummy_manifest, clean_split):
    overlapped = _inject_patient_overlap(clean_split, "train", "val", dummy_manifest)
    audit = audit_split(overlapped, dummy_manifest)
    assert any(
        len(pids) > 0 for pids in audit.overlap_pairs.values()
    ), "overlap_pairs must contain patient IDs"

def test_assert_no_overlap_raises_on_overlap(dummy_manifest, clean_split):
    overlapped = _inject_patient_overlap(clean_split, "train", "val", dummy_manifest)
    with pytest.raises(ValueError, match="overlap"):
        assert_no_patient_overlap(overlapped, dummy_manifest)

# ---------- missing / ghost assignments ----------

def test_audit_detects_missing_assignments(dummy_manifest, clean_split):
    incomplete = {k: list(v) for k, v in clean_split.items()}
    removed = incomplete["train"].pop()
    audit = audit_split(incomplete, dummy_manifest)
    assert not audit.valid
    assert removed in audit.missing_sample_ids, (
        f"{removed!r} should be in missing_sample_ids"
    )

def test_audit_detects_ghost_sample_ids(dummy_manifest, clean_split):
    ghost_id = "totally_fake_ghost_sample_id"
    with_ghost = {k: list(v) for k, v in clean_split.items()}
    with_ghost["train"].append(ghost_id)
    audit = audit_split(with_ghost, dummy_manifest)
    assert not audit.valid
    assert ghost_id in audit.ghost_sample_ids

def test_audit_detects_duplicate_cross_split_assignment(dummy_manifest, clean_split):
    dup_id = clean_split["train"][0]
    with_dup = {k: list(v) for k, v in clean_split.items()}
    with_dup["val"].append(dup_id)
    audit = audit_split(with_dup, dummy_manifest)
    assert not audit.valid
    assert dup_id in audit.duplicate_assignments

# ---------- unknown split names ----------

def test_audit_detects_unknown_split_names(dummy_manifest, clean_split):
    junk = {k: list(v) for k, v in clean_split.items()}
    junk["junk_split"] = ["some_id"]
    audit = audit_split(junk, dummy_manifest)
    assert not audit.valid
    assert "junk_split" in audit.unknown_split_names

def test_audit_detects_missing_required_split_name(dummy_manifest, clean_split):
    missing = {k: list(v) for k, v in clean_split.items() if k != "reliability"}
    audit = audit_split(missing, dummy_manifest)
    assert not audit.valid
    assert "reliability" in audit.missing_split_names
    assert "missing required split names" in audit.message

# ---------- write_split ----------

def test_write_split_creates_splits_csv(tmp_path, dummy_manifest, clean_split):
    paths = write_split(clean_split, dummy_manifest, tmp_path)
    csv_path = paths["csv"]
    assert csv_path.exists()
    import csv as csv_mod
    with csv_path.open() as fh:
        reader = csv_mod.DictReader(fh)
        rows = list(reader)
    assert set(reader.fieldnames) == {"sample_id", "patient_id", "split_name"}
    assert len(rows) == len(dummy_manifest)
    assert {row["split_name"] for row in rows} <= {
        "train", "val", "reliability", "test"
    }

def test_write_split_creates_audit_json(tmp_path, dummy_manifest, clean_split):
    paths = write_split(clean_split, dummy_manifest, tmp_path)
    audit_path = paths["audit"]
    assert audit_path.exists()
    with audit_path.open() as fh:
        data = json.load(fh)
    assert "valid" in data
    assert "total_samples" in data
    assert data["valid"] is True
