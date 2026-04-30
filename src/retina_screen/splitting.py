"""
splitting.py -- Patient-level deterministic splitting for the retinal screening pipeline.

Owns: patient-level split logic, 60/15/15/10 default ratios, split audit, overlap
detection, and split file writing.

Must not contain: image-level splitting, dataset-specific logic, model code, or
ODIR/BRSET/mBRSET conditionals.

Critical invariant: every sample from the same patient must appear in exactly one
split. Any patient overlap between train/val/reliability/test silently inflates
evaluation metrics and invalidates research claims.
"""

from __future__ import annotations

import csv
import json
import logging
import random as _random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from retina_screen.schema import CanonicalSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATIO_TOLERANCE: float = 1e-6

_SPLIT_ORDER: tuple[str, ...] = ("train", "val", "reliability", "test")
_REQUIRED_SPLIT_NAMES: frozenset[str] = frozenset(_SPLIT_ORDER)


class SplitName(str, Enum):
    """Canonical split names for the four-way patient-level split."""

    TRAIN = "train"
    VAL = "val"
    RELIABILITY = "reliability"
    TEST = "test"


DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    SplitName.TRAIN: 0.60,
    SplitName.VAL: 0.15,
    SplitName.RELIABILITY: 0.15,
    SplitName.TEST: 0.10,
}


# ---------------------------------------------------------------------------
# Audit dataclass
# ---------------------------------------------------------------------------


@dataclass
class SplitAudit:
    """Result of auditing a split assignment against the source manifest.

    valid=True only if there are no overlaps, no ghost IDs, no missing IDs,
    no duplicate assignments, and no unknown split names.
    """

    sample_counts: dict[str, int]
    patient_counts: dict[str, int]
    total_samples: int
    total_patients: int
    overlap_pairs: dict[str, list[str]]           # "train/val" -> [patient_ids]
    duplicate_assignments: list[str]              # sample_ids assigned to > 1 split
    duplicate_within_splits: dict[str, list[str]] # split_name -> [dup sample_ids]
    missing_sample_ids: list[str]                 # in manifest but unassigned
    ghost_sample_ids: list[str]                   # assigned but not in manifest
    unknown_split_names: list[str]                # not in SplitName enum
    missing_split_names: list[str]                # required SplitName values absent
    valid: bool
    message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_patients(
    manifest: Sequence[CanonicalSample],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    stratify_on: str | None = None,
    group_on: str = "patient_id",
) -> dict[str, list[str]]:
    """Split a manifest into train/val/reliability/test by patient group.

    All samples from the same patient (identified by ``group_on``) are assigned
    to exactly one split. This is enforced structurally — sample-level shuffling
    is never used.

    Parameters
    ----------
    manifest:
        List of canonical samples to split.
    ratios:
        Per-split patient ratios. Must sum to 1.0 within RATIO_TOLERANCE.
        Defaults to 60/15/15/10.
    seed:
        Seed for the patient-level shuffle. Different seeds produce different
        assignments (with overwhelming probability for realistic dataset sizes).
    stratify_on:
        Optional canonical field name by which to approximate stratification.
        Stratification is best-effort and patient-level (majority vote per patient).
        Does not guarantee exact balance; documented as approximate.
        Fails with ValueError if the field does not exist in CanonicalSample.
    group_on:
        Canonical field used as the patient group key. Stage 4 only permits
        "patient_id"; any other value is rejected to prevent image-level splits.

    Returns
    -------
    dict[str, list[str]]
        Maps split name (str) to list of sample_ids assigned to that split.
    """
    schema_fields = set(CanonicalSample.model_fields.keys())

    # Stage 4 is patient-level only. Do not permit sample_id/image_path or any
    # other field to become the grouping key.
    if group_on != "patient_id":
        raise ValueError(
            f"Stage 4 supports patient-level splitting only; group_on must be "
            f"'patient_id', got {group_on!r}."
        )
    if group_on not in schema_fields:
        raise ValueError(
            f"group_on={group_on!r} is not a field in CanonicalSample. "
            f"Valid fields: {sorted(schema_fields)}"
        )
    if stratify_on is not None and stratify_on not in schema_fields:
        raise ValueError(
            f"stratify_on={stratify_on!r} is not a field in CanonicalSample. "
            f"Valid fields: {sorted(schema_fields)}"
        )

    # Normalize and validate ratios.
    # Use .value for enum keys (str(SplitName.TRAIN) gives "SplitName.TRAIN" in Python 3.12).
    def _key(k: object) -> str:
        return k.value if hasattr(k, "value") else str(k)

    source = ratios if ratios is not None else DEFAULT_SPLIT_RATIOS
    effective_ratios: dict[str, float] = {_key(k): v for k, v in source.items()}
    ratio_keys = set(effective_ratios)
    missing_ratio_keys = sorted(_REQUIRED_SPLIT_NAMES - ratio_keys)
    extra_ratio_keys = sorted(ratio_keys - _REQUIRED_SPLIT_NAMES)
    if missing_ratio_keys or extra_ratio_keys:
        parts: list[str] = []
        if missing_ratio_keys:
            parts.append(f"missing split names: {missing_ratio_keys}")
        if extra_ratio_keys:
            parts.append(f"unsupported split names: {extra_ratio_keys}")
        raise ValueError(
            "Split ratios must define exactly train/val/reliability/test; "
            + "; ".join(parts)
        )
    ratio_sum = sum(effective_ratios.values())
    if abs(ratio_sum - 1.0) > RATIO_TOLERANCE:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {ratio_sum:.8f}, "
            f"tolerance={RATIO_TOLERANCE}). Ratios: {effective_ratios}"
        )

    if not manifest:
        raise ValueError("Manifest is empty; cannot split.")

    # Validate group_on values
    for s in manifest:
        pid = getattr(s, group_on)
        if pid is None or (isinstance(pid, str) and not pid.strip()):
            raise ValueError(
                f"Sample {s.sample_id!r} has blank/None {group_on!r}. "
                f"All samples must have a non-empty group key."
            )

    # Build patient -> samples mapping
    patient_to_samples: dict[str, list[str]] = defaultdict(list)
    for s in manifest:
        pid = str(getattr(s, group_on))
        patient_to_samples[pid].append(s.sample_id)

    unique_patients = sorted(patient_to_samples.keys())
    n_patients = len(unique_patients)

    # Compute per-split patient counts
    counts = _compute_patient_counts(n_patients, effective_ratios)

    # Shuffle patients deterministically
    rng = _random.Random(seed)
    if stratify_on is not None:
        patients_ordered = _stratified_shuffle(
            unique_patients, patient_to_samples, manifest, stratify_on, rng
        )
    else:
        patients_ordered = list(unique_patients)
        rng.shuffle(patients_ordered)

    # Assign patients to splits in canonical order
    ordered_splits = list(_SPLIT_ORDER)
    split_dict: dict[str, list[str]] = {}
    idx = 0
    for split_name in ordered_splits:
        n = counts[split_name]
        batch = patients_ordered[idx: idx + n]
        idx += n
        sample_ids: list[str] = []
        for pid in batch:
            sample_ids.extend(patient_to_samples[pid])
        split_dict[split_name] = sample_ids

    # Log summary
    for split_name in ordered_splits:
        sids = split_dict[split_name]
        assigned_pids = {
            str(getattr(s, group_on))
            for s in manifest
            if s.sample_id in set(sids)
        }
        logger.info(
            "split=%s  samples=%d  patients=%d",
            split_name,
            len(sids),
            len(assigned_pids),
        )
    logger.info(
        "Total: %d patients, %d samples",
        n_patients,
        sum(len(v) for v in split_dict.values()),
    )
    return split_dict


def audit_split(
    split_dict: dict[str, list[str]],
    manifest: Sequence[CanonicalSample],
) -> SplitAudit:
    """Audit a split assignment dict against the original manifest.

    Detects:
    - patient overlap across any of the six split pairs
    - sample IDs assigned to more than one split
    - duplicate sample IDs within a single split
    - ghost sample IDs (in split_dict but absent from manifest)
    - missing sample IDs (in manifest but absent from split_dict)
    - unknown split names (not in SplitName enum)
    - missing required split names (train/val/reliability/test)
    """
    manifest_ids: set[str] = {s.sample_id for s in manifest}
    manifest_patient: dict[str, str] = {s.sample_id: s.patient_id for s in manifest}

    valid_names: set[str] = {sn.value for sn in SplitName}
    unknown_split_names = sorted(k for k in split_dict if k not in valid_names)
    missing_split_names = sorted(valid_names - set(split_dict))

    sample_counts: dict[str, int] = {name: 0 for name in _SPLIT_ORDER}
    patient_counts: dict[str, int] = {name: 0 for name in _SPLIT_ORDER}
    split_patient_sets: dict[str, set[str]] = {name: set() for name in _SPLIT_ORDER}

    # Track all assignments for cross-split duplicate detection
    global_assignment_count: Counter[str] = Counter()
    for split_name, sids in split_dict.items():
        sample_counts[split_name] = len(sids)
        global_assignment_count.update(sids)
        pids = {
            manifest_patient[sid] for sid in sids if sid in manifest_patient
        }
        patient_counts[split_name] = len(pids)
        split_patient_sets[split_name] = pids

    # Cross-split duplicates (same sample_id in > 1 split)
    duplicate_assignments = sorted(
        sid for sid, cnt in global_assignment_count.items() if cnt > 1
    )

    # Within-split duplicates
    duplicate_within_splits: dict[str, list[str]] = {}
    for split_name, sids in split_dict.items():
        within_count = Counter(sids)
        dups = sorted(sid for sid, cnt in within_count.items() if cnt > 1)
        if dups:
            duplicate_within_splits[split_name] = dups

    # Ghost IDs: in split but not in manifest
    all_assigned: set[str] = set(global_assignment_count.keys())
    ghost_sample_ids = sorted(all_assigned - manifest_ids)

    # Missing IDs: in manifest but not assigned to any split
    missing_sample_ids = sorted(manifest_ids - all_assigned)

    # Patient overlap across all six split pairs
    _PAIRS = (
        ("train", "val"),
        ("train", "reliability"),
        ("train", "test"),
        ("val", "reliability"),
        ("val", "test"),
        ("reliability", "test"),
    )
    overlap_pairs: dict[str, list[str]] = {}
    for a, b in _PAIRS:
        common = sorted(
            split_patient_sets.get(a, set()) & split_patient_sets.get(b, set())
        )
        if common:
            overlap_pairs[f"{a}/{b}"] = common

    # Build validity
    issues: list[str] = []
    if overlap_pairs:
        for pair, pids in overlap_pairs.items():
            issues.append(
                f"patient overlap in {pair!r}: "
                f"{len(pids)} patients, e.g. {pids[:3]}"
            )
    if duplicate_assignments:
        issues.append(
            f"duplicate cross-split assignments: {duplicate_assignments[:5]}"
        )
    if ghost_sample_ids:
        issues.append(
            f"ghost sample IDs not in manifest: {ghost_sample_ids[:5]}"
        )
    if missing_sample_ids:
        issues.append(
            f"unassigned sample IDs: {missing_sample_ids[:5]}"
        )
    if unknown_split_names:
        issues.append(f"unknown split names: {unknown_split_names}")
    if missing_split_names:
        issues.append(f"missing required split names: {missing_split_names}")
    if duplicate_within_splits:
        issues.append(
            f"within-split duplicates: {list(duplicate_within_splits.keys())}"
        )

    total_samples = len(manifest)
    total_patients = len({s.patient_id for s in manifest})
    valid = not issues
    message = "OK" if valid else "; ".join(issues)

    return SplitAudit(
        sample_counts=sample_counts,
        patient_counts=patient_counts,
        total_samples=total_samples,
        total_patients=total_patients,
        overlap_pairs=overlap_pairs,
        duplicate_assignments=duplicate_assignments,
        duplicate_within_splits=duplicate_within_splits,
        missing_sample_ids=missing_sample_ids,
        ghost_sample_ids=ghost_sample_ids,
        unknown_split_names=unknown_split_names,
        missing_split_names=missing_split_names,
        valid=valid,
        message=message,
    )


def assert_no_patient_overlap(
    split_dict: dict[str, list[str]],
    manifest: Sequence[CanonicalSample],
) -> None:
    """Raise ValueError with actionable details if any patient appears in multiple splits.

    Check this after every split_patients() call and before training begins.
    """
    audit = audit_split(split_dict, manifest)
    if audit.overlap_pairs:
        lines = []
        for pair, pids in audit.overlap_pairs.items():
            lines.append(
                f"  {pair}: {len(pids)} overlapping patients, "
                f"examples: {pids[:5]}"
            )
        raise ValueError(
            f"{len(audit.overlap_pairs)} patient overlap(s) detected:\n"
            + "\n".join(lines)
        )


def write_split(
    split_dict: dict[str, list[str]],
    manifest: Sequence[CanonicalSample],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write splits.csv and split_audit.json to output_dir.

    splits.csv columns: sample_id, patient_id, split_name.
    split_audit.json: JSON serialisation of SplitAudit.

    Returns a dict with keys "csv" and "audit" mapping to their paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sid_to_sample: dict[str, CanonicalSample] = {s.sample_id: s for s in manifest}

    # splits.csv — rows ordered by canonical split order then sample_id
    csv_path = out / "splits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_id", "patient_id", "split_name"]
        )
        writer.writeheader()
        for split_name in _SPLIT_ORDER:
            for sid in split_dict.get(split_name, []):
                s = sid_to_sample.get(sid)
                writer.writerow(
                    {
                        "sample_id": sid,
                        "patient_id": s.patient_id if s else "",
                        "split_name": split_name,
                    }
                )

    # split_audit.json
    audit = audit_split(split_dict, manifest)
    audit_path = out / "split_audit.json"
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(audit), fh, indent=2)

    logger.info("Split files written to %s", out)
    return {"csv": csv_path, "audit": audit_path}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_patient_counts(
    n_total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """Distribute n_total patients across splits using floor + remainder.

    The last split in canonical order absorbs any rounding remainder.
    Raises ValueError if any split would receive 0 patients.
    """
    # Canonical ordering: _SPLIT_ORDER first, then any extras
    ordered = [s for s in _SPLIT_ORDER if s in ratios]
    ordered += [s for s in sorted(ratios.keys()) if s not in ordered]

    counts: dict[str, int] = {}
    remaining = n_total
    for name in ordered[:-1]:
        c = int(ratios[name] * n_total)  # floor
        counts[name] = c
        remaining -= c
    counts[ordered[-1]] = remaining

    for name, c in counts.items():
        if c < 1:
            raise ValueError(
                f"Split {name!r} would have 0 patients with {n_total} patients total. "
                f"Add more patients or adjust ratios. "
                f"(Minimum ~{len(ordered)} patients needed for a {len(ordered)}-way split "
                f"with these ratios.)"
            )
    return counts


def _stratified_shuffle(
    unique_patients: list[str],
    patient_to_samples: dict[str, list[str]],
    manifest: Sequence[CanonicalSample],
    stratify_on: str,
    rng: _random.Random,
) -> list[str]:
    """Best-effort patient-level stratified shuffle.

    Each patient is assigned a stratum label via majority vote of the
    stratify_on column across all their samples (non-None values only).
    Patients with all-None values get stratum '_unknown'.

    Patients are shuffled independently within each stratum, then concatenated.
    This is approximate and does not guarantee perfect stratum balance.
    It never sacrifices patient grouping for stratification.
    """
    sample_map: dict[str, CanonicalSample] = {s.sample_id: s for s in manifest}

    strata: dict[str, list[str]] = defaultdict(list)
    for pid in unique_patients:
        sids = patient_to_samples[pid]
        values = [
            getattr(sample_map[sid], stratify_on)
            for sid in sids
            if sid in sample_map
        ]
        non_none = [v for v in values if v is not None]
        stratum = (
            str(Counter(non_none).most_common(1)[0][0]) if non_none else "_unknown"
        )
        strata[stratum].append(pid)

    if set(strata.keys()) == {"_unknown"}:
        logger.warning(
            "stratify_on=%r: all patient values are None; "
            "falling back to non-stratified shuffle.",
            stratify_on,
        )

    result: list[str] = []
    for stratum in sorted(strata.keys()):
        group = strata[stratum]
        rng.shuffle(group)
        result.extend(group)
    return result
