#!/usr/bin/env python
"""
scripts/01_make_splits.py -- Build manifest, dataset audit, and patient-level splits.

Thin orchestration script. Business logic lives in src/retina_screen/.

Usage:
    python scripts/01_make_splits.py --config configs/experiment/baseline_odir_dinov2.yaml

Output artifacts (under outputs/splits/{dataset}/{split_id}/):
    canonical_manifest.csv  -- all CanonicalSample fields, one row per sample
    dataset_audit.json      -- label distributions and quality/uncertain counts
    splits.csv              -- sample_id, patient_id, split_name
    split_audit.json        -- overlap checks, patient/sample counts per split
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.core import (
    ensure_dir,
    load_config,
    seed_everything,
    setup_logging,
)
from retina_screen.schema import CanonicalSample
from retina_screen.splitting import assert_no_patient_overlap, split_patients
from retina_screen.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter factory (dict-dispatch avoids dataset-name conditionals)
# ---------------------------------------------------------------------------


def _make_dummy_adapter(cfg: dict):
    from retina_screen.adapters.dummy import DummyAdapter  # noqa: PLC0415
    return DummyAdapter(n_patients=cfg.get("n_patients", 80))


def _make_odir_adapter(cfg: dict):
    from retina_screen.adapters.odir import ODIRAdapter  # noqa: PLC0415
    return ODIRAdapter(dataset_root=cfg.get("dataset_root", "ODIR-5K"))


_ADAPTER_BUILDERS = {
    "dummy": _make_dummy_adapter,
    "odir":  _make_odir_adapter,
}


def _build_adapter(cfg: dict):
    name = cfg.get("dataset", "dummy")
    builder = _ADAPTER_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown dataset={name!r}. Supported: {sorted(_ADAPTER_BUILDERS)}")
    return builder(cfg)


# ---------------------------------------------------------------------------
# Manifest CSV
# ---------------------------------------------------------------------------


def _write_manifest_csv(manifest: list[CanonicalSample], out_path: Path) -> None:
    fieldnames = list(CanonicalSample.model_fields.keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sample in manifest:
            row: dict = {}
            for field in fieldnames:
                val = getattr(sample, field, None)
                if val is not None and hasattr(val, "value"):
                    val = val.value
                row[field] = val
            writer.writerow(row)
    logger.info("Manifest written: %s (%d samples)", out_path, len(manifest))


# ---------------------------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------------------------


def _build_dataset_audit(
    manifest: list[CanonicalSample],
    supported_tasks: list[str],
) -> dict:
    n_total = len(manifest)
    n_patients = len({s.patient_id for s in manifest})

    label_counts: dict[str, dict] = {}
    for task in supported_tasks:
        field = TASK_REGISTRY[task].target_column
        positives = sum(1 for s in manifest if getattr(s, field, None) == 1)
        negatives = sum(1 for s in manifest if getattr(s, field, None) == 0)
        missing = sum(1 for s in manifest if getattr(s, field, None) is None)
        label_counts[task] = {"positives": positives, "negatives": negatives, "missing": missing}

    return {
        "total_samples": n_total,
        "total_patients": n_patients,
        "label_counts": label_counts,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Splits CSV and audit
# ---------------------------------------------------------------------------


def _write_splits_csv(
    split: dict[str, list[str]],
    manifest: list[CanonicalSample],
    out_path: Path,
) -> None:
    sid_to_pid = {s.sample_id: s.patient_id for s in manifest}
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "patient_id", "split_name"])
        writer.writeheader()
        for split_name, sids in split.items():
            for sid in sids:
                writer.writerow({
                    "sample_id": sid,
                    "patient_id": sid_to_pid.get(sid, ""),
                    "split_name": split_name,
                })
    logger.info("Splits CSV written: %s", out_path)


def _build_split_audit(
    split: dict[str, list[str]],
    manifest: list[CanonicalSample],
) -> dict:
    sid_to_pid = {s.sample_id: s.patient_id for s in manifest}
    split_pid_sets: dict[str, set[str]] = {}
    counts: dict[str, dict] = {}

    for split_name, sids in split.items():
        pids = {sid_to_pid[sid] for sid in sids if sid in sid_to_pid}
        split_pid_sets[split_name] = pids
        counts[split_name] = {"samples": len(sids), "patients": len(pids)}

    split_names = list(split.keys())
    overlaps: list[dict] = []
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = split_pid_sets[a] & split_pid_sets[b]
            if overlap:
                overlaps.append({
                    "pair": [a, b],
                    "n_patients": len(overlap),
                    "examples": sorted(overlap)[:5],
                })

    return {
        "counts": counts,
        "patient_overlap_violations": overlaps,
        "valid": len(overlaps) == 0,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical manifest and patient-level splits."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    dataset = cfg.get("dataset", "dummy")

    adapter = _build_adapter(cfg)
    manifest = adapter.build_manifest()
    supported_tasks = adapter.get_supported_tasks()
    n_patients = len({s.patient_id for s in manifest})
    logger.info(
        "Manifest: %d samples, %d patients, tasks=%s",
        len(manifest), n_patients, supported_tasks,
    )

    split_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path("outputs") / "splits" / dataset / split_id)
    logger.info("Output directory: %s", out_dir)

    _write_manifest_csv(manifest, out_dir / "canonical_manifest.csv")

    audit = _build_dataset_audit(manifest, supported_tasks)
    with (out_dir / "dataset_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    logger.info("Dataset audit written.")

    split = split_patients(manifest, seed=seed)
    assert_no_patient_overlap(split, manifest)

    for split_name, sids in split.items():
        pids = {s.patient_id for s in manifest if s.sample_id in set(sids)}
        logger.info("Split %-12s: %d samples, %d patients", split_name, len(sids), len(pids))

    _write_splits_csv(split, manifest, out_dir / "splits.csv")

    split_audit = _build_split_audit(split, manifest)
    with (out_dir / "split_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(split_audit, fh, indent=2)
    logger.info("Split audit valid=%s.", split_audit["valid"])

    if not split_audit["valid"]:
        logger.error("SPLIT AUDIT FAILED — patient overlap detected: %s",
                     split_audit["patient_overlap_violations"])
        sys.exit(1)

    logger.info("Done. Artifacts: %s", out_dir)


if __name__ == "__main__":
    main()
