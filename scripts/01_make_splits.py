#!/usr/bin/env python
"""
scripts/01_make_splits.py -- Build canonical manifest and patient-level splits.

Thin orchestration script. Dataset parsing lives in adapters; splitting logic
lives in src/retina_screen/splitting.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.core import ensure_dir, load_config, seed_everything, setup_logging
from retina_screen.schema import CanonicalSample
from retina_screen.splitting import assert_no_patient_overlap, split_patients, write_split
from retina_screen.tasks import TASK_REGISTRY, LabelQuality

logger = logging.getLogger(__name__)


def _import_from_string(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_dataset_config(cfg: dict[str, Any]) -> dict[str, Any]:
    config_path = cfg.get("dataset_config")
    if config_path:
        return load_config(config_path)
    return {"name": cfg.get("dataset", "dummy")}


def _build_adapter(cfg: dict[str, Any]):
    dataset_cfg = _load_dataset_config(cfg)
    adapter_class = dataset_cfg.get("adapter_class")
    if adapter_class:
        adapter_type = _import_from_string(str(adapter_class))
        kwargs = {
            key: cfg.get(key) if key in cfg else dataset_cfg.get(key)
            for key in ("dataset_root", "metadata_file", "training_images_dir")
            if key in cfg or key in dataset_cfg
        }
        return adapter_type(**kwargs)
    return DummyAdapter(n_patients=cfg.get("n_patients", 80))


def _write_manifest_csv(manifest: list[CanonicalSample], out_path: Path) -> None:
    fieldnames = list(CanonicalSample.model_fields.keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sample in manifest:
            row: dict[str, Any] = {}
            for field in fieldnames:
                value = getattr(sample, field, None)
                if value is not None and hasattr(value, "value"):
                    value = value.value
                row[field] = value
            writer.writerow(row)
    logger.info("Canonical manifest written: %s (%d samples)", out_path, len(manifest))


def _label_coverage(
    manifest: list[CanonicalSample],
    supported_tasks: list[str],
) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for task_name in supported_tasks:
        target_column = TASK_REGISTRY[task_name].target_column
        positives = sum(1 for sample in manifest if getattr(sample, target_column) == 1)
        negatives = sum(1 for sample in manifest if getattr(sample, target_column) == 0)
        missing = sum(1 for sample in manifest if getattr(sample, target_column) is None)
        coverage[task_name] = {
            "positives": positives,
            "negatives": negatives,
            "missing": missing,
        }
    return coverage


def _generic_dataset_audit(
    adapter: Any,
    manifest: list[CanonicalSample],
    supported_tasks: list[str],
) -> dict[str, Any]:
    adapter_audit: dict[str, Any] = {}
    audit_getter = getattr(adapter, "get_dataset_audit", None)
    if callable(audit_getter):
        adapter_audit = audit_getter()

    sex_counts = Counter(
        sample.sex.value if sample.sex is not None else "missing"
        for sample in manifest
    )
    ages = [sample.age_years for sample in manifest if sample.age_years is not None]
    weak_proxy_tasks = [
        task_name
        for task_name in supported_tasks
        if TASK_REGISTRY[task_name].label_quality == LabelQuality.PROXY
    ]

    audit: dict[str, Any] = {
        **adapter_audit,
        "total_samples": len(manifest),
        "total_patients": len({sample.patient_id for sample in manifest}),
        "supported_tasks": supported_tasks,
        "supported_task_coverage": _label_coverage(manifest, supported_tasks),
        "weak_proxy_tasks": weak_proxy_tasks,
        "subgroup_coverage": {
            **adapter_audit.get("subgroup_coverage", {}),
            "sex_counts": dict(sex_counts),
            "age": {
                "available": len(ages),
                "missing": len(manifest) - len(ages),
                "min": min(ages) if ages else None,
                "max": max(ages) if ages else None,
                "mean": sum(ages) / len(ages) if ages else None,
            },
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical manifest and patient-level splits."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    dataset_cfg = _load_dataset_config(cfg)
    dataset_name = str(dataset_cfg.get("name", cfg.get("dataset", "dummy")))

    adapter = _build_adapter(cfg)
    manifest = adapter.build_manifest()
    adapter.validate()
    supported_tasks = adapter.get_supported_tasks()
    logger.info(
        "Manifest built: dataset=%s samples=%d patients=%d tasks=%s",
        dataset_name,
        len(manifest),
        len({sample.patient_id for sample in manifest}),
        supported_tasks,
    )

    split_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.get("split_output_root", "outputs/splits"))
    out_dir = ensure_dir(output_root / dataset_name / split_id)

    manifest_path = out_dir / "canonical_manifest.csv"
    _write_manifest_csv(manifest, manifest_path)

    dataset_audit = _generic_dataset_audit(adapter, manifest, supported_tasks)
    dataset_audit_path = out_dir / "dataset_audit.json"
    with dataset_audit_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset_audit, fh, indent=2)
    logger.info("Dataset audit written: %s", dataset_audit_path)

    split = split_patients(manifest, seed=seed)
    assert_no_patient_overlap(split, manifest)
    split_paths = write_split(split, manifest, out_dir)

    logger.info("Split artifacts written: %s", out_dir)
    logger.info("canonical_manifest=%s", manifest_path)
    logger.info("dataset_audit=%s", dataset_audit_path)
    logger.info("splits=%s", split_paths["csv"])
    logger.info("split_audit=%s", split_paths["audit"])


if __name__ == "__main__":
    main()
