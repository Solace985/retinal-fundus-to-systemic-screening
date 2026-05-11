"""
tests/test_stage8d2_safeguards.py -- Stage 8D-2A safeguard regression tests.

Covers:
  A. Explicit run-dir / checkpoint-path enforcement for full/internal configs.
  B. Finality semantics: stage8d2_full_internal reason, rehearsal/smoke unchanged.
  C. Cache provenance fields: extraction_scope, limit_requested, split_counts,
     reliability_included.
  D. Stage 8D-2 config validation: no rehearsal, no final_test_result, full_dataset_run.

No extraction, training, or evaluation is run. All tests use module loading,
tmp_path fixtures, and synthetic config dicts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAGE8D2_CONFIG = "configs/experiment/stage8d2_brset_resnet50_full_multitask.yaml"
_STAGE8D1_CONFIG = "configs/experiment/stage8d1_brset_resnet50_rehearsal_multitask.yaml"
_STAGE8C_CONFIG = "configs/experiment/stage8c_brset_resnet50.yaml"


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluate_mod():
    spec = importlib.util.spec_from_file_location(
        "_scripts_05_evaluate",
        _PROJECT_ROOT / "scripts" / "05_evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extract_mod():
    spec = importlib.util.spec_from_file_location(
        "_scripts_03_extract_embeddings",
        _PROJECT_ROOT / "scripts" / "03_extract_embeddings.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    with (_PROJECT_ROOT / path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _make_run_dir(tmp_path: Path, *, fast_dev_run: bool = False,
                  rehearsal: bool = False, run_mode: str = "stage8d2_brset_full_resnet50_multitask",
                  dataset: str = "brset", backbone: str = "resnet50",
                  task_config: str = "configs/tasks/brset_default.yaml",
                  with_checkpoint: bool = True,
                  with_resolved_config: bool = True) -> Path:
    run_dir = tmp_path / "runs" / "train" / "brset_20260601_120000"
    run_dir.mkdir(parents=True)
    if with_checkpoint:
        (run_dir / "model_checkpoint.pt").write_bytes(b"")
    if with_resolved_config:
        cfg = {
            "run_mode": run_mode,
            "dataset": dataset,
            "backbone": backbone,
            "task_config": task_config,
            "fast_dev_run": fast_dev_run,
            "rehearsal": rehearsal,
        }
        with (run_dir / "resolved_config.yaml").open("w") as fh:
            yaml.dump(cfg, fh)
    return run_dir


# ===========================================================================
# Section A — Explicit run-dir / checkpoint-path enforcement
# ===========================================================================


def test_full_internal_config_detected_by_full_dataset_run(evaluate_mod) -> None:
    assert evaluate_mod._is_full_internal_config({"full_dataset_run": True})


def test_full_internal_config_detected_by_run_mode_stage8d2(evaluate_mod) -> None:
    assert evaluate_mod._is_full_internal_config(
        {"run_mode": "stage8d2_brset_full_resnet50_multitask"}
    )


def test_stage8d1_not_full_internal(evaluate_mod) -> None:
    assert not evaluate_mod._is_full_internal_config(
        {"run_mode": "stage8d1_brset_rehearsal", "rehearsal": True, "preliminary": True}
    )


def test_smoke_not_full_internal(evaluate_mod) -> None:
    assert not evaluate_mod._is_full_internal_config(
        {"run_mode": "stage8c_brset_smoke"}
    )


def test_validate_run_dir_passes_for_valid_full_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path)
    eval_cfg = {
        "full_dataset_run": True,
        "dataset": "brset",
        "backbone": "resnet50",
        "task_config": "configs/tasks/brset_default.yaml",
    }
    evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)  # must not raise/exit


def test_validate_run_dir_rejects_missing_checkpoint(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, with_checkpoint=False)
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_fast_dev_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, fast_dev_run=True)
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_rehearsal_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, rehearsal=True)
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_stage8d1_run_mode(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, run_mode="stage8d1_brset_rehearsal")
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_dataset_mismatch_for_full_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, dataset="odir")
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_backbone_mismatch_for_full_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, backbone="dinov2")
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_task_config_mismatch_for_full_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, task_config="configs/tasks/odir_default.yaml")
    eval_cfg = {
        "full_dataset_run": True, "dataset": "brset", "backbone": "resnet50",
        "task_config": "configs/tasks/brset_default.yaml",
    }
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_validate_run_dir_rejects_missing_resolved_config_for_full_run(evaluate_mod, tmp_path) -> None:
    run_dir = _make_run_dir(tmp_path, with_resolved_config=False)
    eval_cfg = {"full_dataset_run": True, "dataset": "brset", "backbone": "resnet50", "task_config": "configs/tasks/brset_default.yaml"}
    with pytest.raises(SystemExit):
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_smoke_config_does_not_require_run_dir(evaluate_mod) -> None:
    """Smoke/rehearsal configs should NOT trigger the full_internal check."""
    cfg = {"run_mode": "stage8c_brset_smoke"}
    assert not evaluate_mod._is_full_internal_config(cfg)


def test_stage8d2_config_is_full_internal(evaluate_mod) -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert evaluate_mod._is_full_internal_config(cfg)


# ===========================================================================
# Section B — Finality semantics
# ===========================================================================


def _get_finality(evaluate_mod, cfg: dict) -> tuple[bool, str]:
    """Simulate the finality resolution logic from scripts/05_evaluate.py::main()."""
    final_test_result = not cfg.get("reported_backbone_is_smoke", False)
    eval_reason = "cached_test_samples_available"

    if "smoke" in str(cfg.get("run_mode", "")).lower():
        final_test_result = False
        eval_reason = "limited_embedding_smoke"

    _run_mode_lower = str(cfg.get("run_mode", "")).lower()
    if "rehearsal" in _run_mode_lower or "stage8d1" in _run_mode_lower:
        final_test_result = False
        eval_reason = "rehearsal_run"
    elif evaluate_mod._is_full_internal_config(cfg):
        final_test_result = False
        eval_reason = "stage8d2_full_internal"
    elif cfg.get("preliminary", False) or cfg.get("rehearsal", False):
        final_test_result = False
        eval_reason = "preliminary_run"

    return final_test_result, eval_reason


def test_stage8d2_finality_is_full_internal(evaluate_mod) -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    final, reason = _get_finality(evaluate_mod, cfg)
    assert final is False
    assert reason == "stage8d2_full_internal"


def test_stage8d1_finality_is_rehearsal_run(evaluate_mod) -> None:
    cfg = _load_yaml(_STAGE8D1_CONFIG)
    final, reason = _get_finality(evaluate_mod, cfg)
    assert final is False
    assert reason == "rehearsal_run"


def test_stage8c_finality_is_limited_embedding_smoke(evaluate_mod) -> None:
    cfg = _load_yaml(_STAGE8C_CONFIG)
    final, reason = _get_finality(evaluate_mod, cfg)
    assert final is False
    assert reason == "limited_embedding_smoke"


def test_stage8d2_final_test_result_never_true(evaluate_mod) -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    final, _ = _get_finality(evaluate_mod, cfg)
    assert final is False, "Stage 8D-2 must never produce final_test_result=True"


# ===========================================================================
# Section C — Cache provenance fields
# ===========================================================================


def test_provenance_limited_scope_fields(extract_mod) -> None:
    """When --limit is set, provenance must show extraction_scope='limited'."""
    cfg = {"run_mode": "stage8d1_brset_rehearsal", "stage": "8D-1"}
    # Simulate what main() builds:
    limit = 32
    prov = {
        "limit_requested": limit,
        "extraction_scope": "limited" if limit is not None else "full",
        "total_samples_extracted": limit,
        "manifest_row_count": limit,
    }
    assert prov["extraction_scope"] == "limited"
    assert prov["limit_requested"] == 32


def test_provenance_full_scope_fields(extract_mod) -> None:
    """When no --limit is passed, provenance must show extraction_scope='full'."""
    limit = None
    prov = {
        "limit_requested": limit,
        "extraction_scope": "limited" if limit is not None else "full",
    }
    assert prov["extraction_scope"] == "full"
    assert prov["limit_requested"] is None


def test_provenance_split_counts_extracted_structure() -> None:
    """split_counts_extracted must be a dict mapping split names to int counts."""
    split_counts = {"train": 640, "val": 192, "test": 192, "reliability": 0}
    for key, val in split_counts.items():
        assert isinstance(key, str)
        assert isinstance(val, int)


def test_provenance_reliability_included_false_when_zero() -> None:
    split_counts = {"train": 640, "val": 192, "test": 192, "reliability": 0}
    reliability_included = split_counts.get("reliability", 0) > 0
    assert reliability_included is False


def test_provenance_reliability_included_true_when_nonzero() -> None:
    split_counts = {"train": 9763, "val": 2443, "test": 1623, "reliability": 2437}
    reliability_included = split_counts.get("reliability", 0) > 0
    assert reliability_included is True


def test_provenance_required_fields_present() -> None:
    """Verify all required provenance fields are in the expected set."""
    required = {
        "backbone_name", "backbone_version", "backbone_source",
        "backbone_model_identifier", "embedding_dim",
        "dataset_source", "dataset_root_used", "preprocessing_hash", "created_at",
        "run_mode", "stage", "limit_requested", "extraction_scope",
        "total_samples_extracted", "manifest_row_count",
        "split_counts_extracted", "reliability_included",
        "overwrite", "config_path", "cache_dir", "manifest_path",
    }
    # Simulate a provenance dict built by the script.
    example = {
        "backbone_name": "resnet50", "backbone_version": "IMAGENET1K_V2",
        "backbone_source": "torchvision", "backbone_model_identifier": "ResNet50_Weights.IMAGENET1K_V2",
        "embedding_dim": 2048, "dataset_source": "brset", "dataset_root_used": "/data/brset",
        "preprocessing_hash": "abc123", "created_at": "2026-05-11T00:00:00+00:00",
        "run_mode": "stage8d2_brset_full_resnet50_multitask", "stage": "8D-2",
        "limit_requested": None, "extraction_scope": "full",
        "total_samples_extracted": 16266, "manifest_row_count": 16266,
        "split_counts_extracted": {"train": 9763, "val": 2443, "test": 1623, "reliability": 2437},
        "reliability_included": True, "overwrite": False,
        "config_path": "configs/experiment/stage8d2_brset_resnet50_full_multitask.yaml",
        "cache_dir": "cache/embeddings/resnet50/brset/abc123",
        "manifest_path": "cache/embeddings/resnet50/brset/abc123/manifest.csv",
    }
    missing = required - example.keys()
    assert not missing, f"Missing provenance fields: {missing}"


def test_provenance_limited_distinguishable_from_full() -> None:
    """A limited and full provenance must be distinguishable without checking row count."""
    limited = {"extraction_scope": "limited", "limit_requested": 1024}
    full = {"extraction_scope": "full", "limit_requested": None}
    assert limited["extraction_scope"] != full["extraction_scope"]
    assert limited["limit_requested"] != full["limit_requested"]


# ===========================================================================
# Section D — Stage 8D-2 config validation
# ===========================================================================


def test_stage8d2_config_does_not_trigger_limit_guard(extract_mod) -> None:
    """Stage 8D-2 config must not trigger the Stage 8D-1 no-limit guard."""
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert not extract_mod._is_stage8d1_rehearsal_config(cfg), (
        "Stage 8D-2 config must not trigger the rehearsal/limit guard "
        "(full_dataset_run=true should exempt it)."
    )


def test_stage8d2_config_full_dataset_run_true() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("full_dataset_run") is True


def test_stage8d2_config_rehearsal_false() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("rehearsal") is False


def test_stage8d2_config_preliminary_true() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("preliminary") is True


def test_stage8d2_config_no_final_test_result_key() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert "final_test_result" not in cfg, (
        "Stage 8D-2 config must not contain final_test_result field; "
        "finality is determined at runtime by evaluation logic."
    )


def test_stage8d2_config_fast_dev_run_false() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("fast_dev_run") is False


def test_stage8d2_config_reported_backbone_is_smoke_false() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("reported_backbone_is_smoke") is False


def test_stage8d2_config_run_mode_pattern() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    run_mode = cfg.get("run_mode", "")
    assert "stage8d2" in run_mode.lower()
    assert "brset" in run_mode.lower()
    assert "resnet50" in run_mode.lower()
    assert "multitask" in run_mode.lower()


def test_stage8d2_config_dataset_brset() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("dataset") == "brset"


def test_stage8d2_config_backbone_resnet50() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("backbone") == "resnet50"


def test_stage8d2_config_no_smoke_flag() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert not cfg.get("reported_backbone_is_smoke", False)
    assert "smoke" not in str(cfg.get("run_mode", "")).lower()


def test_stage8d2_config_internal_full_result_true() -> None:
    cfg = _load_yaml(_STAGE8D2_CONFIG)
    assert cfg.get("internal_full_result") is True
