"""
tests/test_evaluate_checkpoint_selection.py -- Regression test for _latest_run_dir priority.

Verifies that scripts/05_evaluate.py::_latest_run_dir() prefers runs/train/ over
runs/fast_dev_run/ when both exist, and falls back to fast_dev_run when train is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evaluate_mod():
    spec = importlib.util.spec_from_file_location(
        "_scripts_05_evaluate",
        _PROJECT_ROOT / "scripts" / "05_evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_checkpoint(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_checkpoint.pt").write_bytes(b"")


def test_train_preferred_over_fast_dev_run(evaluate_mod, tmp_path, monkeypatch) -> None:
    """train run must be returned when both train and fast_dev_run exist."""
    fast_dir = tmp_path / "runs" / "fast_dev_run" / "brset_20260507_old"
    train_dir = tmp_path / "runs" / "train" / "brset_20260511_new"
    _make_checkpoint(fast_dir)
    _make_checkpoint(train_dir)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None, "_latest_run_dir returned None"
    assert "train" in str(result), (
        f"Expected 'train' run to be preferred, got: {result}"
    )
    assert "fast_dev_run" not in str(result), (
        f"fast_dev_run must not shadow train run, got: {result}"
    )


def test_fast_dev_run_fallback_when_no_train(evaluate_mod, tmp_path, monkeypatch) -> None:
    """fast_dev_run must be returned when no train run exists."""
    fast_dir = tmp_path / "runs" / "fast_dev_run" / "brset_20260507_smoke"
    _make_checkpoint(fast_dir)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None, "_latest_run_dir returned None when fast_dev_run should be fallback"
    assert "fast_dev_run" in str(result), (
        f"Expected fast_dev_run fallback, got: {result}"
    )


def test_none_when_no_run_exists(evaluate_mod, tmp_path, monkeypatch) -> None:
    """None must be returned when neither train nor fast_dev_run has a checkpoint."""
    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")
    assert result is None


def test_latest_train_run_selected_by_name(evaluate_mod, tmp_path, monkeypatch) -> None:
    """When multiple train runs exist, the lexicographically latest name is selected."""
    for name in ["brset_20260510_early", "brset_20260511_latest", "brset_20260509_oldest"]:
        _make_checkpoint(tmp_path / "runs" / "train" / name)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None
    assert result.name == "brset_20260511_latest", (
        f"Expected latest-named run, got: {result.name}"
    )


def test_dataset_prefix_filter(evaluate_mod, tmp_path, monkeypatch) -> None:
    """Runs for a different dataset must not be returned."""
    _make_checkpoint(tmp_path / "runs" / "train" / "odir_20260511_run")
    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")
    assert result is None, f"Expected None for brset query when only odir run exists, got: {result}"
