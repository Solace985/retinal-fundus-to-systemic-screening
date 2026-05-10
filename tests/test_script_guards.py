"""
tests/test_script_guards.py -- Verify smoke/rehearsal limit-enforcement guards.

Stage 8C smoke and Stage 8D-1 rehearsal configs must require --limit when running
scripts/03_extract_embeddings.py.

Testing approach
----------------
Stage 8C: subprocess tests (the guard exits before backbone/dataset loading).
Stage 8D-1 detection: unit tests that import and call the guard functions directly
  with synthetic config dicts.  No subprocess with --limit is used because that
  would proceed into backbone/dataset loading.
Stage 8D-1 subprocess: one test verifies the no-limit failure path, which exits
  before backbone/dataset loading.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = "scripts/03_extract_embeddings.py"
_STAGE8C_CONFIG = "configs/experiment/stage8c_brset_resnet50.yaml"
_STAGE8D1_CONFIG = "configs/experiment/stage8d1_brset_resnet50_rehearsal_multitask.yaml"


# ---------------------------------------------------------------------------
# Module loader for unit-testing guard functions directly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extract_mod():
    """Load scripts/03_extract_embeddings.py as an importable module.

    Executed once per test session. Imports the module without calling main(),
    so no extraction, backbone loading, or dataset traversal occurs.
    """
    spec = importlib.util.spec_from_file_location(
        "_scripts_03_extract_embeddings",
        _PROJECT_ROOT / "scripts" / "03_extract_embeddings.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stage 8C subprocess tests (guard exits before backbone/dataset loading)
# ---------------------------------------------------------------------------


def test_stage8c_guard_exits_nonzero_without_limit() -> None:
    """Stage 8C smoke config without --limit must exit nonzero with an informative message."""
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", _STAGE8C_CONFIG],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode != 0, (
        "scripts/03_extract_embeddings.py should exit nonzero when --limit is "
        "omitted for a stage8c_brset_smoke config."
    )
    combined = result.stdout + result.stderr
    assert "Stage 8C" in combined or "smoke" in combined.lower(), (
        f"Exit message should mention 'Stage 8C' or 'smoke', got: {combined[:300]!r}"
    )


def test_stage8c_guard_does_not_trigger_with_limit() -> None:
    """Stage 8C smoke config with --limit must not trigger the guard.

    The script may fail for other reasons (missing data, etc.) after the guard
    passes, but it must not exit due to the limit guard specifically.
    """
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", _STAGE8C_CONFIG, "--limit", "1"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    combined = result.stdout + result.stderr
    assert "Stage 8C smoke configs require --limit" not in combined, (
        "Guard must not trigger when --limit is provided. "
        f"Got output: {combined[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Stage 8D-1 unit tests — _is_stage8d1_rehearsal_config detection logic
# ---------------------------------------------------------------------------


def test_stage8d1_guard_fires_on_stage8d1_runmode(extract_mod) -> None:
    assert extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "stage8d1_brset_rehearsal"}
    )


def test_stage8d1_guard_fires_on_rehearsal_in_runmode(extract_mod) -> None:
    assert extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "rehearsal_brset_convnext"}
    )


def test_stage8d1_guard_fires_on_rehearsal_true_field(extract_mod) -> None:
    assert extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "some_run", "rehearsal": True}
    )


def test_stage8d1_guard_fires_on_preliminary_true_field(extract_mod) -> None:
    assert extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "some_run", "preliminary": True}
    )


def test_stage8d1_guard_does_not_fire_for_stage8d2(extract_mod) -> None:
    assert not extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "stage8d2_brset_full", "preliminary": False, "rehearsal": False}
    )


def test_stage8d1_guard_does_not_fire_for_stage8c_smoke(extract_mod) -> None:
    assert not extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "stage8c_brset_smoke"}
    )


def test_stage8d1_guard_does_not_fire_for_stage8d3_matrix(extract_mod) -> None:
    assert not extract_mod._is_stage8d1_rehearsal_config(
        {"run_mode": "stage8d3_brset_convnext_multitask"}
    )


# ---------------------------------------------------------------------------
# Stage 8D-1 unit tests — full _enforce_stage8a_limit helper
# ---------------------------------------------------------------------------


def test_enforce_limit_raises_systemexit_for_rehearsal_no_limit(extract_mod) -> None:
    """_enforce_stage8a_limit must raise SystemExit for rehearsal config without limit."""
    cfg = {"run_mode": "stage8d1_brset_rehearsal", "rehearsal": True}
    with pytest.raises(SystemExit):
        extract_mod._enforce_stage8a_limit(cfg, limit=None)


def test_enforce_limit_passes_for_rehearsal_with_limit(extract_mod) -> None:
    """_enforce_stage8a_limit must not raise when limit is provided."""
    cfg = {"run_mode": "stage8d1_brset_rehearsal", "rehearsal": True}
    extract_mod._enforce_stage8a_limit(cfg, limit=1024)  # must not raise


def test_enforce_limit_passes_for_full_run_no_limit(extract_mod) -> None:
    """_enforce_stage8a_limit must not raise for a non-guarded full-run config."""
    cfg = {"run_mode": "stage8d2_brset_full", "preliminary": False, "rehearsal": False}
    extract_mod._enforce_stage8a_limit(cfg, limit=None)  # must not raise


# ---------------------------------------------------------------------------
# Stage 8D-1 subprocess test — no-limit exits before backbone/dataset work
# ---------------------------------------------------------------------------


def test_stage8d1_guard_exits_nonzero_without_limit() -> None:
    """Stage 8D-1 rehearsal config without --limit must exit nonzero before data loading."""
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", _STAGE8D1_CONFIG],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode != 0, (
        "scripts/03_extract_embeddings.py should exit nonzero when --limit is "
        "omitted for a stage8d1 rehearsal config."
    )
    combined = result.stdout + result.stderr
    assert "8d-1" in combined.lower() or "rehearsal" in combined.lower(), (
        f"Exit message should mention '8D-1' or 'rehearsal': {combined[:300]!r}"
    )
