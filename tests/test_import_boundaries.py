"""
tests/test_import_boundaries.py -- AST-based static import-boundary checks.

Verifies that the import dependency graph matches the architecture contract
(docs/ai_context/01_architecture_contract.md). These tests pass trivially
when downstream files are empty stubs (no imports = no violations).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src" / "retina_screen"
ADAPTERS_DIR = SRC_ROOT / "adapters"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_imported_modules(file_path: Path) -> list[str]:
    """Return a flat list of module names imported by *file_path*.

    Returns an empty list if the file does not exist or is empty. Syntax errors
    fail loudly so malformed files cannot bypass import-boundary checks.
    """
    if not file_path.exists():
        return []
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(
            f"Could not parse {file_path}: {exc.msg} at line {exc.lineno}, "
            f"column {exc.offset}"
        ) from exc

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Resolve relative imports relative to retina_screen package
                if node.level and node.level > 0:
                    # Relative import: treat as retina_screen.<module>
                    modules.append(f"retina_screen.{node.module or ''}")
                else:
                    modules.append(node.module)
    return modules


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


# ---------------------------------------------------------------------------
# Foundation modules
# ---------------------------------------------------------------------------


def test_schema_imports_no_project_modules():
    """schema.py must only import standard library / third-party packages."""
    mods = get_imported_modules(SRC_ROOT / "schema.py")
    violations = [m for m in mods if m.startswith("retina_screen")]
    assert violations == [], (
        f"schema.py must not import any project module, found: {violations}"
    )


def test_tasks_imports_only_schema():
    """tasks.py may import retina_screen.schema but no other project module."""
    mods = get_imported_modules(SRC_ROOT / "tasks.py")
    violations = [
        m for m in mods
        if m.startswith("retina_screen") and not m.startswith("retina_screen.schema")
    ]
    assert violations == [], (
        f"tasks.py must not import project modules beyond schema, found: {violations}"
    )


def test_feature_policy_imports_only_schema_and_tasks():
    """feature_policy.py may import schema and tasks but no other project modules."""
    mods = get_imported_modules(SRC_ROOT / "feature_policy.py")
    allowed = ("retina_screen.schema", "retina_screen.tasks")
    violations = [
        m for m in mods
        if m.startswith("retina_screen") and not _starts_with_any(m, allowed)
    ]
    assert violations == [], (
        f"feature_policy.py imports unexpected project modules: {violations}"
    )


def test_core_no_forbidden_imports():
    """core.py must not import adapters, model, training, or evaluation."""
    mods = get_imported_modules(SRC_ROOT / "core.py")
    forbidden = (
        "retina_screen.adapters",
        "retina_screen.model",
        "retina_screen.training",
        "retina_screen.evaluation",
    )
    violations = [m for m in mods if _starts_with_any(m, forbidden)]
    assert violations == [], (
        f"core.py must not import {forbidden}, found: {violations}"
    )


# ---------------------------------------------------------------------------
# Adapter boundary: adapters must not import model/training/evaluation/etc.
# ---------------------------------------------------------------------------

_ADAPTER_FILES = [
    f for f in ADAPTERS_DIR.glob("*.py") if f.name != "__init__.py"
] if ADAPTERS_DIR.exists() else []

_ADAPTER_FORBIDDEN = (
    "retina_screen.model",
    "retina_screen.training",
    "retina_screen.evaluation",
    "retina_screen.continual",
    "retina_screen.dashboard_app",
    "retina_screen.reporting",
)


@pytest.mark.parametrize("adapter_file", _ADAPTER_FILES, ids=[f.name for f in _ADAPTER_FILES])
def test_adapter_no_model_training_evaluation(adapter_file: Path):
    """Adapters must not import model, training, evaluation, continual, dashboard, or reporting."""
    mods = get_imported_modules(adapter_file)
    violations = [m for m in mods if _starts_with_any(m, _ADAPTER_FORBIDDEN)]
    assert violations == [], (
        f"{adapter_file.name} must not import {_ADAPTER_FORBIDDEN}, found: {violations}"
    )


# ---------------------------------------------------------------------------
# Model must not import concrete adapters
# ---------------------------------------------------------------------------


def test_model_no_concrete_adapter_imports():
    """model.py must not import any concrete adapter."""
    model_file = SRC_ROOT / "model.py"
    mods = get_imported_modules(model_file)
    violations = [m for m in mods if m.startswith("retina_screen.adapters")]
    assert violations == [], (
        f"model.py must not import adapters, found: {violations}"
    )


# ---------------------------------------------------------------------------
# Evaluation must not import concrete adapters
# ---------------------------------------------------------------------------


def test_evaluation_no_concrete_adapter_imports():
    """evaluation.py must not import any concrete adapter."""
    eval_file = SRC_ROOT / "evaluation.py"
    mods = get_imported_modules(eval_file)
    violations = [m for m in mods if m.startswith("retina_screen.adapters")]
    assert violations == [], (
        f"evaluation.py must not import adapters, found: {violations}"
    )


# ---------------------------------------------------------------------------
# Dashboard must not import training (no live retraining)
# ---------------------------------------------------------------------------


def test_dashboard_no_training_import():
    """dashboard_app.py must not import training.py (dashboard is inference-only)."""
    dashboard_file = SRC_ROOT / "dashboard_app.py"
    mods = get_imported_modules(dashboard_file)
    violations = [m for m in mods if m.startswith("retina_screen.training")]
    assert violations == [], (
        "dashboard_app.py imports training -- this would allow live model updates. "
        f"Found: {violations}"
    )


# ---------------------------------------------------------------------------
# Sanity: the helper itself works correctly
# ---------------------------------------------------------------------------


def test_get_imported_modules_handles_empty_file(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("", encoding="utf-8")
    assert get_imported_modules(empty) == []


def test_get_imported_modules_handles_missing_file(tmp_path):
    missing = tmp_path / "missing.py"
    assert get_imported_modules(missing) == []


def test_get_imported_modules_handles_syntax_error(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def (:", encoding="utf-8")
    with pytest.raises(AssertionError, match="Could not parse"):
        get_imported_modules(bad)
