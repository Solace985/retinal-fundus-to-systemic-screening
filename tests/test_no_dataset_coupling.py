"""tests/test_no_dataset_coupling.py -- Static scan for forbidden dataset-internal vocabulary.

Scans source files outside allowed locations (adapters/configs/docs/tests) for patterns
that should only appear inside dataset adapters. Based on 04_forbidden_patterns.md.
"""
from __future__ import annotations
import ast
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / 'src' / 'retina_screen'

# (pattern, description) -- forbidden outside adapters/configs/docs/tests
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r'diagnostic_keywords', 'ODIR native diagnostic-keyword field'),
    (r'Left-Fundus', 'ODIR native image-field name'),
    (r'Right-Fundus', 'ODIR native image-field name'),
    (r'left_fundus', 'ODIR native column name'),
    (r'right_fundus', 'ODIR native column name'),
    (r'\bCanon\b', 'Camera vendor name (Canon)'),
    (r'\bNikon\b', 'Camera vendor name (Nikon)'),
    (r'\bKowa\b', 'Camera vendor name (Kowa)'),
    (r'\bZeiss\b', 'Camera vendor name (Zeiss)'),
    (r'\bPhelcom\b', 'Camera vendor name (Phelcom)'),
]

DATASET_CONDITIONAL_VALUES: frozenset[str] = frozenset({"odir", "brset", "mbrset"})
DATASET_NAME_VARIABLES: frozenset[str] = frozenset({"dataset", "dataset_name"})


def _get_scannable_files() -> list[Path]:
    scannable: list[Path] = []
    if not SRC_ROOT.exists():
        return scannable
    for py_file in SRC_ROOT.rglob('*.py'):
        rel = py_file.relative_to(SRC_ROOT)
        if rel.parts[0] == 'adapters':
            continue
        scannable.append(py_file)
    scripts_dir = REPO_ROOT / 'scripts'
    if scripts_dir.exists():
        scannable.extend(scripts_dir.rglob('*.py'))
    return scannable


def _scan_file(file_path: Path, pattern: str) -> list[tuple[int, str]]:
    try:
        text = file_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    compiled = re.compile(pattern)
    return [
        (lineno, line.rstrip())
        for lineno, line in enumerate(text.splitlines(), start=1)
        if compiled.search(line)
    ]


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _is_dataset_selector(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in DATASET_NAME_VARIABLES
    if isinstance(node, ast.Attribute):
        return _attribute_path(node)[-2:] == ("dataset", "name")
    return False


def _is_dataset_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in DATASET_CONDITIONAL_VALUES


def _comparison_is_dataset_conditional(node: ast.Compare) -> bool:
    operands = [node.left, *node.comparators]
    has_dataset_selector = any(_is_dataset_selector(operand) for operand in operands)
    has_dataset_literal = any(_is_dataset_literal(operand) for operand in operands)
    has_equality_operator = any(isinstance(op, (ast.Eq, ast.In)) for op in node.ops)
    return has_dataset_selector and has_dataset_literal and has_equality_operator


def _scan_dataset_conditionals(file_path: Path) -> list[tuple[int, str]]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not source.strip():
        return []
    tree = ast.parse(source, filename=str(file_path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            comparisons = [
                child for child in ast.walk(node.test) if isinstance(child, ast.Compare)
            ]
            if any(_comparison_is_dataset_conditional(comp) for comp in comparisons):
                hits.append((node.lineno, ast.unparse(node.test)))
    return hits


_SCANNABLE = _get_scannable_files()

_CASES: list[tuple[Path, str, str]] = [
    (f, pat, desc)
    for f in _SCANNABLE
    for pat, desc in FORBIDDEN_PATTERNS
]

_IDS = [
    f"{c[0].relative_to(REPO_ROOT)}::{c[2]}"
    for c in _CASES
]


@pytest.mark.parametrize('filepath,pattern,description', _CASES, ids=_IDS)
def test_no_forbidden_vocabulary(filepath: Path, pattern: str, description: str) -> None:
    hits = _scan_file(filepath, pattern)
    detail = '; '.join(f'line {ln}: {txt}' for ln, txt in hits)
    assert len(hits) == 0, (
        f"Forbidden {description!r} found in {filepath.relative_to(REPO_ROOT)}: {detail}"
    )


def test_core_files_in_scan() -> None:
    names = {f.name for f in _SCANNABLE}
    for expected in ('core.py', 'schema.py', 'tasks.py', 'feature_policy.py'):
        assert expected in names, f"{expected} missing from scan list"


def test_adapters_excluded_from_scan() -> None:
    for f in _SCANNABLE:
        assert 'adapters' not in f.parts, f"Adapter file {f} should not be scanned"


@pytest.mark.parametrize("filepath", _SCANNABLE, ids=[str(f.relative_to(REPO_ROOT)) for f in _SCANNABLE])
def test_no_dataset_specific_conditionals(filepath: Path) -> None:
    hits = _scan_dataset_conditionals(filepath)
    detail = "; ".join(f"line {ln}: {expr}" for ln, expr in hits)
    assert hits == [], (
        f"Forbidden dataset-specific conditional found in "
        f"{filepath.relative_to(REPO_ROOT)}: {detail}"
    )


def test_dataset_conditional_scan_detects_dataset_name(tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('if dataset_name == "odir":\n    pass\n', encoding="utf-8")
    assert _scan_dataset_conditionals(bad) == [(1, "dataset_name == 'odir'")]


def test_dataset_conditional_scan_detects_dataset_variable(tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('if dataset == "mbrset":\n    pass\n', encoding="utf-8")
    assert _scan_dataset_conditionals(bad) == [(1, "dataset == 'mbrset'")]


def test_dataset_conditional_scan_detects_config_dataset_name(tmp_path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text('if config.dataset.name == "brset":\n    pass\n', encoding="utf-8")
    assert _scan_dataset_conditionals(bad) == [(1, "config.dataset.name == 'brset'")]


def test_dataset_conditional_scan_allows_harmless_lowercase_mentions(tmp_path) -> None:
    ok = tmp_path / "ok.py"
    ok.write_text('message = "odir adapter configured elsewhere"\n', encoding="utf-8")
    assert _scan_dataset_conditionals(ok) == []
