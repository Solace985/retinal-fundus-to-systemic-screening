"""
core.py -- Project-level utilities: config loading, logging, seeding, run directories.

Owns: YAML loading, logging setup, deterministic seeding, run directory creation,
path helpers, git/environment capture, device selection.

Must not contain: schema definitions, task definitions, adapter logic, model
architecture, training loops, or evaluation metrics.
"""

from __future__ import annotations

import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
) -> None:
    """Configure the root logger with an optional file handler.

    Call once at the start of a script or training run.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
    logger.debug("Logging configured at level=%s", level.upper())


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed random, numpy, and torch (when available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.debug("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a plain dict.

    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the YAML root is not a mapping.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping (got {type(cfg).__name__}): {path}"
        )
    logger.debug("Loaded config from %s", path)
    return cfg


def merge_configs(*cfgs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge multiple config dicts; later dicts override earlier ones."""
    merged: dict[str, Any] = {}
    for cfg in cfgs:
        merged.update(cfg)
    return merged


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def make_run_dir(base: str | Path, run_name: str) -> Path:
    """Create and return a timestamped run directory under *base*.

    Directory name: <run_name>_<YYYYMMDD_HHMMSS>.
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory created: %s", run_dir)
    return run_dir


def save_resolved_config(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    """Write the resolved config dict to resolved_config.yaml in *run_dir*."""
    out = Path(run_dir) / "resolved_config.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    logger.info("Resolved config saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the repository root (the directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate project root (no pyproject.toml found)")


# ---------------------------------------------------------------------------
# Git / environment capture
# ---------------------------------------------------------------------------


def capture_git_info() -> dict[str, str]:
    """Capture the current git commit hash and dirty-diff stat.

    Returns a dict with keys 'commit' and 'dirty_diff_stat' on success,
    or 'error' on failure (git not available or not a git repo).
    """
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff_stat = subprocess.check_output(
            ["git", "diff", "--stat"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty_diff_stat"] = diff_stat or "clean"
    except Exception as exc:
        info["error"] = str(exc)
        logger.debug("Git info unavailable: %s", exc)
    return info


def capture_env_info() -> dict[str, str]:
    """Capture Python version and installed versions of key packages."""
    import importlib

    info: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": sys.platform,
    }
    for pkg in ("torch", "numpy", "pandas", "sklearn", "pydantic"):
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            info[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[pkg] = "not_installed"
    return info


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device() -> Any:
    """Return the best available torch device (cuda > mps > cpu).

    Returns a torch.device when torch is installed, or the string 'cpu'
    when torch is not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.debug("Using device: %s", device)
        return device
    except ImportError:
        logger.debug("torch not available; returning 'cpu'")
        return "cpu"
