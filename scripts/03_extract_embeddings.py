#!/usr/bin/env python
"""
scripts/03_extract_embeddings.py -- Extract and cache backbone embeddings.

Thin orchestration script. Business logic lives in src/retina_screen/.

Usage:
    python scripts/03_extract_embeddings.py --config configs/experiment/baseline_odir_dinov2.yaml --limit 32
    python scripts/03_extract_embeddings.py --config configs/experiment/smoke_dummy.yaml --limit 8

--limit N:  extract at most N samples total; split-aware when splits.csv exists.
--overwrite: re-extract even when valid cache entries exist.

Split-aware extraction (when splits.csv exists from scripts/01_make_splits.py):
    Allocates --limit slots across splits proportionally (train 62%, val 19%, test 19%).
    e.g. --limit 32 → train=20, val=6, test=6
    Ensures downstream train/eval scripts have cached samples in each relevant split.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.core import get_device, load_config, seed_everything, setup_logging
from retina_screen.embeddings import (
    BackboneConfig,
    extract_embeddings,
    load_backbone,
    verify_cache_integrity,
)
from retina_screen.preprocessing import PreprocessingConfig, get_preprocessing_hash

logger = logging.getLogger(__name__)

# Default proportional allocation across splits when using --limit.
_SPLIT_ALLOC = {"train": 0.625, "val": 0.1875, "test": 0.1875}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_adapter(cfg: dict):
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


def _build_backbone_config(cfg: dict) -> BackboneConfig:
    backbone_name = cfg.get("backbone", "mock")
    backbone_raw = load_config(Path(f"configs/backbone/{backbone_name}.yaml"))
    return BackboneConfig(
        name=backbone_raw["name"],
        embedding_dim=int(backbone_raw["embedding_dim"]),
        model_type=backbone_raw["model_type"],
        version=backbone_raw.get("version", ""),
    )


def _build_prep_config(cfg: dict) -> PreprocessingConfig:
    prep_name = cfg.get("preprocessing", "default_224")
    prep_raw = load_config(Path(f"configs/preprocessing/{prep_name}.yaml"))
    return PreprocessingConfig(
        image_size=int(prep_raw.get("image_size", 224)),
        mean=tuple(prep_raw.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(prep_raw.get("std", [0.229, 0.224, 0.225])),
        use_clahe=bool(prep_raw.get("use_clahe", False)),
        use_graham=bool(prep_raw.get("use_graham", False)),
        interpolation=str(prep_raw.get("interpolation", "bilinear")),
        random_hflip_p=float(prep_raw.get("random_hflip_p", 0.0)),
        random_rotation_deg=float(prep_raw.get("random_rotation_deg", 0.0)),
        color_jitter=bool(prep_raw.get("color_jitter", False)),
    )


def _latest_splits_csv(dataset: str) -> Path | None:
    """Return the most recent splits.csv for *dataset*, or None."""
    splits_root = Path("outputs") / "splits" / dataset
    if not splits_root.exists():
        return None
    dirs = sorted(splits_root.iterdir(), key=lambda p: p.name, reverse=True)
    for d in dirs:
        p = d / "splits.csv"
        if p.exists():
            return p
    return None


def _load_splits_csv(splits_csv: Path) -> dict[str, list[str]]:
    """Load splits.csv → {split_name: [sample_id, ...]}."""
    split: dict[str, list[str]] = {}
    with splits_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split.setdefault(row["split_name"], []).append(row["sample_id"])
    return split


def _select_samples_split_aware(
    manifest_sids: list[str],
    split: dict[str, list[str]],
    limit: int,
) -> list[str]:
    """Select *limit* sample IDs distributed across splits proportionally.

    Allocates slots to train/val/test by _SPLIT_ALLOC fractions. If a split
    has fewer samples than its slot, remaining slots go to other splits
    (sorted deterministically by split name then sample_id).

    Returns a flat list of selected sample_ids in a deterministic order.
    """
    split_order = ["train", "val", "test", "reliability"]
    alloc_fracs = {k: _SPLIT_ALLOC.get(k, 0.0) for k in split_order}

    # Compute integer slots (floor first, then distribute remainder).
    slots: dict[str, int] = {}
    remaining = limit
    for name in split_order:
        slots[name] = int(limit * alloc_fracs[name])
        remaining -= slots[name]
    # Distribute remainder to the splits with highest fractional part.
    if remaining > 0:
        fractional = sorted(
            split_order, key=lambda n: (limit * alloc_fracs[n]) % 1, reverse=True
        )
        for name in fractional:
            if remaining <= 0:
                break
            slots[name] += 1
            remaining -= 1

    manifest_set = set(manifest_sids)
    selected: list[str] = []
    leftover: int = 0

    per_split: dict[str, list[str]] = {}
    for name in split_order:
        available = sorted(sid for sid in split.get(name, []) if sid in manifest_set)
        want = slots[name] + leftover
        taken = available[:want]
        leftover = want - len(taken)
        per_split[name] = taken
        selected.extend(taken)
        logger.info(
            "Split %-12s: allocated=%d, available=%d, selected=%d",
            name, slots[name], len(available), len(taken),
        )

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache backbone embeddings."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of samples to extract.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if valid cache entries exist.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    backbone_config = _build_backbone_config(cfg)
    prep_config = _build_prep_config(cfg)
    cache_root = Path(cfg.get("cache_root", "cache/embeddings"))
    dataset = cfg.get("dataset", "dummy")

    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: reported_backbone_is_smoke=true. "
            "Using mock backbone, NOT %s.",
            cfg.get("real_backbone_target", "?"),
        )

    device = get_device()
    backbone = load_backbone(backbone_config, device)

    adapter = _build_adapter(cfg)
    manifest = adapter.build_manifest()
    manifest_sids = [s.sample_id for s in manifest]

    # Determine sample selection.
    if args.limit is not None:
        splits_csv = _latest_splits_csv(dataset)
        if splits_csv is not None:
            logger.info("Using split-aware selection from: %s", splits_csv)
            split = _load_splits_csv(splits_csv)
            selected_sids = _select_samples_split_aware(manifest_sids, split, args.limit)
        else:
            logger.warning(
                "splits.csv not found for dataset=%r. "
                "Selecting first %d samples from manifest deterministically. "
                "Run scripts/01_make_splits.py first for split-aware selection.",
                dataset, args.limit,
            )
            selected_sids = manifest_sids[: args.limit]
        sid_set = set(selected_sids)
        manifest_subset = [s for s in manifest if s.sample_id in sid_set]
        logger.info("Selected %d / %d samples for extraction.", len(manifest_subset), len(manifest))
    else:
        manifest_subset = manifest
        logger.info("Extracting all %d samples.", len(manifest_subset))

    manifest_path = extract_embeddings(
        manifest=manifest_subset,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=cache_root,
        device=device,
        image_loader=lambda sample: adapter.load_image(sample.sample_id),
        batch_size=32,
        overwrite=args.overwrite,
        limit=None,  # limit already applied above
    )

    failed = verify_cache_integrity(manifest_path, backbone_config.embedding_dim)
    if failed:
        raise RuntimeError(
            f"Cache integrity verification failed for {len(failed)} sample(s): "
            f"{failed[:10]}"
        )

    logger.info(
        "Extraction complete: %d samples, manifest at %s",
        len(manifest_subset), manifest_path,
    )


if __name__ == "__main__":
    main()
