#!/usr/bin/env python
"""
scripts/03_extract_embeddings.py -- Extract and cache embeddings for a dataset.

Thin orchestration script. No business logic lives here.
Logic is in src/retina_screen/embeddings.py and preprocessing.py.

Usage:
    python scripts/03_extract_embeddings.py --config configs/experiment/smoke_dummy.yaml
    python scripts/03_extract_embeddings.py --config configs/experiment/smoke_dummy.yaml --limit 8
    python scripts/03_extract_embeddings.py --config configs/experiment/smoke_dummy.yaml --overwrite
"""

from __future__ import annotations

import argparse
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
from retina_screen.preprocessing import PreprocessingConfig


def _build_backbone_config(backbone_raw: dict) -> BackboneConfig:
    return BackboneConfig(
        name=backbone_raw["name"],
        embedding_dim=int(backbone_raw["embedding_dim"]),
        model_type=backbone_raw["model_type"],
        version=backbone_raw.get("version", ""),
    )


def _build_prep_config(prep_raw: dict) -> PreprocessingConfig:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache embeddings.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples to extract (default: all).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if a valid cache entry exists.",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    backbone_name = cfg.get("backbone", "mock")
    prep_name = cfg.get("preprocessing", "default_224")
    n_patients = cfg.get("n_patients", 80)
    cache_root = Path(cfg.get("cache_root", "cache/embeddings"))

    backbone_raw = load_config(Path(f"configs/backbone/{backbone_name}.yaml"))
    prep_raw = load_config(Path(f"configs/preprocessing/{prep_name}.yaml"))

    backbone_config = _build_backbone_config(backbone_raw)
    prep_config = _build_prep_config(prep_raw)

    device = get_device()
    backbone = load_backbone(backbone_config, device)

    adapter = DummyAdapter(n_patients=n_patients)
    manifest = adapter.build_manifest()
    logger.info("Manifest: %d samples (limit=%s)", len(manifest), args.limit)

    manifest_path = extract_embeddings(
        manifest=manifest,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=cache_root,
        device=device,
        image_loader=lambda sample: adapter.load_image(sample.sample_id),
        batch_size=32,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    failed = verify_cache_integrity(manifest_path, backbone_config.embedding_dim)
    if failed:
        raise RuntimeError(
            f"Cache integrity verification failed for {len(failed)} sample(s): "
            f"{failed[:10]}"
        )

    n_extracted = min(len(manifest), args.limit) if args.limit is not None else len(manifest)
    logger.info(
        "Extraction complete: %d samples, manifest at %s", n_extracted, manifest_path
    )


if __name__ == "__main__":
    main()
