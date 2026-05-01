#!/usr/bin/env python
"""
scripts/02_verify_backbone_one_image.py -- Verify backbone with one image.

Thin orchestration script. Proves: backbone loads → one image preprocesses →
embedding extracted → saved → reloaded → checksum and dimension verified.

No business logic lives here. Logic is in src/retina_screen/.

Usage:
    python scripts/02_verify_backbone_one_image.py --config configs/experiment/smoke_dummy.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.core import get_device, load_config, seed_everything, setup_logging
from retina_screen.embeddings import (
    BackboneConfig,
    get_cache_dir,
    load_backbone,
    load_embedding,
    save_embedding,
)
from retina_screen.preprocessing import (
    PreprocessingConfig,
    get_preprocessing_hash,
    preprocess_image,
)


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
    parser = argparse.ArgumentParser(
        description="Verify backbone loading and one-image embedding extraction."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
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
    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: reported_backbone_is_smoke=true. "
            "Using mock backbone, NOT %s. Do not report results as %s performance.",
            cfg.get("real_backbone_target", "?"),
            cfg.get("real_backbone_target", "?"),
        )
    logger.info(
        "Backbone loaded: name=%s, model_type=%s, embedding_dim=%d, device=%s",
        backbone_config.name, backbone_config.model_type,
        backbone_config.embedding_dim, device,
    )

    # Build adapter from config and take the first sample.
    def _make_dummy(c: dict):
        from retina_screen.adapters.dummy import DummyAdapter  # noqa: PLC0415
        return DummyAdapter(n_patients=c.get("n_patients", 80))

    def _make_odir(c: dict):
        from retina_screen.adapters.odir import ODIRAdapter  # noqa: PLC0415
        return ODIRAdapter(dataset_root=c.get("dataset_root", "ODIR-5K"))

    _builders = {"dummy": _make_dummy, "odir": _make_odir}
    _ds_name = cfg.get("dataset", "dummy")
    _builder = _builders.get(_ds_name)
    if _builder is None:
        raise ValueError(f"Unknown dataset={_ds_name!r}. Supported: {sorted(_builders)}")
    adapter = _builder(cfg)

    manifest = adapter.build_manifest()
    sample = manifest[0]
    logger.info("Using sample_id=%s (patient_id=%s, dataset=%s)",
                sample.sample_id, sample.patient_id, _ds_name)

    # Preprocess.
    img = adapter.load_image(sample.sample_id)
    tensor = preprocess_image(img, prep_config, mode="extract")
    logger.info("Image preprocessed: input_size=%s, output_shape=%s", img.size, tuple(tensor.shape))

    # Extract embedding.
    batch = tensor.unsqueeze(0).to(device)   # (1, C, H, W)
    with torch.no_grad():
        embedding = backbone(batch).squeeze(0).cpu()   # (embedding_dim,)
    logger.info("Embedding extracted: shape=%s", tuple(embedding.shape))

    # Save and reload.
    prep_hash = get_preprocessing_hash(prep_config)
    cache_dir = get_cache_dir(
        cache_root, backbone_config.name, sample.dataset_source, prep_hash
    )
    cache_path, checksum = save_embedding(embedding, sample.sample_id, cache_dir)
    logger.info("Embedding saved: path=%s, checksum=%s...", cache_path, checksum[:12])

    reloaded = load_embedding(cache_path, checksum, backbone_config.embedding_dim)

    assert torch.allclose(embedding, reloaded), (
        f"Round-trip verification FAILED: embedding and reloaded tensor differ."
    )
    assert embedding.shape == (backbone_config.embedding_dim,), (
        f"Wrong embedding shape {embedding.shape}, expected ({backbone_config.embedding_dim},)."
    )

    logger.info(
        "Verification PASSED: backbone=%s, image_size=%d, embedding_dim=%d, "
        "cache_path=%s, checksum=%s...",
        backbone_config.name, prep_config.image_size,
        backbone_config.embedding_dim, cache_path, checksum[:12],
    )


if __name__ == "__main__":
    main()
