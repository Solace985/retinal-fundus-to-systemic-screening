#!/usr/bin/env python
"""
scripts/04_train.py -- Train a multi-task head on cached embeddings.

Thin orchestration script. Business logic lives in src/retina_screen/.

Usage:
    python scripts/04_train.py --config configs/experiment/baseline_odir_dinov2.yaml
    python scripts/04_train.py --config configs/experiment/baseline_odir_dinov2.yaml --fast_dev_run

--fast_dev_run: 1 epoch, 1 step on available cached train samples. Always saves artifacts.

Prerequisites:
    scripts/01_make_splits.py --config <same_config>
    scripts/03_extract_embeddings.py --config <same_config> [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.core import (
    capture_env_info,
    capture_git_info,
    get_device,
    load_config,
    make_run_dir,
    save_resolved_config,
    seed_everything,
    setup_logging,
)
from retina_screen.data import build_task_targets_and_masks
from retina_screen.embeddings import (
    BackboneConfig,
    load_embedding,
    load_embedding_manifest,
)
from retina_screen.model import MultiTaskHead
from retina_screen.preprocessing import PreprocessingConfig
from retina_screen.training import KendallUncertaintyWeighting, train_one_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_adapter(cfg: dict):
    from retina_screen.adapters.dummy import DummyAdapter  # noqa: PLC0415
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


def _latest_splits_dir(dataset: str) -> Path | None:
    """Return the most recently created splits directory for *dataset*, or None."""
    splits_root = Path("outputs") / "splits" / dataset
    if not splits_root.exists():
        return None
    dirs = sorted(splits_root.iterdir(), key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def _load_splits_csv(splits_csv: Path) -> dict[str, list[str]]:
    """Load splits.csv → {split_name: [sample_id, ...]}."""
    split: dict[str, list[str]] = {}
    with splits_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split.setdefault(row["split_name"], []).append(row["sample_id"])
    return split


def _latest_manifest_path(backbone_name: str, dataset_source: str, prep_hash: str, cache_root: Path) -> Path | None:
    cache_dir = cache_root / backbone_name / dataset_source / prep_hash
    mp = cache_dir / "manifest.csv"
    return mp if mp.exists() else None


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task head on cached embeddings.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--fast_dev_run", action="store_true",
        help="1 epoch, 1 step on all available cached train samples. Always saves artifacts.",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device()

    dataset = cfg.get("dataset", "dummy")
    backbone_config = _build_backbone_config(cfg)
    prep_config = _build_prep_config(cfg)
    cache_root = Path(cfg.get("cache_root", "cache/embeddings"))

    from retina_screen.preprocessing import get_preprocessing_hash  # noqa: PLC0415
    prep_hash = get_preprocessing_hash(prep_config)
    embedding_dim = backbone_config.embedding_dim

    # --- Require splits ---
    splits_dir = _latest_splits_dir(dataset)
    if splits_dir is None or not (splits_dir / "splits.csv").exists():
        logger.error(
            "splits.csv not found. Run first:\n"
            "  python scripts/01_make_splits.py --config %s", args.config
        )
        sys.exit(1)
    logger.info("Loading splits from: %s", splits_dir / "splits.csv")
    split = _load_splits_csv(splits_dir / "splits.csv")

    # --- Require embedding manifest ---
    manifest_path = _latest_manifest_path(
        backbone_config.name, dataset, prep_hash, cache_root
    )
    if manifest_path is None:
        logger.error(
            "Embedding manifest not found. Run first:\n"
            "  python scripts/03_extract_embeddings.py --config %s [--limit 32]", args.config
        )
        sys.exit(1)
    logger.info("Loading embedding manifest from: %s", manifest_path)
    emb_records = load_embedding_manifest(manifest_path)
    cached_ids = {r["sample_id"]: r for r in emb_records}

    # --- Build adapter + manifest ---
    adapter = _build_adapter(cfg)
    manifest_samples = adapter.build_manifest()
    sample_lookup = {s.sample_id: s for s in manifest_samples}
    supported_tasks = adapter.get_supported_tasks()

    # --- Intersect train split with cached embeddings ---
    train_sids_all = split.get("train", [])
    train_cached = [sid for sid in train_sids_all if sid in cached_ids and sid in sample_lookup]
    logger.info(
        "Train split: %d total, %d with cached embeddings",
        len(train_sids_all), len(train_cached),
    )
    if not train_cached:
        logger.error(
            "No cached embeddings found for train split. "
            "Run scripts/03_extract_embeddings.py first."
        )
        sys.exit(1)

    # --- Load embeddings and build targets/masks ---
    embeddings_list: list[torch.Tensor] = []
    valid_samples = []
    for sid in train_cached:
        rec = cached_ids[sid]
        emb = load_embedding(
            Path(rec["cache_path"]),
            rec["checksum"],
            expected_dim=embedding_dim,
        )
        embeddings_list.append(emb)
        valid_samples.append(sample_lookup[sid])

    train_emb = torch.stack(embeddings_list)  # (N, embedding_dim)
    batch = build_task_targets_and_masks(valid_samples, supported_tasks)
    train_targets = {t: torch.tensor(batch.targets[t]) for t in supported_tasks}
    train_masks = {t: torch.tensor(batch.masks[t]) for t in supported_tasks}

    logger.info(
        "Training batch: %d samples, tasks=%s", len(valid_samples), supported_tasks
    )

    # --- Model ---
    model = MultiTaskHead(embedding_dim=embedding_dim, task_names=supported_tasks)
    weighter = KendallUncertaintyWeighting(task_names=supported_tasks)
    params = list(model.parameters()) + list(weighter.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    n_epochs = 1 if args.fast_dev_run else cfg.get("n_epochs", 3)

    # --- Training loop ---
    train_log: list[dict] = []
    for epoch in range(n_epochs):
        result = train_one_step(
            model, optimizer, train_emb, train_targets, train_masks,
            supported_tasks, loss_weighter=weighter,
        )
        row = {"epoch": epoch, "train_loss": result["total_loss"], "lr": 1e-3}
        train_log.append(row)
        logger.info(
            "Epoch %d  loss=%.4f  grad_norm=%.3f", epoch,
            result["total_loss"], result["grad_norm"],
        )
        if args.fast_dev_run:
            break

    # --- Save artifacts ---
    run_dir = make_run_dir(
        Path("runs") / ("fast_dev_run" if args.fast_dev_run else "train"),
        dataset,
    )

    resolved_cfg = {
        **cfg,
        "git": {k: str(v) for k, v in capture_git_info().items()},
        "env": {k: str(v) for k, v in capture_env_info().items()},
        "fast_dev_run": args.fast_dev_run,
        "n_train_cached": len(train_cached),
        "splits_dir": str(splits_dir),
        "embedding_manifest": str(manifest_path),
    }
    save_resolved_config(resolved_cfg, run_dir)

    csv_path = run_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "lr"])
        writer.writeheader()
        writer.writerows(train_log)
    logger.info("Saved train_log.csv")

    torch.save(model.state_dict(), run_dir / "model_checkpoint.pt")
    logger.info("Saved model_checkpoint.pt")

    smoke_meta = {
        "run_mode": cfg.get("run_mode", "train"),
        "reported_backbone_is_smoke": cfg.get("reported_backbone_is_smoke", False),
        "real_backbone_target": cfg.get("real_backbone_target", ""),
        "fast_dev_run": args.fast_dev_run,
        "n_samples_used": len(valid_samples),
        "n_steps": len(train_log),
        "train_loss_final": train_log[-1]["train_loss"] if train_log else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with (run_dir / "smoke_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(smoke_meta, fh, indent=2)
    logger.info("Saved smoke_metadata.json")

    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: reported_backbone_is_smoke=true. "
            "Embeddings are from mock backbone, NOT %s. "
            "Do not report these metrics as real %s results.",
            cfg.get("real_backbone_target", "?"),
            cfg.get("real_backbone_target", "?"),
        )

    logger.info("Training complete. Artifacts at: %s", run_dir)


if __name__ == "__main__":
    main()
