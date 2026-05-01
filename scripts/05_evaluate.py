#!/usr/bin/env python
"""
scripts/05_evaluate.py -- Evaluate a trained model on cached embeddings.

Thin orchestration script. Business logic lives in src/retina_screen/.

Usage:
    python scripts/05_evaluate.py --config configs/experiment/baseline_odir_dinov2.yaml

Requires:
    scripts/01_make_splits.py to have run (splits.csv)
    scripts/03_extract_embeddings.py to have run (embedding manifest)
    scripts/04_train.py to have run (model_checkpoint.pt)

Evaluation split selection:
    1. Test split cached embeddings (preferred)
    2. Val split (if test has no cached samples) — marked as limited_embedding_smoke
    3. Train split (emergency only) — marked as emergency_smoke_only
    Never silently evaluates on a different split than reported.
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
    get_device,
    load_config,
    ensure_dir,
    seed_everything,
    setup_logging,
)
from retina_screen.data import build_task_targets_and_masks
from retina_screen.embeddings import (
    BackboneConfig,
    load_embedding,
    load_embedding_manifest,
)
from retina_screen.evaluation import evaluate_predictions
from retina_screen.model import MultiTaskHead
from retina_screen.preprocessing import PreprocessingConfig, get_preprocessing_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with 04_train.py)
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
    splits_root = Path("outputs") / "splits" / dataset
    if not splits_root.exists():
        return None
    dirs = sorted(splits_root.iterdir(), key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def _load_splits_csv(splits_csv: Path) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {}
    with splits_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split.setdefault(row["split_name"], []).append(row["sample_id"])
    return split


def _latest_run_dir(dataset: str) -> Path | None:
    """Return the most recent run directory containing a model checkpoint."""
    for base in [Path("runs") / "fast_dev_run", Path("runs") / "train"]:
        if not base.exists():
            continue
        dirs = sorted(
            [d for d in base.iterdir() if (d / "model_checkpoint.pt").exists()],
            key=lambda p: p.name,
            reverse=True,
        )
        if dirs:
            return dirs[0]
    return None


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
    parser = argparse.ArgumentParser(description="Evaluate trained model on cached embeddings.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    dataset = cfg.get("dataset", "dummy")
    backbone_config = _build_backbone_config(cfg)
    prep_config = _build_prep_config(cfg)
    prep_hash = get_preprocessing_hash(prep_config)
    cache_root = Path(cfg.get("cache_root", "cache/embeddings"))
    embedding_dim = backbone_config.embedding_dim

    # --- Require splits ---
    splits_dir = _latest_splits_dir(dataset)
    if splits_dir is None or not (splits_dir / "splits.csv").exists():
        logger.error(
            "splits.csv not found. Run first:\n"
            "  python scripts/01_make_splits.py --config %s", args.config
        )
        sys.exit(1)
    split = _load_splits_csv(splits_dir / "splits.csv")

    # --- Require embedding manifest ---
    cache_dir = cache_root / backbone_config.name / dataset / prep_hash
    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        logger.error(
            "Embedding manifest not found. Run first:\n"
            "  python scripts/03_extract_embeddings.py --config %s [--limit 32]", args.config
        )
        sys.exit(1)
    emb_records = load_embedding_manifest(manifest_path)
    cached_ids = {r["sample_id"]: r for r in emb_records}

    # --- Require checkpoint ---
    run_dir = _latest_run_dir(dataset)
    if run_dir is None:
        logger.error(
            "No model_checkpoint.pt found. Run first:\n"
            "  python scripts/04_train.py --config %s --fast_dev_run", args.config
        )
        sys.exit(1)
    checkpoint_path = run_dir / "model_checkpoint.pt"
    logger.info("Using checkpoint: %s", checkpoint_path)

    # --- Build adapter + manifest ---
    adapter = _build_adapter(cfg)
    manifest_samples = adapter.build_manifest()
    sample_lookup = {s.sample_id: s for s in manifest_samples}
    supported_tasks = adapter.get_supported_tasks()

    # --- Select evaluation split (test → val → train, with explicit labelling) ---
    eval_split_name: str
    eval_reason: str
    final_test_result: bool

    test_cached = [
        sid for sid in split.get("test", []) if sid in cached_ids and sid in sample_lookup
    ]
    val_cached = [
        sid for sid in split.get("val", []) if sid in cached_ids and sid in sample_lookup
    ]
    train_cached = [
        sid for sid in split.get("train", []) if sid in cached_ids and sid in sample_lookup
    ]

    if test_cached:
        eval_sids = test_cached
        eval_split_name = "test"
        eval_reason = "cached_test_samples_available"
        final_test_result = True
    elif val_cached:
        eval_sids = val_cached
        eval_split_name = "val"
        eval_reason = "limited_embedding_smoke"
        final_test_result = False
        logger.warning(
            "No cached test samples available. Evaluating on val split instead. "
            "This is NOT a final test result (final_test_result=false)."
        )
    elif train_cached:
        eval_sids = train_cached
        eval_split_name = "train"
        eval_reason = "emergency_smoke_only"
        final_test_result = False
        logger.warning(
            "No cached test or val samples. Evaluating on TRAIN split. "
            "This is an emergency smoke-only result — do NOT use for reporting."
        )
    else:
        logger.error("No cached embeddings available in any split. Cannot evaluate.")
        sys.exit(1)

    logger.info(
        "Evaluation split=%s (%d samples, reason=%s, final_test_result=%s)",
        eval_split_name, len(eval_sids), eval_reason, final_test_result,
    )

    # --- Load embeddings and targets ---
    embeddings_list: list[torch.Tensor] = []
    eval_samples = []
    for sid in eval_sids:
        rec = cached_ids[sid]
        emb = load_embedding(
            Path(rec["cache_path"]), rec["checksum"], expected_dim=embedding_dim,
        )
        embeddings_list.append(emb)
        eval_samples.append(sample_lookup[sid])

    eval_emb = torch.stack(embeddings_list)

    # --- Load model ---
    model = MultiTaskHead(embedding_dim=embedding_dim, task_names=supported_tasks)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    logger.info("Model loaded from checkpoint.")

    # --- Inference ---
    with torch.no_grad():
        preds = model(eval_emb.to(device))
    preds_np = {t: preds[t].cpu().numpy() for t in supported_tasks}

    # --- Evaluate ---
    batch = build_task_targets_and_masks(eval_samples, supported_tasks)
    metrics = evaluate_predictions(
        predictions=preds_np,
        targets=batch.targets,
        masks=batch.masks,
        task_names=supported_tasks,
    )

    for task, results in metrics.items():
        for r in results:
            logger.info(
                "task=%-35s metric=%-10s status=%s value=%s reason=%s n=%d",
                task, r.metric_name, r.status.value,
                f"{r.value:.4f}" if r.value is not None else "N/A",
                r.reason or "-", r.n,
            )

    # --- Save smoke_metrics.json ---
    n_cached_per_split = {
        "test": len(test_cached),
        "val": len(val_cached),
        "train": len(train_cached),
    }
    metrics_dict = {
        task: [
            {"metric": r.metric_name, "value": r.value, "status": r.status.value,
             "reason": r.reason, "n": r.n}
            for r in results
        ]
        for task, results in metrics.items()
    }
    smoke_output = {
        "evaluation_split": eval_split_name,
        "reason": eval_reason,
        "final_test_result": final_test_result,
        "n_eval_samples": len(eval_sids),
        "n_cached_per_split": n_cached_per_split,
        "run_mode": cfg.get("run_mode", "train"),
        "reported_backbone_is_smoke": cfg.get("reported_backbone_is_smoke", False),
        "real_backbone_target": cfg.get("real_backbone_target", ""),
        "checkpoint": str(checkpoint_path),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": metrics_dict,
    }

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_out_dir = ensure_dir(Path("outputs") / "evaluation" / run_id)
    out_path = eval_out_dir / "smoke_metrics.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(smoke_output, fh, indent=2)
    logger.info("Evaluation complete. Metrics saved to: %s", out_path)

    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: backbone is mock, NOT %s. "
            "Do not report these metrics as real model performance.",
            cfg.get("real_backbone_target", "?"),
        )


if __name__ == "__main__":
    main()
