#!/usr/bin/env python
"""
scripts/04_train.py -- Train a multi-task head on cached embeddings.

Thin orchestration script. Business logic lives in src/retina_screen/.

Usage:
    python scripts/04_train.py --config configs/experiment/stage8d2_brset_resnet50_full_multitask.yaml
    python scripts/04_train.py --config configs/experiment/... --fast_dev_run
    python scripts/04_train.py --config configs/experiment/... \
        --training-config configs/training/standard.yaml

--fast_dev_run: 1 epoch, mini-batch, no early stopping. Always saves artifacts.
--training-config: path to training hyperparameter YAML (default: configs/training/standard.yaml).
    Experiment config keys override training config keys.

Prerequisites:
    scripts/01_make_splits.py --config <same_config>
    scripts/03_extract_embeddings.py --config <same_config>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

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
from retina_screen.data import build_metadata_features, build_task_targets_and_masks
from retina_screen.embeddings import (
    BackboneConfig,
    load_embedding,
    load_embedding_manifest,
)
from retina_screen.evaluation import evaluate_predictions
from retina_screen.feature_policy import FeaturePolicy
from retina_screen.model import MultiTaskHead
from retina_screen.preprocessing import PreprocessingConfig
from retina_screen.training import (
    EarlyStopping,
    KendallUncertaintyWeighting,
    compute_class_weights,
    get_kendall_log_sigmas,
    train_one_epoch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_adapter(cfg: dict):
    from retina_screen.adapters.dummy import DummyAdapter  # noqa: PLC0415
    return DummyAdapter(n_patients=cfg.get("n_patients", 80))


def _import_from_string(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_dataset_config(cfg: dict[str, Any]) -> dict[str, Any]:
    config_path = cfg.get("dataset_config")
    if config_path:
        return load_config(config_path)
    return {"name": cfg.get("dataset", "dummy")}


def _build_adapter(cfg: dict[str, Any]):
    dataset_cfg = _load_dataset_config(cfg)
    adapter_class = dataset_cfg.get("adapter_class")
    if adapter_class:
        adapter_type = _import_from_string(str(adapter_class))
        kwargs = {
            key: cfg.get(key) if key in cfg else dataset_cfg.get(key)
            for key in ("dataset_root", "metadata_file", "training_images_dir")
            if key in cfg or key in dataset_cfg
        }
        return adapter_type(**kwargs)
    return _make_dummy_adapter(cfg)


def _latest_splits_dir(dataset: str) -> Path | None:
    splits_root = Path("outputs") / "splits" / dataset
    if not splits_root.exists():
        return None
    dirs = sorted(
        [path for path in splits_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _load_splits_csv(splits_csv: Path) -> dict[str, list[str]]:
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


def _load_split_embeddings(
    sids: list[str],
    cached_ids: dict,
    sample_lookup: dict,
    embedding_dim: int,
) -> tuple[torch.Tensor, list]:
    """Load embeddings and samples for a list of sample IDs."""
    emb_list: list[torch.Tensor] = []
    samples = []
    for sid in sids:
        rec = cached_ids[sid]
        emb = load_embedding(
            Path(rec["cache_path"]),
            rec["checksum"],
            expected_dim=embedding_dim,
        )
        emb_list.append(emb)
        samples.append(sample_lookup[sid])
    return torch.stack(emb_list), samples


def _compute_val_macro_auroc(
    model: torch.nn.Module,
    val_emb: torch.Tensor | None,
    val_targets_np: dict | None,
    val_masks_np: dict | None,
    supported_tasks: list[str],
    device: torch.device,
) -> float | None:
    """Compute macro-average AUROC across binary tasks on the validation split.

    Returns None when val embeddings are unavailable or all binary tasks return NA.
    Used only for training control (early stopping, best-val checkpoint) — not
    a paper-reported metric.
    """
    if val_emb is None or val_targets_np is None:
        return None
    model.eval()
    with torch.no_grad():
        val_preds_pt = model(val_emb.to(device))
    val_preds_np = {t: val_preds_pt[t].cpu().numpy() for t in supported_tasks}
    model.train()
    metrics = evaluate_predictions(val_preds_np, val_targets_np, val_masks_np, supported_tasks)
    aurocs = [
        r.value
        for tn in supported_tasks
        for r in metrics.get(tn, [])
        if r.metric_name == "auroc" and r.status.name == "OK" and r.value is not None
    ]
    if not aurocs:
        logger.warning(
            "Val macro-AUROC: all binary tasks returned NA — "
            "early stopping metric unavailable this epoch."
        )
        return None
    return sum(aurocs) / len(aurocs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task head on cached embeddings.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--training-config", default="configs/training/standard.yaml",
        dest="training_config",
        help="Training hyperparameter config YAML. Experiment config keys override this.",
    )
    parser.add_argument(
        "--fast_dev_run", action="store_true",
        help="1 epoch, mini-batch, no early stopping. Always saves artifacts.",
    )
    args = parser.parse_args()

    setup_logging()

    # --- Load and merge configs (training_config is base; experiment config wins) ---
    training_cfg = load_config(args.training_config)
    experiment_cfg = load_config(args.config)
    cfg: dict[str, Any] = {**training_cfg, **experiment_cfg}

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = get_device()

    dataset = cfg.get("dataset", "dummy")
    backbone_config = _build_backbone_config(cfg)
    prep_config = _build_prep_config(cfg)
    cache_root = Path(cfg.get("cache_root", "cache/embeddings"))

    from retina_screen.preprocessing import get_preprocessing_hash  # noqa: PLC0415
    prep_hash = get_preprocessing_hash(prep_config)
    embedding_dim = backbone_config.embedding_dim

    # --- Training hyperparameters (all from merged config; no hardcoded values) ---
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    batch_size = int(cfg.get("batch_size", 256))
    max_epochs = 1 if args.fast_dev_run else int(cfg.get("max_epochs", 100))
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    lr_min = float(cfg.get("lr_min", 0.0))
    es_patience = int(cfg.get("early_stopping_patience", 10))
    clip_norm = float(cfg.get("gradient_clip_max_norm", 1.0))
    class_w_enabled = bool(cfg.get("class_weighting_enabled", False))
    max_class_weight = float(cfg.get("max_class_weight", 10.0))

    logger.info(
        "Training config: optimizer=%s lr=%.2e wd=%.2e batch=%d max_epochs=%d "
        "warmup=%d scheduler=cosine clip=%.1f class_weighting=%s es_patience=%d",
        optimizer_name, lr, weight_decay, batch_size, max_epochs,
        warmup_epochs, clip_norm, class_w_enabled, es_patience,
    )

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
            "  python scripts/03_extract_embeddings.py --config %s", args.config
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
    logger.info("Train split: %d total, %d with cached embeddings", len(train_sids_all), len(train_cached))
    if not train_cached:
        logger.error("No cached embeddings found for train split. Run scripts/03_extract_embeddings.py first.")
        sys.exit(1)

    # --- Intersect val split with cached embeddings ---
    val_sids_all = split.get("val", [])
    val_cached = [sid for sid in val_sids_all if sid in cached_ids and sid in sample_lookup]
    logger.info("Val split: %d total, %d with cached embeddings", len(val_sids_all), len(val_cached))
    has_val = len(val_cached) > 0
    if not has_val:
        logger.warning("No val embeddings available; early stopping metric will always be None.")

    # --- Load train embeddings + targets/masks ---
    train_emb, train_samples = _load_split_embeddings(train_cached, cached_ids, sample_lookup, embedding_dim)
    train_batch = build_task_targets_and_masks(train_samples, supported_tasks)
    train_targets = {t: torch.tensor(train_batch.targets[t]) for t in supported_tasks}
    train_masks = {t: torch.tensor(train_batch.masks[t]) for t in supported_tasks}
    logger.info("Train embeddings loaded: shape=%s", list(train_emb.shape))

    # --- Load val embeddings + targets/masks (numpy format for evaluate_predictions) ---
    val_emb: torch.Tensor | None = None
    val_targets_np: dict | None = None
    val_masks_np: dict | None = None
    if has_val:
        val_emb, val_samples = _load_split_embeddings(val_cached, cached_ids, sample_lookup, embedding_dim)
        val_batch = build_task_targets_and_masks(val_samples, supported_tasks)
        val_targets_np = val_batch.targets   # dict of numpy arrays (evaluate_predictions interface)
        val_masks_np = val_batch.masks       # dict of numpy arrays
        logger.info("Val embeddings loaded: shape=%s", list(val_emb.shape))

    # --- Mandatory FeaturePolicy gate (train samples) ---
    feature_policy = FeaturePolicy()
    mode = cfg.get("mode", "image_only")
    for sample in train_samples:
        for task_name in supported_tasks:
            metadata = build_metadata_features(sample, task_name, feature_policy, mode)
            if mode == "image_only" and metadata.allowed_fields:
                raise RuntimeError(
                    "FeaturePolicy exposed metadata in image_only mode for "
                    f"task={task_name!r}, sample_id={sample.sample_id!r}"
                )

    logger.info("Training: %d samples, tasks=%s", len(train_samples), supported_tasks)

    # --- Model ---
    model = MultiTaskHead(embedding_dim=embedding_dim, task_names=supported_tasks)
    weighter = KendallUncertaintyWeighting(task_names=supported_tasks)
    params = list(model.parameters()) + list(weighter.parameters())

    # --- Optimizer (from config; no hardcoded defaults) ---
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}. Supported: 'adamw', 'adam'.")

    # --- LR scheduler: linear warmup then cosine annealing ---
    def lr_lambda(epoch: int) -> float:
        # Use (epoch + 1) so the first optimizer step runs at LR > 0 (not 0).
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs
        total_cosine = max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / total_cosine))
        min_factor = lr_min / lr if lr > 0 else 0.0
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Early stopping ---
    early_stopping = EarlyStopping(patience=es_patience, mode="max")

    # --- Seeded generator for reproducible mini-batch shuffling ---
    generator = torch.Generator()
    generator.manual_seed(seed)
    logger.info("Batch shuffling generator seeded with seed=%d", seed)

    # --- Class weights (disabled for standard corrected baseline) ---
    if class_w_enabled:
        class_weights = compute_class_weights(
            train_targets, train_masks, supported_tasks, max_weight=max_class_weight
        )
        logger.info("Class weighting ENABLED: tasks with weights=%s", list(class_weights.keys()))
    else:
        class_weights = None
        logger.info("Class weighting DISABLED (standard corrected baseline).")

    # --- Dynamic CSV fieldnames ---
    task_loss_fields = [f"train_loss_{t}" for t in supported_tasks]
    log_sigma_fields = [f"log_sigma_{t}" for t in supported_tasks]
    csv_fields = [
        "epoch", "train_loss", "lr", "grad_norm", "n_optimizer_steps",
        "val_macro_auroc", *task_loss_fields, *log_sigma_fields, "epoch_time_s",
    ]

    # --- Create run directory before training loop ---
    run_dir = make_run_dir(
        Path("runs") / ("fast_dev_run" if args.fast_dev_run else "train"),
        dataset,
    )
    logger.info("Run directory: %s", run_dir)

    # --- Training loop ---
    train_log: list[dict] = []
    best_val_metric: float | None = None
    best_epoch: int = -1
    early_stop_triggered = False
    early_stop_epoch: int | None = None
    best_checkpoint_saved = False
    epoch_result: dict = {}

    for epoch in range(max_epochs):
        epoch_start = time.monotonic()

        epoch_result = train_one_epoch(
            model, optimizer, train_emb, train_targets, train_masks,
            supported_tasks, batch_size=batch_size, weighter=weighter,
            max_grad_norm=clip_norm, class_weights=class_weights,
            generator=generator,
        )

        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        log_sigmas = get_kendall_log_sigmas(weighter)
        epoch_time = time.monotonic() - epoch_start

        # Validation macro-AUROC for early stopping
        val_macro_auroc = _compute_val_macro_auroc(
            model, val_emb, val_targets_np, val_masks_np, supported_tasks, device
        )

        row: dict = {
            "epoch": epoch,
            "train_loss": epoch_result["avg_total_loss"],
            "lr": current_lr,
            "grad_norm": epoch_result["avg_grad_norm"],
            "n_optimizer_steps": epoch_result["n_optimizer_steps"],
            "val_macro_auroc": val_macro_auroc if val_macro_auroc is not None else "",
            **{f"train_loss_{t}": epoch_result["per_task_avg_loss"].get(t, "") for t in supported_tasks},
            **{f"log_sigma_{t}": log_sigmas.get(t, "") for t in supported_tasks},
            "epoch_time_s": round(epoch_time, 2),
        }
        train_log.append(row)

        logger.info(
            "Epoch %d | loss=%.4f | val_auroc=%s | lr=%.2e | steps=%d | t=%.1fs",
            epoch, epoch_result["avg_total_loss"],
            f"{val_macro_auroc:.4f}" if val_macro_auroc is not None else "NA",
            current_lr, epoch_result["n_optimizer_steps"], epoch_time,
        )
        for tn in supported_tasks:
            logger.info(
                "  task=%-35s train_loss=%.4f log_sigma=%.3f",
                tn,
                epoch_result["per_task_avg_loss"].get(tn, float("nan")),
                log_sigmas.get(tn, float("nan")),
            )

        # Save best-val checkpoint when validation metric improves
        if val_macro_auroc is not None and (
            best_val_metric is None or val_macro_auroc > best_val_metric
        ):
            best_val_metric = val_macro_auroc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_checkpoint.pt")
            best_checkpoint_saved = True
            logger.info(
                "  → Best val AUROC=%.4f at epoch %d (saved model_checkpoint.pt)",
                best_val_metric, epoch,
            )

        # Check early stopping (skip in fast_dev_run)
        if not args.fast_dev_run and early_stopping.step(val_macro_auroc):
            early_stop_triggered = True
            early_stop_epoch = epoch
            logger.info(
                "Early stopping triggered at epoch %d "
                "(patience=%d, best_epoch=%d, best_val_auroc=%s).",
                epoch, es_patience, best_epoch,
                f"{best_val_metric:.4f}" if best_val_metric is not None else "NA",
            )
            break

        if args.fast_dev_run:
            logger.info("fast_dev_run: stopping after 1 epoch.")
            break

    # Always save last-epoch weights (reference; model_checkpoint.pt is best-val)
    torch.save(model.state_dict(), run_dir / "model_last.pt")
    logger.info("Saved model_last.pt (last-epoch reference).")

    # Fallback: if val was always NA, model_checkpoint.pt was never written
    if not best_checkpoint_saved:
        torch.save(model.state_dict(), run_dir / "model_checkpoint.pt")
        logger.warning(
            "No best-val checkpoint saved (val_macro_auroc was always NA). "
            "Saved last-epoch weights as model_checkpoint.pt."
        )

    # --- Save artifacts ---
    actual_epochs_run = len(train_log)
    total_steps = sum(int(r["n_optimizer_steps"]) for r in train_log)
    steps_per_epoch = int(epoch_result.get("n_optimizer_steps", 0)) if epoch_result else 0

    resolved_cfg = {
        **cfg,
        "git": {k: str(v) for k, v in capture_git_info().items()},
        "env": {k: str(v) for k, v in capture_env_info().items()},
        "fast_dev_run": args.fast_dev_run,
        "n_train_cached": len(train_cached),
        "n_val_cached": len(val_cached),
        "splits_dir": str(splits_dir),
        "embedding_manifest": str(manifest_path),
        "training_config_path": str(args.training_config),
    }
    save_resolved_config(resolved_cfg, run_dir)

    csv_path = run_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train_log)
    logger.info("Saved train_log.csv (%d rows).", actual_epochs_run)

    run_metadata = {
        "run_mode": cfg.get("run_mode", "train"),
        "reported_backbone_is_smoke": cfg.get("reported_backbone_is_smoke", False),
        "real_backbone_target": cfg.get("real_backbone_target", ""),
        "fast_dev_run": args.fast_dev_run,
        "n_samples_used": len(train_samples),
        "n_val_cached": len(val_cached),
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "weight_decay_note": (
            "Accepted AdamW default (0.01); implementation_reference.md value "
            "is ambiguous due to PDF truncation artifact."
        ),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "warmup_epochs": warmup_epochs,
        "scheduler": "cosine",
        "lr_min": lr_min,
        "early_stopping_patience": es_patience,
        "gradient_clip_max_norm": clip_norm,
        "class_weighting_enabled": class_w_enabled,
        "max_class_weight": max_class_weight if class_w_enabled else None,
        "seed": seed,
        "actual_epochs_run": actual_epochs_run,
        "early_stopping_triggered": early_stop_triggered,
        "early_stopping_epoch": early_stop_epoch,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_val_metric,
        "n_optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "n_steps": total_steps,  # backward-compat alias
        "train_loss_final": train_log[-1]["train_loss"] if train_log else None,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)
    logger.info("Saved run_metadata.json.")

    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: reported_backbone_is_smoke=true. "
            "Embeddings are from mock backbone, NOT %s. "
            "Do not report these metrics as real %s results.",
            cfg.get("real_backbone_target", "?"),
            cfg.get("real_backbone_target", "?"),
        )

    logger.info(
        "Training complete. epochs=%d total_steps=%d best_val_auroc=%s. Artifacts: %s",
        actual_epochs_run, total_steps,
        f"{best_val_metric:.4f}" if best_val_metric is not None else "NA",
        run_dir,
    )


if __name__ == "__main__":
    main()
