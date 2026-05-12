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
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

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
from retina_screen.evaluation import evaluate_predictions, evaluate_subgroups
from retina_screen.model import MultiTaskHead
from retina_screen.preprocessing import PreprocessingConfig, get_preprocessing_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with 04_train.py)
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


def _latest_run_dir(dataset: str) -> Path | None:
    """Return the most recent run directory containing a model checkpoint."""
    for base in [Path("runs") / "train", Path("runs") / "fast_dev_run"]:
        if not base.exists():
            continue
        dirs = sorted(
            [
                d for d in base.iterdir()
                if d.name.startswith(f"{dataset}_") and (d / "model_checkpoint.pt").exists()
            ],
            key=lambda p: p.name,
            reverse=True,
        )
        if dirs:
            return dirs[0]
    return None


def _is_full_internal_config(cfg: dict[str, Any]) -> bool:
    """Return True for Stage 8D-2+ full/internal configs requiring explicit --run-dir."""
    run_mode = str(cfg.get("run_mode", "")).lower()
    return bool(cfg.get("full_dataset_run", False)) or "stage8d2" in run_mode


def _validate_run_dir_for_eval(run_dir: Path, eval_cfg: dict[str, Any]) -> None:
    """Validate a run directory for evaluation compatibility.

    For full/internal (Stage 8D-2+) configs: all checks are hard errors (sys.exit).
    For smoke/rehearsal fallback: dataset/backbone mismatches are warnings only.
    """
    is_full = _is_full_internal_config(eval_cfg)
    if not run_dir.exists() or not run_dir.is_dir():
        logger.error("Run directory does not exist or is not a directory: %s", run_dir)
        sys.exit(1)
    if not (run_dir / "model_checkpoint.pt").exists():
        logger.error("model_checkpoint.pt not found in run directory: %s", run_dir)
        sys.exit(1)
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        if is_full:
            logger.error(
                "resolved_config.yaml not found in run directory: %s. "
                "Stage 8D-2 full/internal evaluation requires resolved_config.yaml "
                "to verify run compatibility (dataset, backbone, task config, "
                "fast_dev_run, rehearsal status).",
                run_dir,
            )
            sys.exit(1)
        logger.warning(
            "resolved_config.yaml missing in run directory %s; skipping run config checks.", run_dir
        )
        return
    run_cfg = load_config(resolved_path)
    if is_full:
        if run_cfg.get("fast_dev_run", False):
            logger.error(
                "Run directory is a fast_dev_run (smoke) run: %s. "
                "Stage 8D-2 full/internal evaluation requires a full training run.",
                run_dir,
            )
            sys.exit(1)
        if run_cfg.get("rehearsal", False):
            logger.error(
                "Run directory is a rehearsal run (rehearsal=true): %s. "
                "Stage 8D-2 full/internal evaluation requires a full (non-rehearsal) run.",
                run_dir,
            )
            sys.exit(1)
        _run_mode_r = str(run_cfg.get("run_mode", "")).lower()
        if "stage8d1" in _run_mode_r or "rehearsal" in _run_mode_r:
            logger.error(
                "Run directory points to a Stage 8D-1/rehearsal run "
                "(run_mode=%r): %s. Stage 8D-2 full/internal evaluation requires "
                "a Stage 8D-2 run directory.",
                run_cfg.get("run_mode", ""), run_dir,
            )
            sys.exit(1)
        # Hard errors for dataset, backbone, task_config identity.
        _run_dataset = run_cfg.get("dataset")
        _eval_dataset = eval_cfg.get("dataset")
        if _run_dataset and _run_dataset != _eval_dataset:
            logger.error(
                "Dataset mismatch: run directory has dataset=%r, eval config has dataset=%r. "
                "Cannot evaluate Stage 8D-2 with mismatched dataset.",
                _run_dataset, _eval_dataset,
            )
            sys.exit(1)
        if not _run_dataset:
            logger.error(
                "Run directory resolved_config.yaml is missing 'dataset' field: %s. "
                "Cannot verify dataset compatibility for Stage 8D-2.",
                run_dir,
            )
            sys.exit(1)
        _run_backbone = run_cfg.get("backbone")
        _eval_backbone = eval_cfg.get("backbone")
        if _run_backbone and _run_backbone != _eval_backbone:
            logger.error(
                "Backbone mismatch: run directory has backbone=%r, eval config has backbone=%r. "
                "Cannot evaluate Stage 8D-2 with mismatched backbone.",
                _run_backbone, _eval_backbone,
            )
            sys.exit(1)
        if not _run_backbone:
            logger.error(
                "Run directory resolved_config.yaml is missing 'backbone' field: %s. "
                "Cannot verify backbone compatibility for Stage 8D-2.",
                run_dir,
            )
            sys.exit(1)
        _run_task_cfg = run_cfg.get("task_config")
        _eval_task_cfg = eval_cfg.get("task_config")
        if _eval_task_cfg and _run_task_cfg and _run_task_cfg != _eval_task_cfg:
            logger.error(
                "Task config mismatch: run directory has task_config=%r, "
                "eval config has task_config=%r. "
                "Cannot evaluate Stage 8D-2 with mismatched task configuration.",
                _run_task_cfg, _eval_task_cfg,
            )
            sys.exit(1)
        if _eval_task_cfg and not _run_task_cfg:
            logger.error(
                "Run directory resolved_config.yaml is missing 'task_config' field: %s. "
                "Cannot verify task compatibility for Stage 8D-2.",
                run_dir,
            )
            sys.exit(1)
    else:
        if run_cfg.get("dataset") and run_cfg.get("dataset") != eval_cfg.get("dataset"):
            logger.warning(
                "Dataset mismatch: run=%r, eval=%r.", run_cfg.get("dataset"), eval_cfg.get("dataset")
            )
        if run_cfg.get("backbone") and run_cfg.get("backbone") != eval_cfg.get("backbone"):
            logger.warning(
                "Backbone mismatch: run=%r, eval=%r.", run_cfg.get("backbone"), eval_cfg.get("backbone")
            )


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
# Diagnostics helper
# ---------------------------------------------------------------------------


def _build_diagnostics(
    predictions_np: dict,
    targets: dict,
    masks: dict,
    task_names: list[str],
) -> dict:
    """Compute per-task diagnostic summaries from model outputs.

    Binary tasks: polarity check (positive-class mean score vs negative-class mean score).
    Ordinal tasks: predicted vs true class distribution, collapse detection.
    All tasks: valid label count.

    This is for internal diagnostic use only — not a paper metric.
    """
    from retina_screen.tasks import TASK_REGISTRY, TaskType  # noqa: PLC0415

    diag: dict = {}
    for tn in task_names:
        task = TASK_REGISTRY[tn]
        # Explicitly cast to float64 numpy arrays to avoid dtype/scalar surprises
        mask = np.asarray(masks[tn], dtype=np.float64)
        target = np.asarray(targets[tn], dtype=np.float64)
        pred_logits = np.asarray(predictions_np[tn], dtype=np.float64)

        valid_idx = mask >= 0.5  # numpy bool array; robust to int/float mask values
        n_valid = int(np.sum(valid_idx))
        entry: dict = {"n_valid_labels": n_valid}

        if task.task_type == TaskType.BINARY and n_valid > 0:
            valid_logits = pred_logits[valid_idx].ravel()
            scores = 1.0 / (1.0 + np.exp(-valid_logits))
            labels = target[valid_idx].ravel()
            pos_mask = labels >= 0.5   # 1.0 → True (robust to float rounding)
            neg_mask = labels < 0.5    # 0.0 → True
            n_pos = int(np.sum(pos_mask))
            n_neg = int(np.sum(neg_mask))
            entry["n_positives"] = n_pos
            entry["n_negatives"] = n_neg
            entry["pos_mean_score"] = float(np.mean(scores[pos_mask])) if n_pos > 0 else None
            entry["neg_mean_score"] = float(np.mean(scores[neg_mask])) if n_neg > 0 else None
            if entry["pos_mean_score"] is not None and entry["neg_mean_score"] is not None:
                entry["polarity_correct"] = bool(entry["pos_mean_score"] > entry["neg_mean_score"])
                entry["score_gap"] = round(float(entry["pos_mean_score"] - entry["neg_mean_score"]), 4)

        elif task.task_type == TaskType.ORDINAL and n_valid > 0:
            valid_preds = pred_logits[valid_idx]
            if valid_preds.ndim == 2:
                pred_classes = valid_preds.argmax(axis=-1).astype(int)
            else:
                pred_classes = (valid_preds > 0.0).astype(int)
            true_classes = target[valid_idx].ravel().astype(int)
            unique_pred = sorted(set(pred_classes.tolist()))
            unique_true = sorted(set(true_classes.tolist()))
            entry["predicted_class_distribution"] = {
                str(c): int(np.sum(pred_classes == c)) for c in unique_pred
            }
            entry["true_class_distribution"] = {
                str(c): int(np.sum(true_classes == c)) for c in unique_true
            }
            entry["prediction_collapsed"] = len(unique_pred) == 1
            if entry["prediction_collapsed"]:
                entry["collapse_class"] = unique_pred[0]

        diag[tn] = entry
    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on cached embeddings.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--run-dir", default=None, dest="run_dir",
        help="Explicit training run directory containing model_checkpoint.pt and resolved_config.yaml. "
             "Required for full/internal (Stage 8D-2+) runs.",
    )
    parser.add_argument(
        "--checkpoint-path", default=None, dest="checkpoint_path",
        help="Explicit path to model_checkpoint.pt. Alternative to --run-dir; "
             "parent directory is used for resolved_config.yaml checks.",
    )
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

    # --- Resolve checkpoint (explicit selector or fallback discovery) ---
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        _validate_run_dir_for_eval(run_dir, cfg)
        checkpoint_path = run_dir / "model_checkpoint.pt"
    elif args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            logger.error("--checkpoint-path not found: %s", checkpoint_path)
            sys.exit(1)
        run_dir = checkpoint_path.parent
        _validate_run_dir_for_eval(run_dir, cfg)
    else:
        if _is_full_internal_config(cfg):
            logger.error(
                "Stage 8D-2 full/internal evaluation requires explicit --run-dir or "
                "--checkpoint-path to avoid stale checkpoint selection. "
                "Example: --run-dir runs/train/brset_<timestamp>"
            )
            sys.exit(1)
        # Fallback: automatic latest-run discovery (smoke/rehearsal only).
        run_dir = _latest_run_dir(dataset)
        if run_dir is None:
            logger.error(
                "No model_checkpoint.pt found. Run first:\n"
                "  python scripts/04_train.py --config %s", args.config
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
        eval_reason = (
            "limited_embedding_smoke"
            if cfg.get("reported_backbone_is_smoke", False)
            else "cached_test_samples_available"
        )
        final_test_result = not cfg.get("reported_backbone_is_smoke", False)
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

    # Smoke run_mode is never a final result regardless of backbone type.
    # A real backbone + --limit 32 extraction is still a smoke evaluation.
    if "smoke" in str(cfg.get("run_mode", "")).lower():
        final_test_result = False
        eval_reason = "limited_embedding_smoke"

    # Rehearsal and preliminary configs must never produce final results.
    _run_mode_lower = str(cfg.get("run_mode", "")).lower()
    if "rehearsal" in _run_mode_lower or "stage8d1" in _run_mode_lower:
        final_test_result = False
        eval_reason = "rehearsal_run"
    elif _is_full_internal_config(cfg):
        # Stage 8D-2 full/internal: preliminary but explicitly not paper-final.
        final_test_result = False
        eval_reason = "stage8d2_full_internal"
    elif cfg.get("preliminary", False) or cfg.get("rehearsal", False):
        final_test_result = False
        eval_reason = "preliminary_run"

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

    # --- Save metrics and evaluation summary ---
    n_cached_per_split = {
        "test": len(test_cached),
        "val": len(val_cached),
        "train": len(train_cached),
    }

    def _metric_to_dict(r) -> dict:
        d = {
            "metric": r.metric_name,
            "value": r.value,
            "status": r.status.value,
            "reason": r.reason,
            "n": r.n,
            "positives": r.positives,
            "negatives": r.negatives,
        }
        if getattr(r, "per_class_support", None) is not None:
            d["per_class_support"] = r.per_class_support
        return d

    overall_metrics = {
        task: [
            _metric_to_dict(r)
            for r in results
        ]
        for task, results in metrics.items()
    }

    sex_labels = np.array(
        [sample.sex.value if sample.sex is not None else None for sample in eval_samples],
        dtype=object,
    )
    subgroup_results = evaluate_subgroups(
        predictions=preds_np,
        targets=batch.targets,
        masks=batch.masks,
        task_names=supported_tasks,
        subgroup_labels=sex_labels,
    )
    subgroup_metrics = {
        group: {
            task: [_metric_to_dict(r) for r in results]
            for task, results in task_results.items()
        }
        for group, task_results in subgroup_results.items()
    }

    summary = {
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
        "overall_metrics_path": "overall_metrics.json",
        "subgroup_metrics_path": "subgroup_metrics.json",
    }

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_out_dir = ensure_dir(Path(cfg.get("evaluation_output_root", "outputs/evaluation")) / run_id)
    with (eval_out_dir / "overall_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(overall_metrics, fh, indent=2)
    with (eval_out_dir / "subgroup_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(subgroup_metrics, fh, indent=2)
    with (eval_out_dir / "evaluation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with (eval_out_dir / "smoke_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "metrics": overall_metrics}, fh, indent=2)

    # Diagnostics: polarity checks for binary tasks, prediction distribution for ordinal
    try:
        diagnostics = _build_diagnostics(preds_np, batch.targets, batch.masks, supported_tasks)
        with (eval_out_dir / "diagnostics.json").open("w", encoding="utf-8") as fh:
            json.dump(diagnostics, fh, indent=2)
        logger.info("Saved diagnostics.json")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute diagnostics: %s", exc)

    logger.info("Evaluation complete. Metrics saved to: %s", eval_out_dir)

    if cfg.get("reported_backbone_is_smoke", False):
        logger.warning(
            "SMOKE RUN: backbone is mock, NOT %s. "
            "Do not report these metrics as real model performance.",
            cfg.get("real_backbone_target", "?"),
        )


if __name__ == "__main__":
    main()
