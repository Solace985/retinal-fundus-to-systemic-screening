#!/usr/bin/env python
"""
scripts/00_smoke_dummy.py -- Dummy end-to-end smoke script.

Thin orchestration script.  Proves the pipeline architecture runs from
DummyAdapter through split → mock embeddings → model → masked loss →
evaluation → artifacts, without real data or pretrained backbones.

Required artifacts produced:
  runs/dummy_smoke/<run_id>/resolved_config.yaml
  runs/dummy_smoke/<run_id>/train_log.csv
  runs/dummy_smoke/<run_id>/metrics.json

Optional artifact:
  runs/dummy_smoke/<run_id>/model_checkpoint.pt

No business logic lives in this script.  Logic belongs in src/retina_screen/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch

# Allow running without editable-install (pyproject.toml covers this normally).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.core import (
    capture_env_info,
    capture_git_info,
    make_run_dir,
    save_resolved_config,
    seed_everything,
    setup_logging,
)
from retina_screen.data import build_task_targets_and_masks
from retina_screen.evaluation import evaluate_predictions
from retina_screen.model import MultiTaskHead
from retina_screen.splitting import assert_no_patient_overlap, split_patients
from retina_screen.training import KendallUncertaintyWeighting, train_one_step

# ---------------------------------------------------------------------------
# Configuration (change via config YAML in later stages, not by editing this)
# ---------------------------------------------------------------------------

EMBED_DIM: int = 1024
N_PATIENTS: int = 80
N_EPOCHS: int = 3
SEED: int = 42


# ---------------------------------------------------------------------------
# Mock embedding helper (smoke-only; real embeddings come from embeddings.py)
# ---------------------------------------------------------------------------


def _mock_embedding(sample_id: str, dim: int = EMBED_DIM) -> torch.Tensor:
    """Deterministic 1024-dim float32 embedding derived from sample_id SHA-256."""
    seed = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % (2 ** 31)
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(dim, generator=gen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    seed_everything(SEED)
    logger = logging.getLogger(__name__)

    run_dir = make_run_dir("runs/dummy_smoke", "smoke")
    logger.info("Run directory: %s", run_dir)

    # --- Resolved config ---
    config = {
        "stage": "dummy_smoke",
        "n_patients": N_PATIENTS,
        "embedding_dim": EMBED_DIM,
        "n_epochs": N_EPOCHS,
        "seed": SEED,
        # Convert to plain str so yaml.safe_dump can serialise all values.
        "git": {k: str(v) for k, v in capture_git_info().items()},
        "env": {k: str(v) for k, v in capture_env_info().items()},
    }
    save_resolved_config(config, run_dir)

    # --- DummyAdapter ---
    adapter = DummyAdapter(n_patients=N_PATIENTS)
    manifest = adapter.build_manifest()
    task_names = adapter.get_supported_tasks()
    logger.info("Manifest: %d samples | tasks: %s", len(manifest), task_names)

    # --- Patient-level split ---
    split_dict = split_patients(manifest, seed=SEED)
    assert_no_patient_overlap(split_dict, manifest)

    sid_to_sample = {s.sample_id: s for s in manifest}
    train_ids = split_dict["train"]
    test_ids = split_dict["test"]
    train_samples = [sid_to_sample[sid] for sid in train_ids]
    test_samples = [sid_to_sample[sid] for sid in test_ids]
    logger.info("Train samples: %d | Test samples: %d", len(train_ids), len(test_ids))

    # --- Mock embeddings ---
    train_emb = torch.stack([_mock_embedding(sid) for sid in train_ids])
    test_emb = torch.stack([_mock_embedding(sid) for sid in test_ids])

    # --- Targets and masks (via data.py) ---
    train_batch = build_task_targets_and_masks(train_samples, task_names)
    test_batch = build_task_targets_and_masks(test_samples, task_names)

    train_targets = {t: torch.tensor(train_batch.targets[t]) for t in task_names}
    train_masks = {t: torch.tensor(train_batch.masks[t]) for t in task_names}

    # --- Model ---
    model = MultiTaskHead(embedding_dim=EMBED_DIM, task_names=task_names)
    weighter = KendallUncertaintyWeighting(task_names=task_names)
    params = list(model.parameters()) + list(weighter.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    # --- Training loop ---
    train_log: list[dict] = []
    for epoch in range(N_EPOCHS):
        result = train_one_step(
            model, optimizer,
            train_emb, train_targets, train_masks,
            task_names, loss_weighter=weighter,
        )
        log_row = {"epoch": epoch, "train_loss": result["total_loss"], "lr": 1e-3}
        train_log.append(log_row)
        logger.info(
            "Epoch %d  loss=%.4f  grad_norm=%.3f",
            epoch, result["total_loss"], result["grad_norm"],
        )

    # --- Save train_log.csv ---
    csv_path = run_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "lr"])
        writer.writeheader()
        writer.writerows(train_log)
    logger.info("Saved train_log.csv")

    # --- Inference on test split ---
    model.eval()
    with torch.no_grad():
        test_preds = model(test_emb)

    import numpy as np  # noqa: PLC0415
    preds_np = {
        t: test_preds[t].numpy() for t in task_names
    }

    # --- Evaluation ---
    metrics = evaluate_predictions(
        predictions=preds_np,
        targets=test_batch.targets,
        masks=test_batch.masks,
        task_names=task_names,
    )

    for tn, results in metrics.items():
        for r in results:
            logger.info(
                "task=%-30s metric=%-10s status=%s value=%s reason=%s n=%d",
                tn, r.metric_name, r.status.value,
                f"{r.value:.4f}" if r.value is not None else "N/A",
                r.reason or "-", r.n,
            )

    # --- Save metrics.json ---
    metrics_dict = {
        tn: [
            {
                "metric": r.metric_name,
                "value": r.value,
                "status": r.status.value,
                "reason": r.reason,
                "n": r.n,
            }
            for r in results
        ]
        for tn, results in metrics.items()
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, indent=2)
    logger.info("Saved metrics.json")

    # --- Optional checkpoint ---
    torch.save(model.state_dict(), run_dir / "model_checkpoint.pt")
    logger.info("Saved model_checkpoint.pt")

    logger.info("Smoke run complete. Artifacts at: %s", run_dir)


if __name__ == "__main__":
    main()
