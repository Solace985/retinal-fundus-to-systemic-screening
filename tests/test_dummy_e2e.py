from __future__ import annotations
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.data import build_metadata_features, build_task_targets_and_masks
from retina_screen.evaluation import MetricStatus, evaluate_predictions
from retina_screen.feature_policy import FeaturePolicy, ModelInputMode
from retina_screen.model import MultiTaskHead
from retina_screen.splitting import assert_no_patient_overlap, split_patients
from retina_screen.tasks import TASK_REGISTRY, TaskType
from retina_screen.training import (
    KendallUncertaintyWeighting, TaskLossResult,
    compute_masked_task_loss, train_one_step,
)


EMBED_DIM = 1024
N_PATIENTS = 80
SEED = 42


def _mock_emb(sample_id: str, dim: int = EMBED_DIM) -> torch.Tensor:
    seed = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % (2 ** 31)
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(dim, generator=gen)


@pytest.fixture(scope="module")
def e2e():
    adapter = DummyAdapter(n_patients=N_PATIENTS)
    manifest = adapter.build_manifest()
    task_names = adapter.get_supported_tasks()
    split_dict = split_patients(manifest, seed=SEED)
    sid_to = {s.sample_id: s for s in manifest}
    train_ids = split_dict["train"]
    test_ids = split_dict["test"]
    train_samples = [sid_to[sid] for sid in train_ids]
    test_samples = [sid_to[sid] for sid in test_ids]
    train_emb = torch.stack([_mock_emb(sid) for sid in train_ids])
    test_emb = torch.stack([_mock_emb(sid) for sid in test_ids])
    train_batch = build_task_targets_and_masks(train_samples, task_names)
    test_batch = build_task_targets_and_masks(test_samples, task_names)
    train_targets = {t: torch.tensor(train_batch.targets[t]) for t in task_names}
    train_masks = {t: torch.tensor(train_batch.masks[t]) for t in task_names}
    model = MultiTaskHead(embedding_dim=EMBED_DIM, task_names=task_names)
    weighter = KendallUncertaintyWeighting(task_names=task_names)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(weighter.parameters()), lr=1e-3
    )
    step_result = train_one_step(
        model, optimizer, train_emb, train_targets, train_masks,
        task_names, loss_weighter=weighter,
    )
    model.eval()
    with torch.no_grad():
        test_preds = model(test_emb)
    preds_np = {t: test_preds[t].numpy() for t in task_names}
    metrics = evaluate_predictions(
        predictions=preds_np,
        targets=test_batch.targets,
        masks=test_batch.masks,
        task_names=task_names,
    )
    return dict(
        adapter=adapter, manifest=manifest, task_names=task_names,
        split_dict=split_dict, train_ids=train_ids, test_ids=test_ids,
        train_batch=train_batch, test_batch=test_batch,
        train_targets=train_targets, train_masks=train_masks,
        train_emb=train_emb, test_emb=test_emb,
        model=model, weighter=weighter, optimizer=optimizer,
        step_result=step_result, test_preds=test_preds, preds_np=preds_np,
        metrics=metrics,
    )


@pytest.fixture(scope="module")
def smoke_run_dir() -> Path:
    """Run the dummy smoke script once and return the exact new run directory."""
    runs_root = Path("runs") / "dummy_smoke"
    runs_root.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in runs_root.glob("smoke_*") if p.is_dir()}

    # Smoke run names use second-level timestamps; avoid accidental directory reuse.
    time.sleep(1.1)
    result = subprocess.run(
        [sys.executable, "scripts/00_smoke_dummy.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, "Smoke script failed: " + result.stderr[:500]

    after = {p.resolve() for p in runs_root.glob("smoke_*") if p.is_dir()}
    new_runs = sorted(after - before)
    assert len(new_runs) == 1, (
        "Expected exactly one new dummy smoke run directory, "
        f"found {len(new_runs)}. New dirs: {[str(p) for p in new_runs]}. "
        f"stdout={result.stdout[:500]!r} stderr={result.stderr[:500]!r}"
    )
    return new_runs[0]


# --- No real data ---

def test_no_real_data_needed(e2e):
    for s in e2e["manifest"]:
        assert s.dataset_source == "dummy"
        assert s.image_path.startswith("dummy://")


def test_feature_policy_metadata_path_in_e2e(e2e):
    sample = e2e["manifest"][0]
    metadata = build_metadata_features(
        sample,
        "glaucoma",
        FeaturePolicy(),
        ModelInputMode.IMAGE_PLUS_METADATA,
    )

    assert metadata.allowed_fields
    assert set(metadata.values) == set(metadata.allowed_fields)
    assert set(metadata.observation_mask) == set(metadata.allowed_fields)
    assert "dataset_source" not in metadata.allowed_fields
    assert "camera_type" not in metadata.allowed_fields
    for field in metadata.allowed_fields:
        assert metadata.values[field] == getattr(sample, field)
        assert metadata.observation_mask[field] in (0.0, 1.0)


# --- Split safety ---

def test_patient_split_no_overlap(e2e):
    assert_no_patient_overlap(e2e["split_dict"], e2e["manifest"])


def test_all_splits_nonempty(e2e):
    for name, ids in e2e["split_dict"].items():
        assert len(ids) > 0, f"Split {name!r} is empty"


# --- Task masks ---

def test_masks_are_zero_or_one(e2e):
    for tn in e2e["task_names"]:
        for m in e2e["train_batch"].masks[tn]:
            assert m in (0.0, 1.0), f"Mask value {m!r} is not 0 or 1 for task {tn!r}"


def test_at_least_one_missing_label_has_mask_zero(e2e):
    found = False
    for tn in e2e["task_names"]:
        if 0.0 in e2e["train_batch"].masks[tn]:
            found = True
            break
    assert found, "No missing labels found in any task; task masking cannot be tested"


def test_observed_binary_zero_has_mask_one_target_zero(e2e):
    for tn in e2e["task_names"]:
        if TASK_REGISTRY[tn].task_type != TaskType.BINARY:
            continue
        for tgt, mask in zip(e2e["train_batch"].targets[tn], e2e["train_batch"].masks[tn]):
            if tgt == 0.0 and mask == 1.0:
                return  # found one
    pytest.skip("No observed binary-0 labels in this dataset configuration")


def test_masks_align_with_targets(e2e):
    for tn in e2e["task_names"]:
        t_len = len(e2e["train_batch"].targets[tn])
        m_len = len(e2e["train_batch"].masks[tn])
        assert t_len == m_len, f"Target/mask length mismatch for task {tn!r}"


# --- Model output shapes ---

def test_model_forward_has_all_configured_tasks(e2e):
    for tn in e2e["task_names"]:
        assert tn in e2e["test_preds"]


def test_binary_outputs_are_1d(e2e):
    for tn in e2e["task_names"]:
        if TASK_REGISTRY[tn].task_type == TaskType.BINARY:
            assert e2e["test_preds"][tn].ndim == 1, f"{tn!r} should be 1-D"


def test_ordinal_outputs_are_2d(e2e):
    for tn in e2e["task_names"]:
        if TASK_REGISTRY[tn].task_type == TaskType.ORDINAL:
            out = e2e["test_preds"][tn]
            assert out.ndim == 2, f"{tn!r} should be 2-D, got shape {out.shape}"
            assert out.shape[-1] == TASK_REGISTRY[tn].num_classes


# --- Loss and training ---

def test_loss_is_finite(e2e):
    loss = e2e["step_result"]["total_loss"]
    assert math.isfinite(loss), f"Loss is not finite: {loss}"


def test_fully_masked_task_has_valid_count_zero(e2e):
    # Build a batch where all labels for one task are missing
    task_names = e2e["task_names"]
    # Pick the first binary task
    binary_tasks = [t for t in task_names if TASK_REGISTRY[t].task_type == TaskType.BINARY]
    if not binary_tasks:
        pytest.skip("No binary tasks to test")
    tn = binary_tasks[0]
    preds = e2e["model"](e2e["train_emb"])
    all_missing_masks = {t: torch.zeros_like(e2e["train_masks"][t]) if t == tn
                         else e2e["train_masks"][t] for t in task_names}
    result = compute_masked_task_loss(
        preds, e2e["train_targets"], all_missing_masks, task_names
    )
    assert result.valid_counts[tn] == 0


def test_regression_nan_targets_do_not_produce_nan_loss(e2e):
    regression_tasks = [t for t in e2e["task_names"] if TASK_REGISTRY[t].task_type == TaskType.REGRESSION]
    if not regression_tasks:
        pytest.skip("No regression tasks in dummy adapter")
    preds = e2e["model"](e2e["train_emb"])
    result = compute_masked_task_loss(
        preds, e2e["train_targets"], e2e["train_masks"], e2e["task_names"]
    )
    for tn in regression_tasks:
        loss_val = result.losses[tn].item()
        assert not math.isnan(loss_val), f"Regression task {tn!r} produced NaN loss"


def test_kendall_skips_all_missing_tasks_by_count(e2e):
    task_names = e2e["task_names"]
    all_zero_losses = {t: torch.tensor(0.0) for t in task_names}
    all_zero_counts = {t: 0 for t in task_names}
    all_one_counts = {t: 1 for t in task_names}
    zero_result = TaskLossResult(losses=all_zero_losses, valid_counts=all_zero_counts)
    one_result = TaskLossResult(losses=all_zero_losses, valid_counts=all_one_counts)
    zero_total = e2e["weighter"](zero_result)
    one_total = e2e["weighter"](one_result)
    # all-missing should produce a near-zero (grad_fn safe) total
    assert math.isfinite(zero_total.item())
    # non-missing should also be finite
    assert math.isfinite(one_total.item())


def test_kendall_parameters_are_in_optimizer(e2e):
    optimizer_params = {
        id(param)
        for group in e2e["optimizer"].param_groups
        for param in group["params"]
    }
    missing = [
        name
        for name, param in e2e["weighter"].named_parameters()
        if id(param) not in optimizer_params
    ]
    assert missing == [], f"Kendall parameters missing from optimizer: {missing}"


def test_one_optimizer_step_runs(e2e):
    assert math.isfinite(e2e["step_result"]["total_loss"])
    assert math.isfinite(e2e["step_result"]["grad_norm"])


# --- Evaluation ---

def test_evaluation_returns_nonempty_dict(e2e):
    assert len(e2e["metrics"]) > 0


def test_at_least_one_metric_ok_or_na_not_crash(e2e):
    for tn, results in e2e["metrics"].items():
        for r in results:
            assert r.status in (MetricStatus.OK, MetricStatus.NA), (
                f"Task {tn!r} metric {r.metric_name!r} has unexpected status {r.status!r}"
            )


def test_sparse_subgroup_returns_na_not_crash(e2e):
    # With small test split, binary metrics likely return NA — verify it doesn't crash
    found_na = False
    for tn, results in e2e["metrics"].items():
        for r in results:
            if r.status == MetricStatus.NA:
                found_na = True
                assert isinstance(r.reason, str)
                assert len(r.reason) > 0
    # It's OK if found_na is False (all metrics valid), just ensure no crash above


# --- Artifact tests ---

def test_smoke_creates_resolved_config(smoke_run_dir):
    cfg_path = smoke_run_dir / "resolved_config.yaml"
    assert cfg_path.exists()
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict)
    assert cfg.get("stage") == "dummy_smoke"


def test_smoke_creates_metrics_json(smoke_run_dir):
    metrics_path = smoke_run_dir / "metrics.json"
    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
    assert data
    for task_results in data.values():
        assert isinstance(task_results, list)
        for metric in task_results:
            assert {"metric", "value", "status", "reason", "n"} <= set(metric)
            if metric["status"] == "na":
                assert metric["value"] is None
                assert metric["reason"]


def test_smoke_creates_train_log_csv(smoke_run_dir):
    log_path = smoke_run_dir / "train_log.csv"
    assert log_path.exists()
    with log_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert {"epoch", "train_loss", "lr"} <= set(reader.fieldnames or [])
    assert len(rows) > 0


def test_smoke_creates_optional_model_checkpoint(smoke_run_dir):
    # The current Stage 5 smoke script writes this simple optional artifact.
    assert (smoke_run_dir / "model_checkpoint.pt").exists()
