# Project Decisions Log

This file records locked project decisions.

If this file conflicts with older documents, follow `docs/ai_context/00_source_of_truth_order.md`.

Status labels:

- `locked`: do not change without explicit user approval.
- `open`: unresolved; ask before implementing.
- `deferred`: not part of current MVP.

---

## Decision 001 — Use Modular Monolith Architecture

Status: locked

Decision:

Use a modular monolith under:

src/retina_screen/

Rationale:

The project is larger than a single experiment file. It requires dataset adapters, canonical schema, task registry, FeaturePolicy, patient-level splitting, embedding cache, fairness evaluation, continual-learning simulation, explainability, dashboard inference, and reporting.

Consequences:

- Do not collapse implementation into one giant script.
- Do not over-fragment before MVP.
- Split large files only after MVP if a file exceeds size-budget warnings and the split is approved.

Files affected:

docs/architecture.md
src/retina_screen/\*

---

## Decision 002 — Dataset-Specific Logic Lives Only in Adapters and Configs

Status: locked

Decision:

Dataset-specific parsing, native column names, label vocabularies, camera names, file paths, and dataset-specific quirks belong only in:

src/retina*screen/adapters/*
configs/dataset/_
configs/tasks/_
configs/experiment/_
docs/_
tests/\_

Rationale:

The pipeline must remain dataset-agnostic downstream of adapters so BRSET/mBRSET/external datasets can be added without refactoring model, training, evaluation, or dashboard code.

Consequences:

- No dataset-native parsing in `model.py`, `training.py`, `evaluation.py`, `data.py`, `preprocessing.py`, `embeddings.py`, `continual.py`, or `dashboard_app.py`.
- New datasets should require adapter/config/task additions only.

Files affected:

src/retina_screen/adapters/\*
src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/data.py
src/retina_screen/training.py
src/retina_screen/evaluation.py

---

## Decision 003 — Use Canonical Schema and Task Registry

Status: locked

Decision:

All datasets must be converted into a canonical schema. All tasks must be declared through a task registry.

Rationale:

Datasets differ in labels, missingness, DR grading systems, metadata, and supported tasks. Downstream code must operate on canonical fields only.

Consequences:

- `schema.py` is the single source of truth for canonical sample fields.
- `tasks.py` is the single source of truth for task definitions.
- Do not duplicate canonical field lists in docs or adapters.
- Do not hardcode task lists in training/evaluation.

Files affected:

src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/adapters/\*

---

## Decision 004 — FeaturePolicy Is Mandatory

Status: locked

Decision:

All metadata entering the model must pass through `feature_policy.py`.

Rationale:

Metadata can leak targets or create shortcuts. Age cannot be used to predict retinal age. Sex cannot be used to predict sex. Camera and dataset source can become shortcuts.

Consequences:

- FeaturePolicy must block leakage.
- Image-only and image-plus-metadata modes must be separate.
- Metadata dropout is not enough to prevent leakage.

Files affected:

src/retina_screen/feature_policy.py
src/retina_screen/data.py
src/retina_screen/model.py
tests/test_feature_policy.py

---

## Decision 005 — Use 60/15/15/10 Patient-Level Split

Status: locked

Decision:

Use:

train: 60%
validation: 15%
reliability: 15%
test: 10%

All splits are patient-level.

Rationale:

The original project specification used a simpler split, but the issues log introduced a separate reliability split for dashboard subgroup reliability lookup. Later-overrides-earlier precedence makes 60/15/15/10 the active rule.

Consequences:

- No image-level splitting.
- Reliability split is not used for training, early stopping, or model selection.
- Reliability split is used to generate subgroup reliability lookup tables.
- Split audit must prove zero patient overlap.

Files affected:

src/retina_screen/splitting.py
src/retina_screen/evaluation.py
tests/test_patient_split.py
tests/test_split_audit.py

---

## Decision 006 — Build DummyAdapter MVP Before Real Dataset Work

Status: locked

Decision:

The first working pipeline must use DummyAdapter.

Target path:

DummyAdapter
→ canonical schema
→ patient-level split
→ mock embeddings
→ dataloader
→ FeaturePolicy
→ task masks
→ model
→ masked loss
→ evaluation

Rationale:

DummyAdapter catches dataset coupling before ODIR or BRSET logic enters the system.

Consequences:

- Do not implement ODIR training before dummy E2E passes.
- Do not jump to real backbones before mock embedding flow works.

Files affected:

src/retina_screen/adapters/dummy.py
src/retina_screen/data.py
src/retina_screen/model.py
src/retina_screen/training.py
src/retina_screen/evaluation.py
tests/test_dummy_e2e.py

---

## Decision 007 — ODIR-Only Mode Cannot Headline Systemic Prediction

Status: locked

Decision:

If only ODIR is available, the paper is framed as:

multi-condition retinal screening + fairness + cross-site robustness + continual-learning simulation

not as definitive systemic disease prediction.

Rationale:

ODIR diabetes/hypertension labels are weak proxy labels, not structured clinical systemic records.

Consequences:

- ODIR diabetes/hypertension heads may exist.
- They must be reported as weak proxy labels.
- ODIR-only mode must not headline whole-body/systemic prediction.
- CV composite and retinal age gap are secondary/dashboard convenience outputs in ODIR-only mode.

Files affected:

configs/paper/claim_mode.yaml
configs/tasks/odir_default.yaml
src/retina_screen/tasks.py
src/retina_screen/evaluation.py
src/retina_screen/reporting.py
src/retina_screen/dashboard_app.py

---

## Decision 008 — BRSET and mBRSET Are Future Adapter-Only Integrations

Status: locked

Decision:

BRSET/mBRSET remain supported as future adapter/config/task additions.

Rationale:

They may provide stronger structured systemic labels, device metadata, and smartphone transfer. The architecture should support them without refactoring downstream layers.

Consequences:

- `brset.py` and `mbrset.py` may remain stubs until access.
- Adding BRSET/mBRSET must not require changing `model.py`, `training.py`, `evaluation.py`, `continual.py`, or `dashboard_app.py`.

Files affected:

src/retina_screen/adapters/brset.py
src/retina_screen/adapters/mbrset.py
configs/dataset/brset.yaml
configs/dataset/mbrset.yaml
configs/tasks/brset_default.yaml
configs/tasks/mbrset_default.yaml

---

## Decision 009 — Foundation Backbones Stay Frozen by Default

Status: locked

Decision:

RETFound, DINOv2, ConvNeXt, and ResNet are frozen embedding extractors by default.

Rationale:

Frozen backbones reduce compute, improve reproducibility, allow caching, and avoid unnecessary fine-tuning risk.

Consequences:

- `requires_grad = False` for backbone parameters by default.
- Backbone fine-tuning is only allowed as an explicit ablation.

Files affected:

src/retina_screen/embeddings.py
configs/backbone/\*
src/retina_screen/model.py

---

## Decision 010 — OOD Uses PCA-64 Mahalanobis by Default

Status: locked

Decision:

OOD detection uses PCA-64 followed by Mahalanobis distance.

Rationale:

Full-dimensional covariance can be unstable in 1024-dimensional embedding space. PCA-64 stabilizes OOD estimation.

Consequences:

- PCA basis, mean, inverse covariance, and threshold are cached by backbone/dataset/preprocessing hash.
- Threshold is calibrated on validation distances.
- OOD thresholds cannot be reused across different embedding spaces.

Files affected:

configs/ood/pca64_mahalanobis.yaml
src/retina_screen/embeddings.py
src/retina_screen/continual.py
src/retina_screen/dashboard_app.py

---

## Decision 011 — RFMiD Has Limited Roles

Status: locked

Decision:

RFMiD is used for:

1. TCAV/concept-direction source.
2. Secondary external ocular validation.

RFMiD is not a continual-learning stream.

Rationale:

RFMiD label schema differs enough from ODIR’s eight-category structure that using it as a stream would complicate the continual-learning protocol.

Consequences:

- RFMiD config should declare concept/external-validation roles only.
- Continual-learning configs should not use RFMiD as incoming stream unless a later decision changes this.

Files affected:

src/retina_screen/adapters/rfmid.py
configs/dataset/rfmid.yaml
src/retina_screen/explainability.py
src/retina_screen/continual.py

---

## Decision 012 — Dashboard Is Inference-Only

Status: locked

Decision:

The dashboard must never retrain or update model parameters from user uploads.

Rationale:

User uploads are unlabeled, unverified, and unsafe for live medical AI learning.

Consequences:

- Dashboard may run quality/OOD checks, prediction, explanation, reliability lookup, and consent-based logging.
- Dashboard must not call training/optimizer/continual-learning update functions.
- Continual learning remains offline simulation only.

Files affected:

src/retina_screen/dashboard_app.py
src/retina_screen/continual.py
tests/test_import_boundaries.py

---

## Decision 013 — Evaluation Protocol Must Be Pre-Registered

Status: locked

Decision:

Before headline experiments, freeze:

- metrics,
- subgroup columns,
- mitigation methods,
- external datasets,
- bootstrap iterations,
- sparse subgroup thresholds,
- multiple-comparison correction.

Rationale:

Prevents post-hoc cherry-picking.

Consequences:

- `configs/evaluation/preregistered_protocol.yaml` must not be changed after headline experiments begin unless logged.
- Baseline and mitigation runs must use the same harness.

Files affected:

configs/evaluation/preregistered_protocol.yaml
src/retina_screen/evaluation.py
src/retina_screen/reporting.py

---

## Decision 014 — Static-Analysis Tests Run Early

Status: locked

Decision:

`test_no_dataset_coupling.py` and `test_import_boundaries.py` are Stage 1 tests.

Rationale:

They pass trivially on an empty codebase and catch architecture violations as files are added.

Consequences:

- Do not delay architecture static tests until after MVP.
- Do not weaken these tests to make bad code pass.

Files affected:

tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py
docs/ai_context/06_testing_protocol.md

---

## Decision 015 — Kendall Uncertainty Weighting Is Default Multi-Task Loss Weighting

Status: locked

Decision:

Use Kendall uncertainty weighting as the default multi-task loss weighting method.

Rationale:

The implementation reference defines it as the default and it avoids one task dominating purely due to loss scale.

Consequences:

- GradNorm may remain configurable later.
- Manual task weights may be used only through config and documented experiments.

Files affected:

src/retina_screen/training.py
configs/training/standard.yaml
configs/model/multitask_default.yaml

---

## Decision 016 — Continual-Learning Chunk Size Defaults to 250

Status: locked

Decision:

Continual-learning stream chunk size defaults to 250 images.

Rationale:

The implementation reference defines 250 as a meaningful chunk size for observing forgetting/adaptation without making chunks too small.

Consequences:

- This value belongs in continual-learning config.
- It may be tuned only through config and logged.

Files affected:

configs/training/continual.yaml
src/retina_screen/continual.py

---

# Open Decisions

These decisions are not locked yet. Ask before implementing if they become relevant.

## Open 001 — Dashboard Framework

Options:

- Streamlit
- Gradio

Default leaning:

- Streamlit for prototype unless changed later.

Affected files:

src/retina_screen/dashboard_app.py
scripts/08_launch_dashboard.py
requirements.txt

## Open 002 — Exact RFMiD Concept Set for TCAV

Decision needed:

Which RFMiD labels should be used as TCAV concept directions.

Affected files:

src/retina_screen/adapters/rfmid.py
src/retina_screen/explainability.py
configs/dataset/rfmid.yaml

## Open 003 — Clinical Co-Author / Clinical Language Strength

Decision needed:

Whether a clinical co-author is available to strengthen clinical framing.

Default:

If no clinical co-author exists, keep all systemic claims conservative.

Affected files:

configs/paper/claim_mode.yaml
src/retina_screen/reporting.py
src/retina_screen/dashboard_app.py

## Open 004 — Include RetiZero or DINORET

Decision needed:

Whether weights are accessible and worth adding.

Default:

Do not delay MVP. Use DINOv2 + RETFound + ConvNeXt/ResNet baseline first.

Affected files:

configs/backbone/\*
src/retina_screen/embeddings.py

## Open 005 — Full Fine-Tuning / Backbone LoRA Ablation

Decision needed:

Whether to run a later ablation that adds LoRA to the foundation backbone.

Default:

Do not include in MVP. Head-only training first.

Affected files:

src/retina_screen/embeddings.py
src/retina_screen/model.py
src/retina_screen/continual.py
configs/training/continual.yaml
