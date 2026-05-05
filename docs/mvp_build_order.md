# MVP Build Order

This file is a human-facing summary of the MVP implementation sequence.

Authoritative detailed references:

- `docs/ai_context/03_file_generation_order.md` defines exact file generation order.
- `docs/ai_context/06_testing_protocol.md` defines verification gates.
- `docs/ai_context/01_architecture_contract.md` defines file responsibilities and import boundaries.
- `docs/decisions.md` defines locked decisions.

If this file conflicts with the AI context documents or decisions log, follow `docs/ai_context/00_source_of_truth_order.md`.

## MVP Meaning

MVP does not mean skipping architecture.

MVP means every required architectural component exists in a minimal, working, testable form.

The first MVP target is:

DummyAdapter
→ canonical schema
→ patient-level split
→ mock embeddings
→ dataloader
→ FeaturePolicy
→ task masks
→ multi-task model
→ masked loss
→ evaluation smoke test

No real ODIR training, real foundation backbones, dashboard, explainability, continual learning, or paper reporting should begin before this dummy path works.

MVP Stage Summary
Stage 1 — Repository Foundation

Goal:

repository is pip-installable
retina_screen imports
pytest can discover tests

Key files:

pyproject.toml
requirements.txt
environment.yml
.gitignore
package **init**.py files

Gate:

pip install -r requirements.txt
python -c "import retina_screen; print(retina_screen.**version**)"
pytest --collect-only

Exit code 5 from pytest is acceptable only if the sole reason is zero collected tests from empty stubs.

Stage 2 — Core Contracts

Goal:

canonical schema exists
task registry exists
FeaturePolicy exists
static architecture tests exist

Key files:

src/retina_screen/core.py
src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/feature_policy.py
tests/test_schema_tasks_policy.py
tests/test_feature_policy.py
tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py

Gate:

pytest tests/test_schema_tasks_policy.py tests/test_feature_policy.py tests/test_no_dataset_coupling.py tests/test_import_boundaries.py
Stage 3 — Adapter Foundation

Goal:

base adapter contract exists
DummyAdapter emits valid canonical samples
no real dataset is required

Key files:

src/retina_screen/adapters/base.py
src/retina_screen/adapters/dummy.py
tests/test_dummy_adapter.py

Gate:

pytest tests/test_dummy_adapter.py tests/test_no_dataset_coupling.py tests/test_import_boundaries.py
Stage 4 — Splitting and Data Layer

Goal:

patient-level split works
60/15/15/10 split exists
task masks work
missing labels are not negatives

Key files:

src/retina_screen/splitting.py
src/retina_screen/data.py
tests/test_patient_split.py
tests/test_split_audit.py
tests/test_task_masking.py

Gate:

pytest tests/test_patient_split.py tests/test_split_audit.py tests/test_task_masking.py
Stage 5 — Dummy End-to-End MVP

Goal:

dummy pipeline trains and evaluates minimally

Key files:

src/retina_screen/model.py
src/retina_screen/training.py
src/retina_screen/evaluation.py
scripts/00_smoke_dummy.py
tests/test_sparse_subgroup_eval.py
tests/test_dummy_e2e.py

Gate:

pytest tests/test_task_masking.py tests/test_sparse_subgroup_eval.py tests/test_dummy_e2e.py
python scripts/00_smoke_dummy.py
Iteration Pattern

For every stage:

Specify the next file or stage.
Hand off to Claude Code with a constrained prompt.
Claude Code implements only that stage.
Claude Code runs the relevant gate.
User audits the result.
Architectural issues come back to conversation review.
Routine bugs are fixed inside Claude Code.
Decisions log is updated only if a new decision was made.
Move to the next stage only after the gate passes.
Escalate Back to Conversation When

Claude Code must escalate when:

a guardrail conflicts with another rule,
a design decision is not specified,
a protected file needs modification,
a test failure cannot be fixed without changing a contract,
schema/task/FeaturePolicy/splitting/evaluation behavior would need to change.

Routine implementation details such as helper names, log wording, and internal function signatures can be resolved by Claude Code.

Definition of MVP Done

MVP is done when:

Batch 1 repository foundation passes.
Core contract tests pass.
DummyAdapter validates canonical samples.
Patient-level split audit passes.
Task masking test passes.
Sparse subgroup evaluation test passes.
Dummy end-to-end pipeline passes.
scripts/00_smoke_dummy.py runs without real data.

ODIR work begins only after this.

## BRSET/mBRSET Access Update — 2026-05-02

BRSET and mBRSET access approval changes the intended final scientific path,
but not the immediate engineering order.

- Continue ODIR Batch/Stage 7 as the first real-dataset engineering smoke
  because ODIR is local and testable.
- Do not treat ODIR as the final primary dataset anymore.
- Do not implement BRSET/mBRSET parsing until local files are downloaded and
  inspected.
- Insert a BRSET adapter/integration batch after ODIR smoke once local BRSET
  files exist.
- Insert an mBRSET adapter/cross-device batch once local mBRSET files exist.
- BRSET/mBRSET integration should require adapter/config/task/test additions
  only, not downstream refactoring.
- Keep ODIR configs; later add BRSET/mBRSET configs instead of replacing shared
  backbone configs.

BRSET becomes the intended primary scientific dataset once inspected locally.
mBRSET becomes the intended cross-device / smartphone / portable-camera
validation dataset and continual-learning stream candidate.

---

## Stage 6 — Preprocessing and Embedding Cache

Status: completed/accepted.

Key files: src/retina_screen/preprocessing.py, src/retina_screen/embeddings.py,
scripts/02_verify_backbone_one_image.py, scripts/03_extract_embeddings.py,
configs/preprocessing/default_224.yaml, configs/backbone/mock.yaml,
tests/test_cache_manifest.py.

---

## Stage 7 — ODIR Real-Dataset Smoke

Status: completed/accepted.

Role: first real-dataset engineering smoke only. Not the primary scientific dataset.
ODIR confirmed at: data/ODIR-5K/ODIR-5K/ODIR-5K/
Stage 7 tests and gates passed in the final Stage 7 report.

Key files: src/retina_screen/adapters/odir.py, configs/dataset/odir.yaml,
configs/tasks/odir_default.yaml, configs/experiment/baseline_odir_dinov2.yaml,
configs/experiment/baseline_odir_retfound.yaml, scripts/01_make_splits.py,
scripts/04_train.py, scripts/05_evaluate.py, tests/test_odir_adapter.py.

---

## Stage 7.5 — Documentation, Dataset Inventory, Planner Correction

Status: completed after this documentation patch / ready to commit.
Type: documentation, dataset inventory, privacy policy, and planner correction.
Next stage: Stage 8A - real foundation backbone integration.

Deliverables:
- docs/dataset_inventory.md (created)
- docs/decisions.md (Decisions 019–022 appended)
- docs/mvp_build_order.md (this update)
- .gitignore (root-level stray artifact entries added)
- README.md (current development status corrected)
- docs/project_status.md (created)

---

## Stage 8A — Real Foundation Backbone Integration

Goal: implement and verify at least one real backbone (DINOv2-Large, ConvNeXt-Base, ResNet-50).
RETFound is optional pending weight availability.

Key rules:
- Use ODIR --limit 32 for one-image verification and limited smoke only.
- No silent mock fallback. Backbone must load real weights.
- No full ODIR scientific bake-off; ODIR is verification-only at this stage.

Gate: scripts/02_verify_backbone_one_image.py passes with real backbone (not mock).
scripts/03_extract_embeddings.py --limit 32 completes with real embeddings.

Key files: src/retina_screen/embeddings.py (real backbone loading),
configs/backbone/dinov2_large.yaml, configs/backbone/convnext_base.yaml,
configs/backbone/resnet50.yaml, configs/backbone/retfound.yaml.

---

## Stage 8B — BRSET Local Preflight

Goal: read-only inspection of data/brset/ — structure, metadata, labels, laterality, privacy.
Rules: no adapter implementation, no config, no code changes.
Report: patient ID format, laterality field, device/camera field, label columns, missingness,
label distribution, privacy concerns.
Gate: Stage 8B preflight report accepted by user.

---

## Stage 8C — BRSET Adapter, Config, Tasks, Tests

Goal: implement BRSETAdapter + configs/tasks/tests + smoke gates.

Key rules:
- Adapter only. No downstream changes (model.py, training.py, evaluation.py unchanged).
- RETINA_SCREEN_BRSET_ROOT env var for dataset root.
- Synthetic fixture tests; real-data tests guarded by local availability.

Gate: pytest tests/test_brset_adapter.py passes; scripts 01–05 pass with BRSET config.

Key files: src/retina_screen/adapters/brset.py, configs/dataset/brset.yaml,
configs/tasks/brset_default.yaml, tests/test_brset_adapter.py.

---

## Stage 8D — First Real BRSET Baseline + Head Ablations

Goal: real backbone embeddings on BRSET; linear probe baseline; ImageNet-init baseline.

Key rules: frozen backbone; trainable multi-task head only; pre-registered evaluation protocol;
metrics reported as preliminary (not paper-final).

Gate: scripts 01–05 pass end-to-end on BRSET with real backbone; metrics written to outputs/.

---

## Stage 8E — Baseline Visual Diagnostics

Goal: generate plots from Stage 8D artifacts only. No retraining.
Outputs: embedding TSNE/UMAP, calibration curves, task AUC bar charts, attention maps (if ViT).
All marked preliminary until reporting stage confirms them.
Gate: visual outputs present in outputs/ with no retraining.

---

## Stage 8F — mBRSET Preflight + Adapter + Cross-Device Validation

May split into 8F-1 (preflight) and 8F-2 (adapter + validation).

Goal: validate BRSET-trained model on mBRSET (cross-device, smartphone shift).
No training on mBRSET unless explicit continual-learning simulation config.

Key files: src/retina_screen/adapters/mbrset.py, configs/dataset/mbrset.yaml,
configs/tasks/mbrset_default.yaml.

Gate: cross-device evaluation metrics written; domain shift magnitude reported.

---

## Stage 8G — External Dataset Preflights and Adapters

One dataset per substage. Suggested order:
APTOS 2019 → IDRiD → Messidor-2 → EyePACS DR dataset → EyePACS-AIROGS-light-V2 → RFMiD.

Rules:
- Task-compatible validation only. No label invention. No fine-tuning on external data.
- Document label-mapping confidence for each dataset.
- RFMiD: TCAV concept source + secondary external validation only (not continual-learning stream,
  per Decision 011).
- EyePACS dataset variant and RFMiD metadata root layout must be confirmed during preflight.

Gate per dataset: preflight report + adapter + smoke evaluation accepted.

---

## Resuming Original Planner After Stage 8G

After external dataset validation, resume original conceptual sequence:

- Fairness mitigation: reweighted/Group-DRO configs; fairness-gap CSV output.
- Continual learning: offline simulation; replay buffer; mBRSET as stream candidate.
- Explainability: ViT rollout, Grad-CAM, metadata attribution, RFMiD TCAV.
- Reporting: paper-ready tables and figures.
- Dashboard: inference-only; quality/OOD gates; reliability lookup.
