# File Generation Order

Follow this order unless explicitly instructed otherwise.

Do not jump to ODIR, real backbones, dashboard, continual learning, or paper reporting before the DummyAdapter MVP path works.

This file defines generation order only. Detailed testing gates are defined in:

- `docs/ai_context/06_testing_protocol.md`

## Stage 0 — Control Documents

Create or finalize:

1. `docs/architecture.md`
2. `docs/decisions.md`
3. `docs/ai_context/00_source_of_truth_order.md`
4. `docs/ai_context/01_architecture_contract.md`
5. `docs/ai_context/02_guardrails_compressed.md`
6. `docs/ai_context/03_file_generation_order.md`
7. `docs/ai_context/04_forbidden_patterns.md`
8. `docs/ai_context/05_adapter_contract.md`
9. `docs/ai_context/06_testing_protocol.md`

Expected result:

- Architecture is frozen.
- Source-of-truth precedence is clear.
- Adapter contract is defined.
- Forbidden patterns are defined.
- Testing gates are defined.

## Stage 1 — Repository Foundation

Create:

1. `README.md`
2. `CLAUDE.md`
3. `PROTECTED_FILES.md`
4. `pyproject.toml`
5. `requirements.txt`
6. `environment.yml`
7. `.gitignore`
8. package directories and `__init__.py` files

Expected result:

- Repository can be installed as a package.
- Coding agents know what they can and cannot modify.
- Basic tooling is ready.

Recommended early tests:

1. `tests/test_no_dataset_coupling.py`
2. `tests/test_import_boundaries.py`

These are static-analysis tests. They should pass trivially on an empty or near-empty codebase and begin catching violations as files are added.

## Stage 2 — Core Contracts

Create:

1. `src/retina_screen/core.py`
2. `src/retina_screen/schema.py`
3. `src/retina_screen/tasks.py`
4. `src/retina_screen/feature_policy.py`

Create or update tests:

1. `tests/test_schema_tasks_policy.py`
2. `tests/test_feature_policy.py`
3. `tests/test_no_dataset_coupling.py`
4. `tests/test_import_boundaries.py`

Test order:

1. `test_schema_tasks_policy.py`
2. `test_feature_policy.py`
3. `test_no_dataset_coupling.py`
4. `test_import_boundaries.py`

Expected result:

- Canonical schema exists.
- Task registry exists.
- FeaturePolicy exists.
- Static architecture tests exist.
- No dataset-specific logic has leaked into core modules.

## Stage 3 — Adapter Foundation

Create:

1. `src/retina_screen/adapters/base.py`
2. `src/retina_screen/adapters/dummy.py`

Create or update tests:

1. `tests/test_dummy_adapter.py`

Test order:

1. `test_schema_tasks_policy.py`
2. `test_dummy_adapter.py`
3. `test_no_dataset_coupling.py`
4. `test_import_boundaries.py`

Expected result:

- Base adapter contract exists.
- DummyAdapter produces valid canonical samples.
- DummyAdapter does not require real dataset files.

## Stage 4 — Splitting and Data Layer

Create:

1. `src/retina_screen/splitting.py`
2. `src/retina_screen/data.py`

Create or update tests:

1. `tests/test_patient_split.py`
2. `tests/test_split_audit.py`
3. `tests/test_task_masking.py`

Test order:

1. `test_patient_split.py`
2. `test_split_audit.py`
3. `test_task_masking.py`
4. `test_no_dataset_coupling.py`
5. `test_import_boundaries.py`

Expected result:

- Patient-level split exists.
- Default split is train/val/reliability/test = 60/15/15/10.
- Split audit proves zero patient overlap.
- Data layer can build task masks.
- Missing labels are not treated as negative labels.

Expected artifacts after split script exists:

- `outputs/splits/<dataset_or_dummy>/<split_id>/splits.csv`
- `outputs/splits/<dataset_or_dummy>/<split_id>/split_audit.json`

## Stage 5 — Dummy Model/Training/Evaluation MVP

Create:

1. `src/retina_screen/model.py`
2. `src/retina_screen/training.py`
3. `src/retina_screen/evaluation.py`
4. `scripts/00_smoke_dummy.py`

Create or update tests:

1. `tests/test_task_masking.py`
2. `tests/test_sparse_subgroup_eval.py`
3. `tests/test_dummy_e2e.py`

Target pipeline:

- DummyAdapter
- patient split
- mock embeddings
- dataloader
- feature policy
- task masks
- model
- masked loss
- evaluation

Test order:

1. `test_task_masking.py`
2. `test_sparse_subgroup_eval.py`
3. `test_dummy_e2e.py`
4. `test_no_dataset_coupling.py`
5. `test_import_boundaries.py`

Expected result:

- Dummy end-to-end pipeline runs without real data.
- Loss is finite.
- Task masks are applied.
- Evaluation outputs at least one valid overall metric when dummy labels support it.
- Sparse subgroup logic returns NA where appropriate.

Expected artifacts:

- `runs/dummy_smoke/<run_id>/resolved_config.yaml`
- `runs/dummy_smoke/<run_id>/metrics.json`
- `runs/dummy_smoke/<run_id>/train_log.csv`

## Stage 6 — Preprocessing and Embeddings

Create:

1. `src/retina_screen/preprocessing.py`
2. `src/retina_screen/embeddings.py`
3. `scripts/02_verify_backbone_one_image.py`
4. `scripts/03_extract_embeddings.py`

Create or update tests:

1. `tests/test_cache_manifest.py`

Test order:

1. `test_cache_manifest.py`
2. `test_no_dataset_coupling.py`
3. `test_import_boundaries.py`

Verification:

- one image through one backbone
- embedding file saved
- manifest written
- checksum reload works
- broken cache is not silently skipped

Expected artifacts:

- `cache/embeddings/<backbone>/<dataset>/<preprocessing_hash>/<sample_id>.pt`
- `cache/embeddings/<backbone>/<dataset>/<preprocessing_hash>/manifest.csv`

## Stage 7 — ODIR Adapter and First Real Dataset

Create:

1. `src/retina_screen/adapters/odir.py`
2. `configs/dataset/odir.yaml`
3. `configs/tasks/odir_default.yaml`
4. `configs/experiment/baseline_odir_dinov2.yaml`
5. `configs/experiment/baseline_odir_retfound.yaml`
6. `scripts/01_make_splits.py`
7. `scripts/04_train.py`
8. `scripts/05_evaluate.py`

Test order:

1. `test_dummy_e2e.py`
2. `test_patient_split.py`
3. `test_split_audit.py`
4. `test_task_masking.py`
5. `test_cache_manifest.py`
6. `test_no_dataset_coupling.py`
7. `test_import_boundaries.py`

Verification:

- ODIR manifest builds.
- ODIR weak-proxy labels are marked.
- Split audit passes.
- One-image preprocessing passes.
- One-image backbone verification passes.
- Embeddings extract.
- Baseline trains.
- Evaluation outputs overall and subgroup metrics.

Expected artifacts:

- `outputs/splits/odir/<split_id>/splits.csv`
- `outputs/splits/odir/<split_id>/split_audit.json`
- `cache/embeddings/<backbone>/odir/<preprocessing_hash>/manifest.csv`
- `runs/<run_id>/resolved_config.yaml`
- `runs/<run_id>/train_log.csv`
- `runs/<run_id>/metrics.json`
- `outputs/evaluation/<run_id>/overall_metrics.csv`
- `outputs/evaluation/<run_id>/subgroup_metrics.csv`

## Stage 8 — Fairness Mitigation

Create or update:

1. `configs/training/reweighted.yaml`
2. `configs/training/group_dro.yaml`
3. `configs/experiment/fairness_odir_reweighted.yaml`
4. `configs/experiment/fairness_odir_groupdro.yaml`
5. `src/retina_screen/training.py` only if mitigation hooks are not already implemented
6. `src/retina_screen/evaluation.py` only if fairness outputs are not already implemented

Verification:

- baseline and mitigation use the same split.
- baseline and mitigation use the same evaluation harness.
- fairness gap table is produced.
- performance-vs-fairness trade-off output is produced.

Expected artifacts:

- `outputs/evaluation/<run_id>/fairness_gaps.csv`
- `outputs/evaluation/<run_id>/calibration.csv`
- `outputs/evaluation/<run_id>/reliability_lookup.csv`

## Stage 9 — External Validation

Create:

1. `src/retina_screen/adapters/external_dr.py`
2. `configs/dataset/external_dr.yaml`
3. `configs/tasks/external_dr.yaml`

Verification:

- external datasets evaluate only supported tasks.
- no external fine-tuning occurs.
- label mapping metadata is recorded.
- unsupported tasks are skipped.

Expected artifact:

- `outputs/evaluation/<run_id>/external_validation.csv`

## Stage 10 — Continual Learning

Create:

1. `src/retina_screen/continual.py`
2. `configs/training/continual.yaml`
3. `configs/ood/pca64_mahalanobis.yaml`
4. `configs/experiment/continual_odir.yaml`
5. `scripts/06_run_continual.py`

Verification:

- source-test remains frozen.
- stream and incoming-test are separated.
- OOD gate runs before update eligibility.
- replay buffer maintains subgroup balance.
- naive fine-tuning baseline is included.

Expected artifacts:

- `outputs/continual/<run_id>/forgetting_curve.csv`
- `outputs/continual/<run_id>/adaptation_curve.csv`
- `outputs/continual/<run_id>/ood_flag_rates.csv`
- `registry/<model_version>.json`

## Stage 11 — Explainability

Create:

1. `src/retina_screen/explainability.py`
2. `src/retina_screen/adapters/rfmid.py`
3. `configs/dataset/rfmid.yaml`

Verification:

- ViT attention rollout works for selected samples.
- ConvNet Grad-CAM works for ConvNet baseline.
- Metadata attribution respects FeaturePolicy.
- RFMiD concept role is not used as continual stream.

Expected artifacts:

- `outputs/explainability/<run_id>/attention_examples/`
- `outputs/explainability/<run_id>/metadata_attribution.csv`

## Stage 12 — Dashboard

Create:

1. `src/retina_screen/dashboard_app.py`
2. `scripts/08_launch_dashboard.py`

Verification:

- dashboard is inference-only.
- low-quality/OOD states are surfaced.
- reliability lookup uses generated table.
- no training loop is called.
- dashboard language uses screening/research wording, not diagnosis wording.

## Stage 13 — Reporting

Create:

1. `src/retina_screen/reporting.py`
2. `scripts/07_generate_paper_outputs.py`

Verification:

- paper tables and figures are generated from evaluation outputs.
- no manual notebook-only result generation is required.
- claim mode agrees with dataset scenario.

Expected artifacts:

- `outputs/paper/tables/`
- `outputs/paper/figures/`

## Global Rule

If a later stage requires changing a protected earlier file, stop and ask for approval unless the user explicitly requested that exact protected file.

Protected examples:

- `schema.py`
- `tasks.py`
- `feature_policy.py`
- `adapters/base.py`
- `splitting.py`
- `evaluation.py`
- `docs/architecture.md`
- `docs/decisions.md`
- `docs/ai_context/*`
