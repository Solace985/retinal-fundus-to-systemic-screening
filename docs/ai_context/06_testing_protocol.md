## `docs/ai_context/06_testing_protocol.md`

# Testing Protocol

This file defines the verification gates for implementation.

Do not move to the next stage if the current stage's gate fails.

## General Testing Rules

- Tests should target research-validity failures, not just runtime errors.
- Silent leakage bugs are worse than crashes.
- Each test must be independently runnable.
- Tests must not depend on execution order.
- Tests must not depend on shared state from previous tests.
- DummyAdapter tests must pass before real dataset work.
- Patient split tests must pass before training.
- Task masking tests must pass before multi-task training.
- Sparse subgroup tests must pass before fairness claims.
- Cache manifest tests must pass before large embedding extraction.

## Test Failure Quality

Failure messages should be actionable.

Bad failure:

```text
AssertionError: False is not true

Good failure:

AssertionError: 47 patients overlap between train and validation splits, including ['P0001', 'P0023', 'P0148']

Every high-risk test should report:

what invariant failed,
where it failed,
example offending IDs/fields where possible.
Stage 1 — Core Contract and Static-Analysis Tests

Files:

schema.py
tasks.py
feature_policy.py
static repository checks

Required tests:

tests/test_schema_tasks_policy.py
tests/test_feature_policy.py
tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py

Static-analysis tests:

test_no_dataset_coupling.py
test_import_boundaries.py

These should pass trivially on an empty or near-empty codebase and begin catching violations as code is added.

Must verify:

canonical sample requires required fields defined in schema.py
missing values are explicit
task registry entries map to valid canonical fields
task types are valid
label-quality/headline flags exist where needed
FeaturePolicy blocks age → retinal age
FeaturePolicy blocks sex → sex prediction
FeaturePolicy can produce image-only metadata mask
FeaturePolicy handles unknown/missing metadata safely
no dataset-internal vocabulary appears outside allowed files
forbidden imports are not present
Stage 2 — Adapter Tests

Files:

adapters/base.py
adapters/dummy.py

Required tests:

tests/test_dummy_adapter.py

Must verify:

DummyAdapter builds manifest
all dummy samples validate against schema
patient IDs exist
supported tasks are declared
stratification columns are declared
missing labels exist in at least some dummy samples
no native dataset assumptions are required
Stage 3 — Split Tests

File:

splitting.py

Required tests:

tests/test_patient_split.py
tests/test_split_audit.py

Must verify:

split names are train/val/reliability/test
split ratio defaults to 60/15/15/10
grouping is by patient_id
no patient appears in more than one split
split audit reports patient counts and sample counts
split audit fails loudly on overlap

Example assertion pattern:

overlap = set(train_pids) & set(val_pids)
assert len(overlap) == 0, (
    f"patient overlap between train and val: "
    f"{len(overlap)} patients, examples={sorted(list(overlap))[:10]}"
)

Repeat equivalent checks for all split pairs.

Stage 4 — Data and Task-Masking Tests

File:

data.py

Required tests:

tests/test_task_masking.py

Must verify:

missing labels generate mask = 0
observed labels generate mask = 1
missing binary labels are not encoded as negative class
masked tasks contribute zero loss/gradient
metadata tensors respect FeaturePolicy
paired-eye batching uses patient_id, not random pairing

Example masked-loss assertion pattern:

loss_with_missing = compute_masked_loss(raw_loss, mask_zero)
assert loss_with_missing.item() == 0.0, (
    "missing label produced non-zero masked loss"
)

For batched losses:

assert task_masks.shape == targets.shape, (
    f"task mask shape {task_masks.shape} does not match target shape {targets.shape}"
)
Stage 5 — Dummy End-to-End Test

Files:

model.py
training.py
evaluation.py
scripts/00_smoke_dummy.py

Required test:

tests/test_dummy_e2e.py

Must verify full dummy pipeline:

DummyAdapter
→ split
→ mock embeddings
→ dataloader
→ feature policy
→ task masks
→ model forward
→ masked loss
→ one training step
→ evaluation output

Success criteria:

no crash
loss is finite
task masks are applied
task masks have expected batch/task shape
targets and masks align by task
metadata tensor passes through FeaturePolicy with expected shape
model forward output contains at least one configured task
evaluation creates a non-empty metrics dictionary
evaluation computes at least one valid overall metric where the dummy data supports it
sparse subgroup logic does not crash
no real dataset files are needed

Expected output artifacts:

runs/dummy_smoke/<run_id>/resolved_config.yaml
runs/dummy_smoke/<run_id>/metrics.json
runs/dummy_smoke/<run_id>/train_log.csv
Stage 6 — Evaluation Validity Tests

File:

evaluation.py

Required test:

tests/test_sparse_subgroup_eval.py

Must verify:

subgroup with n < 30 returns NA
subgroup with positives < 5 returns NA
subgroup with negatives < 5 returns NA
single-class subgroup does not compute AUC
NA metrics include reason
overall metrics still compute where valid

Example invalid-subgroup assertion:

metric = compute_subgroup_auc(y_true, y_score, min_n=30, min_pos=5, min_neg=5)
assert metric.status == "NA"
assert metric.reason in {"sparse_subgroup", "insufficient_class_counts", "single_class_subgroup"}
Stage 7 — Cache and Embedding Tests

Files:

preprocessing.py
embeddings.py

Required test:

tests/test_cache_manifest.py

Must verify:

mock or small tensor embedding is saved
manifest row is written
checksum is computed
checksum reload succeeds
missing file is detected
checksum mismatch is detected
broken cache is not silently skipped

Before full extraction, also verify:

scripts/02_verify_backbone_one_image.py

Must confirm:

model loads
one image preprocesses
embedding dimension matches config
embedding saves and reloads
Stage 8 — ODIR Gate

Before ODIR training:

Dummy end-to-end test passes.
Split audit passes.
FeaturePolicy tests pass.
Task masking tests pass.
Static coupling/import tests pass.

ODIR-specific verification:

ODIR manifest builds.
ODIR patient IDs are valid.
left/right eyes share patient ID where applicable.
ODIR supported tasks are declared.
ODIR weak-proxy labels are marked.
ODIR split audit has zero patient overlap.
Stage 9 — Baseline Training Gate

Before trusting baseline numbers:

embeddings were generated from verified backbone
cache manifest passed
split is patient-level
training config is saved
resolved config is logged
seed is logged
model checkpoint is saved
validation metrics are produced
evaluation harness runs on held-out test split

Expected output artifacts:

runs/<run_id>/resolved_config.yaml
runs/<run_id>/train_log.csv
runs/<run_id>/metrics.json
runs/<run_id>/model_checkpoint.pt
outputs/evaluation/<run_id>/overall_metrics.csv
outputs/evaluation/<run_id>/subgroup_metrics.csv
Stage 10 — Fairness Gate

Before fairness claims:

preregistered protocol exists
subgroup columns are frozen
min subgroup size rules are enforced
sparse groups return NA
baseline and mitigation use same split
baseline and mitigation use same evaluation harness
fairness gap output is generated
calibration metrics are included for binary tasks

Expected output artifacts:

outputs/evaluation/<run_id>/fairness_gaps.csv
outputs/evaluation/<run_id>/calibration.csv
outputs/evaluation/<run_id>/reliability_lookup.csv
Stage 11 — External Validation Gate

Before cross-site claims:

external adapter supports the evaluated task
label mapping metadata is recorded
no fine-tuning occurs
unsupported tasks are skipped
external performance drop is reported honestly

Expected output artifact:

outputs/evaluation/<run_id>/external_validation.csv
Stage 12 — Continual-Learning Gate

Before continual-learning claims:

source_test is frozen and never enters replay
stream split and incoming_test split are separate
PCA-64 OOD stats are fit on the correct embedding space
OOD threshold is calibrated on validation split
low-quality/OOD samples are excluded from updates
replay buffer maintains subgroup balance
naive fine-tuning baseline is run
forgetting/adaptation curves are produced

Expected output artifacts:

outputs/continual/<run_id>/forgetting_curve.csv
outputs/continual/<run_id>/adaptation_curve.csv
outputs/continual/<run_id>/ood_flag_rates.csv
registry/<model_version>.json
Stage 13 — Explainability Gate

Before explainability claims:

ViT attention rollout works for selected samples
ConvNet Grad-CAM works for ConvNet baseline
metadata attribution respects FeaturePolicy
RFMiD concept role is not used as continual stream
generated explanations are described as attention-based/attribution-based, not clinical proof

Expected output artifacts:

outputs/explainability/<run_id>/attention_examples/
outputs/explainability/<run_id>/metadata_attribution.csv
Stage 14 — Dashboard Gate

Before dashboard demo:

dashboard does not call training loop
model version is displayed or logged
prediction output says screening/research use, not diagnosis
low-quality/OOD cases are visibly downgraded
reliability lookup uses generated reliability table
sparse reliability cells show insufficient data
optional logging requires consent flag
Stage 15 — Reporting Gate

Before paper outputs:

tables are generated from evaluation outputs
figures are generated programmatically
headline metrics include confidence intervals
limitations are reflected where label quality is weak
claim_mode agrees with dataset scenario

Expected output artifacts:

outputs/paper/tables/
outputs/paper/figures/
Intra-Stage Debugging Order

When multiple tests exist in a stage, run the most local test first.

Recommended order:

schema/tasks/policy tests
→ static coupling/import tests
→ adapter tests
→ split tests
→ task masking tests
→ sparse subgroup tests
→ dummy E2E
→ cache tests
→ real dataset tests

If test_dummy_e2e.py fails, run these first:

test_schema_tasks_policy.py
test_feature_policy.py
test_dummy_adapter.py
test_split_audit.py
test_task_masking.py
test_sparse_subgroup_eval.py

Do not debug the full pipeline before isolating contract-level failures.
```
