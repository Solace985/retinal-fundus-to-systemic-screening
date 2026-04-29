# Forbidden Patterns

This file lists patterns that should trigger review failure unless they appear in an explicitly allowed location.

Allowed locations for dataset-specific vocabulary:

- `src/retina_screen/adapters/*`
- `configs/*`
- `docs/*`
- `tests/*` when testing coupling rules

Dataset names such as `odir`, `brset`, and `mbrset` are not globally forbidden. They may appear in config names, experiment names, logs, adapter-resolution code, and documentation.

What is forbidden outside adapters/configs/docs/tests is dataset-internal vocabulary: native column names, native image-field names, native camera values, native diagnostic-string names, and dataset-specific conditional logic.

## Dataset-Internal Vocabulary Outside Allowed Locations

Forbidden outside adapters/configs/docs/tests:

```text
diagnostic_keywords
Left-Fundus
Right-Fundus
left_fundus
right_fundus
Canon
Nikon
Kowa
Zeiss
Phelcom

Checked by:

tests/test_no_dataset_coupling.py

Allowed alternative:

Use canonical schema fields such as:
- sample.image_path
- sample.eye_laterality
- sample.camera_type
- sample.dataset_source
- sample.diabetes
- sample.hypertension

Native vocabulary must be translated into canonical fields inside the adapter.

Dataset Conditional Logic

Forbidden outside adapter/config-resolution code:

if dataset_name == "odir":
    ...

if dataset_name == "brset":
    ...

if dataset_name == "mbrset":
    ...

if config.dataset.name == "odir":
    ...

Checked by:

tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py

Allowed alternatives:

# In config-resolution / adapter factory code only:
adapter_class = import_from_string(config.dataset.adapter_class)
adapter = adapter_class(config.dataset)

# In generic pipeline code:
supported_tasks = adapter.get_supported_tasks()
stratification_columns = adapter.get_stratification_columns()

Generic code should ask the adapter/registry/config what is available. It should not branch on dataset identity.

Image-Level Splitting

Forbidden:

train_test_split(images)
train_test_split(sample_ids)

Checked by:

tests/test_patient_split.py
tests/test_split_audit.py

Allowed alternative:

groups = manifest["patient_id"]
# Use group-aware splitting so all samples from a patient stay in one split.

Required invariant:

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(reliability_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(reliability_patients)
assert val_patients.isdisjoint(test_patients)
assert reliability_patients.isdisjoint(test_patients)
Missing Labels as Negatives

Forbidden:

label = 0 if missing else label

Checked by:

tests/test_task_masking.py

Allowed alternative:

target = placeholder_value
mask = 0

# Later:
loss = raw_loss * mask

Missing labels contribute zero gradient.

Metadata Leakage

Forbidden:

age_years used to predict retinal_age
sex used to predict sex
dataset_source used for disease prediction without explicit policy
camera_type used for disease prediction without explicit policy

Checked by:

tests/test_feature_policy.py
tests/test_task_masking.py

Allowed alternative:

allowed_metadata = feature_policy.allowed_fields(task_name, model_input_mode)
metadata_tensor = build_metadata_tensor(sample, allowed_metadata)

All metadata must pass through FeaturePolicy.

Dashboard Retraining

Forbidden in dashboard code:

train(...)
fit(...)
optimizer.step()
update_model(...)
run_continual_update(...)

Checked by:

tests/test_import_boundaries.py

Allowed alternative:

with torch.no_grad():
    predictions = model(batch)

Dashboard may run inference, quality/OOD checks, explanations, and reliability lookup. It must not update parameters.

MC-Dropout Inference

Forbidden:

model.train()

when used only to activate MC-Dropout.

Checked by:

tests/test_import_boundaries.py

Allowed alternative:

model.eval()
activate_dropout_layers_only(model)
# Keep BatchNorm frozen.

MC-Dropout must not accidentally put BatchNorm into training mode.

Sparse or Single-Class Subgroup AUC

Forbidden:

roc_auc_score(y_true, y_score)

without checking subgroup validity.

Checked by:

tests/test_sparse_subgroup_eval.py

Allowed alternative:

if n < min_subgroup_size:
    return NA(reason="sparse_subgroup")

if positives < min_positives or negatives < min_negatives:
    return NA(reason="insufficient_class_counts")

if only_one_class_present:
    return NA(reason="single_class_subgroup")

Invalid subgroup metrics should be reported as NA with a reason.

External Validation Fine-Tuning

Forbidden:

train on primary dataset
fine-tune on external dataset
report as external validation

Checked by:

tests/test_import_boundaries.py
evaluation review

Allowed alternative:

train on primary dataset
evaluate directly on external dataset without retraining

Fine-tuning external datasets belongs only to continual-learning simulation, not external validation.

Silent Cache Skipping

Forbidden:

if not path.exists():
    continue

Checked by:

tests/test_cache_manifest.py

Allowed alternative:

if not path.exists() or checksum_mismatch(path):
    logger.warning("Cache entry invalid; re-extracting sample_id=%s", sample_id)
    reextract_sample(sample_id)

Missing or corrupted cache files must be logged and repaired if possible.

OOD Threshold Reuse Across Embedding Spaces

Forbidden:

threshold = load_threshold("retfound_threshold.json")
# then using it for DINOv2 embeddings

Checked by:

tests/test_cache_manifest.py
continual-learning review

Allowed alternative:

cache/ood_stats/{backbone}/{dataset}/{preprocessing_hash}/threshold.json

OOD stats must be keyed by the exact embedding space.

Diagnostic Claim Language

Avoid in code comments, dashboard text, and paper-facing outputs:

diagnosis
diagnosed with diabetes
diagnosed with hypertension
definitive systemic disease detection
whole-body diagnosis
clinically identified lesion

Checked by:

manual review
dashboard review

Allowed alternative:

screening probability
proxy systemic label
risk marker
research-use output
requires clinical confirmation
attention-based explanation
Business Logic Inside Scripts

Forbidden:

implementing training loop in scripts/04_train.py
implementing metrics in scripts/05_evaluate.py
parsing ODIR columns in scripts
building model architecture in scripts

Checked by:

tests/test_import_boundaries.py
manual review

Allowed alternative:

# Script:
config = load_config(args.config)
run_training(config)

# Source package:
# training.py owns the training implementation.

Scripts should load configs and call src/retina_screen/*.

```
