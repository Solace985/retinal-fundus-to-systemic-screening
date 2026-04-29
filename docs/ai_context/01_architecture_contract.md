# Architecture Contract

This is the enforceable architecture summary for coding agents.

Do not redesign the repository unless explicitly instructed.

## Core Architecture

The project is a modular monolith under:

```text
src/retina_screen/

Each file owns one major layer of the pipeline. The architecture is dataset-agnostic downstream of adapters.

Dataset Boundary

Dataset-specific logic may exist only in:

src/retina_screen/adapters/*
configs/dataset/*
configs/tasks/*
configs/experiment/* when selecting experiment presets
docs/*
tests/* when testing dataset-coupling rules

Dataset-specific logic must not exist in:

schema.py
feature_policy.py
splitting.py
data.py
preprocessing.py
embeddings.py
model.py
training.py
evaluation.py
continual.py
explainability.py
dashboard_app.py
reporting.py

A dataset-agnostic pipeline does not mean dataset names never appear. It means dataset-specific structure, native columns, parsing rules, label vocabularies, camera names, and path conventions are isolated behind adapters/configuration.

File Responsibilities and Anti-Responsibilities
core.py

Owns:

YAML/config loading
logging setup
seeding
run directory creation
path helpers
git commit/diff capture
environment capture
device selection

Must not contain:

schema definitions
task definitions
adapter parsing logic
model architecture
training loops
evaluation metrics
schema.py

Owns:

canonical sample schema
canonical enums
missing-value conventions
manifest contract types where needed

Must not contain:

ODIR/BRSET/mBRSET parsing logic
experiment-specific task lists
training logic
evaluation logic
tasks.py

Owns:

task definitions
task registry
task type metadata
target-column mapping to canonical schema
label-quality metadata
headline eligibility metadata

Must not contain:

native dataset column names
dataset parsing logic
model heads
loss execution code
feature_policy.py

Owns:

metadata access control
leakage prevention
image-only vs image-plus-metadata mode rules
task-specific blocked/allowed metadata fields

Must not contain:

model architecture
task loss computation
dataset parsing logic

Mandatory behavior:

age cannot be used to predict retinal age
sex cannot be used to predict sex
dataset source and camera type require explicit policy
adapters/*

Own:

native dataset reading
native label parsing
native path resolution
native-to-canonical mapping
supported-task declaration
stratification-column declaration
quality-column declaration
patient ID extraction
eye laterality extraction where available

Must not contain:

training logic
evaluation metrics
model architecture
fairness computation
dashboard inference
continual-learning updates
splitting.py

Owns:

patient-level splitting
60/15/15/10 train/val/reliability/test split
split audit
zero-overlap verification

Must not contain:

image-level random splitting
dataset-native parsing logic
model/training code
data.py

Owns:

embedding-backed datasets
collate functions
task-mask construction
metadata tensor construction
FeaturePolicy application
paired-eye batching
source balancing where needed

Must not contain:

native dataset parsing
model architecture
optimizer logic
metric computation
preprocessing.py

Owns:

retinal crop
resize
optional CLAHE
optional Graham preprocessing
normalization
training augmentations
deterministic validation/test transforms

Must not contain:

dataset-native column names
training loops
model task heads
evaluation metrics
embeddings.py

Owns:

backbone loading
one-image verification
frozen embedding extraction
cache path construction
manifest writing
checksum validation
cache repair
embedding-space metadata needed for OOD

Must not contain:

task losses
fairness metrics
dashboard UI code
native dataset parsing outside adapter-provided inputs
model.py

Owns:

multi-task head
metadata branch
fusion trunk
optional paired-eye attention
task heads
MC-dropout inference path

Must not contain:

adapter imports
dataset-specific conditionals
optimizer/training loop
evaluation harness
paper reporting
training.py

Owns:

training loop
validation loop
task-masked losses
Kendall uncertainty weighting
reweighted training
Group DRO
early stopping
checkpointing

Must not contain:

native dataset parsing
direct concrete adapter usage
paper figure generation
dashboard inference logic
evaluation.py

Owns:

binary metrics
multiclass/ordinal metrics
regression metrics
calibration metrics
bootstrap confidence intervals
subgroup stratification
fairness gaps
sparse subgroup handling
external validation evaluation
reliability lookup generation

Must not contain:

training loop
optimizer updates
paper figure rendering
dashboard UI logic
native dataset parsing
continual.py

Owns offline continual-learning simulation:

PCA-64 Mahalanobis OOD
replay buffer
subgroup balancing
stream/test separation
LoRA/head-adapter update orchestration
naive fine-tuning baseline
forgetting/adaptation curves

Must not contain:

live dashboard retraining
user-upload model updates
native dataset parsing
dashboard UI code
explainability.py

Owns:

ViT attention rollout
ConvNet Grad-CAM
metadata attribution after FeaturePolicy
optional RFMiD-based TCAV

Must not contain:

clinical causality claims
model training loop
dataset-native parsing
dashboard_app.py

Owns research dashboard inference:

upload image
optional metadata entry
quality/OOD warning flow
prediction display
uncertainty display
explanation overlay
reliability lookup
optional consent-based logging

Must not contain:

model training
optimizer steps
continual-learning updates from uploads
diagnostic medical-device claims
reporting.py

Owns:

paper-ready table generation
paper-ready figure generation
consumption of evaluation outputs

Must not contain:

model training
raw metric computation if already owned by evaluation.py
dataset parsing logic
Explicit Per-File Dependency Graph

This dependency graph is the intended import boundary. Keep imports acyclic.

Foundation modules
schema.py
  imports: standard library, pydantic/dataclasses/typing only

tasks.py
  may import: schema.py

feature_policy.py
  may import: schema.py, tasks.py

core.py
  may import: standard library, yaml/omegaconf/hydra if used, logging, random/numpy/torch utilities
  must not import: adapters, model, training, evaluation
Adapter modules
adapters/base.py
  may import: schema.py, tasks.py

adapters/dummy.py
  may import: adapters/base.py, schema.py, tasks.py

adapters/odir.py
  may import: adapters/base.py, schema.py, tasks.py

adapters/external_dr.py
  may import: adapters/base.py, schema.py, tasks.py

adapters/rfmid.py
  may import: adapters/base.py, schema.py, tasks.py

adapters/brset.py
  may import: adapters/base.py, schema.py, tasks.py

adapters/mbrset.py
  may import: adapters/base.py, schema.py, tasks.py

Adapters must not import:

model.py
training.py
evaluation.py
continual.py
dashboard_app.py
reporting.py
Data pipeline modules
splitting.py
  may import: schema.py

data.py
  may import: schema.py, tasks.py, feature_policy.py

preprocessing.py
  may import: core.py only for logging/config helpers if needed

embeddings.py
  may import: schema.py, preprocessing.py, core.py

These modules must not import concrete adapters unless explicitly resolving adapter classes through config in a controlled factory.

Model/training/evaluation modules
model.py
  may import: tasks.py, schema.py if needed

training.py
  may import: data.py, model.py, tasks.py, evaluation.py for validation metrics only

evaluation.py
  may import: schema.py, tasks.py

Forbidden:

model.py importing adapters/*
evaluation.py importing adapters/odir.py
training.py parsing native dataset files
Advanced pipeline modules
continual.py
  may import: data.py, model.py, training.py, evaluation.py, embeddings.py, core.py

explainability.py
  may import: model.py, embeddings.py, preprocessing.py, feature_policy.py

dashboard_app.py
  may import: preprocessing.py, embeddings.py, model.py, explainability.py, evaluation.py for reliability lookup utilities only

reporting.py
  may import: evaluation.py output readers and core/path helpers

Forbidden:

dashboard_app.py importing training.py for live updates
reporting.py recomputing metrics that evaluation.py should own
continual.py being triggered by dashboard uploads
MVP Rule

Do not implement advanced features before the dummy end-to-end pipeline works.

First major milestone:

DummyAdapter
→ split
→ mock embeddings
→ dataloader
→ feature policy
→ task masks
→ model
→ masked loss
→ evaluation

ODIR implementation comes after this.

File Size Budget

These are calibration anchors, not hard limits. If a file exceeds the warning threshold, raise the split question instead of silently producing a massive file.

File	Expected MVP size	Warning threshold
core.py	250–500 LOC	700 LOC
schema.py	250–500 LOC	700 LOC
tasks.py	250–500 LOC	700 LOC
feature_policy.py	200–400 LOC	600 LOC
adapters/base.py	150–300 LOC	450 LOC
adapters/dummy.py	150–300 LOC	450 LOC
splitting.py	250–500 LOC	700 LOC
data.py	400–650 LOC	850 LOC
preprocessing.py	300–600 LOC	800 LOC
embeddings.py	500–850 LOC	1100 LOC
model.py	500–850 LOC	1100 LOC
training.py	600–950 LOC	1200 LOC
evaluation.py	700–1100 LOC	1300 LOC
continual.py	700–1100 LOC	1300 LOC
explainability.py	400–800 LOC	1000 LOC
dashboard_app.py	350–700 LOC	900 LOC
reporting.py	300–600 LOC	800 LOC

If evaluation.py, training.py, embeddings.py, or continual.py exceeds the warning threshold, propose a split but do not perform it unless approved.

Producer / Consumer Contracts
File	Consumes	Produces
adapters/*	raw dataset files	canonical manifests/samples
splitting.py	canonical manifest	split CSV + split audit JSON
preprocessing.py	raw image path/tensor + preprocessing config	preprocessed image tensor
embeddings.py	canonical manifest + split + preprocessing/backbone config	.pt embeddings + embedding manifest CSV
data.py	embedding manifest + canonical metadata + split CSV	model-ready batches
model.py	model config + task registry	predictions/logits per task
training.py	batches + model + training config	checkpoints, train logs, validation metrics
evaluation.py	predictions/checkpoints + evaluation config	overall metrics, subgroup metrics, fairness gaps, reliability lookup
continual.py	trained model + embeddings + stream config	continual-learning logs, forgetting/adaptation curves, adapter/checkpoint registry updates
explainability.py	model + image/embedding + explanation config	attention/Grad-CAM/attribution outputs
dashboard_app.py	trained model + reliability lookup + OOD stats	UI outputs + optional consent logs
reporting.py	evaluation outputs	paper-ready tables and figures
Future Dataset Rule

Adding a new dataset should require only:

a new adapter file or adapter class,
dataset YAML,
task YAML if new tasks exist,
registry/config additions where needed,
embedding extraction rerun.

Adding BRSET or mBRSET must not require changes to:

model.py
training.py
evaluation.py
continual.py
dashboard_app.py
```
