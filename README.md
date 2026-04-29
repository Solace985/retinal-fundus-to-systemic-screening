# Retinal Fundus Foundation-Model Pipeline

A reproducible research pipeline for retinal fundus image analysis using frozen foundation-model embeddings, multi-task prediction heads, fairness auditing, cross-site validation, continual-learning simulation, explainability, and a research dashboard.

The project is designed for a medical AI research paper targeting a mid-to-top-tier venue depending on dataset access and final experimental strength.

## Project Goal

This repository builds a dataset-agnostic retinal screening pipeline that can support:

- multi-condition retinal disease prediction,
- foundation-model backbone comparison,
- patient-level data splitting,
- task-masked multi-task learning,
- fairness and subgroup auditing,
- external/cross-site validation,
- OOD-aware continual-learning simulation,
- image-level explainability,
- subgroup reliability lookup,
- and a research-use dashboard.

The initial MVP is built with a `DummyAdapter` first, then extended to ODIR-5K, and later to BRSET/mBRSET if dataset access is available.

## Current Development Status

Architecture: finalized  
Implementation status: early MVP build  
Current priority: DummyAdapter end-to-end pipeline

First MVP target:

DummyAdapter
→ canonical schema
→ patient-level split
→ mock embeddings
→ dataloader
→ feature policy
→ task masks
→ multi-task model
→ masked loss
→ evaluation smoke test

`

Real datasets and real foundation backbones should be added only after the dummy path works.

## Architecture Summary

The codebase follows a modular monolith design under:

src/retina_screen/

Main layers:

Dataset Adapter
→ Canonical Schema
→ Patient-Level Split
→ Preprocessing
→ Frozen Backbone Embeddings
→ Embedding Cache
→ Data Loader + FeaturePolicy + Task Masks
→ Multi-Task Head
→ Training Loop
→ Evaluation / Fairness / Reliability
→ Continual Learning / Explainability / Dashboard / Reporting

Dataset-specific logic is allowed only inside adapters and configuration files. Downstream modules must consume canonical schema fields, task registry definitions, feature-policy outputs, and config-driven behavior.

## Repository Structure

retinal_fundus_to_systemic_screening/
│
├── configs/ # YAML experiment, dataset, model, training, OOD, and evaluation configs
├── docs/ # project specification, guardrails, decisions, architecture, and AI context docs
├── src/retina_screen/ # main Python package
├── scripts/ # thin CLI entrypoints
├── tests/ # unit and architecture-validity tests
├── data/ # local datasets, gitignored
├── cache/ # embeddings and cache artifacts, gitignored
├── runs/ # training runs and checkpoints, gitignored
├── registry/ # small model/version metadata
└── outputs/ # generated metrics, tables, plots, and paper outputs

## Core Design Rules

The most important rules are:

1. Dataset-specific logic belongs only in `src/retina_screen/adapters/*` and config files.
2. All downstream code must use the canonical schema.
3. Splitting must be patient-level, never image-level.
4. The default split is `train / validation / reliability / test = 60 / 15 / 15 / 10`.
5. Missing labels must use task masks and must never be treated as negative labels.
6. Metadata must pass through `FeaturePolicy` before reaching the model.
7. Foundation backbones are frozen by default.
8. ODIR-only experiments must not be framed as definitive systemic disease prediction.
9. The dashboard is inference-only and must not retrain from user uploads.
10. Evaluation must handle sparse/single-class subgroups safely.

For complete rules, see:

docs/ai_context/
docs/architecture.md
docs/decisions.md
PROTECTED_FILES.md
AGENTS.md

## Setup

Create and activate an environment, then install the package in editable mode.

### Using pip

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

On Linux/macOS:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

### Using conda

conda env create -f environment.yml
conda activate retinal-screening
pip install -e .

## Running Tests

Run the full test suite:

pytest

Run architecture/static checks:

pytest tests/test_no_dataset_coupling.py tests/test_import_boundaries.py

Run core contract tests:

pytest tests/test_schema_tasks_policy.py tests/test_feature_policy.py

## MVP Execution Order

Implementation should follow the order defined in:

docs/ai_context/03_file_generation_order.md

High-level order:

1. Core contracts
2. Base adapter + DummyAdapter
3. Patient-level split and data layer
4. Dummy model/training/evaluation MVP
5. Preprocessing and embedding cache
6. ODIR adapter
7. Real embedding extraction
8. Baseline training
9. Fairness/reliability outputs
10. Continual learning, explainability, dashboard, reporting

## Planned CLI Scripts

The scripts are thin entrypoints. Real logic belongs in `src/retina_screen/`.

scripts/00_smoke_dummy.py
scripts/01_make_splits.py
scripts/02_verify_backbone_one_image.py
scripts/03_extract_embeddings.py
scripts/04_train.py
scripts/05_evaluate.py
scripts/06_run_continual.py
scripts/07_generate_paper_outputs.py
scripts/08_launch_dashboard.py

Example future usage:

python scripts/00_smoke_dummy.py
python scripts/04_train.py --config configs/experiment/baseline_odir_retfound.yaml
python scripts/05_evaluate.py --config configs/experiment/baseline_odir_retfound.yaml

## Dataset Notes

The architecture is built to support multiple retinal datasets through adapters.

Initial datasets:

- `DummyAdapter` for smoke testing.
- `ODIRAdapter` for open-access MVP experiments.
- `external_dr.py` for DR-focused external validation datasets.
- `RFMiDAdapter` for concept directions and secondary ocular validation.
- `BRSETAdapter` and `mBRSETAdapter` as future integrations after dataset access.

Dataset files should be placed under `data/`, which is intentionally gitignored.

## Research-Framing Notes

In ODIR-only mode, diabetes and hypertension labels are treated as weak proxy labels, not definitive structured systemic diagnoses.

The ODIR-only paper framing should emphasize:

multi-condition retinal screening

- fairness auditing
- cross-site robustness
- continual-learning simulation

rather than strong whole-body/systemic disease prediction.

Stronger systemic-proxy framing becomes possible only if BRSET/mBRSET access is available and properly integrated.

## For Coding Agents

Coding agents must read:
AGENTS.md
PROTECTED_FILES.md
docs/ai_context/
docs/architecture.md
docs/decisions.md

Do not redesign the architecture.
Do not modify protected files unless explicitly instructed.
Do not add dataset-specific logic outside adapters/configs/docs/tests.
Build MVP-first and run tests after each stage.

## License

License to be decided before public release.
