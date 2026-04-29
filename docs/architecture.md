# Retinal Foundation-Model Pipeline — Final Architecture

Status: locked v1 architecture  
Project: retinal_fundus_to_systemic_screening  
Purpose: build a dataset-agnostic retinal foundation-model pipeline for multi-condition screening, fairness auditing, cross-site validation, continual-learning simulation, explainability, and dashboard demonstration.

This architecture is a modular monolith. It is not a single-file prototype and not an over-fragmented microservice-style project. Each file owns one major responsibility. Splitting into more files is allowed only after MVP if a file becomes genuinely too large or unstable.

Important: The Architecture is to be referred to every single time there are any changes made in the files to ensure proper integration among them and overall structural integrity of the codebase.

## Final Directory Structure

```text
retinal_fundus_to_systemic_screening/
│
├── README.md
├── CLAUDE.md
├── PROTECTED_FILES.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── experiment/
│   │   ├── smoke_dummy.yaml
│   │   ├── baseline_odir_dinov2.yaml
│   │   ├── baseline_odir_retfound.yaml
│   │   ├── baseline_odir_convnext.yaml
│   │   ├── fairness_odir_reweighted.yaml
│   │   ├── fairness_odir_groupdro.yaml
│   │   └── continual_odir.yaml
│   │
│   ├── dataset/
│   │   ├── dummy.yaml
│   │   ├── odir.yaml
│   │   ├── external_dr.yaml
│   │   ├── rfmid.yaml
│   │   ├── brset.yaml
│   │   └── mbrset.yaml
│   │
│   ├── backbone/
│   │   ├── mock.yaml
│   │   ├── dinov2_large.yaml
│   │   ├── retfound.yaml
│   │   ├── convnext_base.yaml
│   │   └── resnet50.yaml
│   │
│   ├── tasks/
│   │   ├── odir_default.yaml
│   │   ├── external_dr.yaml
│   │   ├── brset_default.yaml
│   │   └── mbrset_default.yaml
│   │
│   ├── model/
│   │   ├── multitask_default.yaml
│   │   ├── multitask_no_metadata.yaml
│   │   └── multitask_no_cross_attention.yaml
│   │
│   ├── preprocessing/
│   │   ├── default_224.yaml
│   │   ├── default_512.yaml
│   │   └── smartphone_robust.yaml
│   │
│   ├── training/
│   │   ├── standard.yaml
│   │   ├── reweighted.yaml
│   │   ├── group_dro.yaml
│   │   └── continual.yaml
│   │
│   ├── ood/
│   │   └── pca64_mahalanobis.yaml
│   │
│   ├── evaluation/
│   │   └── preregistered_protocol.yaml
│   │
│   └── paper/
│       └── claim_mode.yaml
│
├── docs/
│   ├── project_specification.pdf
│   ├── implementation_reference.pdf
│   ├── architecture_guardrails.pdf
│   ├── issues_and_solutions.md
│   ├── project_plan.md
│   ├── mvp_build_order.md
│   ├── architecture.md
│   ├── decisions.md
│   └── ai_context/
│       ├── 00_source_of_truth_order.md
│       ├── 01_architecture_contract.md
│       ├── 02_guardrails_compressed.md
│       ├── 03_file_generation_order.md
│       ├── 04_forbidden_patterns.md
│       ├── 05_adapter_contract.md
│       └── 06_testing_protocol.md
│
├── src/
│   └── retina_screen/
│       ├── __init__.py
│       ├── core.py
│       ├── schema.py
│       ├── tasks.py
│       ├── feature_policy.py
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── dummy.py
│       │   ├── odir.py
│       │   ├── external_dr.py
│       │   ├── rfmid.py
│       │   ├── brset.py
│       │   └── mbrset.py
│       │
│       ├── splitting.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── embeddings.py
│       ├── model.py
│       ├── training.py
│       ├── evaluation.py
│       ├── reporting.py
│       ├── continual.py
│       ├── explainability.py
│       └── dashboard_app.py
│
├── scripts/
│   ├── 00_smoke_dummy.py
│   ├── 01_make_splits.py
│   ├── 02_verify_backbone_one_image.py
│   ├── 03_extract_embeddings.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_run_continual.py
│   ├── 07_generate_paper_outputs.py
│   └── 08_launch_dashboard.py
│
├── tests/
│   ├── __init__.py
│   ├── test_schema_tasks_policy.py
│   ├── test_feature_policy.py
│   ├── test_dummy_adapter.py
│   ├── test_patient_split.py
│   ├── test_split_audit.py
│   ├── test_dummy_e2e.py
│   ├── test_task_masking.py
│   ├── test_cache_manifest.py
│   ├── test_sparse_subgroup_eval.py
│   ├── test_no_dataset_coupling.py
│   └── test_import_boundaries.py
│
├── data/                 # gitignored
├── cache/                # gitignored
├── runs/                 # gitignored
├── registry/             # tracked small JSON metadata only
└── outputs/              # generated tables, plots, reliability lookup, paper-ready results
```
