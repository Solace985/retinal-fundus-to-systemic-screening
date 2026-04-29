# Protected Files

These files define project architecture, research validity, leakage controls, evaluation rules, and agent behavior.

Coding agents must not modify protected files as a side effect of another task.

A protected file may be modified only if the user explicitly asks to edit that exact file or exact decision.

Example allowed:

Implement `src/retina_screen/schema.py`

Example not allowed:

While implementing `training.py`, also changed `schema.py`.

If a change appears necessary in a protected file, stop and ask for approval first.

## Tier 1 — Architecture and Agent-Control Files

These files control project architecture, source-of-truth precedence, repository-level agent behavior, guardrails, implementation order, and documentation authority.

CLAUDE.md
AGENTS.md
GEMINI.md
PROTECTED_FILES.md
docs/architecture.md
docs/decisions.md
docs/ai_context/00_source_of_truth_order.md
docs/ai_context/01_architecture_contract.md
docs/ai_context/02_guardrails_compressed.md
docs/ai_context/03_file_generation_order.md
docs/ai_context/04_forbidden_patterns.md
docs/ai_context/05_adapter_contract.md
docs/ai_context/06_testing_protocol.md

Rules:

- Do not rewrite these during normal coding.
- Do not “improve” them without explicit instruction.
- Do not change source-of-truth precedence.
- Do not change file generation order.
- Do not relax forbidden patterns.
- Do not change testing gates unless explicitly requested.

## Tier 2 — Core Data Contract Files

These files define the canonical internal schema, task definitions, feature permissions, leakage controls, and shared data contracts used across the system.

src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/feature_policy.py

Rules:

- Do not modify these while implementing downstream files unless explicitly requested.
- Do not duplicate their definitions elsewhere.
- Do not add dataset-specific native vocabulary here.
- Do not change canonical field names casually.
- Do not weaken FeaturePolicy leakage controls.

## Tier 3 — Adapter Boundary Files

These files define dataset adapter boundaries, dataset-specific parsing behavior, and the public interface through which raw datasets enter the canonical project schema.

src/retina_screen/adapters/base.py
src/retina_screen/adapters/dummy.py
src/retina_screen/adapters/odir.py
src/retina_screen/adapters/external_dr.py
src/retina_screen/adapters/rfmid.py
src/retina_screen/adapters/brset.py
src/retina_screen/adapters/mbrset.py

Rules:

- Concrete dataset parsing belongs only inside adapter files.
- Do not change the public adapter interface unless explicitly instructed.
- Do not expose dataset-native columns through public adapter methods.
- BRSET/mBRSET stubs should remain stubs until access is available.

## Tier 4 — Leakage, Splitting, and Evaluation-Critical Files

These files can silently invalidate research claims if modified incorrectly. They control dataset loading behavior, patient-level splitting, task masking, missing-label handling, evaluation logic, subgroup rules, and continual-learning constraints.

src/retina_screen/splitting.py
src/retina_screen/data.py
src/retina_screen/evaluation.py
src/retina_screen/continual.py

Rules:

- Do not change patient-level split logic casually.
- Do not change the 60/15/15/10 split default unless a decision is logged.
- Do not weaken task masking.
- Do not treat missing labels as negatives.
- Do not change sparse subgroup rules casually.
- Do not allow dashboard/live continual learning.

## Tier 5 — Experiment-Validity Configs

These files affect paper claims, evaluation protocol, OOD detection behavior, training defaults, ablation validity, and claim boundaries.

configs/evaluation/preregistered_protocol.yaml
configs/paper/claim_mode.yaml
configs/ood/pca64_mahalanobis.yaml
configs/model/multitask_default.yaml
configs/model/multitask_no_metadata.yaml
configs/model/multitask_no_cross_attention.yaml
configs/training/standard.yaml
configs/training/reweighted.yaml
configs/training/group_dro.yaml
configs/training/continual.yaml

Rules:

- Do not change evaluation protocol after headline experiments begin.
- Do not change claim mode to allow unsupported systemic claims.
- Do not change OOD method defaults without recording the decision.
- Do not change baseline training defaults just to improve a result.

## Tier 6 — Architecture-Enforcement Tests

These tests enforce architectural boundaries, leakage prevention, split correctness, masking behavior, subgroup evaluation safety, cache reproducibility, end-to-end dummy execution, dataset-decoupling rules, and import boundaries.

tests/test_feature_policy.py
tests/test_patient_split.py
tests/test_split_audit.py
tests/test_task_masking.py
tests/test_sparse_subgroup_eval.py
tests/test_cache_manifest.py
tests/test_dummy_e2e.py
tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py

Rules:

- Do not weaken these tests to make implementation pass.
- Do not delete architecture checks.
- If a test is failing because the test is too strict, explain the false positive and ask before changing it.
- Prefer fixing source code over weakening tests.

## Normal Modification Rule

If the user explicitly asks for a protected file by name, editing that file is allowed. The agent should still confirm the scope of the change before proceeding.

If the user asks for a different file, for example, “implement training.py”, then editing `schema.py`, `tasks.py`, `feature_policy.py`, the adapter base file, the splitting utility, the evaluation harness, the continual-learning protocol, or any other protected file is not allowed unless the user separately approves.

A change to a protected file as a side effect of an unrelated task is a violation of this rule, regardless of how reasonable the change appears.

## Required Agent Behavior

Before modifying a protected file, the agent must confirm:

1. Which user instruction requires the change.
2. Whether the change affects downstream files that depend on the protected contract.
3. Which tests must be rerun after the change.
4. Whether the change should be recorded in `docs/decisions.md`.

If explicit approval is not already given, stop and ask. Do not modify a protected file as a side effect of an unrelated task. Do not make a protected change “for completeness” or “to keep things consistent.”
