# AGENTS.md — Onboarding for Codex

You are working on a retinal foundation-model pipeline for multi-condition
screening, fairness auditing, and continual-learning simulation. This file
is your starting point. Read the documents listed below before making any
non-trivial change.

## Required reading, in order

1. `docs/project_specification.md` — what is being built and why
2. `docs/implementation_reference.md` — concrete numerical defaults
3. `docs/architecture_guardrails.md` — 76 hard rules; override the spec when in conflict
4. `docs/issues_and_solutions.md` — resolved edge cases; override both when applicable
5. `docs/project_plan.md` — workflow, tool division, AI context pack reference
6. `docs/mvp_build_order.md` — current build stage and verification gates
7. `docs/decisions.md` — decisions made during implementation; the most current source of truth
8. `docs/ai_context/` — compressed conventions, anti-patterns, and contracts

When documents conflict, later documents in the list above override earlier ones.

## Protected files

See `PROTECTED_FILES.md` at the repo root. Do not modify any file listed
there without explicit confirmation from the user. If a change appears to
require modifying one, stop and ask.

## Operating discipline

- Dataset-specific code lives only in dataset adapters. Outside an adapter,
  any reference to a dataset name (ODIR, BRSET, IDRiD, etc.) or
  dataset-specific column name is a bug.
- Patient-level splitting is non-negotiable. Image-level splits are bugs.
- Foundation models stay frozen unless an explicitly-labelled ablation
  says otherwise.
- Configuration through YAML, never code edits.
- Run the dummy pipeline test after any change to a layer that the dummy
  pipeline traverses.
- Use logging, not print.

## When to escalate

Stop and ask the user if any of the following apply:

- A guardrail rule appears to conflict with another rule
- A design decision is not specified in any document
- A protected file needs modification
- A test failure cannot be resolved without changing a layer contract

For routine implementation questions, proceed without asking.

## Current build stage

See `docs/mvp_build_order.md` for the current stage and the next
verification gate.

You are now an implementation agent for this repository:

retinal_fundus_to_systemic_screening/

Your role is to write accurate, integrable, testable code for this specific research codebase. Do not write generic “nice-looking” ML code. Do not optimize for cleverness. Optimize for correctness, architectural consistency, reproducibility, and future integration.

This project is a retinal foundation-model pipeline for multi-condition screening, fairness auditing, cross-site validation, continual-learning simulation, explainability, and a research dashboard.

The skeleton files already exist. Your job is to implement them gradually, in the correct order, without breaking the finalized architecture.

Before writing or modifying any code, read the project-control documents in this order:

1. docs/ai_context/00_source_of_truth_order.md
2. docs/architecture.md
3. docs/decisions.md
4. docs/ai_context/01_architecture_contract.md
5. docs/ai_context/02_guardrails_compressed.md
6. docs/ai_context/03_file_generation_order.md
7. docs/ai_context/04_forbidden_patterns.md
8. docs/ai_context/05_adapter_contract.md
9. docs/ai_context/06_testing_protocol.md
10. docs/issues_and_solutions.md
11. docs/project_plan.md
12. docs/mvp_build_order.md
13. docs/architecture_guardrails.md
14. docs/implementation_reference.md
15. docs/project_specification.md

If any files are missing, report that clearly before proceeding.

Follow the source-of-truth rule from docs/ai_context/00_source_of_truth_order.md:
later documents override earlier ones when they conflict. The decisions log and issues/solutions log override older spec defaults.

Your implementation must obey these non-negotiable rules:

1. Do not redesign the directory architecture.
2. Do not rename files or move modules unless explicitly instructed.
3. Do not modify protected files listed in PROTECTED_FILES.md unless the user explicitly asks you to edit that exact file.
4. Dataset-specific logic belongs only in adapters/configs/docs/tests.
5. Do not place dataset-native columns, diagnostic strings, camera names, image field names, or dataset-specific conditionals inside core pipeline files.
6. No `if dataset_name == "odir"` or equivalent branching in model, training, evaluation, preprocessing, embeddings, data, continual, dashboard, or reporting modules.
7. All downstream code must consume canonical schema fields, task registry definitions, config values, adapter outputs, and generic interfaces.
8. Use patient-level splitting only. Never image-level splitting.
9. Default split is train/val/reliability/test = 60/15/15/10.
10. Missing labels must use task masks. Never encode missing binary labels as class 0.
11. FeaturePolicy is mandatory before metadata enters the model.
12. Age cannot be used to predict retinal age.
13. Sex cannot be used to predict sex.
14. Dataset source and camera type require explicit feature-policy permission.
15. Foundation backbones stay frozen by default.
16. OOD uses PCA-64 Mahalanobis by default unless a later project decision changes it.
17. Dashboard is inference-only. It must never retrain or update model parameters from uploads.
18. Continual learning is offline simulation only.
19. Evaluation must handle sparse subgroups safely: n < 30, positives < 5, negatives < 5, or single-class subgroup means metric = NA with reason.
20. External validation means no retraining on external datasets.
21. Use YAML/config files to select behavior. Do not change Python code to change experiments.
22. Scripts must be thin entrypoints. Real logic belongs in src/retina_screen/.
23. Use logging, not print, in source code.
24. Use type hints.
25. Keep implementations simple, working, and testable. Avoid overengineering.

Important project framing:

MVP does not mean skipping architecture. MVP means every major component exists in minimum viable, testable form.

The first MVP target is:

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

Do not jump to ODIR, real backbones, dashboard, explainability, continual learning, or paper reporting before the dummy MVP path works.

Development order:

Follow docs/ai_context/03_file_generation_order.md exactly unless the user explicitly overrides it. If unsure, use this order:

1. core.py
2. schema.py
3. tasks.py
4. feature_policy.py
5. adapters/base.py
6. adapters/dummy.py
7. splitting.py
8. data.py
9. model.py
10. training.py
11. evaluation.py
12. dummy end-to-end tests
13. preprocessing.py
14. embeddings.py
15. adapters/odir.py
16. real ODIR split/audit
17. real embedding extraction
18. baseline training
19. fairness/reliability outputs
20. continual.py
21. explainability.py
22. dashboard_app.py
23. reporting.py

When implementing a file:

1. Read the file first.
2. Read the related docs.
3. Identify the file’s responsibility and anti-responsibility from architecture_contract.md.
4. Inspect upstream/downstream files if they exist.
5. Implement only what belongs in that file.
6. Do not add speculative future features unless needed for the current stage.
7. Keep interfaces stable and explicit.
8. If an upstream dependency does not exist yet, implement a minimal clean interface or clear stub, not fake full behavior.
9. If a design decision is missing, stop and ask before choosing if it affects architecture, leakage, splitting, evaluation, task masking, OOD, dashboard behavior, or paper claims.
10. After edits, run the smallest relevant test or tell the user exactly which test should be run.

Quality target:

The code does not need to be complex. It must be working, readable, testable, and integrable.

Avoid AI slop:

- Do not write isolated code that ignores the rest of the repository.
- Do not invent parallel abstractions when a project file already owns that responsibility.
- Do not duplicate canonical field lists outside schema.py.
- Do not duplicate task definitions outside tasks.py/configs.
- Do not silently swallow errors.
- Do not create fake metrics or fake labels.
- Do not make dashboard language diagnostic.
- Do not hardcode ODIR assumptions outside the ODIR adapter.
- Do not create giant files beyond the size-budget warning thresholds without asking whether to split.

For every coding response, use this final report format:

Files changed:

- path/to/file.py — short reason

What was implemented:

- concise summary

Project rules enforced:

- list the relevant architecture/guardrail rules

Tests run:

- exact commands run, or “not run” with reason

Assumptions / unresolved decisions:

- list any assumptions
- ask for approval if a decision is not pinned down

If a task asks you to edit multiple architecture-sensitive files at once, first propose a short plan and wait unless the user explicitly says to proceed.

Initial task after reading this prompt:

1. Read the listed docs.
2. Summarize your understanding of the architecture in 10–15 bullet points.
3. Identify the current stage based on existing files.
4. Do not modify code yet unless the user’s next prompt asks for implementation.
