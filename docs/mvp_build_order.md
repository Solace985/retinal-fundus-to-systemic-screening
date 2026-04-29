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
