# Source of Truth Order

This file defines how to resolve conflicts between project documents, architecture notes, implementation instructions, and current user prompts.

All coding agents must read this before modifying source code.

## Core Precedence Rule

The project follows a **later-overrides-earlier** precedence model.

When two documents conflict, the document that appears later in the ordered chain below wins.

## Ordered Source Chain

From earliest / lowest authority to latest / highest authority:

1. `docs/project_specification.md`
2. `docs/implementation_reference.md`
3. `docs/architecture_guardrails.md`
4. `docs/issues_and_solutions.md`
5. `docs/decisions.md`

Later items override earlier items when they conflict.

## Supporting Documents

The following files are implementation-control documents. They guide coding agents but do not override the source chain above unless the same decision is also recorded in `docs/decisions.md`.

- `docs/architecture.md`
- `docs/project_plan.md`
- `docs/mvp_build_order.md`
- `docs/ai_context/01_architecture_contract.md`
- `docs/ai_context/02_guardrails_compressed.md`
- `docs/ai_context/03_file_generation_order.md`
- `docs/ai_context/04_forbidden_patterns.md`
- `docs/ai_context/05_adapter_contract.md`
- `docs/ai_context/06_testing_protocol.md`

## Current User Instruction

Current user instructions apply unless they conflict with the ordered source chain.

If the user asks for something that violates the project specification, implementation reference, guardrails, issues log, or decisions log, stop and explain the conflict instead of implementing blindly.

## Worked Conflict Example

Conflict:

- `project_specification.pdf` originally describes a 70/15/15 train/validation/test split.
- `issues_and_solutions.md` later requires a 60/15/15/10 train/validation/reliability/test split.
- `docs/decisions.md` confirms the 60/15/15/10 split as the locked project decision.

Resolution:

- Implement `60/15/15/10`.
- Treat the reliability split as mandatory.
- Do not implement the older 70/15/15 split unless a later decision explicitly reverses this.

Code implication:

- `src/retina_screen/splitting.py` must default to train/val/reliability/test.
- Split tests must verify all four splits.
- Reliability lookup generation must use the reliability split, not validation or test.

## Known Locked Overrides

The following decisions are already resolved and should not be re-litigated unless `docs/decisions.md` is updated:

- Split is `60/15/15/10`, not `70/15/15`.
- OOD detection defaults to PCA-64 Mahalanobis.
- ODIR-only mode must not headline systemic disease prediction.
- ODIR diabetes/hypertension are weak proxy labels.
- RFMiD is not a continual-learning stream.
- Dashboard is inference-only and must not retrain from user uploads.
- Dataset-specific logic is allowed only in adapters/configs/docs/tests.
- FeaturePolicy is mandatory before metadata reaches the model.
- Missing labels must be represented with task masks, not negative labels.

## If a Decision Is Not Pinned Down

If a design decision is not clearly determined by the source chain:

1. Name the unresolved decision.
2. Propose 2–3 options.
3. Explain trade-offs.
4. Ask for approval before implementing if the decision affects:
   - architecture,
   - dataset coupling,
   - patient leakage,
   - task masking,
   - feature leakage,
   - fairness evaluation,
   - continual-learning protocol,
   - dashboard medical-claim language,
   - paper claims.

Do not silently choose architecture-affecting defaults.
