## `docs/ai_context/02_guardrails_compressed.md`

# Compressed Guardrails

This file is the short, high-signal version of the project guardrails.

Severity tags:-

- `[HARD]`: correctness or research-validity rule. Do not violate.
- `[SOFT]`: strong project preference. Deviations require explanation.
- `[NORM]`: style/convention. Prefer following unless there is a practical reason.

## Claim Control

- [HARD] Do not overclaim whole-body disease prediction.
- [HARD] ODIR-only mode is not a systemic-disease headline paper.
- [HARD] ODIR diabetes/hypertension are weak proxy labels, not definitive systemic diagnoses.
- [HARD] Cardiovascular composite is not a clinical cardiovascular event predictor.
- [HARD] Retinal age gap is not novel by itself; novelty is integration into the broader pipeline.
- [HARD] Dashboard and paper-facing outputs must use screening/research language.
- [HARD] Avoid diagnostic language such as "diagnosed with diabetes" or "whole-body diagnosis."

## Dataset Rules

- [HARD] Dataset-specific logic belongs only in adapters/configs/docs/tests.
- [HARD] No native dataset columns, camera names, label vocabularies, or parsing rules in model/training/evaluation/data/preprocessing/embeddings/dashboard.
- [HARD] New datasets should require adapter/config/task additions only.
- [HARD] External datasets evaluate only tasks they actually support.
- [HARD] DR grade harmonization happens inside adapters.
- [HARD] Approximate label mappings must be recorded and surfaced.

## Splitting Rules

- [HARD] Never split at image level.
- [HARD] Split at patient level only.
- [HARD] Default split is train/val/reliability/test = 60/15/15/10.
- [HARD] Reliability split is for dashboard reliability lookup only.
- [HARD] No patient may appear in more than one split.
- [HARD] Source-test set in continual learning is frozen forever.

## Leakage Rules

- [HARD] FeaturePolicy is mandatory before metadata reaches the model.
- [HARD] Age cannot be used to predict retinal age.
- [HARD] Sex cannot be used to predict sex.
- [HARD] Missing labels are not negative labels.
- [HARD] Dataset source and camera type can become shortcuts; use only with explicit policy/ablation.
- [HARD] Metadata dropout does not replace leakage prevention.
- [HARD] Image-only and image-plus-metadata modes must be clearly separated.

## Preprocessing Rules

- [HARD] Backbone-specific preprocessing must be respected.
- [HARD] Validation/test/external preprocessing must be deterministic.
- [HARD] Augmentation is training-only.
- [SOFT] CLAHE and Graham preprocessing should be configurable, not hardcoded.
- [HARD] Horizontal flip must not corrupt laterality.
- [SOFT] Smartphone augmentations should be used only for explicit smartphone/domain-robustness experiments.

## Embedding and Cache Rules

- [HARD] Foundation backbones are frozen by default.
- [HARD] Verify one image before full embedding extraction.
- [HARD] Cache embeddings by backbone × dataset × preprocessing hash × sample.
- [HARD] Do not silently skip broken cache files.
- [HARD] OOD thresholds are backbone/preprocessing-specific.
- [HARD] Do not reuse OOD thresholds across embedding spaces.
- [HARD] PCA-64 Mahalanobis is the default OOD method unless a later decision changes it.

## Model and Training Rules

- [HARD] Task heads are created from task registry/config.
- [HARD] Task masking must be correct.
- [HARD] Missing binary labels must never become 0.
- [SOFT] Use Kendall uncertainty weighting by default unless config says otherwise.
- [SOFT] Cross-attention is optional and must be ablated.
- [SOFT] Multi-task learning should have single-task or ablation comparison where needed.
- [HARD] MC-Dropout must activate dropout without putting BatchNorm into training mode.

## Evaluation and Fairness Rules

- [HARD] Report overall and subgroup metrics.
- [HARD] Sparse subgroup metrics must be NA, not invalid numbers.
- [HARD] AUC is undefined for single-class groups.
- [HARD] Use confidence intervals for headline metrics.
- [HARD] Use the same evaluation harness for baseline and mitigation models.
- [HARD] External validation is no-retraining evaluation.
- [SOFT] Report weak external performance honestly with context rather than hiding it.
- [HARD] Pre-register evaluation protocol before headline experiments.

## Continual-Learning Rules

- [HARD] Continual learning is offline simulation only.
- [HARD] Dashboard uploads do not update model parameters.
- [HARD] Stream split and target test split must be separate.
- [HARD] Replay buffer must be subgroup-balanced.
- [HARD] OOD and quality gates happen before update eligibility.
- [HARD] Always compare against naive fine-tuning baseline.

## Explainability and Dashboard Rules

- [HARD] Attention maps are not proof of clinical reasoning.
- [HARD] Use attention rollout for ViT backbones.
- [HARD] Use Grad-CAM for ConvNet baselines.
- [HARD] Reliability lookup must be precomputed, not invented.
- [HARD] Low-quality/OOD inputs must get downgraded confidence or soft-decline warnings.
- [HARD] Dashboard must not behave like a diagnostic medical device.

## Reproducibility Rules

- [HARD] Experiments are changed by YAML, not Python edits.
- [HARD] Log resolved config, seed, git commit, dirty diff, environment, dataset hash, manifests, metrics, and checkpoints.
- [SOFT] Final headline experiments should use the seed sweep where feasible.
- [HARD] Scripts are thin; real logic stays in `src/retina_screen/`.
- [NORM] Prefer logging over print in source code.
- [NORM] Prefer explicit type hints.
