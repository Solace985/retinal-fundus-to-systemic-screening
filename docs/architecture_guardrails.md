PDF To Markdown Converter
Debug View
Result View
Below is the implementation guardrail list you should keep beside you before writing the
architecture or code. This is not a general “what could go wrong” list. These are the exact failure
points that can damage the paper, inflate results, create leakage, or make the system impossible
to extend later.

The core rule: before coding anything, force every file/module to obey them.

A. Paper-scope and claim-control issues

1. Do not overclaim “whole-body disease prediction”
   The retina can encode systemic signals, but your public datasets do not support unlimited
   systemic prediction. If BRSET/mBRSET arrives, you can claim public-data systemic proxy
   screening for diabetes, hypertension, retinal age gap, and related metadata-supported conditions.
   If only ODIR is available, you must not make systemic disease prediction the headline because
   ODIR’s diabetes/hypertension labels are weaker and noisier. The project spec itself states that
   public-data systemic prediction must be limited to realistic labels, while true endpoints like
   MACE, eGFR, HbA1c, Alzheimer’s risk, and longitudinal outcomes require restricted datasets.

Implementation rule:

Create a claim_mode or dataset_scenario config:

paper_claim_mode:
scenario: odir_only | brset_only | brset_mbrset
allow_systemic_headline: false
allow_cv_composite_headline: false

For ODIR-only, the architecture can still have diabetes/hypertension heads, but the paper must
frame them as weak systemic proxy labels , not clinical systemic prediction.

2. Choose the paper narrative based on dataset access before
   writing results
   If BRSET/mBRSET is available, the paper can be “retinal foundation-model screening for ocular

systemic proxy prediction with fairness and deployment readiness.”
If only ODIR is available, the stronger narrative is:

A reproducible, fair, continually-updatable foundation-model pipeline for multi-condition
retinal screening, validated across populations.

The implementation reference explicitly says that if BRSET/mBRSET are unavailable, systemic
prediction should be dropped as the primary claim, cardiovascular composite should become
only a dashboard convenience, smartphone transfer should be dropped, and ODIR’s paired-eye
multi-label structure should become central.

Implementation rule:

Keep the code flexible, but keep the paper framing conditional:

ODIR-only = ocular multi-condition + fairness + cross-site robustness

continual learning.
BRSET = systemic proxy + device fairness + structured metadata.
BRSET + mBRSET = systemic proxy + smartphone transfer + device-domain
robustness. 3. Do not thinly execute too many contributions
The full project has many attractive parts: foundation-model bake-off, multi-task learning,
fairness, mitigation, continual learning, OOD, uncertainty, explainability, dashboard, TCAV,
CycleGAN, vision-language extension, RetiZero, etc.

That is dangerous. A paper with five weak contributions looks worse than a paper with three
strong ones. The implementation reference says the non-droppable parts are: foundation-model
bake-off, fairness audit with mitigation, continual-learning simulation, cross-population
external validation, and image-level explainability. Optional items like vision-language
extension, CycleGAN, TCAV, RetiZero, dashboard polish, and extra ablations should be
dropped first under pressure.

Implementation rule:

Your first architecture should support optional modules, but not depend on them.

Mandatory:

dataset adapter
canonical schema
embedding extraction
multi-task head
evaluation/fairness harness
cross-site validation
continual-learning simulation
image-level explainability

Optional:

TCAV
CycleGAN
vision-language alignment
RetiZero
high-polish dashboard
extra ablation sweeps

4. Pre-register the evaluation protocol before headline
   experiments
   Do not run experiments, look at results, then decide which subgroup, metric, or dataset to report.
   That creates post-hoc cherry-picking. The implementation reference explicitly says the subgroup
   list, mitigation methods, external datasets, metrics, bootstrap count, and significance thresholds
   must be frozen before headline experiments.

Implementation rule:

Before training the final model, create:

configs/evaluation/preregistered_protocol.yaml

It must freeze:

metrics:
binary: [roc_auc, pr_auc, accuracy, precision, recall, f1,
balanced_accuracy, ece, brier]
multiclass: [macro_auc, macro_f1, weighted_f1, quadratic_kappa,
accuracy]
regression: [mae, rmse, r2, pearson]

subgroups:

sex
age_band
camera_type
device_class
dataset_source
mitigation_methods:

baseline
reweighted*loss
group_dro
bootstrap_iterations: 1000
min_subgroup_size: 30
min_positives: 5
multiple_comparison_correction: benjamini_hochberg_fdr_0*

After this file is frozen, do not modify it because one metric looks bad.

5. Define one quantitative headline claim early
   The paper needs one number that reviewers can remember. For example:

Worst-subgroup AUC gap reduced from X to Y with less than 2% overall
AUC loss.

or:

Naive continual learning caused X% source-test AUC drop, while replay

OOD-gated LoRA reduced forgetting to Y%.
The implementation reference explicitly says reviewers need one such anchor number, otherwise
the integration story becomes harder to defend.

Implementation rule:

By the end of the first proper baseline phase, decide which of these will be the headline:

fairness gap reduction
continual-learning forgetting reduction
cross-site robustness degradation analysis
foundation-model ranking under fairness stratification

Do not wait until the end of the paper.

B. Dataset and adapter design issues 6. All dataset-specific logic must live only inside dataset
adapters
This is one of the most important architecture rules. ODIR-specific column names, BRSET-
specific label names, camera values, diagnostic strings, and path structures must never leak into
training, evaluation, model, or dashboard code. The project spec says all dataset-specific
structure must live in exactly one place: the dataset adapter.

Implementation rule:

This is allowed:

src/dataset/odir_adapter.py knows ODIR column names.
src/dataset/brset_adapter.py knows BRSET column names.

This is forbidden:

forbidden inside training/eval/model code
if df["diagnostic_keywords"].contains("hypertension"):
...

Instead:

sample.hypertension
sample.dr_grade
sample.camera_type

Everything downstream only sees canonical fields.

7. Build the canonical schema before any real model code
   If the schema is unstable, every downstream file becomes unstable. The schema is the contract
   between datasets and the whole pipeline. The spec says every adapter must output a fixed, well-
   typed schema, and missing fields must be explicitly None or NaN, never absent.

Implementation rule:

Before writing training code, define:

src/dataset/schema.py

It must include:

sample_id
patient_id
dataset_source
image_path
eye_laterality

age_years
sex
ethnicity
camera_type
hospital_site
device_class
dr_grade
diabetes
hypertension
hypertensive_retinopathy
glaucoma
cataract
amd
pathological_myopia
drusen
smoking
obesity
insulin_use
diabetes_duration_years
focus_quality
illumination_quality
image_field_quality
artefact_presence
education_level
insurance_status

Even if ODIR does not provide a field, it should exist as None.

8. Use a task registry, not hardcoded task lists
   Tasks must be first-class objects. The project spec says tasks should have declared name, type,
   target column, loss function, primary metric, and metadata.

Implementation rule:

Never do this inside the training loop:

tasks = ["diabetes", "hypertension", "glaucoma"]

Instead define:

src/dataset/tasks.py

Each task should include:

TaskDefinition(
name="diabetes",
task_type="binary",
target_column="diabetes",
loss="bce",
primary_metric="roc_auc",
allow_metadata=["age_years", "sex"], # controlled by feature
policy
)

This is what allows BRSET/mBRSET tasks to be added later without rewriting the model.

9. Patient-level splitting is non-negotiable
   Do not split by image. Retinal datasets often contain left/right eyes from the same patient, and
   sometimes repeated visits. If one eye goes into train and the other eye goes into test, the model
   has effectively seen the patient. The implementation reference explicitly warns that image-level
   splitting can inflate AUC by 5–15 percentage points.

Implementation rule:

The split function must group by patient_id.

Forbidden:

train_test_split(images)

Required:

GroupShuffleSplit(groups=patient_id)

Before training, generate a split audit:

train_patients ∩ val_patients = 0
train_patients ∩ test_patients = 0
val_patients ∩ test_patients = 0

No patient overlap is allowed.

10. Left/right eye handling must be explicit
    ODIR is bilateral. That is useful, but it also creates leakage risk if not handled properly. The
    model must know whether the sample is left eye, right eye, or paired-eye. The patient-level split
    must keep both eyes in the same split. Cross-attention must only combine eyes from the same
    patient within the same split.

Implementation rule:

Represent eye structure explicitly:

patient_id = P
left_eye_sample_id = P001_L
right_eye_sample_id = P001_R
eye_laterality = left/right
paired_available = true/false

Do not let the dataloader randomly pair samples. Paired-eye batches must be constructed using
patient IDs.

11. DR grade harmonisation must happen inside adapters
    Different datasets may use ICDR, SDRG, adjudicated DR grades, or simplified categories. The
    implementation reference explicitly says BRSET’s ICDR and SDRG must be unified into a

canonical 0–4 ICDR scale at the adapter level, and that APTOS, Messidor-2, and IDRiD must be
mapped carefully.

Implementation rule:

Each adapter should output:

dr_grade = 0/1/2/3/
dr_grade_source_scheme = "ICDR" | "SDRG" | "Messidor" | ...
dr_grade_mapping_confidence = "exact" | "approximate"

Do not hide approximate mappings. They must be documented.

12. ODIR diabetes/hypertension labels are noisy
    ODIR’s diabetes and hypertension labels are not the same as structured clinical diagnosis
    records. They are derived from diagnostic text/category information. The project spec identifies
    this as a weakness and says it should be reported transparently, with sensitivity analysis, and not
    treated as definitive systemic evidence.

Implementation rule:

For ODIR:

tasks:
diabetes:
label_quality: weak_proxy
allowed_as_headline: false

hypertension:
label_quality: weak_proxy
allowed_as_headline: false

For BRSET/mBRSET:

tasks:
diabetes:
label_quality: structured_or_metadata

allowed_as_headline: true

13. Camera/device confounding must not be misreported
    BRSET has a cleaner Canon-vs-Nikon device experiment. ODIR has multiple cameras, but
    camera may be confounded with hospital/site. The project spec says ODIR camera effects should
    be reported as confounded with site, not as clean device invariance.

Implementation rule:

For ODIR:

camera_type != pure device experiment
camera_type may include site/hospital bias

Report:

camera/site-stratified robustness

Not:

clean device-invariance experiment

14. External validation datasets only evaluate supported
    tasks
    IDRiD, Messidor-2, APTOS, and EyePACS mainly support DR evaluation. They cannot
    evaluate every systemic task. The project spec says external datasets should be used only for
    tasks where labels exist.

Implementation rule:

The evaluation harness must check:

if task not in adapter.get_supported_tasks():
skip_task_for_dataset()

Do not report fake missing metrics. Do not impute labels.

15. Subgroup fairness requires enough samples and positives
    Fairness metrics break when subgroups are too small or contain only one class. The
    implementation reference says sparse subgroups should be reported as NA if they have fewer than
    30 samples or fewer than 5 positives.

Implementation rule:

For every subgroup-task pair:

if n < 30: report NA
if positives < 5: report NA
if negatives < 5: report NA

Do not calculate AUC for single-class groups. That produces invalid numbers.

C. Leakage and shortcut-learning issues 16. Add a Feature Policy / Leakage Guard layer
This is my strongest architecture addition. The current design allows metadata like age and sex to
enter the metadata branch. But the model also predicts age and sex. That creates direct leakage.

If age is input and retinal age is target, the model can cheat.

If sex is input and sex is target, the model can cheat.

If camera/source is input, the model may learn dataset shortcuts instead of retinal disease
features.

Implementation rule:

Create:

src/dataset/feature_policy.py

It controls which metadata fields are allowed for each task.

Example:

feature_policy:
retinal_age:
blocked_inputs: [age_years]
sex:
blocked_inputs: [sex]
diabetes_image_only:
blocked_inputs: [age_years, sex, dataset_source, camera_type]
diabetes_clinical_mode:
allowed_inputs: [age_years, sex, device_class]

This layer must run before the model receives metadata.

17. Separate image-only scientific mode from image-plus-
    metadata deployment mode
    A reviewer will ask: “Is the model learning retinal signal, or just using age/sex metadata?” You
    need two modes.

Mode A: Image-only scientific mode

Used to prove retinal images contain signal.

input = image embedding only

Mode B: Image + metadata deployment mode

Used for best practical performance.

input = image embedding + allowed metadata

Implementation rule:

Every main experiment should clearly state which mode it uses:

model_input_mode: image_only | image_plus_metadata

Never mix the two in the same result table without labeling.

18. Cardiovascular composite leakage must be prevented
    The implementation reference explicitly warns that if the CV composite is constructed from
    labels the model also predicts, training the model to predict the composite leaks information. It
    says the composite should either be computed from variables the model does not predict, or
    treated as a separate held-out target.

Implementation rule:

For ODIR-only:

Do not train CV composite as a headline target.
Use it only as dashboard-derived proxy if needed.

For BRSET/mBRSET:

If CV composite uses age, sex, diabetes, hypertension, smoking,
obesity,
then be clear whether the model is predicting:

the composite directly from image, or
the component labels separately.
Avoid circular training like:

model predicts diabetes
model predicts hypertension

CV score is built from those predictions
model is also trained to predict CV score

That is not an independent target.

19. Dataset source as a feature can become a shortcut
    The implementation reference warns that dataset-source one-hot features can let the model
    “game” the source signal instead of learning image features.

Implementation rule:

If dataset_source is used during multi-source training, define why.

Allowed:

used for domain correction during training

Dangerous:

used at inference as normal predictive metadata

Safer options:

zero-mask dataset_source at inference
use gradient reversal
run ablation with and without dataset_source

20. Camera type can also become a shortcut
    Camera type may correlate with site, population, disease severity, or dataset. If the model sees
    camera type, it might use it as a shortcut.

Implementation rule:

For disease prediction, run at least two variants:

image_only
image_plus_allowed_metadata_without_camera
image_plus_allowed_metadata_with_camera

If performance jumps only when camera is included, inspect whether the model is exploiting
acquisition bias.

21. Metadata branch attribution is only meaningful if
    leakage is controlled
    If the model uses age to predict age, SHAP/Integrated Gradients will simply confirm leakage.
    Metadata explainability is only useful after task-specific metadata masking exists.

Implementation rule:

Do not implement metadata SHAP before implementing FeaturePolicy.

D. Preprocessing and augmentation issues 22. Backbone-specific preprocessing must be correct
RETFound and DINOv2 may require different image size, normalization, and preprocessing. The
implementation reference explicitly warns that RETFound should use official model-card
normalization, not generic ImageNet stats, and that DINOv2 expects 224×224 for the standard
checkpoint.

Implementation rule:

Do not have one universal preprocessing config.

Use:

configs/backbone/retfound.yaml
configs/backbone/dinov2_large.yaml
configs/backbone/resnet50.yaml

Each must specify:

input_size:
normalization_mean:
normalization_std:
expects_rgb:
requires_center_crop:

23. Validation and test preprocessing must be deterministic
    Augmentation belongs only in training. The implementation reference says validation/test time
    should use only deterministic preprocessing: crop, resize, CLAHE, normalization.

Implementation rule:

train_transform = deterministic preprocessing + augmentation
val_transform = deterministic preprocessing only
test_transform = deterministic preprocessing only
external_transform = deterministic preprocessing only

Never augment validation/test images.

24. CLAHE and Graham preprocessing must be
    configurable, not hardcoded
    CLAHE or Graham preprocessing may help some datasets/backbones and hurt others. If
    hardcoded, you cannot fairly compare backbones.

Implementation rule:

preprocessing:
crop_retinal_circle: true
clahe: true | false
graham: true | false

Run ablation later if needed. Do not bake it permanently into the image loader.

25. Horizontal flip must not corrupt laterality
    The implementation reference warns that fundus laterality must be handled carefully when using
    horizontal flip.

Implementation rule:

Either:

disable horizontal flip when laterality is used

or:

flip image and update laterality consistently

Do not flip images while keeping incorrect left/right semantics.

26. Smartphone augmentation must not destroy clinical-
    camera performance
    The implementation reference says smartphone augmentation can hurt clinical-camera
    performance if too aggressive, and must be validated.

Implementation rule:

Use smartphone-style augmentation only in experiments that explicitly target portable-device
robustness.

Track:

clinical-camera test AUC before smartphone augmentation
clinical-camera test AUC after smartphone augmentation

If the drop is large, do not use it in the main model.

27. Image-quality gate must not become fake sophistication
    The quality gate is useful, but ODIR may not have true quality labels. If quality is proxy-derived
    from blur/brightness, say so. Do not present it as clinically validated image-quality assessment.

Implementation rule:

Quality labels must have a source flag:

quality_label_source = explicit_dataset_label | proxy_blur_brightness
| manual_subset

Dashboard should say:

quality flag based on proxy image-quality metrics

when appropriate.

E. Embedding extraction and cache issues 28. Freeze foundation models by default
The project is built around frozen foundation models and lightweight heads. The spec explicitly
gives the rationale: full fine-tuning is compute-heavy, risks catastrophic forgetting, and weakens
reproducibility.

Implementation rule:

for p in backbone.parameters():
p.requires_grad = False

No backbone fine-tuning unless running a clearly labeled ablation.

29. Verify every backbone on one image before full
    extraction
    Do not launch full-dataset extraction until each model loads and produces a valid embedding for
    one sample. The implementation reference says model identifiers must be verified before Phase
    2, and unavailable models should be removed or deferred.

Implementation rule:

For every backbone:

load model
load one image
run preprocessing
extract embedding
confirm embedding dimension
save one .pt file
reload .pt file

Only then run full extraction.

30. Cache embeddings by backbone, dataset, sample, and
    preprocessing version
    If preprocessing changes, old embeddings become invalid. If backbone changes, OOD thresholds
    become invalid. The spec says embeddings should be cached using content/config versioning.

Implementation rule:

Embedding path should include:

cache/embeddings/{backbone_name}/{dataset_source}/{preprocessing_hash}
/{sample_id}.pt

Manifest must include:

sample_id
patient_id
dataset_source
embedding_path
embedding_dim
backbone_name
backbone_version
preprocessing_hash
created_at
checksum

31. Never silently skip broken cache files
    The implementation reference says missing or checksum-mismatched files should be re-extracted
    and logged, not silently skipped.

Implementation rule:

Create:

verify_cache()

It should:

walk manifest
check file exists
check checksum
re-extract only broken/missing samples
log every repair

32. OOD thresholds are backbone-specific
    Mahalanobis distance must be computed in the same embedding space used for inference. The
    implementation reference warns that if the backbone changes, the OOD threshold must be
    recomputed.

Implementation rule:

OOD files must be stored like:

ood_stats/{backbone}/{dataset}/{preprocessing_hash}/mean.pt
ood_stats/{backbone}/{dataset}/{preprocessing_hash}/cov_inv.pt
ood_stats/{backbone}/{dataset}/{preprocessing_hash}/threshold.json

Do not reuse RETFound OOD thresholds for DINOv2 embeddings.

33. Store enough information for explainability
    If you only store CLS embeddings, training becomes easy, but attention rollout or patch-level
    analysis may need original image + model attention maps. Decide early what explainability
    needs.

Implementation rule:

For the main model, cache CLS embeddings. For explainability, either:

re-run backbone on selected images

or optionally store:

pooled patch tokens
attention maps for selected examples

Do not store massive patch tokens for every image unless necessary.

F. Model architecture issues 34. Run a signal audit before building the full multi-task
model
Before investing in multi-task heads, fairness mitigation, LoRA, and dashboard, prove that the
dataset labels are learnable.

Implementation rule:

First run:

DINOv2 embeddings + logistic regression
DINOv2 embeddings + shallow MLP
ResNet/ConvNeXt embeddings + logistic regression

Check:

are labels above random?
are subgroup counts usable?
are patient splits clean?
are embeddings correctly cached?

If this fails, the full architecture will not save the paper.

35. Task masking must be implemented correctly
    Different datasets have different labels. The multi-task model must not treat missing labels as
    zero. The spec says each sample carries a per-task mask, and unobserved labels contribute zero
    gradient.

Implementation rule:

For each batch:

y[task] = label or placeholder
mask[task] = 1 if label exists else 0
loss[task] = raw_loss \* mask

Never encode missing binary labels as 0.

36. Loss weighting must not let one task dominate
    DR grade, diabetes, hypertension, sex, age, and ocular diseases have different loss scales and
    label densities. If unmanaged, one task can dominate gradients.

Implementation rule:

Use one of:

Kendall uncertainty weighting
GradNorm
manual normalized task weights

The implementation reference sets Kendall uncertainty weighting as the default and defines
initial log sigma values.

Track per-task loss curves separately. If one task’s loss dominates, fix before trusting results.

37. Cross-attention should be optional and ablated
    Paired-eye cross-attention is promising, especially for ODIR, but it should not be hardwired as if
    guaranteed useful. The risk register already notes that paired-eye attention may provide no
    benefit and can be dropped if needed.

Implementation rule:

model:
use_paired_eye_attention: true | false

Run:

single-eye baseline
paired-eye mean pooling
paired-eye cross-attention

Only keep cross-attention if it helps or gives interpretable value.

38. Multi-task learning needs a single-task baseline
    If the multi-task model performs well, reviewers may ask whether it helped or simply added
    complexity.

Implementation rule:

For at least core tasks, run:

single-task head
multi-task head

Report whether multi-task improves, hurts, or changes fairness gaps.

39. Metadata dropout is not a substitute for leakage
    prevention
    The spec includes random metadata field dropout during training. That helps robustness, but it
    does not prevent leakage. If age is sometimes present while predicting age, the model can still
    cheat.

Implementation rule:

Use metadata dropout only after FeaturePolicy blocks forbidden fields.

40. MC-Dropout must be implemented carefully
    The implementation reference warns that BatchNorm must remain in eval mode even when
    Dropout is active for MC-Dropout.

Implementation rule:

Do not simply call:

model.train()

during inference.

Instead:

model.eval()
activate only dropout layers
keep BatchNorm frozen

Or use functional dropout inside a custom MC inference path.

G. Training and reproducibility issues 41. Use the default training hyperparameters first
Do not randomly tune before having a stable baseline. The implementation reference defines
defaults: AdamW, weight decay 0.01, LR 1e-4, cosine schedule, 5 warmup epochs, 100 max
epochs, patience 10, batch size 256, gradient clipping 1.0, and seed sweep [42, 1337, 2024,
7, 99].

Implementation rule:

First baseline must use defaults. Tune only after baseline works.

42. Dataloader worker seeds must be controlled
    The implementation reference warns that torch.manual_seed() alone does not seed
    dataloader workers.

Implementation rule:

Use:

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
worker_init_fn
torch.Generator()
cudnn deterministic settings

Otherwise your runs will not reproduce.

43. Do not change code to change experiments
    The spec says configurations should control datasets, tasks, hyperparameters, backbones, and
    evaluation protocols.

Implementation rule:

Experiments should be launched with config files:

configs/experiment/baseline_odir_dinov2.yaml
configs/experiment/baseline_odir_retfound.yaml
configs/experiment/fairness_groupdro_odir_retfound.yaml

Not by editing Python files.

44. Log everything needed to reproduce a result
    The implementation reference requires logging resolved config, git commit hash, dirty diff,
    environment, dataset version/hash, seeds, and manifests.

Implementation rule:

Every run directory should contain:

resolved_config.yaml
metrics.json
train_log.csv
git_commit.txt
git_diff.patch
pip_freeze.txt
dataset_manifest_hash.txt
embedding_manifest.csv
model_checkpoint.pt

If you cannot reproduce a number, it should not go into the paper.

H. Evaluation and fairness issues 45. Always report overall + subgroup metrics
The paper’s distinctive strength is fairness. The spec says every metric should be reported overall
and stratified by sex, age band, acquisition device, and source population where available.

Implementation rule:

Evaluation output should always include:

overall_metrics.csv
subgroup_metrics_by_sex.csv
subgroup_metrics_by_age_band.csv
subgroup_metrics_by_camera.csv
subgroup_metrics_by_dataset.csv

fairness_gaps.csv

Do not leave fairness as a later notebook.

46. Fairness metrics must handle undefined cases
    If a subgroup has all positives or all negatives, AUC is undefined. The implementation reference
    says to skip and report sample count/positive count rather than returning NaN as if it were a real
    metric.

Implementation rule:

For every metric:

if metric undefined:
value = "NA"
reason = "single_class_or_sparse_subgroup"

Also store:

n
positives
negatives

47. Use confidence intervals, not only point estimates
    AUC 0.82 vs 0.84 means little without uncertainty. The implementation reference defines 1000
    bootstrap iterations, stratified resampling, DeLong for paired AUCs, bootstrap for cross-dataset
    AUC, and Benjamini-Hochberg correction.

Implementation rule:

Main result tables should include:

metric_mean
ci_lower
ci_upper

For subgroup gaps:

gap
gap_ci_lower
gap_ci_upper
p_value_corrected

48. Mitigation must be evaluated with the exact same
    harness
    Group DRO or reweighted loss is only meaningful if evaluated identically to the baseline. The
    spec says baseline subgroup gaps must be reported first, then mitigated models should be re-
    evaluated identically and plotted as performance-vs-fairness trade-offs.

Implementation rule:

Same data split. Same metrics. Same subgroup thresholds. Same bootstrap process.

Only variable:

training method = baseline vs reweighted vs group_dro

49. External validation must be no-retraining evaluation
    Cross-site validation is not “train on ODIR and fine-tune on APTOS.” It must be train on
    primary dataset, evaluate directly on external datasets. The spec says models trained on the
    primary substrate should be evaluated without retraining on external datasets.

Implementation rule:

train: ODIR/BRSET
test_external: IDRiD/APTOS/Messidor/EyePACS
fine_tune_external: false

Fine-tuning belongs only in continual-learning simulation, not external validation.

50. Label schema differences must be documented in
    external validation
    APTOS, IDRiD, Messidor-2, and EyePACS may not use perfectly identical grading conventions.
    The adapter must record mapping scheme and confidence.

Implementation rule:

Every external evaluation table should include:

dataset
native_label_scheme
canonical_mapping
mapping_type: exact/approximate

I. Continual-learning issues 51. Continual learning must be simulation-only, not live
dashboard retraining
The project spec explicitly says the dashboard should not automatically retrain on user uploads
because user-uploaded data may be unlabeled, unstructured, unreliable, and unsafe for online
learning. Continual learning should be demonstrated in controlled simulation using labelled
datasets.

Implementation rule:

Dashboard:

inference only
optional consent-based logging
no parameter updates

Continual-learning experiment:

offline simulation using labelled chunks

52. Keep source-test set frozen across all continual-learning
    updates
    To measure forgetting, you need one original source-domain test set that is never touched. The
    implementation reference says source-test must remain frozen across all updates.

Implementation rule:

source_train = used for initial training
source_val = used for thresholds/tuning
source_test = frozen forever
stream_chunks = used for updates
incoming_test = held-out target-domain test

Never let source-test enter replay buffer.

53. Stream data and target-domain test data must be
    separated
    If you update on IDRiD chunk 1 and then test on the same images, adaptation numbers are
    meaningless.

Implementation rule:

Each external dataset used in continual learning must be split into:

stream_split
test_split

Update only on stream split. Evaluate adaptation only on test split.

54. Replay buffer must be subgroup-balanced
    The spec says the replay buffer should maintain source-distribution samples with subgroup
    balancing. The implementation reference defines a 2000-sample buffer, subgroup minimum
    quota, and 1:2 new-to-replay mixing ratio.

Implementation rule:

Replay buffer must track:

sample_id
patient_id
embedding
labels
sex
age_band
camera_type
dataset_source

It must prevent subgroup collapse:

minimum subgroup quota >= 5% where possible

55. LoRA should target head linear layers first, not the
    backbone
    The implementation reference warns that LoRA should target head linear layers by default, not
    the foundation backbone.

Implementation rule:

Default:

LoRA target = fusion trunk + task heads
backbone = frozen

Only run backbone LoRA as a labeled ablation:

experiment_type: full_lora_ablation

56. OOD gating must happen before continual-learning
    updates
    The spec says new samples above Mahalanobis threshold should be routed to review, not used
    for LoRA updates.

Implementation rule:

Incoming sample path:

image-quality gate
↓
embedding extraction
↓
Mahalanobis OOD score
↓
if low quality: reject from update
if OOD: review bucket

if accepted: eligible for LoRA update

57. Always compare against naive continual learning
    The implementation reference says you must compare against vanilla fine-tuning without replay
    or OOD gating, because that demonstrates catastrophic forgetting.

Implementation rule:

Continual-learning results must include:

naive_finetune_no_replay_no_ood
lora_replay_ood_quality_gated

Without the naive baseline, the continual-learning protocol has no proof of value.

58. Version every model and adapter
    The spec says every adapter checkpoint should have a version identifier, timestamp, config, path,
    and evaluation metrics, and the dashboard should log which model version produced each
    prediction.

Implementation rule:

Create:

registry/model_registry.json

Each entry:

{
"model_version": "retfound_odir_v1_seed42",
"backbone": "retfound",
"tasks": ["dr_grade", "diabetes", "hypertension"],
"checkpoint_path": "...",

"config_path": "...",
"metrics_path": "...",
"created_at": "...",
"parent_version": null
}

J. Explainability and dashboard issues 59. Attention maps are not proof of clinical reasoning
Attention rollout and Grad-CAM are useful visual aids, but they do not prove causality or true
clinical reasoning.

Implementation rule:

In paper/dashboard language:

attention-based explanation

Not:

the model clinically identified the lesion

Keep explanation claims modest.

60. Use correct explanation method per backbone
    ViT-based models use attention rollout. ConvNet baselines use Grad-CAM. The project spec
    explicitly separates image-level explanation into attention rollout for ViTs and Grad-CAM for
    ConvNets.

Implementation rule:

if backbone_type == vit:
use attention_rollout
if backbone_type == convnet:
use gradcam

Do not force Grad-CAM onto ViT embeddings without a justified method.

61. Subgroup reliability lookup must be precomputed, not
    improvised
    The dashboard should say how well the model performed on patients similar to the uploaded
    case. The implementation reference says this requires a precomputed subgroup-conditional
    reliability table with AUC and CI per subgroup cell, and sparse cells should show insufficient
    data.

Implementation rule:

Create:

reliability/{model_version}/subgroup_reliability.csv

Columns:

sex
age_band
device_class
camera_type
task
auc
auc_lower
auc_upper
sample_count
positive_count
status

Dashboard must not invent reliability for sparse groups.

62. Dashboard must soft-decline low-quality/OOD images
    If the model sees a bad input, the correct behavior is not confident prediction. The spec says low-
    quality inputs should receive a soft-decline/follow-up warning, and OOD inputs should get a
    downgraded confidence flag.

Implementation rule:

Dashboard result states:

quality_status = acceptable | low_quality
ood_status = in_distribution | possible_ood
confidence_flag = high | moderate | low_quality | low_ood

Low-quality/OOD cases should not look like normal predictions.

63. Dashboard must not behave like a diagnostic medical
    device
    This is a research dashboard. It should not claim clinical diagnosis.

Implementation rule:

Use language like:

screening probability
research-use output
requires clinical confirmation

Avoid:

diagnosed with diabetes
diagnosed with hypertension

K. Architecture discipline issues 64. Build dummy adapter early and run it repeatedly
The spec says the dummy adapter should return synthetic samples in the canonical schema and
validate the pipeline before real data is loaded.

Implementation rule:

Before ODIR:

DummyAdapter → preprocessing → embedding mock → model → loss →
evaluation

After ODIR:

Run DummyAdapter again periodically.

If downstream code breaks on dummy data, it has become dataset-coupled.

65. Prevent coupling creep aggressively
    The project spec warns that the most common architecture failure is writing ODIR-specific
    columns or label values outside the ODIR adapter. It says that if downstream components need
    dataset quirks, the canonical schema or task registry should be extended instead.

Implementation rule:

Search your code regularly for:

ODIR
BRSET
diagnostic_keywords
left_fundus
right_fundus
Canon
Nikon

Kowa
Zeiss

Outside adapter/config files, most of these should not appear.

66. BRSET/mBRSET integration should require adapter +
    task-registry changes only
    The spec says that when BRSET/mBRSET arrive, the canonical schema, preprocessing,
    embedding extraction, model head, training loop, evaluation harness, continual-learning
    protocol, explainability stack, and dashboard should remain unchanged. Only adapters, task
    definitions, enums, configs, and new embedding extraction runs should change.

Implementation rule:

If adding BRSET forces you to edit train.py, heads.py, or metrics.py, the architecture is
wrong.

67. Use configuration to select experiments, not conditionals
    inside code
    Avoid code like:

if dataset == "odir":
...
elif dataset == "brset":
...

outside adapters.

Implementation rule:

Use config:

dataset:
adapter_class: ODIRAdapter
tasks:
include: [normal, diabetes, glaucoma, cataract, amd, hypertension,
myopia, other]

Then generic code reads config and registry.

L. Model comparison issues 68. Compare backbones under identical downstream
protocol
The foundation-model bake-off is only valid if every backbone uses the same split, same tasks,
same evaluation harness, and equivalent downstream head.

Implementation rule:

For RETFound, DINOv2, ResNet/ConvNeXt:

same patient split
same task registry
same head architecture except input dimension
same training hyperparameters unless justified
same evaluation metrics
same subgroup analysis

Do not tune one backbone heavily and leave another untuned.

69. Include an ImageNet baseline
    A foundation-model paper needs to show whether retinal foundation models actually help. The
    spec includes ConvNeXt-Base or ResNet-50 ImageNet baseline for this reason.

Implementation rule:

Minimum bake-off:

DINOv2-Large
RETFound
ResNet-50 or ConvNeXt-Base

DINORET/RetiZero are valuable but optional if unavailable.

70. Do not let unavailable models delay the project
    The implementation reference says model identifiers must be verified and inaccessible models
    adjusted out of the bake-off.

Implementation rule:

If DINORET/RetiZero weights are not quickly accessible:

mark as unavailable
defer
continue with DINOv2 + RETFound + ImageNet baseline

Do not burn weeks chasing optional models.

M. Statistical and result-reporting issues 71. Do not report only best seed
The implementation reference defines a five-seed sweep for headline experiments.

Implementation rule:

For final headline experiments:

seeds = [42, 1337, 2024, 7, 99]
report mean ± std

Single-seed results are fine for development, not final paper claims.

72. Calibration matters because this is screening
    A model with high AUC but poor calibration can be dangerous in a screening dashboard. The
    implementation reference requires Brier score, ECE, and reliability diagrams with 10 bins.

Implementation rule:

For every binary task:

ROC-AUC
PR-AUC
Brier score
ECE
reliability diagram

Do not rely only on AUC.

73. Report bad cross-site performance honestly
    If external validation drops badly, that is not automatically a failure. In medical AI, showing
    cross-site degradation rigorously can itself be a valuable result. The spec says honest reporting of
    worse external numbers differentiates the paper from benchmark-overfitting work.

Implementation rule:

Do not hide weak external datasets. Report:

primary-test performance
external-test performance
absolute drop
relative drop

possible causes

N. Medical and biological interpretation
issues 74. Retinal age gap is not novel by itself
The spec says retinal age prediction itself is crowded; the novelty is using retinal age gap as part
of a fairness-audited, continually-updatable, deployable pipeline.

Implementation rule:

Do not frame the paper as:

we invented retinal age prediction

Frame it as:

retinal age gap included as one output in a deployable multi-task
screening pipeline

75. Hypertension/diabetes from retinal image must be
    framed as risk/proxy, not diagnosis
    Especially in ODIR-only mode, the model cannot diagnose systemic disease. It predicts dataset
    labels that may correlate with systemic disease.

Implementation rule:

Use terms:

screening probability
proxy systemic label

risk marker
requires clinical confirmation

Avoid:

diagnosis
definitive hypertension detection
whole-body disease detection

76. Explain ocular disease labels carefully
    ODIR labels include broad categories like “Other,” which is clinically heterogeneous. “Other”
    may include many unrelated conditions.

Implementation rule:

For broad labels:

label_type = heterogeneous_category

Do not overinterpret “Other” performance as a specific disease detector.

This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports
