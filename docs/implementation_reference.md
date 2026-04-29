PDF To Markdown Converter
Debug View
Result View
Implementation Reference — Companion to Project Specification
Implementation Reference
Companion to the Project Specification
Concrete specifications filling gaps in the main document. Use alongside the spec when generating code
or making implementation decisions.

1. Execution Discipline
   Three principles override individual implementation choices when they conflict.

1.1 Depth Over Breadth
Five thinly-executed contributions look worse than three rigorously-executed ones. If time pressure
mounts, drop in this order: (1) vision-language extension, (2) CycleGAN device-shift augmentation, (3)
TCAV concept testing, (4) RetiZero in the bake-off, (5) dashboard polish, (6) optional ablations. Never drop:
the foundation-model bake-off, the fairness audit with mitigation, the continual-learning simulation,
cross-population external validation, image-level explainability.

1.2 Pre-Registration of the Evaluation Protocol
Before running any headline experiment, freeze in a config file: the list of subgroups for stratification, the
list of mitigation methods, the list of external datasets and their roles, the list of metrics, the bootstrap
iteration count, the significance thresholds. Do not change the protocol after seeing results. Reviewers
can detect post-hoc cherry-picking and weight integrity highly.

1.3 One Specific Quantitative Claim
Identify, before Phase 2 ends, one number that beats published baselines. Examples: 'our continual-
learning protocol reduces forgetting from X% (naive baseline) to Y%', or 'subgroup AUC gap reduced from
Z (baseline) to W (mitigated) at <2% overall AUC cost'. Reviewers anchor on at least one such number;
without one, the integration story is harder to defend.

2. Cardiovascular Risk Composite — Concrete Definition
   The CV composite is a single regression target derived from available labels in the primary dataset, used
   as a soft proxy for cardiovascular risk. It is not a clinical risk score. It is a defined function of available
   variables, predicted from the image as a whole-body summary.

2.1 Default Formulation (BRSET / mBRSET)
A simplified Framingham-style logistic risk index (range 0-1) computed deterministically from age, sex,
hypertension, diabetes, smoking (where available), and obesity (where available). Specific weights are
pulled from published Framingham or ASCVD coefficient tables and applied without re-fitting. The
composite is then computed for every patient and serves as a regression target.

2.2 Fallback Formulation (ODIR-only)
Without smoking and obesity, the composite reduces to a function of age, sex, predicted diabetes, and
predicted hypertension. In ODIR-only scenarios, the composite is constructed from predicted labels (a
stage-2 derivation) rather than ground-truth labels, and is reported with explicit caveat. In this scenario,
deprioritise the composite as a headline output and frame it as a derived dashboard convenience rather
than a primary target.

2.3 Validation
The composite's external validity is checked by computing it from ground-truth labels on a held-out
cohort, then comparing against the model's image-only prediction of the composite. Report MAE and
Pearson correlation. Do not claim the composite predicts cardiovascular events; explicitly state in the
paper that it is a proxy.

3. Architectural Hyperparameters
   Default values for every numeric choice that the main specification leaves abstract. Override in the config
   file when justified.

Component Parameter Default value
Image branch Input embedding dim (RETFound,
DINOv2-L)
1024
Image branch Input embedding dim (DINOv2-B
baseline)
768
Metadata branch Hidden dim 64
Metadata branch Activation GELU
Metadata branch Dropout (p) on input fields at training 0.
Cross-attention Number of heads 4
Cross-attention Head dim 256 (1024/4)
Cross-attention Layer norm Pre-norm (before attention)
Cross-attention Residual Yes, around the attention block
Fusion trunk Layer 1 Linear → LayerNorm → GELU → Dropout(0.3)
Fusion trunk Layer 1 dim input_dim → 256
Fusion trunk Layer 2 Linear → LayerNorm → GELU → Dropout(0.3)
Fusion trunk Layer 2 dim 256 → 128
Task head (binary) Architecture Linear(128,64) → GELU → Dropout(0.2) →
Linear(64,1)
Task head (multiclass) Architecture Linear(128,64) → GELU → Dropout(0.2) →
Linear(64,C)
Task head (regression) Architecture Linear(128,64) → GELU → Dropout(0.2) →
Linear(64,1)
MC-Dropout Forward passes at inference 30
Ensemble (alt) Number of seeds 5
Loss weighting Method (default) Kendall uncertainty weighting
Loss weighting Initial sigma per task 1.0 (log_sigma = 0.0) 4. Training Hyperparameters
Parameter Default value
Optimiser AdamW
Weight decay 0.
Base learning rate (head training) 1e- 4
LR schedule Cosine annealing with linear warmup
Warmup epochs 5
Max epochs 100
Early stopping metric Validation macro-AUC across binary tasks
Early stopping patience 10 epochs
Batch size (cached embeddings) 256
Gradient clipping (max norm) 1.
Mixed precision fp16 where supported
Random seed (default run) 42
Headline-experiment seed sweep [42, 1337, 2024, 7, 99]
LoRA rank (head adapters) 8
LoRA alpha 16
LoRA dropout 0.
LoRA target modules Linear layers in fusion trunk + task heads
Group DRO step size (eta_q) 0.
Reweighted-loss inverse-frequency cap 10x (clip max weight) 5. Continual Learning Protocol — Concrete Specifications
5.1 Replay Buffer
Buffer size: 2,000 samples (cached embeddings + labels + stratification fields).
Initial population: random sample of source-distribution training set, stratified to match training
subgroup proportions.
Update policy at each chunk arrival: reservoir sampling, but with explicit per-subgroup minimum
quotas to prevent any subgroup falling below 5% of buffer.
Mixing ratio per LoRA update step: 1 part new chunk : 2 parts replay buffer (tunable). Higher
replay ratio = more preservation, less adaptation.
5.2 OOD Threshold Calibration
Compute Mahalanobis distance distribution on validation set during initial training.
Set threshold at the 95th percentile of validation distances. Inputs above threshold are flagged
OOD.
Re-calibrate threshold once per LoRA update cycle using current model embeddings.
Report threshold and flag-rate per chunk in the simulation logs.
5.3 Simulation Stream Definition
Chunk size: 250 images per chunk (calibrated so chunks are statistically meaningful but small
enough that catastrophic forgetting can be observed).
Stream sequence: an explicit list defined in config, e.g., [mBRSET-1, IDRiD-1, mBRSET-2,
Messidor-1, IDRiD-2, mBRSET-3].
Held-out portions: each external dataset is pre-split into 'stream' (used as incoming chunks) and
'test' (frozen, used to measure post-update adaptation).
Source-test set: the original primary-substrate held-out test, kept frozen across all updates to
measure forgetting.
Update frequency: one LoRA update per arriving chunk, single epoch over (chunk + replay
sample). Tune up if learning curves stall.
5.4 Naive Baseline
To prove the protocol's value, run a naive comparison: vanilla fine-tuning of the head on each chunk, no
replay, no OOD gating, full LR. Report identical curves (source-test and incoming-test). The expected
pattern: naive fine-tuning shows steep source-test drop (catastrophic forgetting); the protocol shows
preserved source-test plus rising incoming-test.

6. Image Augmentation Specifications
   6.1 Standard Training Augmentation
   Library: torchvision.transforms.v2 or albumentations.
   Horizontal flip: applied with p=0.5. Note that fundus images have laterality; ensure the laterality
   label is flipped consistently if used.
   Rotation: ±15 degrees, p=0.5.
   ColorJitter: brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0 (avoid hue shifts that distort
   retinal red channel).
   RandomResizedCrop: scale=(0.85, 1.0), ratio=(0.95, 1.05). Conservative because retinal framing
   is informative.
   6.2 Smartphone-Robustness Augmentation (when training for portable-device
   generalisation)
   Applied with p=0.5 (mixed regime so half the batch sees smartphone-style perturbations).
   Motion blur: kernel size 3-7 random.
   Gaussian blur: sigma 0.5-2.0.
   JPEG compression: quality 30-85.
   Brightness perturbation: ±0.25 (heavier than standard).
   Random shadow / illumination gradient (albumentations RandomShadow): p=0.3.
   Coarse dropout (random patches blacked out): max 4 holes, max size 32x32.
   6.3 Validation / Test Time
   No augmentation. Only deterministic preprocessing (crop, resize, CLAHE, normalisation).

7. Statistical Testing Protocol
   Bootstrap iterations for all metric confidence intervals: 1,000.
   Bootstrap method: stratified resampling preserving class balance per fold.
   AUC difference test: DeLong's test for paired AUCs (same patients).
   Cross-dataset AUC differences: bootstrap-based (independent samples).
   Subgroup gap significance: two-sided bootstrap test on max-min difference.
   Multiple comparison correction: Benjamini-Hochberg FDR at q=0.05 across the family of
   subgroup tests within each task.
   Empty / sparse subgroups: report 'NA' for any subgroup with fewer than 30 samples or fewer
   than 5 positives. Do not extrapolate. Document the threshold in the methods section.
   Calibration: Brier score and ECE with 10 reliability bins; reliability diagram plotted for each task.
8. Caching and Manifest Specifications
   8.1 Embedding Cache Layout
   cache/ embeddings/ {backbone}/ # e.g. retfound, dinov2*large
   {dataset}/ # e.g. odir, brset, idrid {sample_id}.pt #
   torch tensor: shape (D,) for CLS only manifest*{backbone}.csv # global manifest
   per backbone

8.2 Manifest CSV Columns
sample_id (str): primary key.
dataset_source (str): 'odir', 'brset', etc.
file_path (str): absolute path to .pt file.
embedding_dim (int): 1024 for RETFound, etc.
backbone_name (str): 'retfound_cfp' etc.
backbone_version (str): release tag or commit hash.
preprocessing_version (str): hash of preprocessing config.
created_at (ISO timestamp).
checksum (str): SHA256 of the tensor bytes for corruption detection.
8.3 Cache Recovery
If a cache file is missing or its checksum mismatches, re-extract that single sample. Provide a verify_cache()
utility that walks the manifest and re-extracts mismatches. Never silently skip; log every recomputation.

9. Model Zoo — Exact Identifiers
   Use these exact strings when downloading. Verify availability at implementation time; if any model has
   been moved or renamed, update this section first.

Model Source Identifier (verify at use time) Variant
RETFound Hugging
Face /
GitHub
rmaphoh/RETFound_MAE (or
YukunZhou/RETFound_MAE_meh)
MAE-
pretrained,
color fundus
RETFound
v
Hugging
Face
Check rmaphoh organisation for newest Use latest
stable
DINOv2-
Large
Hugging
Face
facebook/dinov2-large ViT-L/14,
1024 - dim
DINOv2-
Base
Hugging
Face
facebook/dinov2-base ViT-B/14,
768 - dim (alt
baseline)
DINORET GitHub Search for 'DINORET' release; verify weights public Retinal-
specific
RetiZero GitHub /
Hugging
Face
Verify weights publicly accessible Retinal-
specific
(optional)
ConvNeXt-
Base
torchvision torchvision.models.convnext_base(weights='IMAGENET1K_V1') ImageNet
baseline
ResNet- 50 torchvision torchvision.models.resnet50(weights='IMAGENET1K_V2') Alt ImageNet
baseline
Verification step: before Phase 2 begins, confirm each chosen model's weights are publicly downloadable
and that the loading code from the model card runs end-to-end on a single sample. Note any models
requiring authentication or institutional approval and adjust the bake-off list accordingly.

10. Configuration Schema
    Hydra-style YAML. The top-level config selects datasets, backbone, tasks, and training mode. Sub-configs
    live in their own files.

Top-level: configs/experiment/baseline_odir_retfound.yaml defaults: - dataset: odir -
backbone: retfound - tasks: odir_default - training: standard - evaluation:
full_subgroup seed: 42 output_dir: runs/baseline_odir_retfound wandb_project:
retfound_pipeline # overrides go here

10.1 dataset/_.yaml fields
name (str), adapter_class (str), root_path (str)
split: train_ratio, val_ratio, test_ratio, stratify_on, group_on=patient_id
preprocessing: include_clahe (bool), include_graham (bool), input_size (int)
10.2 backbone/_.yaml fields
name, hf_identifier, embedding_dim, freeze (bool, default true)
input_size (int), normalisation_stats (mean, std lists)
10.3 tasks/_.yaml fields
List of tasks each with: name, type (binary/multiclass/ordinal/regression), target_column,
num_classes (if applicable), loss (bce/ce/mse/ordinal), pos_weight (optional), primary_metric
(auc/macro_auc/mae)
10.4 training/_.yaml fields
Mode: standard / group_dro / reweighted / multi_source / continual
All hyperparameters from section 4.
LoRA fields (rank, alpha, dropout, target_modules) when mode=continual.
10.5 evaluation/\*.yaml fields
subgroup_columns (list, e.g., ['sex', 'age_band', 'camera_type'])
age_bins (list of bin edges)
min_subgroup_size (default 30)
min_positives_for_metric (default 5)
bootstrap_iterations (default 1000)
external_datasets (list of adapter configs to evaluate against) 11. Reproducibility Checklist
Every run logs the resolved config (post-Hydra-merge) to the run directory.
Every run logs the git commit hash and a 'git diff' patch if the working tree is dirty.
Every run logs the environment (pip freeze output).
All RNG seeds set at process start: random, numpy, torch, torch.cuda; cudnn deterministic flag
enabled.
Dataset version pinned in config (hash of the labels.csv used).
Cache manifests are committed to a small metadata-only repository (not the cache contents).
environment.yml (conda) and requirements.txt (pip) both committed.
README in the repo root: how to obtain each dataset, expected directory layout, how to
reproduce headline numbers.
All headline experiments run with the seed sweep [42, 1337, 2024, 7, 99]; report mean and
standard deviation.
Optional: Dockerfile for full reproducibility. 12. Subgroup-Conditional Reliability Lookup
The dashboard surfaces, at inference time, 'how well has the model performed on patients like you?' This
requires a precomputed table.

12.1 Construction
Computed once per trained model on the held-out validation set.
Subgroup definition: cross product of (sex × age_band × device_class × camera_type) where
each dimension is configurable.
For each cell with sample size ≥ 30 and ≥5 positives per task, compute per-task AUC with 95%
bootstrap CI.
For cells below threshold, store 'insufficient data' flag. Dashboard surfaces this honestly.
12.2 Storage Format
CSV (or parquet) keyed by subgroup tuple, with columns per task for AUC, AUC_lower, AUC_upper,
sample_count, positive_count. Loaded into memory at dashboard startup.

12.3 Inference-Time Lookup
The dashboard reads user-supplied metadata (sex, age, device class) and looks up the matching cell. If
found, surface the AUC + CI alongside the prediction. If the cell is too sparse or absent, surface 'No
reliability data for your specific subgroup; using overall validation AUC as fallback.'

13. Published Baselines to Match or Beat
    Concrete numbers from prior work, useful as targets and as anchor points for the paper's headline claim.
    Update with current numbers from arXiv at writeup time.

Source Task Reported metric Notes
Nakayama et al. 2024 (BRSET
paper)
DR binary on BRSET AUC ~0.95 (ConvNeXt V2) Single-task baseline
Nakayama et al. 2024 Diabetes on BRSET AUC ~0.87 Single-task baseline
Khan et al. 2025 (BRSET
multi-task)
Hypertension on BRSET AUC ~0.79 (best of 6
backbones)
Sex-stratified results
provided
RetBench (OMIA 2025) Avg systemic AUC external RetiZero 0.92, RETFound 0.88 Across multiple datasets
Ninomiya et al. 2026 (Comm
Med)
Retinal age MAE 2.78 years Multi-task with HbA1c
integration
FairMedFM (NeurIPS 2024) Fairness gap on BRSET Reported per backbone Linear probe baselines,
beatable with end-to-end
heads
RETFound Plus (npj DM 2026) 5 - year systemic risk c-
index
+4-10% over RETFound Time-aware extension
The realistic target for at least one headline number: match RetBench's foundation-model comparison
ranking under fairness-stratified protocol (showing fairness gaps not visible in RetBench's aggregate

numbers), OR demonstrate that the LoRA continual-learning protocol holds source-test AUC within 2%
while improving the worst-subgroup AUC by 5%+.

14. ODIR-Only Pivot Narratives
    If both BRSET and mBRSET are unavailable, the paper's framing must be reset, not just toned down. Pick
    ONE of the following narratives and rewrite the abstract / introduction around it. Do not try to do both.

14.1 Pivot A — Multi-Condition Ocular Screening
Headline: 'A reproducible, fair, continually-updatable foundation-model pipeline for multi-condition
retinal screening, validated across populations.' Drop systemic claims from the headline. Cover ODIR's
eight conditions (diabetes, glaucoma, cataract, AMD, hypertension, pathological myopia, normal, other)
as multi-label classification with paired-eye cross-attention as a real architectural contribution. Cross-
population external validation on IDRiD/Messidor-2/APTOS/EyePACS for DR. Same fairness, continual-
learning, dashboard, and explainability components as before. This is the more interesting paper of the
two pivots.

14.2 Pivot B — Foundation-Model Robustness Benchmark
Headline: 'A fairness- and robustness-aware benchmark for retinal foundation models, with deployment-
ready continual-learning protocol.' Centre the bake-off and cross-site evaluation as the contribution. Train
on ODIR, exhaustively evaluate on every external dataset, quantify performance degradation by
population, device, image quality, and subgroup, with the continual-learning protocol as the proposed
mitigation. More conservative, more incremental, but solidly publishable.

14.3 What to Drop in Either Pivot
Systemic prediction as a primary claim.
Cardiovascular composite as a primary output (keep as dashboard convenience).
Smartphone-domain transfer (no companion smartphone dataset available).
Within-dataset device natural experiment claims (ODIR's cameras are confounded with site).
Socioeconomic stratification.
14.4 What Becomes Stronger in Pivot A
Paired-eye cross-attention is a more central architectural contribution because ODIR is fully
bilateral.
Multi-label across eight categories is a richer ocular classification surface.
Chinese cohort fairness analysis fills a real literature gap. 15. Known Gotchas
Patient-level splitting: do NOT split at image level. Same patient appearing in train and test
inflates AUC by 5-15 percentage points and is the most common reproducibility failure in retinal
AI.
DR grade harmonisation: BRSET's ICDR and SDRG must be unified into one canonical 0-4 ICDR
scale at the adapter level. Document the mapping. APTOS uses ICDR; Messidor-2 uses
adjudicated grades that map cleanly; IDRiD uses ICDR.
Eye laterality consistency: when applying horizontal flip augmentation, do not flip the laterality
label. Either keep the laterality coupled to the post-flip image, or do not flip at all.
RETFound preprocessing: RETFound was pretrained with specific normalisation constants. Use
the constants from the official model card, not generic ImageNet stats. Verify before extraction.
DINOv2 input size: DINOv2 expects 224x224 for the standard checkpoint. Mismatched sizes
silently degrade performance.
MC-Dropout and BatchNorm: BatchNorm layers must be in eval() mode at inference even when
Dropout is in train() mode. Use torch.nn.functional.dropout or a custom inference path.
LoRA target modules: target the head Linear layers, NOT the foundation backbone (foundation
stays fully frozen by default). Only target the backbone if explicitly running a 'full LoRA' ablation.
Fairness metric edge cases: when a subgroup has 100% one class, AUC is undefined. Skip and
report sample-size-and-positive-count rather than NaN.
Dataset source as a feature: when included as a one-hot input during multi-source training, the
model can game the source signal (predict source, ignore image). Mitigate with gradient-
reversal layer or by zero-masking the source one-hot at inference.
Cardiovascular composite leakage: if the composite is constructed from labels that the model
also predicts, training the model to predict the composite leaks information. Either compute the
composite ONLY from variables the model does not predict, or compute it from ground-truth
labels and use it as a separate held-out target.
OOD detection on cached embeddings: Mahalanobis must be computed on the SAME
embedding space as inference. If the backbone changes, the OOD threshold must be
recomputed.
Random seed scope: a single torch.manual_seed() does not seed the dataloader workers. Use
generator + worker_init_fn.
Smartphone augmentation strength: too aggressive and the model's clinical-camera
performance drops. Validate that smartphone-augmented training preserves clinical-camera
test AUC within acceptable bounds.
End of implementation reference. Use alongside the main project specification.

This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports
