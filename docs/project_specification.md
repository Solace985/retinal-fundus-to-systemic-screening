Retinal Foundation Model Pipeline — Project Specification
Project Specification
Retinal Foundation Model Pipeline for Multi-Condition Screening,
Fairness Auditing, and Deployable Continual Learning
NTCC In-House Research Programme — Amity Institute of Biotechnology, Amity University
Comprehensive design, decision log, and implementation plan
Version: Initial consolidated draft
Table of Contents
Table of Contents
Table of Contents
Executive Summary
Project Motivation
Problem Statement and Research Questions
Novelty Positioning Against the Literature
4.1 Foundation-Model Comparison on Retinal Tasks
4.2 Systemic Prediction from Retinal Images........................................................................................
4.3 BRSET-Specific Published Work
4.4 Retinal Age Gap as a Biomarker
4.5 Fairness in Medical Imaging Foundation Models
4.6 Continual Learning with LoRA in Medical Imaging
4.7 Deployment Dashboards
4.8 Synthesis
Decisions Made and Rationale
5.1 Scope of Systemic Prediction
5.2 Frozen Foundation Models with Lightweight Heads
5.3 Foundation-Model Bake-off Rather Than Single Backbone
5.4 Multi-Task Joint Learning
5.5 Fairness as First-Class Output
5.6 Continual Learning Through Simulation, Not Live Deployment
5.7 BRSET as First-Choice Primary, ODIR-5K as Fallback
5.8 Build Dataset-Agnostic Pipeline From Day One.............................................................................
5.9 Deployment Dashboard as Required Artefact...............................................................................
5.10 Image-Quality Gate Before Inference
5.11 Out-of-Distribution Gating With Mahalanobis Distance
5.12 Cross-Site External Validation as Required Reporting
5.13 Explainability at Three Levels
5.14 Calibrated Uncertainty With MC-Dropout or Light Ensembles
Constraints, Risks, and Workarounds
6.1 Dataset Access
6.2 Compute
6.3 Public-Data Label Ceiling
6.4 Label Noise on ODIR-5K
6.5 Within-Dataset Device Confounding
6.6 User-Uploaded Data is Unlabelled and Unstructured
6.7 Smartphone Image Quality Heterogeneity
6.8 Foundation Model Memory and Throughput
6.9 Patient Identity and Train-Test Leakage
6.10 Cross-Dataset Label Schema Heterogeneity
Datasets
7.1 Primary-Substrate Candidates
7.1.1 BRSET v1.0.1
7.1.2 mBRSET v1.0
7.2 Fallback Primary...........................................................................................................................
7.2.1 ODIR-5K
7.3 External Validation Datasets
7.4 Dataset Role Matrix
7 .5 ODIR-Only Scenario Analysis
7.5.1 What Survives Unchanged
7.5.2 What Weakens
7.5.3 What Becomes Stronger........................................................................................................
7.5.4 Realistic Publication Tier
Prerequisite Knowledge
8.1 Conceptual Background
8.2 Domain Knowledge
8.3 Practical Skills
8.4 Tools Worth Learning Before Implementation Starts
System Architecture
9.1 Architectural Principles
9.2 Layer 1: Dataset Adapter..............................................................................................................
9.2.1 Responsibility
9.2.2 Interface
9.2.3 Concrete Adapters
9.2.4 Dataset-Coupling Status
9.3 Layer 2: Canonical Sample Schema...............................................................................................
9.3.1 Responsibility
9.3.2 Schema Definition
9.3.3 Conventions
9.3.4 Dataset-Coupling Status
9.4 Layer 3: Preprocessing Pipeline
9.4.1 Responsibility
9.4.2 Operations
9.4.3 Quality Filtering.....................................................................................................................
9.4.4 Dataset-Coupling Status
9.5 Layer 4: Embedding Extraction
9.5.1 Responsibility
9.5.2 Backbones
9.5.3 Caching Strategy
9.5.4 Dataset-Coupling Status
9.6 Layer 5: Multi-Task Head..............................................................................................................
9.6.1 Responsibility
9.6.2 Architecture Specification
9.6.3 Dataset-Coupling Status
9.7 Layer 6: Training Loop
9.7.1 Responsibility
9.7.2 Specifications
9.7.3 Multi-Source Training
9.7.4 Dataset-Coupling Status
9.8 Layer 7: Evaluation and Fairness Harness
9.8.1 Responsibility
9.8.2 Per-Task Metrics
9.8.3 Subgroup Stratification
9.8.4 Fairness Metrics
9.8.5 Cross-Population Transfer Evaluation
9.8.6 Mitigation Ablation
9.8.7 Dataset-Coupling Status
9.9 Layer 8: Continual Learning Protocol
9.9.1 Responsibility
9.9.2 LoRA Adapter Architecture
9.9.3 Replay Buffer
9.9.4 Out-of-Distribution Detection
9.9.5 Image-Quality Gate
9.9.6 Simulation Protocol
9.9.7 Versioned Model Registry
9.9.8 Baseline Comparison
9.9.9 Dataset-Coupling Status
9.10 Layer 9: Explainability Stack
9.10.1 Responsibility
9.10.2 Image-Level Explanation......................................................................................................
9.10.3 Metadata-Level Attribution
9.10.4 Concept-Based Testing (optional)
9.10.5 Subgroup-Stratified Attention Analysis
9.10.6 Dataset-Coupling Status
9.11 Layer 10: Deployment Dashboard
9.11.1 Responsibility
9.11.2 Tech Stack
9.11.3 User Flow
9.11.4 Versioning
9.11.5 Dataset-Coupling Status
Implementation Plan
10.1 Phase 1: Foundation Infrastructure (Days 1-7)............................................................................
10.1.1 Goals
10.1.2 Tasks
10.1.3 Deliverables
10.2 Phase 2: Core Pipeline (Days 8-21)
10.2.1 Goals
10.2.2 Tasks
10.2.3 Deliverables
10.3 Phase 3: Advanced Features (Days 22-35)
10.3.1 Goals
10.3.2 Tasks
10.3.3 Deliverables
10.4 Phase 4: Integration and Paper Preparation (Days 36-49)
10.4.1 Goals
10.4.2 If BRSET / mBRSET Access Has Arrived
10.4.3 If BRSET / mBRSET Access Has Not Arrived
10.4.4 Paper Preparation
10.4.5 Deliverables
ODIR-to-BRSET Modification Path
11.1 What Changes
11.2 What Does Not Change
11.3 Time Budget for Modification
11.4 Risk of Coupling Creep
Publication Strategy
12.1 Target Venues by Scenario
12.2 Title Candidates
12.3 Submission Checklist
Risk Register
Open Questions and Items Requiring Decision
Glossary

1. Executive Summary
   This document specifies a research project that builds a deployable, fair, continually-updatable retinal
   screening pipeline using publicly released foundation models. A single colour fundus photograph is fed to
   a pretrained retinal foundation model, whose frozen embedding is consumed by a small multi-task head
   that simultaneously predicts ocular conditions, demographic and biological-age signals, and a constructed
   cardiovascular risk composite. The same architecture is wrapped in a fairness auditing harness that
   stratifies every metric by sex, age band, acquisition device, and source population, and a continual-
   learning protocol that uses LoRA adapters with subgroup-balanced replay and out-of-distribution
   detection to allow the deployed model to be safely updated over time. A web dashboard surfaces
   predictions with attention-based explanations and per-subgroup reliability annotations.

The research contribution is the integration. No single component is unprecedented; their combination
on reproducible public data, with the fairness and continual-learning scaffolding as first-class outputs
rather than discussion-section asides, is genuinely under-served in the published literature. The work
targets a venue between npj Digital Medicine and Medical Image Analysis depending on dataset
availability.

The dataset strategy is layered. The first-choice primary substrate is BRSET and its smartphone companion
mBRSET, both hosted on PhysioNet under credentialed access. The fallback primary is ODIR-5K, which is
open access and supports most of the planned contributions at a moderately reduced strength. External
validation uses IDRiD, Messidor-2, APTOS 2019, and EyePACS for diabetic retinopathy generalisation
across populations. Implementation begins immediately on ODIR-5K with the pipeline structured for
adapter-level dataset swappability, so that BRSET and mBRSET can be plugged in with minimal refactoring
if and when access is granted.

2. Project Motivation
   A colour fundus photograph is one of the cheapest, fastest, least invasive medical images that exists. It is
   acquired in seconds by a technician, requires no contrast agent or pharmacological preparation, and
   increasingly can be captured on smartphone-attached portable cameras. The retina is the only anatomical
   site where the living microvasculature can be observed directly without surgery, which means it carries
   information about systemic vascular and neural processes that affect the heart, kidneys, brain, and
   metabolic systems.

Most published retinal artificial intelligence work has historically focused on detecting eye-specific
diseases, especially diabetic retinopathy and glaucoma. That problem is, for benchmark purposes, largely
solved, with multiple papers reporting AUCs above 0.95 on standard datasets. The frontier has shifted in
two directions. First, foundation models pretrained on millions of unlabeled retinal images, beginning with
RETFound from Moorfields Eye Hospital published in Nature in 2023, have demonstrated that a single
pretrained backbone encodes information sufficient to predict cardiovascular events, kidney function, and

neurological conditions. Second, the field has begun to acknowledge that benchmark performance on a
single dataset is not a sufficient measure of clinical readiness, and that fairness across demographic and
acquisition-device subgroups, robustness to distribution shift, and the ability to update a deployed model
over time without losing previous performance are the practical gating factors for real-world adoption.

The clinical motivation for this project is the same one driving the broader retinal-foundation-model field.
In low- and middle-income contexts, including parts of India that the researcher will eventually deploy
work in, screening for cardiovascular disease, diabetes, and hypertension is constrained by the availability
of laboratory testing, blood-pressure measurement, and specialist examination. A fundus camera,
including a portable or smartphone-attached one, is dramatically cheaper to deploy and operate than the
alternative pathway. If a single inexpensive photograph can be reliably converted into a multi-organ risk
profile with appropriate uncertainty quantification, the implications for population-level screening are
substantial.

The methodological motivation for this project is the gap between published benchmark results and
deployable systems. The work attempts to close that gap on reproducible public data, so that the pipeline,
evaluation protocol, and deployment artefacts are something other researchers can replicate, criticise,
and extend without privileged access to restricted hospital datasets.

3. Problem Statement and Research Questions
   The project investigates the following research questions, in order of centrality.

Can publicly released retinal foundation models, used with frozen weights, support a single
multi-task pipeline that simultaneously predicts diabetic retinopathy grade, additional ocular
conditions, demographic variables, biological retinal age, and a constructed cardiovascular risk
composite, on reproducible public data?
Across multiple foundation model backbones evaluated under identical downstream protocol,
which encode the most useful signal for these tasks, and how do they differ when evaluation is
stratified across sex, age band, acquisition device, and source population?
Does the model's predictive performance survive realistic distribution shifts, specifically transfer
across populations (Brazilian, Chinese, Indian, French, US cohorts) and across imaging devices
(clinical fundus camera, multi-vendor heterogeneity, and where data permits, smartphone
capture)?
Can a deployed model be updated safely over time using a LoRA-adapter-based protocol with
subgroup-balanced replay and out-of-distribution gating, such that performance on the original
training distribution is preserved while underrepresented subgroups improve?
Can the model's predictions be made interpretable to a clinical reader at inference time, both
through visual attention explanations and through subgroup-conditional reliability annotations
that quantify how well the model has historically performed on patients similar to the one being
screened?
The first question establishes feasibility of multi-task retinal screening on public data. The second is the
foundation-model comparison contribution. The third is the cross-distribution robustness contribution.
The fourth is the continual-learning contribution. The fifth is the explainability and trust contribution. Each
is addressed through a specific component of the pipeline described in the architecture section.

4. Novelty Positioning Against the Literature
   This section summarises the literature review conducted during project planning and identifies, for each
   component of the work, what has already been published and where genuine gaps remain. The novelty
   claim is grounded in the integration of components, not in any single component being unprecedented.

4.1 Foundation-Model Comparison on Retinal Tasks
RETFound from Moorfields Eye Hospital, published in Nature in 2023, established the retinal foundation
model paradigm. RetiZero, DINORET, VisionFM, and EyeFM have followed. RetBench, published as an
OMIA workshop paper at MICCAI 2025, conducted a head-to-head comparison of these models on
ophthalmic tasks including some systemic prediction. The npj Digital Medicine paper introducing
RETFound Plus (January 2026) extended the model to time-aware risk prediction. The Global RETFound
initiative announced at ARVO 2025 indicates ongoing scale-up.

The gap that remains: none of these works combines foundation-model comparison with simultaneous
fairness stratification across demographics and acquisition devices, on reproducible public data. RetBench
compares models but does not stratify within each model. FairMedFM stratifies fairness but uses general-
purpose foundation models on shallow protocols. The combination is genuinely open.

4.2 Systemic Prediction from Retinal Images........................................................................................
Strong systemic prediction results from RETFound-family models exist for heart failure, myocardial
infarction, ischaemic stroke, Parkinson's disease, and Alzheimer's disease, but are based almost
exclusively on restricted cohorts: MEH-AlzEye, UK Biobank, or private hospital systems. HyMNet
(Bioengineering 2024) used RETFound and demographics for hypertension prediction on a private Saudi
cohort. White et al. (Diabetes, Obesity and Metabolism 2024) prospectively validated a UK-Biobank-
trained algorithm for blood pressure, HbA1c, and eGFR estimation in Kenya, using restricted training data.

The gap that remains: a fully reproducible public-data systemic-prediction pipeline that anyone in the
world can replicate. Without privileged access to UK Biobank or hospital cohorts, the public-data version
of this thesis has not been executed at meaningful scale.

4.3 BRSET-Specific Published Work
The BRSET dataset paper itself (Nakayama et al., PLOS Digital Health 2024) presented baseline ConvNeXt
V2 results on diabetic retinopathy, diabetes, and sex prediction. Khan et al. (MDPI Bioengineering 2025)

ran a seven-task benchmark using six conventional CNN and ViT architectures, with gradient-based
saliency maps. FairMedFM (NeurIPS 2024) included BRSET as one of seventeen medical datasets in a
fairness benchmark, but used only general-purpose foundation models with linear probing.

The gap that remains: BRSET has not been worked on with retinal-specific foundation models, with
rigorous camera-stratified evaluation, with the continual-learning protocol described here, or with
deployment-oriented dashboard wrapping. Each of these is a real opportunity if BRSET access is granted.

4.4 Retinal Age Gap as a Biomarker
Two recent and very relevant works: Ninomiya et al. (Communications Medicine, April 2026) presented
an ensemble multi-task model integrating fundus images and HbA1c during training, achieving retinal age
MAE of 2.78 years, and demonstrating that the retinal age gap associates with diabetes, cardiac disease,
and stroke. A separate npj Digital Medicine cross-population paper from June 2025 used UK Biobank plus
Chinese cohorts and used BRSET for external validation, showing that 56 of 159 disease groups had altered
retinal age gap distributions.

The gap that remains: retinal age prediction itself is no longer novel. The novelty available is using the
retinal age gap as one of several simultaneously predicted outputs in a fairness-audited, continually-
updatable, deployable pipeline. The work of building it into a practical screening artefact has not been
done.

4.5 Fairness in Medical Imaging Foundation Models
FairMedFM (NeurIPS 2024) is the current reference benchmark, evaluating eleven foundation models
across seventeen datasets with AUC gaps, equalised odds, and expected calibration error gaps. Mehta et
al. (MLMIR 2024) studied uncertainty fairness on brain imaging. Raumanns et al. 2024 examined how
single-task vs multi-task learning affects fairness on dermoscopic data.

The gap that remains: almost no fairness work focuses specifically on retinal foundation models, and none
combines retinal-specific foundation models, systemic and ocular endpoints, camera and population
stratification, and mitigation experiments under one roof.

4.6 Continual Learning with LoRA in Medical Imaging
LoRA-based continual learning is an active 2024-2025 research area in general machine learning, with FM-
LoRA (CVPR 2025), SD-LoRA (ICLR 2025), InfLoRA (CVPR 2024), Progressive LoRA (ACL 2025), and
Contrastive Regularization over LoRA (ACM MM 2025). For medical imaging, ViT-LoRA (EMBC 2025),
CELoRA (OMIA 2024 for ophthalmic foundation models), and Decentralized LoRA Augmented Transformer
(arXiv 2025) for federated retinal imaging exist. MIRAGE (npj Digital Medicine September 2025) lists LoRA
as future work for OCT foundation models.

The gap that remains: a LoRA-based safe-update protocol for retinal foundation models specifically,
including subgroup-balanced replay buffer and Mahalanobis-distance OOD detection, demonstrated in
simulation on public datasets, has not been published.

4.7 Deployment Dashboards
The deployment literature in retinal AI exists but is dominated by FDA-cleared point-of-care diabetic
retinopathy systems (IDx-DR, EyeArt) that do not handle systemic prediction. ReVision (December 2025)
is an efficiency-oriented foundation model not released as a system end users can interact with. No
published work known to the project team presents an interactive multi-task deployment dashboard with
subgroup-conditional reliability annotations surfaced at inference time.

The gap that remains: a working interactive dashboard with the described feature set is genuinely novel
as a deployment artefact, even if any individual component (inference UI, attention overlay, OOD scoring)
has been done in isolation.

4.8 Synthesis
Five of the seven analysed components have real, current gaps in the published literature as of April 2026.
Two (foundation-model comparison and retinal age prediction) are crowded but can still contribute under
the integration framing. The combination of all seven into one coherent paper, executed on public data
with an emphasis on reproducibility and deployability, has not been published. This is the novelty claim.

5. Decisions Made and Rationale
   This section logs the substantive decisions taken during planning, with the reasoning behind each. They
   are ordered by their structural importance to the project.

5.1 Scope of Systemic Prediction
Decision: predict the realistic public-data systemic panel rather than overclaim whole-body capability.
Rationale: the most ambitious systemic endpoints (MACE events, eGFR, HbA1c continuous, Alzheimer's
risk, cognitive decline, anaemia from haemoglobin) require longitudinal outcomes or paired laboratory
values that exist only in restricted datasets, principally UK Biobank and a handful of private hospital
cohorts. UK Biobank has been explicitly ruled out for this project. The realistic public-data systemic panel
comprises: diabetes binary, hypertension binary (with the limitation that public datasets label this binary
rather than offering continuous blood pressure), insulin use and diabetes duration where available, retinal
age regression and retinal age gap as a derived biological biomarker, a cardiovascular risk composite
constructed from age, sex, and predicted comorbidities, and where mBRSET is available, a broader panel
including smoking, obesity, and microvascular complication history.

5.2 Frozen Foundation Models with Lightweight Heads
Decision: use foundation models with frozen weights and train only lightweight task heads, with optional
LoRA adapters for the continual-learning protocol. Rationale: full fine-tuning of a ViT-Large backbone
requires substantial labelled data and risks catastrophic forgetting of the priors learned during the
foundation model's self-supervised pretraining on millions of images. Frozen extraction preserves the
foundation model's generalisation, makes the entire downstream pipeline reproducible on a free Colab
GPU, allows embeddings to be cached once and reused across all downstream experiments, and matches
current best practice in retinal foundation model research.

5.3 Foundation-Model Bake-off Rather Than Single Backbone
Decision: evaluate at least three retinal-specific or general-purpose foundation models (RETFound,
DINORET, DINOv2-Large) plus an ImageNet-pretrained baseline (ConvNeXt-Base or ResNet-50), under
identical downstream protocol. Add RetiZero if weights are accessible. Rationale: RetBench has shown
that the choice of foundation model materially affects downstream performance, and that the retinal-
specific advantage is not uniform across tasks. A single-backbone paper would be rejected as redundant.
The bake-off framing allows fairness audits to be layered on top, producing a contribution that is
independently novel even if the systemic prediction story were weaker.

5.4 Multi-Task Joint Learning
Decision: train a single multi-task model that predicts all targets jointly through a shared trunk and task-
specific heads, rather than independent per-task models. Rationale: the targets are biologically correlated
(diabetes and hypertension are comorbid, hypertensive retinopathy is downstream of hypertension,
retinal age modulates everything else). Joint training forces the shared representation to respect this
structure and lets gradients from label-rich tasks regularise label-sparse tasks. It also produces, at
inference time, a single forward pass that returns the full screening panel, which is the deployment-
relevant configuration.

5.5 Fairness as First-Class Output
Decision: every metric reported in the paper is reported overall and stratified across sensitive subgroups
(sex, age band, acquisition device where available, source dataset). Mitigation experiments using Group
DRO and reweighted loss are run as a planned ablation, not a discussion-section gesture. Rationale: this
is the single most under-served axis in retinal AI methodology and the place where the paper can make a
genuinely distinctive contribution. Most existing fairness benchmarks (FairMedFM included) report
aggregate fairness metrics on shallow models. A fairness-stratified, retinal-foundation-model evaluation
with mitigation trade-off curves is a publishable contribution on its own.

5.6 Continual Learning Through Simulation, Not Live Deployment
Decision: the continual-learning protocol is demonstrated in controlled simulation using existing labelled
public datasets fed sequentially as a simulated incoming stream, rather than through a live deployed
system that automatically retrains on user uploads. The dashboard logs uploaded images for potential
future research but does not perform online parameter updates. Rationale: naive online learning from
user uploads is broken in medical AI for well-understood reasons (label uncertainty, distribution gaming,
feedback loops, audit trail loss, regulatory non-compliance). The simulation framing produces a
defensible, reproducible research contribution about the protocol. The live continual learning vision can
be deferred to a future deployment study with clinical-partner labellers, mentioned in the discussion.

5.7 BRSET as First-Choice Primary, ODIR-5K as Fallback
Decision: pursue PhysioNet credentialing for BRSET and mBRSET as the first-choice primary substrate,
with ODIR-5K as the open-access fallback if credentialing is delayed or denied. Rationale: BRSET uniquely
provides paired systemic, demographic, and device metadata in a single public dataset, with a natural
Canon-vs-Nikon two-camera experiment for device invariance, plus the mBRSET smartphone companion
for clinical-camera-to-portable-camera transfer. ODIR-5K is open access, large enough (5,000 paired-eye
patients), multi-camera, and supports most planned contributions at moderately reduced strength. The
BRSET-specific contributions that ODIR cannot fully replace are the structured systemic labels, the clean
within-population device experiment, and the clinical-to-smartphone transfer story.

5.8 Build Dataset-Agnostic Pipeline From Day One.............................................................................
Decision: begin implementation immediately on ODIR-5K with the pipeline architected so that all dataset-
specific code lives behind a single adapter interface, allowing BRSET and mBRSET to be plugged in by
writing additional adapter classes if and when access arrives. Rationale: waiting for credentialing is the
worst option because it sacrifices weeks of build time without making the future code more flexible.
Building dataset-agnostic infrastructure is the same work whether the first dataset is ODIR or BRSET. The
discipline of routing all dataset-specific logic through the adapter is the protection, not the choice of which
dataset comes first.

5.9 Deployment Dashboard as Required Artefact...............................................................................
Decision: build a working web dashboard with attention-based explainability, image-quality gating, OOD
flagging, version-tagged predictions, and subgroup-conditional reliability annotations. Use Streamlit or
Gradio for the prototype. Rationale: the dashboard converts the work from a benchmark study into a
deployment artefact reviewers can interact with, which is highly unusual in retinal AI papers and directly
demonstrates the practical relevance of the fairness and continual-learning contributions.

5.10 Image-Quality Gate Before Inference
Decision: implement a small image-quality classifier as a front-end gate that scores every input on focus,
illumination, image field, and artefacts before main-model inference. Train on quality labels available in
BRSET or proxy-derive on ODIR. Rationale: smartphone images and real-world clinical inputs vary
dramatically in quality. Routing low-quality inputs to a soft-decline branch with clinical-follow-up flagging
avoids both false reassurance and the model attempting predictions on inputs it has no business
processing. This is also a defensible regulatory-readiness gesture.

5.11 Out-of-Distribution Gating With Mahalanobis Distance
Decision: at inference, compute the Mahalanobis distance of the foundation-model embedding to the
training-distribution embedding cluster. Flag inputs above a calibrated threshold as out-of-distribution.
Use the same flag to gate inclusion of new inputs into the continual-learning replay buffer. Rationale: this
is a well-validated lightweight OOD detection method that operates entirely in embedding space, requires
no extra training, and produces a continuous score that is interpretable to clinicians (closer to training
distribution = more reliable prediction).

5.12 Cross-Site External Validation as Required Reporting
Decision: every model trained on the primary substrate is evaluated, without retraining, on at least three
external datasets covering distinct populations: IDRiD (Indian), Messidor-2 (French), APTOS 2019 (Indian,
second cohort), and EyePACS (US heterogeneous). For DR grade where labels exist universally; for
systemic targets only where labels exist in the external dataset. Rationale: this is a hygiene requirement
for any deployment-oriented paper and the most common failure mode of medical AI is cross-site
degradation. Reporting external numbers honestly, including where they are worse, is what differentiates
the paper from a benchmark-overfitting submission.

5.13 Explainability at Three Levels
Decision: provide attention rollout or Grad-CAM heatmaps at the image level, SHAP or integrated
gradients on the metadata branch, and where time permits, a TCAV-style concept activation analysis
against clinically defined retinal concepts (arteriolar narrowing, vessel tortuosity, optic disc cupping).
Surface the image-level explanation in the dashboard. Rationale: clinician trust is the binding constraint
on adoption. Attention overlays demonstrate where the model looked, metadata attribution separates
image-driven from metadata-driven predictions, and concept testing moves the explanation from where
to which clinically meaningful features. Doing all three is unusual; doing it in concert with fairness
reporting is more so.

5.14 Calibrated Uncertainty With MC-Dropout or Light Ensembles
Decision: every prediction at inference time comes with a confidence interval produced via either Monte
Carlo Dropout or a small five-seed ensemble of the multi-task head. Rationale: clinical decisions made on

point estimates without uncertainty are unsafe. The Bayesian or ensemble interval lets the dashboard
surface a high/moderate/low confidence flag, which directly shapes how clinicians use the prediction.

6. Constraints, Risks, and Workarounds
   6.1 Dataset Access
   Constraint: BRSET and mBRSET sit behind PhysioNet credentialed access requiring CITI 'Data or Specimens
   Only Research' training (already completed) and review of a research-purpose statement (submitted).
   Approval typically takes one to four weeks. Workaround: ODIR-5K, IDRiD, Messidor-2, APTOS 2019,
   EyePACS, RFMiD, and AIROGS are all open access and can be used immediately. The pipeline is being built
   dataset-agnostically so that BRSET and mBRSET can be slotted in without refactoring.

6.2 Compute
Constraint: training large foundation models from scratch is infeasible on student-budget hardware.
Workaround: the entire pipeline is structured around frozen-backbone embedding extraction.
Embeddings are computed once per backbone per image and cached to disk. All downstream training
operates on cached feature vectors and runs on a free Colab T4 or modest local GPU in minutes per task.

6.3 Public-Data Label Ceiling
Constraint: the most ambitious systemic endpoints (MACE events, eGFR, HbA1c continuous, true
cardiovascular outcomes) require longitudinal outcomes or laboratory values that exist only in restricted
datasets. UK Biobank is unavailable. Workaround: scope the systemic-prediction claim to what public data
realistically supports — diabetes, hypertension, retinal age gap, constructed cardiovascular composite,
and where mBRSET arrives, smoking, obesity, and microvascular history. Frame the paper's whole-body
claim honestly at this scope rather than overclaiming.

6.4 Label Noise on ODIR-5K
Constraint: ODIR-5K's diabetes and hypertension labels are derived from clinical text keywords rather than
structured comorbidity records, making them noisier than BRSET's labels. Workaround: report label-noise
analysis transparently, perform sensitivity tests, and frame the systemic-prediction component on ODIR
as a demonstration rather than a definitive claim. If BRSET arrives, repeat the experiment with cleaner
labels for direct comparison.

6.5 Within-Dataset Device Confounding
Constraint: BRSET's Canon and Nikon images come from the same population and institutions, supporting
a clean device-invariance natural experiment. ODIR-5K's three cameras (Canon, Zeiss, Kowa) are

confounded with hospital site. Workaround: with BRSET, exploit the natural experiment directly. Without
BRSET, report camera effects as confounded with site, not as a clean invariance test, and pivot the
deployment robustness story toward cross-population transfer (which ODIR supports cleanly).

6.6 User-Uploaded Data is Unlabelled and Unstructured
Constraint: a public dashboard that accepts uploads cannot assume user-supplied diagnostic labels are
present, accurate, or well-formatted. Naively training on user uploads is unsafe. Workaround: the live
dashboard is inference-only with optional structured metadata entry. The continual-learning protocol is
demonstrated in simulation on existing labelled public datasets, not on user uploads. The dashboard logs
uploaded images with consent for future offline research, but does not retrain. A clinical-partnership
labelling pathway is mentioned as future work, framed honestly.

6.7 Smartphone Image Quality Heterogeneity
Constraint: smartphone-acquired fundus images vary widely in quality due to motion, illumination, off-
axis capture, and lower-resolution sensors. Workaround: image-quality classifier as a front-end gate;
aggressive smartphone-style augmentation during training (JPEG compression, motion blur, illumination
perturbation, colour shift); device-class-conditional uncertainty calibration so that smartphone-input
predictions come with appropriately wider confidence intervals; if mBRSET is available, separate LoRA
adapters per device class so the model can route to the appropriate adapter at inference.

6.8 Foundation Model Memory and Throughput
Constraint: ViT-Large foundation models (RETFound is approximately 300M parameters) require sufficient
GPU memory for inference and produce 1024-dimensional embeddings per image, accumulating to non-
trivial disk space at dataset scale. Workaround: extract embeddings in batches with mixed-precision
inference; cache as compressed PyTorch tensors; pre-compute once per backbone per dataset; store only
the CLS token plus optionally a small number of pooled spatial tokens for explainability.

6.9 Patient Identity and Train-Test Leakage
Constraint: BRSET and ODIR contain multiple images per patient (left eye and right eye, sometimes
multiple visits). A naive image-level split risks the same patient appearing in train and test, inflating
performance. Workaround: all train/validation/test splits are at patient level. The split utility takes a
patient identifier column and ensures no patient appears in more than one fold.

6.10 Cross-Dataset Label Schema Heterogeneity
Constraint: different datasets use different DR grading systems (ICDR vs Scottish DRG vs simplified five-
class), different label vocabularies, and different conventions for missingness. Workaround: explicit label
harmonisation through a canonical task registry. The adapter layer maps each dataset's native labels into

the canonical schema. Where harmonisation is approximate (e.g., binary diabetic retinopathy across
systems), this is documented and reported as a known limitation.

7. Datasets
   This section consolidates every dataset relevant to the project, the role each plays, and the strategic logic
   of the multi-dataset configuration. The datasets are presented in three groups: primary-substrate
   candidates, external-validation set, and supporting / scale datasets.

7.1 Primary-Substrate Candidates
7.1.1 BRSET v1.0.1
Source: PhysioNet credentialed access. Size: 16,266 images from 8,524 Brazilian patients. Images are
paired left and right eye where available. Cameras: Canon CR2 (10,592 images) and Nikon NF5050 (5,
images), captured at the same institutions on the same population. Labels include diabetic retinopathy
grade in both ICDR and Scottish DRG systems, hypertensive retinopathy, age-related macular
degeneration, drusen, increased optic cup-to-disc ratio, vascular abnormalities, structural anatomical
labels (macula, optic disc, vessels), quality control labels (focus, illumination, image field, artefacts), and
demographic and clinical metadata (age, sex, nationality, diabetes binary, hypertension binary, insulin
use, diabetes duration). Access requires CITI Data or Specimens Only Research training and PhysioNet
Credentialed Health Data Use Agreement.

Strategic role: first-choice primary substrate. The Canon/Nikon two-camera natural experiment, the
structured systemic labels, and the demographic richness are not replicated by any other public dataset.

7.1.2 mBRSET v1.0
Source: PhysioNet credentialed access (same DUA as BRSET). Size: 5,164 smartphone fundus images from
1,291 patients in Itabuna, Bahia, Brazil. Single device: Phelcom Eyer (smartphone-mounted handheld
camera). Labels include systemic hypertension, vascular disease, acute myocardial infarction history,
nephropathy, neuropathy, diabetic foot, smoking, alcohol consumption, obesity, insulin use and timing,
oral diabetes treatment, diabetes duration, education level, insurance status, ethnicity (mixed European,
African, and Native American Brazilian ancestry), and standard ocular labels (DR grade by ICDR, macular
edema, image quality).

Strategic role: smartphone-domain companion to BRSET, but also independently strong. Its systemic and
lifestyle label panel is in some respects richer than BRSET's. If both are available, mBRSET serves as both
the smartphone-transfer test set and a secondary training source. If only mBRSET is available, it can stand
as the primary substrate with a smartphone-first framing.

7.2 Fallback Primary...........................................................................................................................
7.2.1 ODIR-5K
Source: Grand Challenge / Kaggle, open access (registration). Size: approximately 5,000 patients, 10,
paired-eye images. Eight diagnostic categories: Normal, Diabetes, Glaucoma, Cataract, AMD,
Hypertension, Pathological Myopia, Other. Labels are derived from clinical diagnostic text keywords by
expert readers under quality control. Cameras: Canon, Zeiss, Kowa (multi-vendor, multi-hospital across
China). Demographic metadata: age and sex.

Strategic role: open-access fallback primary. Supports foundation-model bake-off, multi-label paired-eye
learning (cross-attention is naturally well-supported because the dataset is bilateral by design), fairness
audits across sex and age, and Chinese-cohort cross-population validation. Limitations: text-derived
systemic labels are noisier than BRSET's structured records; cameras are confounded with hospital site;
no smartphone companion; no socioeconomic variables.

7.3 External Validation Datasets
Dataset Access Size Population Labels Role
IDRiD Open (IEEE
DataPort)
516 images Indian DR grade, DME grade,
lesion segmentation
South Asian DR
external
validation
Messidor- 2 Open
(request)
1,
images
French Adjudicated DR grade European DR
external
validation
APTOS 2019 Open (Kaggle) 3,
images
Indian (Aravind
Eye Hosp.)
DR grade 0- 4 Second Indian
DR cohort
EyePACS
(Kaggle)
Open (Kaggle) ~88,
images
US
heterogeneous
DR grade Large-scale DR
validation,
scale
experiments
RFMiD 1 + 2 Open
(Zenodo)
3,200 + 860
images
Indian 45 - 49 multi-label ocular Multi-label
ocular breadth,
hypertension
as one class
AIROGS Open
(registration)
~113,
images
Multi-country Glaucoma binary Optional
scale/SSL
experiments
These are not training datasets in the primary-substrate sense. Each serves as a held-out test cohort
probing a specific generalisation axis: population (Indian, French, US, Chinese as ODIR), device class, or
label-distribution shift.

7.4 Dataset Role Matrix
The following matrix specifies what each dataset is used for at each stage of the pipeline. Roles marked
with \* are conditional on dataset access being granted.

Stage BRSET* mBRSET* ODIR-5K IDRiD Messido
r- 2
APTO
S
EyePACS RFMi
D
Primary
training
Yes (1st
choice)
Co-
primary or
secondary
Yes
(fallback)
No No No No No
Fairness
stratificati
on
Yes (rich) Yes (rich) Yes
(sex/age)
No No No No No
Camera
natural
experimen
t
Yes
(Canon/Niko
n)
No (single
device)
Partial
(confounded
)
No No No No Partia
l
Smartpho
ne domain
transfer
Train side Smartpho
ne test
side
No No No No No No
Cross-
population
external
External
target
External
target
Chinese
cohort
Indian
cohort
French
cohort
Indian
cohor
t 2
US cohort Indian
cohor
t 3
Continual-
learning
stream
Source/strea
m
Stream Source/strea
m
Stream Stream Strea
m
Stream/sca
le
Strea
m
Retinal age
training
Yes Yes Yes (limited) If age
availabl
e
If age
available
No No No
7 .5 ODIR-Only Scenario Analysis
If neither BRSET nor mBRSET access is granted within the project timeline, the pipeline transitions to ODIR-
5K as primary. This section specifies what changes.

7.5.1 What Survives Unchanged
Foundation-model bake-off (RETFound, DINORET, DINOv2, optionally RetiZero, plus an
ImageNet baseline).
Multi-label paired-eye prediction with cross-attention between left and right eye embeddings
(in fact strengthened by ODIR's bilateral-by-design structure).
Fairness audit across sex and age.
Cross-population external validation across IDRiD, APTOS, Messidor-2, EyePACS for diabetic
retinopathy.
Continual-learning simulation (the simulation feeds external datasets sequentially, regardless of
which is primary).
Deployment dashboard with explainability, image-quality gating, OOD flagging, version-tagged
predictions.
Calibrated uncertainty via MC-Dropout or light ensembling.
Retinal age regression and retinal age gap as a derived biomarker.
7.5.2 What Weakens
Systemic prediction shifts from structured-label to text-derived-label, which must be reported
transparently. Diabetes and hypertension predictions on ODIR are weaker evidence than they
would be on BRSET.
Within-dataset device-invariance natural experiment is replaced by a confounded multi-camera
multi-hospital analysis. The contribution survives but at lower rigour.
Smartphone-domain transfer disappears entirely as a contribution unless an open-access
portable-camera dataset can be located.
Socioeconomic stratification (education, insurance) is not available.
Microvascular complication and cardiovascular history labels (nephropathy, neuropathy, MI,
vascular disease, smoking, obesity) are not available, narrowing the systemic panel substantially.
7.5.3 What Becomes Stronger........................................................................................................
Paired-eye cross-attention becomes a more central architectural contribution because every
patient in ODIR has both eyes labelled, whereas BRSET pairing is partial.
The Chinese cohort fills a real gap in retinal AI fairness literature, which is dominated by UK, US,
and Brazilian cohorts.
ODIR's eight-class multi-label space is broader than BRSET's primary disease set.
7.5.4 Realistic Publication Tier
With BRSET plus mBRSET: target Nature Communications Medicine, npj Digital Medicine, Lancet Digital
Health, or top MICCAI / Medical Image Analysis. With mBRSET only: similar tier, framing shifted to
smartphone-first systemic screening. With BRSET only: Nature Communications Medicine / npj Digital
Medicine tier. With ODIR-5K only: MICCAI main track, Medical Image Analysis, IEEE Journal of Biomedical
and Health Informatics, Computers in Biology and Medicine. The work is publishable in every scenario;
the venue tier scales with dataset richness.

8. Prerequisite Knowledge
   This section lists the conceptual and practical knowledge required to execute the project. It is provided so
   that learning gaps can be identified and addressed early.

8.1 Conceptual Background
Convolutional neural networks and Vision Transformers at the level of forward pass, attention
mechanism, patch embedding, and CLS token.
Self-supervised learning paradigms (masked image modelling as in MAE / RETFound, contrastive
learning as in DINO / DINOv2) at conceptual level. Implementation is not required because
foundation model weights are pretrained.
Transfer learning concepts: frozen backbone, linear probing, full fine-tuning, parameter-efficient
fine-tuning (LoRA).
Multi-task learning fundamentals: shared trunk, task heads, loss weighting, gradient
interference.
Standard classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC) and regression
metrics (MAE, RMSE, R-squared, MAPE).
Fairness metrics: demographic parity, equalised odds, AUC gap, expected calibration error gap.
Group DRO and reweighted loss as mitigation strategies.
Uncertainty quantification: MC-Dropout, deep ensembles, prediction intervals, calibration plots.
Explainability methods: Grad-CAM, attention rollout, integrated gradients, SHAP, TCAV.
8.2 Domain Knowledge
Retinal anatomy: optic disc, macula, fovea, retinal vasculature (arterioles vs venules), nerve fibre
layer.
Diabetic retinopathy grading: ICDR (none / mild / moderate / severe / proliferative) and the
Scottish DRG variants.
Hypertensive retinopathy: arteriolar narrowing, A/V nicking, cotton-wool spots, optic disc
swelling.
Age-related macular degeneration: drusen, geographic atrophy, choroidal neovascularisation.
Glaucoma: cup-to-disc ratio, retinal nerve fibre layer thinning.
Conceptual link from retinal microvasculature to systemic conditions (cardiovascular risk, kidney
function).
8.3 Practical Skills
Python programming, NumPy, pandas, basic data wrangling.
PyTorch: dataset and dataloader patterns, model definition with nn.Module, training loop,
optimisers, schedulers, mixed-precision training, checkpointing.
Hugging Face Hub for downloading pretrained foundation models (RETFound, DINOv2).
scikit-learn: train/test splits with stratification and group-aware splitting, baseline classifiers.
Matplotlib / seaborn for plotting; plotly for interactive dashboard plots.
Streamlit or Gradio for the dashboard.
Git and GitHub for version control.
Basic Docker for reproducibility (optional but desirable).
8.4 Tools Worth Learning Before Implementation Starts
Hugging Face PEFT library for LoRA adapters.
Weights & Biases (or MLflow) for experiment tracking.
Captum or SHAP for explainability.
Fairlearn for fairness metrics and Group DRO baselines. 9. System Architecture
The architecture is described as ten layers, ordered from data input to user-facing inference. Each layer
has a single well-defined responsibility, a fixed interface to the next layer, and a clear dataset-coupling
status (dataset-aware, partially dataset-aware, or dataset-agnostic). This separation is the central
structural decision that allows the project to begin on ODIR-5K immediately while preserving the option
to integrate BRSET and mBRSET later without refactoring.

9.1 Architectural Principles
Single source of dataset-specific logic. All knowledge of how a particular dataset is structured on
disk, how its labels are encoded, and what its column names are, lives in exactly one place: the
dataset adapter for that dataset. No other code in the system references dataset-specific
column names, file paths, or label vocabularies.
Canonical sample schema. Every adapter outputs samples in a fixed, well-typed schema. Every
downstream component consumes this schema. Missing fields are explicitly None or NaN, never
absent.
Task registry. Tasks (diabetes, hypertension, DR grade, retinal age, sex, etc.) are first-class
registered objects with declared name, type (binary, multiclass, ordinal, regression), target
column in the canonical schema, and metadata. The training loop and evaluation harness read
from the registry; no hardcoded task lists exist anywhere else.
Caching by content hash. Embeddings, preprocessed images, and split assignments are cached
on disk keyed by the hash of their inputs and the version of the producing code. Re-runs do not
recompute unchanged artefacts.
Configuration through declarative files, not code edits. Hyperparameters, dataset paths,
backbone choices, task selection, and evaluation protocols live in YAML or Hydra configurations.
Code is dataset- and task-agnostic; configurations select what gets used in a given run.
Reproducibility by default. Every random number generator is seeded; every run logs the
configuration, code commit hash, and environment. Embeddings are cached so identical
configurations produce identical outputs across runs.
9.2 Layer 1: Dataset Adapter..............................................................................................................
9.2.1 Responsibility
The dataset adapter is the only component in the system that knows what a particular dataset looks like
on disk. It reads the raw label files, image directories, and any auxiliary metadata, and exposes a uniform
interface to the rest of the pipeline.

9.2.2 Interface
Each adapter is a class implementing the following methods. Method names are illustrative; concrete
signatures will be settled at implementation time.

list_samples(): returns a list of unique sample identifiers (typically image_id or patient_id_eye).
load_sample(sample_id): returns a dict in the canonical sample schema with the image path or
tensor and all available label and metadata fields filled in (None for missing).
get_supported_tasks(): returns the subset of registered tasks that this adapter can supply labels
for, with mapping from task name to canonical-schema column.
get_demographic_columns(): returns the list of demographic and acquisition columns available
for fairness stratification (e.g., sex, age, camera, hospital_site, ethnicity_proxy if present).
get_quality_columns(): returns the list of image-quality columns available.
get_patient_id(sample_id): returns the patient-level identifier for group-aware splitting.
9.2.3 Concrete Adapters
ODIRAdapter: parses the diagnostic keyword strings into the eight-class multi-label vector; reads
age and sex from the demographic columns; flags multi-camera origin where extractable from
filename or metadata.
BRSETAdapter (deferred until access granted): reads the labels.csv, maps DR_ICDR and
DR_SDRG into a unified DR grade column, exposes camera_type as Canon/Nikon binary, exposes
diabetes / hypertension / hypertensive_retinopathy / insulin_use / diabetes_duration as
separate task columns.
mBRSETAdapter (deferred until access granted): reads the broader systemic label set, exposes
single-device-class flag, includes the socioeconomic columns (educational_level, insurance) as
fairness stratifiers.
IDRiDAdapter, MessidorAdapter, APTOSAdapter, EyePACSAdapter, RFMiDAdapter: each reads
its native label file, exposes only the tasks it supports (typically just DR grade), and returns
sex/age as None where unavailable.
DummyAdapter: returns synthetic samples in the canonical schema for end-to-end pipeline
testing without real data. Used to validate the pipeline works before any real dataset is loaded.
9.2.4 Dataset-Coupling Status
Fully dataset-aware. This is the only layer where dataset-specific code lives. Every other layer is dataset-
agnostic by construction.

9.3 Layer 2: Canonical Sample Schema...............................................................................................
9.3.1 Responsibility
The canonical schema is the data contract between the adapter layer and everything else. It is defined
once, version-controlled, and treated as a stable interface.

9.3.2 Schema Definition
A sample is a dictionary (or pydantic / dataclass instance) with the following fields:

sample_id (str, required)
patient_id (str, required, used for group-aware splits)
dataset_source (str enum, required, e.g., 'odir', 'brset', 'mbrset', 'idrid')
image_path (str, required, absolute path to the source file)
eye_laterality (str enum or None: 'left', 'right', or None)
age_years (float or None)
sex (str enum or None: 'male', 'female', 'other', or None)
ethnicity (str or None, dataset-defined values)
camera_type (str or None, dataset-defined values)
hospital_site (str or None, where exposed)
device_class (str enum or None: 'clinical_camera', 'smartphone')
Task labels — one field per registered task, value is the label or None: dr_grade, diabetes,
hypertension, hypertensive_retinopathy, drusen, glaucoma, cataract, amd,
pathological_myopia, smoking, obesity, insulin_use, diabetes_duration_years, etc.
Image-quality labels: focus_quality, illumination_quality, image_field_quality, artefact_presence
(str enums or None)
socioeconomic fields where available: education_level, insurance_status (str or None)
9.3.3 Conventions
Missing values are explicitly None for objects and NaN for numeric, never absent. Code
downstream uses isna() / is None checks consistently.
Categorical fields use a fixed string vocabulary recorded in a central enum module. Adapters
that need to map dataset-specific values to canonical values do so within their own code.
DR grade is harmonised to a 0-4 ICDR scale at the adapter level; the adapter records the source
grading system in a derived 'dr_grade_source_scheme' field for auditability.
9.3.4 Dataset-Coupling Status
Dataset-agnostic by definition. The schema is the same regardless of which datasets feed it.

9.4 Layer 3: Preprocessing Pipeline
9.4.1 Responsibility
Convert raw fundus images of varied resolution, illumination, and quality into a standardised tensor
representation suitable for foundation-model embedding extraction.

9.4.2 Operations
Retinal circle detection and cropping: detect the circular field of view, crop to the inscribed
square, mask the corners. This removes the variable amount of black border that different
cameras produce and standardises framing.
Resize to backbone-required input size: typically 224 by 224 or 512 by 512 pixels, depending on
the foundation model variant.
CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the green channel for vessel-
contrast enhancement, with parameters tuned against held-out validation.
Optional Graham preprocessing (Gaussian-blurred subtraction) as in the classic Kaggle Diabetic
Retinopathy approach. Configurable on/off.
Per-channel normalisation using ImageNet statistics (or RETFound-specific statistics if
documented in the foundation model release).
For training (only), data augmentation: horizontal flip (with care, since fundus images have
laterality), small rotations, brightness and contrast jitter, optional ColorJitter, optional CutOut.
For training where smartphone robustness is sought, an additional augmentation regime
simulating smartphone artefacts: stronger JPEG compression, motion blur, illumination
perturbation, simulated colour shift.
9.4.3 Quality Filtering.....................................................................................................................
A small image-quality classifier is trained on whatever quality labels are available (BRSET ships explicit
ones; on ODIR these can be proxy-derived from artefact metadata or sharpness statistics). At inference,
low-quality images are flagged. During training, low-quality images can be optionally excluded from the
gradient signal but kept in the test set with appropriate uncertainty inflation.

9.4.4 Dataset-Coupling Status
Dataset-agnostic. The same preprocessing pipeline applies to every adapter's output. Adapter-specific
quirks (e.g., a particular dataset that needs unusual cropping) are handled inside the adapter's
load_sample, not in this layer.

9.5 Layer 4: Embedding Extraction
9.5.1 Responsibility
Convert preprocessed images into fixed-dimensional embeddings using each foundation-model
backbone, and cache these embeddings to disk so all downstream training operates on cached tensors
rather than reprocessing raw images.

9.5.2 Backbones
RETFound (color fundus variant): ViT-Large, MAE-pretrained on ~1.6M unlabeled retinal images
at Moorfields. Output: 1024-dim CLS token plus optional pooled patch tokens. Weights from the
official Hugging Face release.
DINORET: retinal-specific DINO-pretrained model. Output dimension and access: per the
published release; expected ViT-Base or Large scale.
DINOv2-Large: general-purpose Meta DINOv2, used as a strong general-purpose baseline to test
whether retinal-specific pretraining is necessary. 1024-dim output.
RetiZero (optional): if weights are released and accessible, included in the bake-off. Per
RetBench it is a strong systemic-prediction backbone.
ConvNeXt-Base or ResNet-50 (ImageNet-pretrained): non-foundation baseline to quantify the
foundation-model advantage.
9.5.3 Caching Strategy
Embeddings stored as PyTorch tensors (.pt) in a directory keyed by backbone name and
preprocessing config hash.
File naming: {backbone}/{dataset_source}/{sample_id}.pt
A manifest CSV per backbone records every cached embedding with sample_id, dataset_source,
file_path, embedding_dim, backbone_version, preprocessing_version.
Mixed-precision (fp16) inference for memory efficiency; embeddings stored as fp32 to avoid
downstream precision issues.
9.5.4 Dataset-Coupling Status
Dataset-agnostic. The extractor receives preprocessed tensors and produces embeddings; it does not
know or care which dataset they came from.

9.6 Layer 5: Multi-Task Head..............................................................................................................
9.6.1 Responsibility
The trainable model. Takes cached foundation-model embeddings plus optional metadata as input and
produces a vector of task-specific predictions through a shared trunk and per-task heads.

9.6.2 Architecture Specification
Image Embedding Branch

Input: cached CLS token from the chosen foundation-model backbone (1024-dim for RETFound and
DINOv2-Large; backbone-dependent for others). No further transformation; passed directly to the fusion
stage.

Metadata Branch

Input: a tabular feature vector built from age (continuous, normalised), sex (one-hot or learned 2-dim
embedding), camera type (one-hot over the union vocabulary across datasets, with all-zeros encoding
'unknown'), dataset source (one-hot, used during multi-source training to allow the model to learn
dataset-specific corrections), eye laterality (one-hot left/right/unknown). The vector is passed through a
2 - layer MLP with hidden size 64 and GELU activation, producing a 64-dim metadata embedding. Metadata
fields are dropped at random with probability 0.3 during training to ensure the model can degrade
gracefully when metadata is unavailable at inference.

Paired-Eye Cross-Attention (where applicable)

When both left and right eye embeddings are available for the same patient (ODIR is fully bilateral by
design; BRSET partially), they are processed through a single cross-attention block. Left-eye embedding is
treated as the query, right-eye as the key and value (and vice versa for the mirror direction); the two
outputs are averaged. Output dimension matches input. When only one eye is available, the cross-
attention reduces to identity. This component is optional and toggled by configuration.

Fusion

Concatenate the (cross-attended where applicable) image embedding and the metadata embedding into
a single vector. Pass through a shared trunk: Linear(input_dim, 256), LayerNorm, GELU, Dropout(0.3),
Linear(256, 128), LayerNorm, GELU, Dropout(0.3). The 128-dim trunk output feeds all task heads.

Task Heads

One small head per registered task, each consuming the 128-dim trunk output:

Binary classification heads (diabetes, hypertension, hypertensive retinopathy, drusen, sex when
target, glaucoma, cataract, etc.): Linear(128, 64), GELU, Dropout(0.2), Linear(64, 1), sigmoid at
inference.
Multiclass / ordinal heads (DR grade): Linear(128, 64), GELU, Dropout(0.2), Linear(64,
num_classes), softmax at inference. For ordinal targets, additional ordinal-loss formulation (e.g.,
CORN or cumulative-link) is configurable.
Regression heads (retinal age, diabetes_duration_years, derived cardiovascular composite
score): Linear(128, 64), GELU, Dropout(0.2), Linear(64, 1).
Task Masking

Each batch sample carries a per-task mask indicating whether that task's label is observed for that sample
(since different datasets supply different label subsets). The total loss is the sum of per-task losses, each
weighted by the mask sum (so unobserved labels contribute zero gradient). This is the mechanism that
allows multi-dataset training where each dataset supplies only some tasks.

Loss Weighting

The total loss is a weighted sum of per-task losses. Weights are learned per Kendall et al.'s uncertainty-
based weighting: each task has a learned log-variance parameter, and the weighted loss is L_total = sum
over tasks of (1/(2*sigma_t^2) * L_t + log(sigma_t)). Alternative: GradNorm, where weights are adjusted
to balance task gradient magnitudes. Choice between the two is configurable.

Uncertainty

Dropout layers at the trunk and task-head level are kept active at inference for MC-Dropout uncertainty
estimation: 30 stochastic forward passes produce a distribution over predictions, from which a 95%
credible interval is derived. Alternative: train a small ensemble (5 seeds) of the head-only network;
ensemble variance produces the interval. MC-Dropout is the default for compute reasons; the ensemble
is run as an ablation.

9.6.3 Dataset-Coupling Status
Dataset-agnostic. The architecture and forward pass do not know which dataset any given sample came
from. Dataset-source one-hot is passed as a feature, but the architecture itself is generic.

9.7 Layer 6: Training Loop
9.7.1 Responsibility
Train the multi-task head end to end on the cached embeddings, with proper validation, checkpointing,
and experiment tracking.

9.7.2 Specifications
Optimiser: AdamW with weight decay 0.01.
Learning rate: 1e-4 with cosine annealing schedule, 5-epoch warmup.
Batch size: 256 (cached embeddings are small, so memory is not the binding constraint; gradient
noise tuning is).
Epochs: up to 100 with early stopping on validation macro-AUC across tasks; patience 10.
Mixed precision (fp16) training where supported.
Gradient clipping at 1.0.
Random seed fixed and logged. Five-seed runs for the headline experiments to report mean and
standard deviation.
Train/validation/test split: 70/15/15 at patient level. Stratification on the most label-dense task
(typically DR grade).
Experiment tracking: Weights & Biases or MLflow logs every training run with full config, code
commit hash, and metric curves.
9.7.3 Multi-Source Training
When multiple primary datasets are available (e.g., BRSET plus mBRSET if both are credentialed), a multi-
source mode mixes batches from each dataset proportional to size, with task masking handling the
differing label availability. Adapter-output samples carry their dataset_source field, and a small auxiliary
classifier head on the trunk predicts source as a side task to encourage source-invariant features (gradient-
reversal layer optional, configurable).

9.7.4 Dataset-Coupling Status
Dataset-agnostic. The training loop iterates over a generic data loader produced by the harness; dataset
specifics are upstream.

9.8 Layer 7: Evaluation and Fairness Harness
9.8.1 Responsibility
Compute every metric the paper will report, both overall and stratified across sensitive subgroups. This is
the single most important layer for the paper's contribution and is treated with proportionate care.

9.8.2 Per-Task Metrics
Binary tasks: ROC-AUC, PR-AUC, accuracy, precision, recall, F1, balanced accuracy, expected
calibration error, Brier score.
Ordinal / multiclass tasks (DR grade): macro-AUC, macro-F1, weighted F1, quadratic Cohen's
kappa, accuracy.
Regression tasks (retinal age): MAE, RMSE, R-squared, Pearson correlation. Retinal age gap =
predicted_age - chronological_age is computed as a derived field and reported separately.
Bootstrap 95% confidence intervals on all metrics (1000 resamples).
9.8.3 Subgroup Stratification
Every metric is recomputed within every cell of the cross product of available stratification dimensions:

Sex (male / female / other / unknown)
Age band (e.g., <40, 40-55, 55-70, 70+)
Camera or device class (where available)
Dataset source (when evaluating across multiple datasets)
Hospital site (where exposed by the adapter)
Education / insurance (only on mBRSET if available)
9.8.4 Fairness Metrics
AUC gap: max - min AUC across subgroups for each binary task.
Demographic parity difference: max - min positive prediction rate across subgroups.
Equalised odds gap: max - min TPR and FPR across subgroups.
Calibration gap: max - min ECE across subgroups.
Statistical significance testing on subgroup gaps: bootstrap-based or DeLong's test for AUC
differences.
9.8.5 Cross-Population Transfer Evaluation
Models trained on the primary substrate are evaluated on each external dataset without retraining. Per-
dataset overall metrics and per-subgroup metrics are reported. Performance degradation from primary-
test to external-test is quantified and discussed per task. Where labels are absent in an external dataset
(e.g., systemic labels in IDRiD), only the supported tasks are evaluated.

9.8.6 Mitigation Ablation
After reporting the baseline subgroup gaps, the model is retrained with subgroup-robust objectives:

Group DRO: minimax over subgroup-conditional losses.
Reweighted loss: per-sample weights inversely proportional to subgroup size.
Subgroup-balanced replay buffer (overlapping with the continual-learning protocol).
Each mitigated model is re-evaluated identically and the trade-off curve (overall performance vs subgroup
gap) is plotted. The plot is the central evidence for the fairness contribution.

9.8.7 Dataset-Coupling Status
Dataset-agnostic. The harness reads available stratification columns from the canonical schema and
produces metrics for whatever subgroups exist in the data. New datasets contribute their stratification
dimensions automatically.

9.9 Layer 8: Continual Learning Protocol
9.9.1 Responsibility
Demonstrate, in simulation, that the deployed model can be updated over time as new data arrives,
without losing performance on previously seen distributions and with targeted improvement on
underrepresented subgroups.

9.9.2 LoRA Adapter Architecture
Low-Rank Adaptation modules are inserted into the trainable head (and optionally into the foundation
backbone if compute allows). Each LoRA module replaces a target linear layer's weight update with W' =
W + (B \* A) where A is rank x in_features and B is out_features x rank, with rank typically 8 or 16. Only A
and B are trained; the original W stays frozen. Multiple adapters can be stored on disk and loaded
selectively at inference.

9.9.3 Replay Buffer
A buffer of source-distribution samples (specifically, their cached embeddings plus labels and stratification
fields) maintained per subgroup with explicit balancing. When updating on new incoming data, the buffer
is sampled with deliberate oversampling of underrepresented subgroups, so the update simultaneously
preserves previous performance and improves underserved groups.

9.9.4 Out-of-Distribution Detection
Mahalanobis distance from each new sample's embedding to the source-distribution embedding cluster.
The cluster's mean and covariance are computed once on the training set. New samples with distance
above a calibrated threshold are flagged as OOD and routed to a 'review bucket' rather than fed into the
LoRA update. Threshold is calibrated on a held-out validation set to achieve, say, 95% true-distribution
acceptance.

9.9.5 Image-Quality Gate
Independent of OOD detection, every incoming sample is scored by the image-quality classifier from Layer

Low-quality samples are excluded from the update regardless of their OOD score.
9.9.6 Simulation Protocol
Concrete experimental design: train the initial model on the primary substrate (BRSET train split, or ODIR
train split). The simulated incoming stream is a temporally-ordered sequence of held-out chunks from
external datasets (e.g., mBRSET chunk 1, IDRiD chunk 1, mBRSET chunk 2, Messidor-2 chunk, IDRiD chunk
2, ... where each chunk is, say, 100-500 images). After each chunk arrives, image-quality gating and OOD
gating are applied; surviving samples enter the LoRA update with a replay-buffer mix. After every update,
the model is re-evaluated on (a) the original primary-test set (to measure forgetting) and (b) a held-out
portion of the incoming domain (to measure adaptation). Curves of these two metrics over update steps
form the central evidence.

9.9.7 Versioned Model Registry
Every adapter checkpoint is given a version identifier and timestamp, and stored in a registry mapping
version -> file path -> training configuration -> evaluation metrics. The dashboard logs which model
version produced each prediction. Rolling back to any prior version is a single config edit.

9.9.8 Baseline Comparison
To prove the protocol works, we compare against a naive continual-learning baseline (vanilla fine-tuning
on each chunk without replay or OOD gating) and show the catastrophic forgetting it produces. The Group
DRO and reweighted-loss mitigation experiments from Layer 7 also serve as comparison points for the
fairness-improvement claim.

9.9.9 Dataset-Coupling Status
Dataset-agnostic. The simulation protocol consumes whatever sequence of cached embeddings and
labels the harness supplies; specifics of which datasets they came from live only in the configuration file.

9.10 Layer 9: Explainability Stack
9.10.1 Responsibility
Generate, for every prediction, interpretable visual and feature-level explanations that a clinician can
inspect. Surface these explanations in the dashboard.

9.10.2 Image-Level Explanation......................................................................................................
Attention rollout for ViT-based foundation models. Aggregates attention across layers to
produce a heatmap over the input image showing which patches the model attended to.
Grad-CAM for the ConvNet baseline. Class-activation map produced from the final convolutional
feature map.
Both are rendered as semitransparent overlays on the input fundus image in the dashboard.
9.10.3 Metadata-Level Attribution
Integrated gradients or SHAP applied to the metadata branch. Quantifies how much each
metadata feature (age, sex, camera, etc.) contributed to a given prediction.
Reported alongside image-level attention so the clinician can see whether the prediction was
driven by the image, the metadata, or both.
9.10.4 Concept-Based Testing (optional)
TCAV (Testing with Concept Activation Vectors) against clinically defined retinal concepts: arteriolar
narrowing, vessel tortuosity, optic disc cupping, microaneurysms, exudates, drusen. Concept directions
are derived from labelled examples (from RFMiD or hand-curated subsets). For each prediction, the cosine
similarity between the model's representation and each concept direction quantifies how much the

prediction relies on that clinically meaningful feature. This is an optional component that can be deferred
if time is tight.

9.10.5 Subgroup-Stratified Attention Analysis
As a fairness probe, attention maps are aggregated within subgroups (e.g., 'Canon images, female, 50-65')
and compared across subgroups. Systematic differences in where the model looks across demographics
or devices are flagged as potential bias signals. This is reported in the paper but does not surface in the
dashboard.

9.10.6 Dataset-Coupling Status
Dataset-agnostic. Operates on the trained model and a given input.

9.11 Layer 10: Deployment Dashboard
9.11.1 Responsibility
A web interface that loads the trained, version-tagged model and provides a working inference experience
for clinicians, researchers, and reviewers. This is both a research artefact (something reviewers can
interact with) and a demonstration of the deployment-readiness narrative.

9.11.2 Tech Stack
Streamlit or Gradio for the prototype. Both render Python directly into a web UI with minimal scaffolding
and support image upload, parameter controls, and inline plotting. If a more polished system is needed
later, FastAPI plus a React front end is the natural upgrade. For the NTCC paper, Streamlit or Gradio is
sufficient and demonstrably professional.

9.11.3 User Flow
User uploads a fundus image and optionally enters age, sex, and device class.
Image-quality classifier scores the upload. Low-quality inputs receive a soft-decline message
recommending clinical follow-up; the user can override and proceed at their discretion.
Foundation-model embedding extracted (cached if previously seen).
OOD detector scores the embedding. Out-of-distribution inputs receive a confidence-flag
downgrade.
Multi-task head produces predictions for the full panel. MC-Dropout produces 95% confidence
intervals.
Attention rollout produces an explanation heatmap; rendered as overlay.
Subgroup-conditional reliability annotation: for the user's reported subgroup (sex, age band,
device class), the dashboard surfaces 'In validation, the model achieved AUC X with confidence
interval Y for patients similar to you.'
All outputs displayed in a single result panel with confidence flag (high / moderate / low_OOD),
prediction values with intervals, attention overlay, metadata attribution bar chart, and subgroup
reliability.
Optional: with explicit consent, the upload (image + metadata + prediction + version tag) is
logged to a research database for future offline study. No automatic retraining occurs.
9.11.4 Versioning
The dashboard reads from the model registry; the active version is shown in the UI and can be selected
by an administrator to roll forward or back. Every prediction is tagged with the model version that
produced it.

9.11.5 Dataset-Coupling Status
Dataset-agnostic. Operates on the trained model. New datasets that have updated the model produce no
change in dashboard code.

10. Implementation Plan
    This section specifies what to build, in what order, with what artefacts produced at each step. The plan is
    organised in four phases. Phase 1 and 2 are entirely doable on ODIR-5K and the open-access external
    datasets; they constitute the core deliverable. Phase 3 adds the advanced contributions. Phase 4
    integrates BRSET and mBRSET if access arrives, or alternatively polishes the ODIR-only paper.

10.1 Phase 1: Foundation Infrastructure (Days 1-7)............................................................................
10.1.1 Goals
Have the project repository, environment, dataset-agnostic infrastructure, and a working ODIR adapter in
place. Demonstrate the pipeline can run end-to-end on synthetic data using the dummy adapter, and on
real ODIR samples for the first few stages.

10.1.2 Tasks
Initialise git repository with directory structure: data/ (gitignored), src/dataset/,
src/preprocessing/, src/models/, src/training/, src/eval/, src/cl/, src/explain/, src/dashboard/,
configs/, notebooks/, docs/.
Set up Python environment: PyTorch with CUDA, Hugging Face Hub and PEFT, scikit-learn,
pandas, numpy, matplotlib, seaborn, plotly, streamlit, captum, shap, fairlearn, pydantic, hydra-
core, pyyaml, tqdm, wandb.
Define the canonical sample schema as a dataclass or pydantic model in src/dataset/schema.py.
Define the central enum module for categorical vocabularies (sex, camera_type, device_class,
etc.).
Define the task registry in src/dataset/tasks.py: each task is an object with name, type
(binary/multiclass/ordinal/regression), target column, loss function, primary metric, and
optional pos_weight or num_classes.
Implement the abstract DatasetAdapter base class with the interface methods listed in section
9.2.2.
Implement DummyAdapter: returns synthetic samples in the canonical schema for end-to-end
pipeline testing.
Implement ODIRAdapter: parses the labels file, extracts age and sex, parses diagnostic keywords
into the eight-class multi-label vector. Map labels to the canonical schema. Verify on a sample of
records that fields are correctly populated.
Download ODIR-5K to local storage. Verify image counts and label availability.
Write unit tests for ODIRAdapter (not exhaustive; just enough to catch obvious regressions).
Implement preprocessing pipeline (Layer 3) as a configurable transforms chain. Make every
operation toggleable from a YAML config.
Implement patient-level split utility: takes the manifest, the ratio, and a stratification target;
returns train / val / test sample-id lists with no patient appearing in more than one split.
10.1.3 Deliverables
Working git repo, environment, configurations.
Canonical schema, task registry, base adapter, ODIR adapter, dummy adapter.
Preprocessing pipeline.
Splits for ODIR train/val/test, saved as a CSV mapping sample_id to split.
10.2 Phase 2: Core Pipeline (Days 8-21)
10.2.1 Goals
End-to-end training of the multi-task model on ODIR with frozen RETFound and DINOv2 backbones, with
proper evaluation including subgroup stratification. By the end of Phase 2, the project has produced its
first headline numbers.

10.2.2 Tasks
Implement embedding extraction (Layer 4). Load RETFound from Hugging Face. Run on every
ODIR image with mixed-precision inference. Cache to disk. Generate the manifest CSV.
Repeat for DINOv2-Large. Cache separately.
(Optional this phase) Implement DINORET extraction if weights are accessible; otherwise defer
to Phase 4 polish.
Implement the multi-task head (Layer 5) with image branch, metadata branch, fusion trunk, task
heads. Implement task masking and uncertainty-weighted loss.
Implement the training loop (Layer 6). Use a Hydra or YAML-driven configuration to select
backbone, tasks, hyperparameters.
Run baseline training: ODIR + RETFound, all eight ODIR tasks plus age regression and sex
prediction. Report per-task AUC / accuracy on validation. Iterate hyperparameters until
performance is stable and reasonable.
Run baseline training: ODIR + DINOv2-Large. Compare to RETFound.
Run ImageNet-baseline training: extract embeddings from ConvNeXt-Base or ResNet- 50
ImageNet weights, train the same multi-task head. Compare to foundation model results.
Implement the evaluation harness (Layer 7) including per-task metrics, bootstrap CIs, and
subgroup stratification by sex and age band.
Implement cross-population external evaluation. Adapt IDRiDAdapter, MessidorAdapter,
APTOSAdapter, EyePACSAdapter for the DR-grade task. Run inference of ODIR-trained model on
each, report per-dataset metrics.
Implement the paired-eye cross-attention component as an optional architectural variant. Run
an ablation comparing with vs without.
Implement Group DRO and reweighted-loss training as alternative training modes. Run
mitigation experiments and produce the trade-off curve plot.
10.2.3 Deliverables
Cached embeddings for RETFound and DINOv2 on ODIR (and external sets where useful).
Trained multi-task models for at least three backbone choices.
Baseline subgroup-stratified evaluation tables. The first headline result of the paper.
Cross-population transfer table for DR grade.
Mitigation trade-off curve.
10.3 Phase 3: Advanced Features (Days 22-35)
10.3.1 Goals
Add the continual-learning protocol, the explainability stack, and the deployment dashboard. By the end
of Phase 3, the full system as described in this document is implemented.

10.3.2 Tasks
Implement the LoRA adapter setup using Hugging Face PEFT or a custom minimal
implementation. Validate by training a single LoRA adapter on a held-out chunk of ODIR and
verifying it improves performance on that chunk.
Implement the per-subgroup replay buffer.
Implement Mahalanobis OOD detection. Compute training-set embedding mean and covariance
once and cache. Calibrate threshold on a held-out validation set.
Implement the image-quality classifier. Train on whatever quality labels are available; use ODIR's
diagnostic keyword 'low quality' tags or proxy-derived blur / brightness metrics if explicit labels
are absent.
Implement the simulation protocol (Layer 8). The 'incoming stream' is a temporally-ordered
sequence of held-out chunks from external datasets. Run the simulation, log the source-test and
incoming-test performance curves, plot.
Implement the naive baseline (vanilla fine-tune without replay or OOD gating) and compare.
Show catastrophic forgetting in the baseline and its absence in the protocol.
Implement attention rollout for the ViT backbone (Layer 9). Render heatmaps as overlays on
input images.
Implement Grad-CAM for the ConvNet baseline.
Implement metadata-branch SHAP or integrated gradients.
Implement subgroup-stratified attention analysis (aggregate attention maps within subgroups;
compare).
(Optional) Implement TCAV for clinically defined retinal concepts. Concept directions derived
from RFMiD or hand-labelled examples.
Build the deployment dashboard (Layer 10). Streamlit application that loads the trained model
and registry, accepts uploads, runs the full inference pipeline, displays results, supports consent-
based logging.
Build the subgroup-conditional reliability lookup table. Pre-compute per-subgroup AUC + CI on
the held-out validation set; the dashboard reads from this table at inference.
10.3.3 Deliverables
Trained LoRA adapters for the simulation protocol.
Continual-learning curves: source-test preservation and incoming-test adaptation, with naive
baseline comparison.
Explainability components: attention rollout, Grad-CAM, metadata SHAP.
Working Streamlit dashboard.
10.4 Phase 4: Integration and Paper Preparation (Days 36-49)
10.4.1 Goals
If BRSET and mBRSET access has arrived, integrate them and rerun the pipeline with them as primary. If
not, finalise the ODIR-only version of the paper. Either way, prepare paper figures, tables, and writing.

10.4.2 If BRSET / mBRSET Access Has Arrived
Implement BRSETAdapter and mBRSETAdapter. Each is a single file. Total time: under two days.
Add BRSET- and mBRSET-specific tasks to the registry: hypertensive_retinopathy, insulin_use,
diabetes_duration_years, smoking, obesity, vascular_disease, mi_history, nephropathy,
neuropathy.
Run embedding extraction on BRSET and mBRSET for each backbone. Cache.
Re-run the full Phase 2 training with BRSET as primary, including the Canon-vs-Nikon device-
invariance natural experiment as a fairness stratification dimension.
Re-run the smartphone-domain transfer experiment: train on BRSET, test on mBRSET, and the
reverse.
Re-run the continual-learning simulation with BRSET as source distribution and mBRSET /
external datasets as the incoming stream.
Update all paper figures and tables with BRSET-primary numbers.
10.4.3 If BRSET / mBRSET Access Has Not Arrived
Frame the paper around ODIR-5K as primary with the modified narrative described in section
7.5.
Add additional ablations and analyses to strengthen the ODIR-only paper: deeper paired-eye
experiments, more cross-population transfer points, larger sweeps over backbones.
Discuss BRSET / mBRSET as future-work validation in the discussion section.
10.4.4 Paper Preparation
Generate every figure programmatically from the evaluation harness output. Subgroup-
stratified bar charts, mitigation trade-off curves, continual-learning curves, attention overlay
examples.
Write the paper sections: introduction with literature positioning per section 4 of this
document, methods describing the architecture, results, discussion.
Compile the limitations section honestly: label-noise on ODIR if applicable, confounded-camera-
with-site if applicable, absence of true longitudinal outcomes, etc.
Produce the dashboard demo (screen recording or live URL) for inclusion as supplementary
material.
Release code repository under an appropriate open-source licence with full reproduction
instructions.
10.4.5 Deliverables
Final paper draft with all figures and tables.
Public code repository.
Live or recorded dashboard demonstration.
Submission-ready package for the chosen venue. 11. ODIR-to-BRSET Modification Path
This section specifies, in concrete terms, exactly what changes when BRSET (and optionally mBRSET)
access arrives mid-project. The architecture is designed so this modification is genuinely small: a single
new adapter file, a few additions to the task registry, and re-runs of training and evaluation. No layer
beyond the adapter changes. No refactoring is required.

11.1 What Changes
Add src/dataset/brset_adapter.py implementing the BRSETAdapter class. Approximately 200-
400 lines including label harmonisation.
Add src/dataset/mbrset_adapter.py implementing the mBRSETAdapter class. Similar scope.
Extend the task registry with BRSET- and mBRSET-only tasks: hypertensive_retinopathy,
insulin_use, diabetes_duration_years, smoking, obesity, vascular_disease, mi_history,
nephropathy, neuropathy. These are added as new TaskDefinition entries; no changes to
existing ones.
Extend the central enum module with any new categorical values (e.g., 'phelcom_eyer' as a new
device_class enum, or new ethnicity values).
Extend stratification logic in the evaluation harness to include camera_type as a fairness
dimension. The harness already iterates over available stratification columns, so this is a
configuration change, not a code change.
Run embedding extraction on BRSET and mBRSET images for each backbone. New cached
embedding files appear under existing cache directory structure.
Re-run training with BRSET as primary substrate. Configuration change only; same training
script.
Re-run evaluation. Same harness, new dataset_source values trigger additional subgroup cells.
11.2 What Does Not Change
Canonical sample schema: unchanged. New label columns are added but no existing ones are
renamed or restructured.
Preprocessing pipeline: unchanged. Operates on standardised tensors regardless of source.
Embedding extraction: unchanged. Same backbones, same caching code.
Multi-task head architecture: unchanged. New tasks appear as additional task heads
automatically registered from the task registry.
Training loop: unchanged. Iterates over registered tasks, applies task masking.
Evaluation harness: unchanged. Reads available stratification columns, computes subgroup
metrics for whichever are present.
Continual learning protocol: unchanged. Consumes whatever sequence of cached embeddings
the configuration specifies.
Explainability stack: unchanged. Operates on trained model and inputs.
Dashboard: unchanged. Loads the active model version; the model version is what changed.
11.3 Time Budget for Modification
Estimated effort: two to four days from access granted to BRSET-primary results regenerated. The
bottleneck is embedding extraction wall time, not coding. Coding the two adapters and updating the task
registry is half a day at most.

11.4 Risk of Coupling Creep
The dataset-agnostic discipline only holds if it is enforced. The single most common way the discipline
breaks down is when, mid-implementation, a developer writes ODIR-specific column names or label
values into code outside the ODIR adapter. Avoiding this is a question of habit:

Every time you find yourself writing 'if df["hypertension_keyword"] == ...' or hardcoding an ODIR
label vocabulary, stop. That logic belongs in the adapter.
Every time the multi-task head, training loop, or evaluation harness needs to know about a
specific dataset's quirk, ask whether the canonical schema or task registry can be extended
instead.
Run the dummy adapter end-to-end periodically to catch coupling regressions. If a downstream
component breaks on dummy data, it is coupling against the real dataset.
Followed consistently for the first two weeks, this discipline becomes automatic, and the BRSET
integration genuinely is the half-day operation described above.

12. Publication Strategy
    12.1 Target Venues by Scenario
    With BRSET plus mBRSET as primary substrate: target Nature Communications Medicine, npj Digital
    Medicine, Lancet Digital Health, or top MICCAI / Medical Image Analysis. The combination of multi-task
    systemic prediction, fairness audit with device natural experiment, smartphone-domain transfer, and
    continual-learning protocol is competitive at this tier.

With BRSET only or mBRSET only as primary substrate: similar tier, with framing adjusted. mBRSET-only
emphasises smartphone-first deployment narrative.

With ODIR-5K only as primary substrate: target MICCAI main track, Medical Image Analysis, IEEE Journal
of Biomedical and Health Informatics, Computers in Biology and Medicine. Solid mid-tier publication with
the methodology contributions intact.

Workshop-tier fallback: MICCAI workshops (OMIA, FAIRMI, ML4H), MIDL, Medical Imaging Meets
NeurIPS. Suitable as a venue to surface intermediate results or specific sub-contributions.

12.2 Title Candidates
Beyond the Eye: Foundation-Model-Based Retinal Screening for Systemic Disease with Cross-Site
and Fairness Evaluation
One Photograph, Many Organs: A Reproducible Pipeline for Multi-Task Retinal Screening with
Deployment-Grade Fairness, Robustness, and Continual-Learning Evaluation
Fair, Robust, and Continually Updatable: A Foundation-Model Pipeline for Multi-Condition
Retinal Screening on Public Data
Multi-Condition Retinal Screening with Foundation Models: A Deployable, Fair, and Continually-
Updatable Pipeline (recommended for ODIR-only scenario)
12.3 Submission Checklist
Headline numbers (per-task AUC etc.) overall and stratified.
Cross-population transfer table for DR grade and any other transferable tasks.
Subgroup fairness gap tables and mitigation trade-off curve.
Continual-learning curves with baseline comparison.
Sample attention overlays from the dashboard.
Limitations section: dataset-specific label-noise, public-data label ceiling, simulation-not-
deployment for continual learning, single-investigator labelling absence, etc.
Reproducibility appendix: full hyperparameter table, environment specification, dataset access
instructions.
Public code repository released under permissive licence.
Dashboard demonstration (live URL or recording). 13. Risk Register
The following risks have been identified, in approximate order of impact. Each lists likelihood, impact, and
the planned response.

Risk Likelihood Impact Response
BRSET / mBRSET PhysioNet access
denied or significantly delayed
Low-Medium High Pivot to ODIR-5K primary;
pipeline already designed for
swap; paper drops to mid-tier
venue
RETFound or DINORET weights become
unavailable or incompatible
Low Medium Use DINOv2-Large plus
ImageNet baseline; reduce
bake-off scope
Compute budget insufficient for
embedding extraction at full scale
Low Medium Reduce per-image resolution;
subset large datasets (EyePACS)
for scale experiments
Adapter coupling creep (dataset-
specific code leaks into other layers)
Medium Medium Periodic dummy-adapter end-
to-end runs; code review
discipline
Continual learning protocol fails to
show clear preservation+adaptation
curves
Medium High Tune LoRA rank, replay buffer
ratio, OOD threshold; fall back
to simpler protocol if needed
Cross-population transfer numbers are
too low to be publishable
Medium Medium Honest reporting; this would
itself be a contribution if
quantified rigorously
Paired-eye cross-attention provides no
benefit
Medium Low Drop the component; simplify
architecture
Dashboard performance issues with
foundation-model inference
Low-Medium Low Pre-cache embeddings for
demo images; use smaller
backbone for live demo if
needed
Time pressure forces scope reduction High Medium Phase 3 components are all
individually droppable; Phase
1+2 alone is a publishable
workshop paper
Reviewer challenges on label noise
(especially ODIR
diabetes/hypertension)
High if ODIR-
only
Medium Pre-emptive label-noise
sensitivity analysis; transparent
limitations section 14. Open Questions and Items Requiring Decision
Items that remain open and need to be decided either now or before the corresponding implementation
step is reached.

Cardiovascular composite score formulation: Framingham, QRISK, ASCVD, or a custom proxy.
Decide before Phase 2 evaluation.
Whether to include RetiZero in the bake-off (depends on weights accessibility). Investigate
during Phase 2.
Whether to include TCAV-based concept testing (Phase 3 optional). Decide based on time
budget at the start of Phase 3.
Choice between MC-Dropout and small ensemble for uncertainty (default MC-Dropout,
ensemble as ablation). Confirm during Phase 2.
Choice between Streamlit and Gradio for the dashboard. Decide at the start of Phase 3.
Whether to include the vision-language alignment component (entirely optional, deferred
extension). Default: do not include in initial paper; revisit only if all other components are
complete and time remains.
Whether to attempt a CycleGAN-style synthetic device-shift augmentation in addition to BRSET's
natural Canon/Nikon experiment. Default: do not include; add only if the natural experiment
alone is judged insufficient evidence.
Final paper title (currently four candidates listed). Decide closer to submission.
Whether to invite a clinical co-author (an ophthalmologist or endocrinologist) for clinical
validation of the prediction panel framing. Discuss with NTCC supervisor. 15. Glossary
Term Definition
AUC Area Under the (ROC) Curve. Standard binary-classification metric.
BRSET Brazilian Multilabel Ophthalmological Dataset. 16,266 images from 8,524 patients, hosted
on PhysioNet.
CITI Collaborative Institutional Training Initiative. Provider of the human-subjects research
training course required for PhysioNet credentialed access.
CLAHE Contrast Limited Adaptive Histogram Equalisation. Preprocessing technique to enhance
local contrast.
CLS token The classification token output by a Vision Transformer; used as the image-level
embedding.
DINORET DINO-pretrained retinal-specific foundation model.
DINOv2 General-purpose self-supervised foundation model from Meta AI.
DR / ICDR / SDRG Diabetic Retinopathy. ICDR = International Clinical DR; SDRG = Scottish DR Grading.
DUA Data Use Agreement. Legal contract governing how a dataset can be used.
ECE Expected Calibration Error. Measure of how well predicted probabilities match observed
frequencies.
GradNorm Gradient Normalisation. Loss-balancing strategy for multi-task learning.
Group DRO Group Distributionally Robust Optimisation. Fairness-aware training that minimises the
worst-group loss.
Kendall
uncertainty
weighting
Multi-task loss weighting using learned per-task variance.
LoRA Low-Rank Adaptation. Parameter-efficient fine-tuning method that adds trainable low-
rank matrices to frozen layers.
MAE Mean Absolute Error (regression metric); also Masked Autoencoder (self-supervised
pretraining).
Mahalanobis
distance
Distance metric accounting for variable correlations; used here for OOD detection.
mBRSET Mobile BRSET. Smartphone-acquired companion dataset to BRSET, hosted on PhysioNet.
MC-Dropout Monte Carlo Dropout. Inference-time uncertainty estimation by keeping dropout active
across multiple forward passes.
NTCC Non-Teaching Credit Course. Amity University's in-house research internship programme.
ODIR-5K Ocular Disease Intelligent Recognition dataset. ~5,000 paired-eye images from Chinese
hospitals.
OOD Out-of-Distribution. Inputs that differ significantly from the training distribution.
ORCID Open Researcher and Contributor ID. Persistent identifier used to disambiguate
researchers.
PEFT Parameter-Efficient Fine-Tuning. Hugging Face library covering LoRA and similar.
RETFound Retinal foundation model from Moorfields Eye Hospital (Nature 2023).
RetiZero Retinal foundation model. Strong systemic-prediction performance per RetBench.
SHAP Shapley Additive exPlanations. Feature attribution method.
TCAV Testing with Concept Activation Vectors. Concept-based explainability method.
UK Biobank Large-scale UK cohort study with extensive paired retinal and systemic data. Restricted
access; out of scope for this project.
ViT Vision Transformer. Transformer architecture applied to image patches.
End of document.
