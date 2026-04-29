## `docs/ai_context/05_adapter_contract.md`

# Dataset Adapter Contract

Adapters are the only dataset-aware source modules.

Every dataset adapter must convert native dataset structure into canonical project structure.

## Adapter Location

Adapters live in:

```text
src/retina_screen/adapters/

Current adapters:

base.py
dummy.py
odir.py
external_dr.py
rfmid.py
brset.py
mbrset.py
Schema Source of Truth

Do not duplicate the canonical field list in this document.

The single source of truth for canonical fields is:

src/retina_screen/schema.py

Adapter implementations must read and satisfy the schema defined in that file.

If the schema changes, adapters must adapt to the schema. This document should not be updated with a copied field list.

Adapter Responsibilities

Each adapter is responsible for:

Reading native dataset metadata.
Reading native labels.
Resolving image paths.
Mapping native labels to canonical task fields.
Mapping native demographic/acquisition fields to canonical fields.
Providing patient IDs for group splitting.
Providing eye laterality where available.
Exposing supported tasks.
Exposing available stratification columns.
Exposing image-quality columns or quality-label source where available.
Returning canonical samples or canonical manifest rows.
Validating that required canonical fields are present.
Recording label mapping confidence where mappings are approximate.
Adapter Must Not

Adapters must not:

train models
compute model losses
run evaluation metrics
perform fairness analysis
perform continual-learning updates
generate dashboard predictions
expose native dataset vocabulary through public interfaces
force downstream modules to know native column names
Required Public Interface

Concrete adapters must implement the base adapter contract.

Expected methods or equivalent behavior:

build_manifest()
load_sample(sample_id)
load_image(sample_id)
get_supported_tasks()
get_stratification_columns()
get_quality_columns()
get_patient_id(sample_id)
validate()

Exact signatures may be refined in adapters/base.py, but the semantic contract must hold.

Do not add new public adapter methods unless the base class is intentionally extended.

Minimal Adapter Pseudocode

This is conceptual pseudocode, not final implementation.

class ExampleAdapter(DatasetAdapter):
    def build_manifest(self) -> list[CanonicalSample]:
        native_table = self._load_native_metadata()
        samples = []

        for row in native_table:
            sample = self._map_native_row_to_canonical_sample(row)
            samples.append(sample)

        return samples

    def load_sample(self, sample_id: str) -> CanonicalSample:
        row = self._lookup_native_row(sample_id)
        return self._map_native_row_to_canonical_sample(row)

    def load_image(self, sample_id: str) -> ImageLike:
        sample = self.load_sample(sample_id)
        return load_image_from_path(sample.image_path)

    def get_supported_tasks(self) -> set[str]:
        return {"task_name_a", "task_name_b"}

    def get_stratification_columns(self) -> list[str]:
        return ["sex", "age_band", "camera_type"]

    def get_patient_id(self, sample_id: str) -> str:
        return self.load_sample(sample_id).patient_id

    def validate(self) -> None:
        manifest = self.build_manifest()
        validate_against_schema(manifest)

Private helper methods may use native dataset columns. Public outputs must be canonical.

Supported Tasks

get_supported_tasks() returns only tasks with real labels in that dataset.

External validation datasets must not pretend to support unavailable tasks.

Example:

APTOS supports DR grade.
It does not support diabetes/hypertension systemic labels.
Evaluation must skip unsupported tasks.
Stratification Columns

Adapters expose available canonical stratification fields, such as:

sex
age_band
camera_type
device_class
dataset_source
hospital_site
ethnicity
education_level
insurance_status

Only fields actually available in canonical form should be exposed.

Patient Identity

Adapters must provide stable patient IDs.

Rules:

Left and right eye from the same person must share the same patient_id.
Repeated visits from the same person must share the same patient_id when identifiable.
If a dataset has only image-level data and no patient ID, the adapter must document this limitation and create conservative IDs only if justified.
Eye Laterality

If laterality is available, use canonical values defined in schema.py.

Paired-eye logic must use patient ID and laterality.

Downstream dataloaders must not randomly pair samples.

DR Grade Harmonization

DR grade harmonization happens inside adapters.

Adapters should output canonical DR grade fields defined in schema.py.

Adapters must also record:

native/source grading scheme,
whether mapping is exact or approximate,
mapping confidence where applicable.

Approximate mappings must not be hidden.

Failure Modes to Avoid

If you want to add a public method like:

get_canon_camera_samples()
get_odir_diagnostic_keywords()
get_brset_diabetes_columns()

stop.

That is a sign the adapter contract is being bent.

Allowed alternative:

Keep native helper methods private inside the adapter.
Map the result to canonical fields.
Let downstream code use canonical fields and config-selected behavior.
ODIR Adapter Rules

ODIR-specific parsing is allowed only in odir.py.

ODIR diabetes/hypertension labels should be marked as weak proxy labels through task/config metadata.

ODIR camera/site effects must not be reported as clean device-invariance experiments.

BRSET/mBRSET Adapter Rules

Before access is granted, brset.py and mbrset.py may remain stubs.

When implemented, they should require no changes to:

model.py
training.py
evaluation.py
continual.py
dashboard_app.py

BRSET/mBRSET integration should be adapter + tasks + configs + embedding extraction only.

RFMiD Adapter Rules

RFMiD roles:

TCAV/concept direction source
secondary external ocular validation

RFMiD is not a continual-learning stream unless a later decision explicitly changes this.

Dummy Adapter Rules

DummyAdapter must:

produce synthetic canonical samples
include multiple patients
include at least some missing labels
include at least two subgroup values where possible
include enough structure to test splitting, task masking, feature policy, and evaluation

DummyAdapter is used to detect dataset coupling. If downstream code breaks on DummyAdapter, downstream code likely depends on real dataset quirks.
```
