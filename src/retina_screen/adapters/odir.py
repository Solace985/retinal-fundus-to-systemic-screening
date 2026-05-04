"""
adapters/odir.py -- ODIR-5K dataset adapter.

ODIR-specific native vocabulary is confined to this file and ODIR configs/tests.
Downstream pipeline code consumes only CanonicalSample fields and task registry
names.

Stage 7 scope
-------------
ODIR-5K is used as a first real-dataset engineering smoke test, not a final
paper baseline. Only Training Images are used. Testing Images are excluded
because they are unlabeled for this supervised smoke stage. The duplicated
nested ODIR-5K/ODIR-5K folder is intentionally ignored.

Label policy
------------
- dr_grade is unsupported; ODIR has no reliable direct 0-4 ordinal DR column.
- D maps to canonical diabetes as a weak retinal-proxy label.
- H maps to canonical hypertension as a weak proxy only when hypertension is
  already present in TASK_REGISTRY. It is not routed to hypertensive_retinopathy
  in Stage 7.
- Suspected/uncertain terms map to None, never to a positive label.
- Ambiguous/unmapped positive-row eye labels map to None, not 0.
- Missing labels remain None.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from retina_screen.adapters.base import DatasetAdapter
from retina_screen.core import project_root
from retina_screen.schema import CanonicalSample, EyeLaterality, Sex
from retina_screen.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native ODIR constants (private to adapter boundary)
# ---------------------------------------------------------------------------

_COL_ID = "ID"
_COL_AGE = "Patient Age"
_COL_SEX = "Patient Sex"
_COL_LEFT_IMAGE = "Left-Fundus"
_COL_RIGHT_IMAGE = "Right-Fundus"
_COL_LEFT_KEYWORDS = "Left-Diagnostic Keywords"
_COL_RIGHT_KEYWORDS = "Right-Diagnostic Keywords"

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        _COL_ID,
        _COL_AGE,
        _COL_SEX,
        _COL_LEFT_IMAGE,
        _COL_RIGHT_IMAGE,
        _COL_LEFT_KEYWORDS,
        _COL_RIGHT_KEYWORDS,
        "D",
        "G",
        "C",
        "A",
        "H",
        "M",
        "O",
    }
)

_POSITIVE_KEYWORDS: dict[str, frozenset[str]] = {
    "glaucoma": frozenset({"glaucoma"}),
    "cataract": frozenset({"cataract"}),
    "amd": frozenset(
        {
            "dry age-related macular degeneration",
            "wet age-related macular degeneration",
        }
    ),
    "pathological_myopia": frozenset({"pathological myopia"}),
    "diabetes": frozenset(
        {
            "mild nonproliferative retinopathy",
            "moderate non proliferative retinopathy",
            "severe nonproliferative retinopathy",
            "proliferative diabetic retinopathy",
            "diabetic retinopathy",
        }
    ),
    "hypertension": frozenset({"hypertensive retinopathy"}),
}

_UNCERTAIN_KEYWORDS: frozenset[str] = frozenset(
    {
        "suspected glaucoma",
        "suspected",
    }
)
_NORMAL_KEYWORDS: frozenset[str] = frozenset({"normal fundus"})
_LOW_QUALITY_KEYWORDS: frozenset[str] = frozenset({"low image quality"})

_BASE_TASK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("glaucoma", "G"),
    ("cataract", "C"),
    ("amd", "A"),
    ("pathological_myopia", "M"),
    ("diabetes", "D"),
)
_OPTIONAL_TASK_COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "hypertension",
        "H",
        "ODIR H is available but hypertension is not registered.",
    ),
    (
        "other_ocular",
        "O",
        "ODIR O is available but other_ocular is not registered.",
    ),
)


def _parse_keywords(raw: Any) -> list[str]:
    """Parse an ODIR diagnostic-keyword cell into normalized tokens."""
    if raw is None or pd.isna(raw):
        return []
    text = str(raw).replace("，", ",")
    return [token.strip().lower() for token in text.split(",") if token.strip()]


def _infer_eye_label(
    patient_label: int | None,
    eye_tokens: list[str],
    task_name: str,
) -> int | None:
    """Infer a per-eye label from a patient-level flag and eye keywords."""
    if patient_label is None:
        return None
    if patient_label == 0:
        return 0

    token_set = set(eye_tokens)
    if token_set & _UNCERTAIN_KEYWORDS:
        return None
    if token_set & _POSITIVE_KEYWORDS.get(task_name, frozenset()):
        return 1
    if token_set and token_set <= _NORMAL_KEYWORDS:
        return 0
    return None


def _parse_onehot(value: Any) -> int | None:
    """Parse an ODIR one-hot value as 0/1/None."""
    if value is None or pd.isna(value):
        return None
    parsed = int(value)
    if parsed not in (0, 1):
        raise ValueError(f"ODIR one-hot labels must be 0/1, got {value!r}")
    return parsed


class ODIRAdapter(DatasetAdapter):
    """Adapter for ODIR-5K Training Images and data.xlsx metadata."""

    DATASET_SOURCE: str = "odir"

    def __init__(
        self,
        dataset_root: str | Path = "ODIR-5K",
        metadata_file: str = "data.xlsx",
        training_images_dir: str = "Training Images",
    ) -> None:
        env_root = os.environ.get("RETINA_SCREEN_ODIR_ROOT")
        resolved_root = Path(env_root) if env_root else Path(dataset_root)
        if not resolved_root.is_absolute():
            resolved_root = project_root() / resolved_root

        self._root = resolved_root
        self._metadata_path = resolved_root / metadata_file
        self._images_dir = resolved_root / training_images_dir

        if not self._root.exists():
            raise FileNotFoundError(
                f"ODIR dataset root not found: {self._root}. "
                "Use dataset_root or RETINA_SCREEN_ODIR_ROOT."
            )
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"ODIR metadata file not found: {self._metadata_path}")
        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"ODIR training image directory not found: {self._images_dir}"
            )

        # Audit provenance: excluded and duplicate directories
        _testing_dir = self._root / "Testing Images"
        self._excluded_testing_dirs: list[str] = (
            [str(_testing_dir)] if _testing_dir.exists() else []
        )
        _parent_training = self._root.parent / training_images_dir
        self._excluded_duplicate_dirs: list[str] = (
            [str(self._root.parent)] if _parent_training.exists() else []
        )

        try:
            self._df = pd.read_excel(self._metadata_path, sheet_name=0, dtype={_COL_ID: str})
        except ImportError as exc:
            raise RuntimeError(
                "ODIR xlsx loading requires openpyxl. It is intentionally not "
                "added by Stage 7; add it to project dependencies before "
                "reproducible Stage 8+ runs."
            ) from exc

        missing_columns = sorted(_REQUIRED_COLUMNS - set(self._df.columns))
        if missing_columns:
            raise ValueError(
                f"ODIR metadata is missing required columns: {missing_columns}"
            )

        self._supported_tasks, self._native_task_columns, self._unsupported = (
            self._resolve_supported_tasks()
        )
        self._excluded_by_reason: Counter[str] = Counter()
        self._missing_image_references: list[str] = []
        self._unknown_diagnostic_terms: Counter[str] = Counter()
        self._uncertain_diagnostic_terms: Counter[str] = Counter()
        self._low_quality_keyword_counts: Counter[str] = Counter()
        self._ambiguous_per_eye_count: Counter[str] = Counter()

        self._manifest = self._build_manifest_internal()
        self._sample_lookup = {sample.sample_id: sample for sample in self._manifest}

        # Audit provenance: count image files in Training Images not referenced by manifest
        _referenced_names = {Path(s.image_path).name for s in self._manifest}
        _all_image_names = {
            p.name for p in self._images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        self._unreferenced_image_count: int = len(_all_image_names - _referenced_names)

    def build_manifest(self) -> list[CanonicalSample]:
        """Return ODIR Training Images as canonical samples."""
        return list(self._manifest)

    def load_sample(self, sample_id: str) -> CanonicalSample:
        """Return one canonical sample by sample_id."""
        try:
            return self._sample_lookup[sample_id]
        except KeyError:
            raise KeyError(f"ODIR sample_id not found: {sample_id!r}") from None

    def load_image(self, sample_id: str) -> Image.Image:
        """Load the PIL image for a canonical ODIR sample."""
        sample = self.load_sample(sample_id)
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"ODIR image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def get_supported_tasks(self) -> list[str]:
        """Return Stage 7 task names that are registered and mappable."""
        return list(self._supported_tasks)

    def get_stratification_columns(self) -> list[str]:
        """Return canonical stratification fields available for ODIR."""
        return ["sex", "age_years", "dataset_source", "eye_laterality"]

    def get_quality_columns(self) -> list[str]:
        """No schema-valid image-quality label is populated in Stage 7."""
        return []

    def get_dataset_audit(self) -> dict[str, Any]:
        """Return ODIR-specific audit details for dataset_audit.json."""
        ages = [
            float(sample.age_years)
            for sample in self._manifest
            if sample.age_years is not None
        ]
        sex_counts = Counter(
            sample.sex.value if sample.sex is not None else "missing"
            for sample in self._manifest
        )
        return {
            "dataset_root_used": str(self._root),
            "metadata_path_used": str(self._metadata_path),
            "image_dir_used": str(self._images_dir),
            "excluded_testing_directories": self._excluded_testing_dirs,
            "excluded_duplicate_directories": self._excluded_duplicate_dirs,
            "unreferenced_image_files": self._unreferenced_image_count,
            "metadata_rows": int(len(self._df)),
            "candidate_eye_samples": int(len(self._df) * 2),
            "valid_image_samples": len(self._manifest),
            "missing_image_references": len(self._missing_image_references),
            "missing_image_reference_examples": self._missing_image_references[:10],
            "excluded_samples_by_reason": dict(self._excluded_by_reason),
            "unique_patients": len({sample.patient_id for sample in self._manifest}),
            "left_sample_count": sum(
                1 for sample in self._manifest if sample.eye_laterality == EyeLaterality.LEFT
            ),
            "right_sample_count": sum(
                1 for sample in self._manifest if sample.eye_laterality == EyeLaterality.RIGHT
            ),
            "subgroup_coverage": {
                "sex_counts": dict(sex_counts),
                "age": {
                    "available": len(ages),
                    "missing": len(self._manifest) - len(ages),
                    "min": min(ages) if ages else None,
                    "max": max(ages) if ages else None,
                    "mean": sum(ages) / len(ages) if ages else None,
                },
            },
            "weak_proxy_warnings": [
                "ODIR D is a retinal diabetic-retinopathy proxy for diabetes, not structured clinical diabetes.",
                "ODIR H is a hypertensive-retinopathy proxy for hypertension, not structured clinical hypertension.",
            ],
            "available_but_unsupported": {
                "dr_grade": "ODIR has no reliable direct 0-4 ordinal DR grade column.",
                **self._unsupported,
            },
            "unknown_diagnostic_terms": dict(self._unknown_diagnostic_terms),
            "uncertain_diagnostic_terms": dict(self._uncertain_diagnostic_terms),
            "low_quality_keyword_counts": dict(self._low_quality_keyword_counts),
            "ambiguous_per_eye_count": dict(self._ambiguous_per_eye_count),
            "mapping_warnings": [
                "Suspected or uncertain diagnostic terms are masked as None.",
                "Normal fundus is used as negative only when it is the only eye keyword.",
                "ODIR other_ocular is a broad catch-all and is non-headline.",
            ],
        }

    def _resolve_supported_tasks(
        self,
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        tasks: list[str] = []
        columns: dict[str, str] = {}
        unsupported: dict[str, str] = {}

        for task_name, native_column in _BASE_TASK_COLUMNS:
            if task_name not in TASK_REGISTRY:
                raise ValueError(
                    f"Required ODIR Stage 7 task {task_name!r} is not in TASK_REGISTRY"
                )
            tasks.append(task_name)
            columns[task_name] = native_column

        for task_name, native_column, reason in _OPTIONAL_TASK_COLUMNS:
            if task_name in TASK_REGISTRY:
                tasks.append(task_name)
                columns[task_name] = native_column
            else:
                unsupported[task_name] = reason

        return tasks, columns, unsupported

    def _build_manifest_internal(self) -> list[CanonicalSample]:
        samples: list[CanonicalSample] = []

        for _, row in self._df.iterrows():
            row_id = str(row[_COL_ID]).strip()
            patient_id = f"odir_{row_id}"
            age_years = self._parse_age(row.get(_COL_AGE))
            sex = self._parse_sex(row.get(_COL_SEX))
            left_keywords = _parse_keywords(row.get(_COL_LEFT_KEYWORDS))
            right_keywords = _parse_keywords(row.get(_COL_RIGHT_KEYWORDS))
            self._record_keyword_terms(left_keywords)
            self._record_keyword_terms(right_keywords)

            eye_specs = (
                ("L", _COL_LEFT_IMAGE, left_keywords, EyeLaterality.LEFT),
                ("R", _COL_RIGHT_IMAGE, right_keywords, EyeLaterality.RIGHT),
            )
            for side, image_column, eye_tokens, laterality in eye_specs:
                filename = str(row.get(image_column, "")).strip()
                sample_id = f"odir_{row_id}_{side}"
                if not filename:
                    self._excluded_by_reason["blank_image_reference"] += 1
                    self._missing_image_references.append(f"{sample_id}:<blank>")
                    continue

                image_path = self._images_dir / filename
                if not image_path.exists():
                    self._excluded_by_reason["missing_image_reference"] += 1
                    self._missing_image_references.append(f"{sample_id}:{filename}")
                    continue

                label_kwargs = self._compute_eye_labels(row, eye_tokens)
                samples.append(
                    CanonicalSample(
                        sample_id=sample_id,
                        patient_id=patient_id,
                        dataset_source=self.DATASET_SOURCE,
                        image_path=str(image_path),
                        eye_laterality=laterality,
                        age_years=age_years,
                        sex=sex,
                        image_quality_label=None,
                        **label_kwargs,
                    )
                )

        logger.info(
            "ODIR manifest built: %d valid samples, %d excluded",
            len(samples),
            sum(self._excluded_by_reason.values()),
        )
        return samples

    def _compute_eye_labels(
        self,
        row: pd.Series,
        eye_tokens: list[str],
    ) -> dict[str, int | None]:
        labels: dict[str, int | None] = {}

        for task_name in self._supported_tasks:
            native_column = self._native_task_columns[task_name]
            patient_label = _parse_onehot(row.get(native_column))
            if task_name in _POSITIVE_KEYWORDS:
                eye_label = _infer_eye_label(patient_label, eye_tokens, task_name)
                if patient_label == 1 and eye_label is None:
                    self._ambiguous_per_eye_count[task_name] += 1
            else:
                eye_label = patient_label

            target_column = TASK_REGISTRY[task_name].target_column
            labels[target_column] = eye_label

        return labels

    def _record_keyword_terms(self, tokens: list[str]) -> None:
        known_terms = (
            set().union(*_POSITIVE_KEYWORDS.values())
            | set(_UNCERTAIN_KEYWORDS)
            | set(_NORMAL_KEYWORDS)
            | set(_LOW_QUALITY_KEYWORDS)
        )
        for token in tokens:
            if token in _UNCERTAIN_KEYWORDS:
                self._uncertain_diagnostic_terms[token] += 1
            if token in _LOW_QUALITY_KEYWORDS:
                self._low_quality_keyword_counts[token] += 1
            if token not in known_terms:
                self._unknown_diagnostic_terms[token] += 1

    @staticmethod
    def _parse_age(raw: Any) -> float | None:
        if raw is None or pd.isna(raw):
            return None
        return float(raw)

    @staticmethod
    def _parse_sex(raw: Any) -> Sex:
        value = str(raw).strip().lower()
        if value == "female":
            return Sex.FEMALE
        if value == "male":
            return Sex.MALE
        return Sex.UNKNOWN
