"""
adapters/odir.py -- ODIR-5K dataset adapter.

Translates ODIR-5K native metadata (data.xlsx) into CanonicalSample objects.

Native vocabulary is confined to this file.  Downstream code uses only
canonical schema fields and config-selected task names.

ODIR-5K label notes
-------------------
- One xlsx row per patient; each row provides left and right eye image filenames
  and per-eye diagnostic keywords, plus patient-level one-hot disease columns.
- D-column (diabetes) = diabetic retinopathy manifestation — retinal proxy for
  systemic diabetes.  Mapped to canonical ``diabetes`` with PROXY label quality
  per Decision 007.
- H-column (hypertension) = hypertensive retinopathy — direct fundus finding.
  Mapped to canonical ``hypertensive_retinopathy``.
- dr_grade is NOT supported: ODIR has no 0-4 ordinal DR grade column.
- Testing images (1,000 files, patient IDs 1000+) have no ground-truth labels
  in data.xlsx.  Only the training set (3,500 patients, 7,000 eye samples) is
  used by this adapter.
- The nested ODIR-5K/ODIR-5K/ folder is a duplicate; this adapter uses the
  outer paths only.

Per-eye label logic
-------------------
Rather than blindly copying patient-level one-hot labels to both eyes, this
adapter uses per-eye diagnostic keywords where the mapping is unambiguous:

1. patient_label == 0  →  eye_label = 0  (explicit observed negative)
2. patient_label == 1  →  inspect this eye's keywords:
   a. keyword in _POSITIVE_KEYWORDS[task]  →  eye_label = 1
   b. keyword in _UNCERTAIN_KEYWORDS       →  eye_label = None (mask = 0)
   c. keywords are exclusively normal      →  eye_label = 0
   d. otherwise (ambiguous)               →  eye_label = None (mask = 0)

For ``other_ocular`` (O column, catch-all category), keyword matching is not
attempted.  The patient-level label propagates to both eyes directly.

image_quality_label
-------------------
The schema uses ``Optional[ImageQualityLabel]`` (enum: GOOD/USABLE/REJECT/UNKNOWN).
ODIR keywords contain "low image quality" but this does not map cleanly to the
enum.  ``image_quality_label`` is left as None for all ODIR samples.
Low-quality keyword counts are reported in dataset_audit.json by script 01.

openpyxl dependency
-------------------
ODIR metadata is stored as .xlsx.  This adapter reads it via pandas
``read_excel()``, which requires openpyxl to be installed.  openpyxl is not yet
declared in pyproject.toml and should be added before Stage 8 deployment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from retina_screen.adapters.base import DatasetAdapter
from retina_screen.core import project_root
from retina_screen.schema import (
    CanonicalSample,
    EyeLaterality,
    Sex,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword mappings (private to this module)
# ---------------------------------------------------------------------------

# Tokens that positively confirm a specific disease in one eye.
# "suspected" terms are NOT included here (see _UNCERTAIN_KEYWORDS).
_POSITIVE_KEYWORDS: dict[str, frozenset[str]] = {
    "glaucoma": frozenset({"glaucoma"}),
    "cataract": frozenset({"cataract"}),
    "amd": frozenset({
        "dry age-related macular degeneration",
        "wet age-related macular degeneration",
    }),
    "pathological_myopia": frozenset({"pathological myopia"}),
    "diabetes": frozenset({
        "mild nonproliferative retinopathy",
        "moderate non proliferative retinopathy",
        "severe nonproliferative retinopathy",
        "proliferative diabetic retinopathy",
        "diabetic retinopathy",
    }),
    "hypertensive_retinopathy": frozenset({"hypertensive retinopathy"}),
}

# Tokens that indicate uncertain/suspected findings.
# Map to None rather than 1 so they remain masked.
_UNCERTAIN_KEYWORDS: frozenset[str] = frozenset({
    "suspected glaucoma",
    "suspected",
})

# Tokens that indicate the eye is normal (no finding).
_NORMAL_KEYWORDS: frozenset[str] = frozenset({"normal fundus"})

# Tasks that support per-eye keyword inference.
_KEYWORD_TASKS: frozenset[str] = frozenset(_POSITIVE_KEYWORDS.keys())

# Supported tasks (in order; must all exist in TASK_REGISTRY).
_SUPPORTED_TASKS: list[str] = [
    "glaucoma",
    "cataract",
    "amd",
    "pathological_myopia",
    "diabetes",
    "hypertensive_retinopathy",
    "other_ocular",
]

# Native one-hot column → canonical task field (patient-level).
_ONEHOT_COLUMN: dict[str, str] = {
    "glaucoma":               "G",
    "cataract":               "C",
    "amd":                    "A",
    "pathological_myopia":    "M",
    "diabetes":               "D",
    "hypertensive_retinopathy": "H",
    "other_ocular":           "O",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_keywords(raw: Any) -> list[str]:
    """Parse a cell value into a list of normalised keyword tokens.

    Handles:
    - None / NaN → empty list
    - Full-width comma ``，`` normalised to ``,``
    - Each token stripped and lowercased
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).replace("，", ",")
    return [tok.strip().lower() for tok in text.split(",") if tok.strip()]


def _infer_eye_label(
    patient_label: int | None,
    eye_tokens: list[str],
    task: str,
) -> int | None:
    """Infer the per-eye binary label for *task* given the patient-level label
    and this eye's parsed keyword tokens.

    Returns
    -------
    0, 1, or None
        None means the label is ambiguous or uncertain; downstream masks it out.
    """
    if patient_label == 0:
        # Explicit patient-level negative → this eye is also negative.
        return 0

    # patient_label == 1: use keyword evidence to determine per-eye status.
    token_set = set(eye_tokens)

    # Check for uncertain keywords first.
    if token_set & _UNCERTAIN_KEYWORDS:
        return None

    # Check for positive disease keyword.
    if token_set & _POSITIVE_KEYWORDS.get(task, frozenset()):
        return 1

    # If keywords are exclusively "normal fundus" (no abnormal terms), the
    # eye is explicitly reported as normal despite the patient-level positive.
    if token_set and token_set <= _NORMAL_KEYWORDS:
        return 0

    # Ambiguous: no clear positive or explicit normal.
    return None


# ---------------------------------------------------------------------------
# ODIRAdapter
# ---------------------------------------------------------------------------


class ODIRAdapter(DatasetAdapter):
    """Dataset adapter for ODIR-5K.

    Parameters
    ----------
    dataset_root:
        Path to the ODIR-5K root directory containing ``data.xlsx`` and the
        ``Training Images`` folder.  If relative, resolved against the project
        root; may be overridden by the ``RETINA_SCREEN_ODIR_ROOT`` environment
        variable.
    metadata_file:
        Name of the metadata xlsx file inside *dataset_root*.
    """

    DATASET_SOURCE: str = "odir"

    def __init__(
        self,
        dataset_root: str | Path = "ODIR-5K",
        metadata_file: str = "data.xlsx",
    ) -> None:
        # Resolve dataset root (env override → argument → relative to project root).
        env_root = os.environ.get("RETINA_SCREEN_ODIR_ROOT")
        if env_root:
            resolved = Path(env_root)
        else:
            resolved = Path(dataset_root)
            if not resolved.is_absolute():
                resolved = project_root() / resolved

        if not resolved.exists():
            raise FileNotFoundError(
                f"ODIRAdapter: dataset_root not found: {resolved}. "
                f"Set RETINA_SCREEN_ODIR_ROOT or pass the correct path."
            )

        self._root = resolved
        self._images_dir = resolved / "Training Images"
        self._meta_path = resolved / metadata_file

        if not self._meta_path.exists():
            raise FileNotFoundError(
                f"ODIRAdapter: metadata file not found: {self._meta_path}"
            )
        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"ODIRAdapter: training images folder not found: {self._images_dir}"
            )

        # Read metadata once at init.
        logger.info("Loading ODIR metadata from %s", self._meta_path)
        self._df: pd.DataFrame = pd.read_excel(
            self._meta_path, sheet_name=0, dtype={"ID": str}
        )
        logger.info(
            "ODIR metadata loaded: %d patient rows, columns: %s",
            len(self._df),
            list(self._df.columns),
        )

        # Build sample-id → row lookup for fast load_sample().
        self._sample_lookup: dict[str, CanonicalSample] = {}
        manifest = self._build_manifest_internal()
        for sample in manifest:
            self._sample_lookup[sample.sample_id] = sample

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_manifest(self) -> list[CanonicalSample]:
        """Return all training eye samples as CanonicalSample objects.

        Returns 7,000 samples for the full dataset (3,500 patients × 2 eyes).
        The list is ordered by patient ID (ascending), left eye before right.
        """
        return list(self._sample_lookup.values())

    def load_sample(self, sample_id: str) -> CanonicalSample:
        """Return the CanonicalSample for *sample_id*.

        Raises
        ------
        KeyError
            If *sample_id* is not found in this adapter's manifest.
        """
        try:
            return self._sample_lookup[sample_id]
        except KeyError:
            raise KeyError(
                f"ODIRAdapter: sample_id {sample_id!r} not found in manifest."
            )

    def load_image(self, sample_id: str) -> Image.Image:
        """Load and return the PIL Image for *sample_id*.

        Raises
        ------
        KeyError
            If *sample_id* is not in the manifest.
        FileNotFoundError
            If the image file does not exist on disk.
        """
        sample = self.load_sample(sample_id)
        img_path = Path(sample.image_path)
        if not img_path.exists():
            raise FileNotFoundError(
                f"ODIRAdapter: image file not found: {img_path}"
            )
        return Image.open(img_path).convert("RGB")

    def get_supported_tasks(self) -> list[str]:
        return list(_SUPPORTED_TASKS)

    def get_stratification_columns(self) -> list[str]:
        return ["sex", "age_years", "dataset_source", "eye_laterality"]

    def get_quality_columns(self) -> list[str]:
        return ["image_quality_label"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_manifest_internal(self) -> list[CanonicalSample]:
        """Build the full list of CanonicalSample objects from the DataFrame."""
        samples: list[CanonicalSample] = []

        for _, row in self._df.iterrows():
            row_id = str(row["ID"]).strip()
            patient_id = f"odir_{row_id}"

            age_raw = row.get("Patient Age")
            age_years: float | None = float(age_raw) if pd.notna(age_raw) else None

            sex_raw = str(row.get("Patient Sex", "")).strip().lower()
            if sex_raw == "female":
                sex = Sex.FEMALE
            elif sex_raw == "male":
                sex = Sex.MALE
            else:
                sex = Sex.UNKNOWN

            left_kw = _parse_keywords(row.get("Left-Diagnostic Keywords"))
            right_kw = _parse_keywords(row.get("Right-Diagnostic Keywords"))

            for laterality, filename_col, eye_tokens, lat_enum in [
                ("L", "Left-Fundus",  left_kw,  EyeLaterality.LEFT),
                ("R", "Right-Fundus", right_kw, EyeLaterality.RIGHT),
            ]:
                sample_id = f"odir_{row_id}_{laterality}"
                filename = str(row.get(filename_col, "")).strip()
                image_path = str(self._images_dir / filename)

                label_kwargs = self._compute_eye_labels(row, eye_tokens)

                sample = CanonicalSample(
                    sample_id=sample_id,
                    patient_id=patient_id,
                    dataset_source=self.DATASET_SOURCE,
                    image_path=image_path,
                    eye_laterality=lat_enum,
                    age_years=age_years,
                    sex=sex,
                    image_quality_label=None,
                    **label_kwargs,
                )
                samples.append(sample)

        return samples

    def _compute_eye_labels(
        self,
        row: pd.Series,
        eye_tokens: list[str],
    ) -> dict[str, int | None]:
        """Return a dict of canonical task field → inferred per-eye label value."""
        result: dict[str, int | None] = {}

        for task in _SUPPORTED_TASKS:
            native_col = _ONEHOT_COLUMN[task]
            raw_val = row.get(native_col)

            # Parse patient-level label from the one-hot column.
            if pd.isna(raw_val):
                patient_label: int | None = None
            else:
                patient_label = int(raw_val)

            if task in _KEYWORD_TASKS:
                eye_label = _infer_eye_label(patient_label, eye_tokens, task)
            else:
                # other_ocular: no keyword inference; propagate patient-level label.
                eye_label = patient_label

            # Map task name to the canonical CanonicalSample field name.
            # All supported tasks have target_column == task name for ODIR.
            from retina_screen.tasks import TASK_REGISTRY  # noqa: PLC0415
            field_name = TASK_REGISTRY[task].target_column
            result[field_name] = eye_label

        return result
