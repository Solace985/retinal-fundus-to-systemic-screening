"""
adapters/dummy.py -- Deterministic synthetic adapter for pipeline testing.

DummyAdapter produces valid CanonicalSample objects without any real dataset
files. It is used to validate the full pipeline (splitting, data loading,
feature policy, task masking, model, evaluation) before real datasets are
connected.

Design invariants:
- Deterministic for a given (n_patients, seed) configuration.
- Includes bilateral eye structure (some patients have L and R samples).
- Includes subgroup variation: sex, age, device class.
- Includes a mix of observed labels and missing labels (None, never 0).
- Requires no real fundus images; load_image() returns a synthetic PIL image.
- Contains no ODIR/BRSET/mBRSET native vocabulary.
"""

from __future__ import annotations

import hashlib
import logging
import random as _random
from typing import Any

from retina_screen.adapters.base import DatasetAdapter
from retina_screen.schema import (
    CanonicalSample,
    DeviceClass,
    EyeLaterality,
    ImageQualityLabel,
    Sex,
)
from retina_screen.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


class DummyAdapter(DatasetAdapter):
    """Synthetic adapter that generates canonical samples without real data.

    Parameters
    ----------
    n_patients:
        Number of synthetic patients to generate. The first half get both
        eyes (L+R); the second half get a left eye only.
    seed:
        Seed for the label-value random number generator.
        Sample IDs and structural choices (laterality, age, sex, device) are
        seed-independent so that the manifest structure is stable.
    """

    DATASET_SOURCE: str = "dummy"

    _SUPPORTED_TASKS: list[str] = [
        "dr_grade",
        "glaucoma",
        "cataract",
        "diabetes",
        "hypertension",
    ]

    _STRATIFICATION_COLUMNS: list[str] = [
        "sex",
        "age_years",
        "device_class",
        "dataset_source",
    ]

    _QUALITY_COLUMNS: list[str] = [
        "image_quality_score",
        "image_quality_label",
    ]

    # Fraction of binary labels to leave as None (missing / not observed).
    _MISSING_RATE: float = 0.35

    def __init__(self, n_patients: int = 20, seed: int = 42) -> None:
        if n_patients < 4:
            raise ValueError(
                f"DummyAdapter requires n_patients >= 4 for meaningful splits, "
                f"got {n_patients}"
            )
        self._n_patients = n_patients
        self._seed = seed
        self._manifest: list[CanonicalSample] | None = None
        self._index: dict[str, CanonicalSample] | None = None

    # ------------------------------------------------------------------
    # DatasetAdapter interface
    # ------------------------------------------------------------------

    def build_manifest(self) -> list[CanonicalSample]:
        """Return the full list of synthetic canonical samples (cached)."""
        if self._manifest is None:
            self._manifest = self._generate()
            self._index = {s.sample_id: s for s in self._manifest}
        return list(self._manifest)

    def load_sample(self, sample_id: str) -> CanonicalSample:
        """Return the canonical sample for *sample_id*."""
        if self._index is None:
            self.build_manifest()
        try:
            return self._index[sample_id]  # type: ignore[index]
        except KeyError:
            raise KeyError(
                f"DummyAdapter: sample_id {sample_id!r} not found. "
                f"Available IDs: {sorted(self._index)[:5]} ..."
            ) from None

    def load_image(self, sample_id: str) -> Any:
        """Return a deterministic synthetic PIL Image for *sample_id*.

        Verifies the sample exists first. The image is a solid-colour
        64x64 RGB image whose colour is derived from the sample_id SHA-256.
        No real fundus image file is required.
        """
        # Raises KeyError if sample not found — consistent with contract.
        _ = self.load_sample(sample_id)
        from PIL import Image  # noqa: PLC0415

        h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
        color = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
        return Image.new("RGB", (64, 64), color=color)

    def get_supported_tasks(self) -> list[str]:
        return list(self._SUPPORTED_TASKS)

    def get_stratification_columns(self) -> list[str]:
        return list(self._STRATIFICATION_COLUMNS)

    def get_quality_columns(self) -> list[str]:
        return list(self._QUALITY_COLUMNS)

    # ------------------------------------------------------------------
    # Private generation
    # ------------------------------------------------------------------

    def _generate(self) -> list[CanonicalSample]:
        """Generate the deterministic synthetic manifest."""
        rng = _random.Random(self._seed)
        samples: list[CanonicalSample] = []

        n_bilateral = self._n_patients // 2  # first half get both eyes
        n_clinical = (self._n_patients * 3) // 4  # first 75% get clinical device

        for i in range(self._n_patients):
            pid = f"dummy_P{i + 1:04d}"
            sex = Sex.FEMALE if i % 2 == 0 else Sex.MALE
            age = round(25.0 + i * 2.5, 1)
            device = DeviceClass.CLINICAL if i < n_clinical else DeviceClass.SMARTPHONE
            lateralities = (
                [EyeLaterality.LEFT, EyeLaterality.RIGHT]
                if i < n_bilateral
                else [EyeLaterality.LEFT]
            )

            for lat in lateralities:
                sid = f"{pid}_{lat.value[0].upper()}"
                iq_score = round(0.5 + rng.random() * 0.5, 2)
                iq_label = (
                    ImageQualityLabel.GOOD if iq_score >= 0.75 else ImageQualityLabel.USABLE
                )

                samples.append(
                    CanonicalSample(
                        sample_id=sid,
                        patient_id=pid,
                        dataset_source=self.DATASET_SOURCE,
                        image_path=f"dummy://{sid}",
                        eye_laterality=lat,
                        sex=sex,
                        age_years=age,
                        device_class=device,
                        camera_type="dummy_fundus_camera",
                        image_quality_score=iq_score,
                        image_quality_label=iq_label,
                        # dr_grade is always observed (primary ophthalmic outcome)
                        dr_grade=rng.randint(0, 4),
                        # Binary labels have a chance of being missing
                        glaucoma=self._maybe_binary(rng),
                        cataract=self._maybe_binary(rng),
                        diabetes=self._maybe_binary(rng),
                        hypertension=self._maybe_binary(rng),
                    )
                )

        self._ensure_supported_label_coverage(samples)

        logger.debug(
            "DummyAdapter generated %d samples for %d patients (seed=%d)",
            len(samples),
            self._n_patients,
            self._seed,
        )
        return samples

    def _ensure_supported_label_coverage(self, samples: list[CanonicalSample]) -> None:
        """Ensure every declared dummy task has at least one observed label."""
        for idx, task_name in enumerate(self._SUPPORTED_TASKS):
            target_column = TASK_REGISTRY[task_name].target_column
            if any(getattr(sample, target_column) is not None for sample in samples):
                continue
            sample = samples[idx % len(samples)]
            setattr(sample, target_column, self._default_observed_label(target_column))

    def _default_observed_label(self, target_column: str) -> int:
        """Return a deterministic observed label for dummy task coverage."""
        if target_column == "dr_grade":
            return 0
        return 1

    def _maybe_binary(self, rng: _random.Random) -> int | None:
        """Return 0 or 1 with probability (1 - MISSING_RATE), else None."""
        if rng.random() < self._MISSING_RATE:
            return None
        return rng.randint(0, 1)
