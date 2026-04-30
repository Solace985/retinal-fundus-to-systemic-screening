"""
adapters/base.py -- Abstract base class for all dataset adapters.

Adapters are the only dataset-aware source modules. Each adapter translates
native dataset structure into canonical CanonicalSample objects.

This file defines the public contract that every concrete adapter must satisfy.
Concrete adapters may add private helpers but must not expose native dataset
vocabulary through public methods.

Allowed imports: standard library, retina_screen.schema, retina_screen.tasks.
Must not import: model, training, evaluation, data, preprocessing, embeddings,
continual, dashboard_app, reporting, or any concrete adapter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from retina_screen.schema import CanonicalSample
from retina_screen.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Abstract base class for retinal dataset adapters.

    Concrete adapters must implement all abstract methods. Downstream code
    (splitting, data, model, evaluation) must only call the public interface
    defined here; it must not branch on adapter identity or dataset name.
    """

    # ------------------------------------------------------------------
    # Abstract interface (every concrete adapter must implement these)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_manifest(self) -> list[CanonicalSample]:
        """Build and return the full list of canonical samples.

        The returned list must be stable across repeated calls (same order,
        same content for the same adapter configuration).
        """
        ...

    @abstractmethod
    def load_sample(self, sample_id: str) -> CanonicalSample:
        """Return the canonical sample for *sample_id*.

        Raises
        ------
        KeyError
            If *sample_id* is not found in this adapter's manifest.
        """
        ...

    @abstractmethod
    def load_image(self, sample_id: str) -> Any:
        """Load and return the raw image for *sample_id*.

        Returns a ``PIL.Image.Image`` by convention. Downstream preprocessing
        transforms are applied by ``preprocessing.py``, not here.

        Raises
        ------
        KeyError
            If *sample_id* is not found.
        FileNotFoundError or IOError
            If the underlying image file is unavailable.
        """
        ...

    @abstractmethod
    def get_supported_tasks(self) -> list[str]:
        """Return the task names that this dataset can provide labels for.

        All returned names must exist in ``TASK_REGISTRY``.
        Tasks where this dataset has no labels must not be included.
        """
        ...

    @abstractmethod
    def get_stratification_columns(self) -> list[str]:
        """Return canonical schema field names usable for stratified splitting.

        All returned names must be fields in ``CanonicalSample``.
        """
        ...

    @abstractmethod
    def get_quality_columns(self) -> list[str]:
        """Return canonical schema field names that carry image quality signals.

        All returned names must be fields in ``CanonicalSample``.
        Returns an empty list if no quality information is available.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers (concrete adapters may override for efficiency)
    # ------------------------------------------------------------------

    def get_patient_id(self, sample_id: str) -> str:
        """Return the patient ID for *sample_id*.

        Default implementation delegates to ``load_sample()``. Adapters may
        override this for efficiency when patient ID lookup is cheaper than
        building a full canonical sample.

        Raises
        ------
        KeyError
            If *sample_id* is not found.
        """
        return self.load_sample(sample_id).patient_id

    def validate(self) -> None:
        """Validate adapter outputs against the project data contracts.

        Checks enforced:
        - manifest is non-empty
        - every manifest item is a CanonicalSample
        - sample IDs are unique and non-empty
        - patient IDs are non-empty
        - all supported tasks exist in ``TASK_REGISTRY``
        - all supported task target columns exist in ``CanonicalSample``
        - all stratification columns exist in ``CanonicalSample``
        - all quality columns exist in ``CanonicalSample``

        Raises
        ------
        ValueError
            With an actionable message identifying the first violation found.
        """
        cls_name = type(self).__name__
        schema_fields = set(CanonicalSample.model_fields.keys())

        # --- Manifest non-empty ---
        manifest = self.build_manifest()
        if not manifest:
            raise ValueError(f"{cls_name}.build_manifest() returned an empty list")

        for idx, sample in enumerate(manifest):
            if not isinstance(sample, CanonicalSample):
                raise ValueError(
                    f"{cls_name}.build_manifest() item at index {idx} is not a "
                    f"CanonicalSample; got {type(sample).__name__}"
                )

        # --- Unique, non-empty sample IDs ---
        sample_ids: list[str] = [s.sample_id for s in manifest]
        seen: set[str] = set()
        duplicates: list[str] = []
        for sid in sample_ids:
            if sid in seen:
                duplicates.append(sid)
            seen.add(sid)
        if duplicates:
            raise ValueError(
                f"{cls_name}: duplicate sample IDs found: {duplicates[:5]}"
                + (" ..." if len(duplicates) > 5 else "")
            )

        for s in manifest:
            if not s.sample_id:
                raise ValueError(f"{cls_name}: a sample has an empty sample_id")
            if not s.patient_id:
                raise ValueError(
                    f"{cls_name}: sample {s.sample_id!r} has an empty patient_id"
                )

        # --- Supported tasks in registry ---
        for task in self.get_supported_tasks():
            if task not in TASK_REGISTRY:
                raise ValueError(
                    f"{cls_name}: supported task {task!r} is not in TASK_REGISTRY. "
                    f"Register it in tasks.py first."
                )
            target_column = TASK_REGISTRY[task].target_column
            if target_column not in schema_fields:
                raise ValueError(
                    f"{cls_name}: supported task {task!r} targets "
                    f"{target_column!r}, which is not a field in CanonicalSample"
                )

        # --- Stratification columns in schema ---
        for col in self.get_stratification_columns():
            if col not in schema_fields:
                raise ValueError(
                    f"{cls_name}: stratification column {col!r} is not a field "
                    f"in CanonicalSample"
                )

        # --- Quality columns in schema ---
        for col in self.get_quality_columns():
            if col not in schema_fields:
                raise ValueError(
                    f"{cls_name}: quality column {col!r} is not a field "
                    f"in CanonicalSample"
                )

        logger.info(
            "%s.validate() passed — %d samples, %d patients",
            cls_name,
            len(manifest),
            len({s.patient_id for s in manifest}),
        )
