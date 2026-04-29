"""
feature_policy.py -- Metadata access-control policy (leakage prevention).

Owns: model input mode definitions, per-task metadata blocking rules,
globally restricted field gating, and the FeaturePolicy class.

Must not contain: model architecture, task loss computation, dataset parsing,
or training loop code. Must fail closed on unknown tasks and unknown fields.

Key rules enforced here:
- age_years cannot be used to predict retinal_age
- sex cannot be used to predict the sex task
- dataset_source and camera_type are blocked by default; require explicit allow
- image_only mode returns no metadata
- unknown task names raise ValueError (fail closed)
- unknown metadata fields in explicit_allow raise ValueError (fail closed)
"""

from __future__ import annotations

from enum import Enum

from retina_screen.schema import CANONICAL_METADATA_FIELDS
from retina_screen.tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------


class ModelInputMode(str, Enum):
    """Controls which metadata fields may be sent to the model."""

    IMAGE_ONLY = "image_only"
    IMAGE_PLUS_METADATA = "image_plus_metadata"
    CLINICAL_DEPLOYMENT = "clinical_deployment"
    FAIRNESS_ABLATION = "fairness_ablation"


# ---------------------------------------------------------------------------
# Module-level constants (derived from schema; do not duplicate field lists)
# ---------------------------------------------------------------------------

# Fields that require explicit permission even in non-image_only modes.
# They are in CANONICAL_METADATA_FIELDS but excluded from all default bases.
_RESTRICTED_FIELDS: frozenset[str] = frozenset({"dataset_source", "camera_type"})

# Safe metadata: everything recognised minus the restricted subset.
_DEFAULT_METADATA: frozenset[str] = CANONICAL_METADATA_FIELDS - _RESTRICTED_FIELDS

# Base allowed fields per mode (before task-specific blocking).
_MODE_BASES: dict[str, frozenset[str]] = {
    ModelInputMode.IMAGE_ONLY: frozenset(),
    ModelInputMode.IMAGE_PLUS_METADATA: _DEFAULT_METADATA,
    # Clinical deployment exposes only clinically interpretable demographics.
    ModelInputMode.CLINICAL_DEPLOYMENT: frozenset(
        {"age_years", "sex", "eye_laterality", "image_quality_score", "image_quality_label"}
    ),
    # Fairness ablation uses the same base as image_plus_metadata.
    ModelInputMode.FAIRNESS_ABLATION: _DEFAULT_METADATA,
}

# Per-task hard blocks: these fields are NEVER allowed as inputs for that task,
# regardless of mode, to prevent label-leakage.
_TASK_BLOCKS: dict[str, frozenset[str]] = {
    "retinal_age": frozenset({"age_years"}),  # age cannot predict retinal age
    "sex": frozenset({"sex"}),                # sex cannot predict sex
}


# ---------------------------------------------------------------------------
# FeaturePolicy
# ---------------------------------------------------------------------------


class FeaturePolicy:
    """Determines which metadata fields may be used as model inputs.

    Usage::

        policy = FeaturePolicy()
        allowed = policy.allowed_fields("glaucoma", "image_plus_metadata")

    The policy fails closed:
    - Unknown task names raise ValueError.
    - Unknown fields in explicit_allow raise ValueError.
    - image_only always returns an empty set.
    """

    #: Fields that require explicit allow; never in any default mode base.
    RESTRICTED_FIELDS: frozenset[str] = _RESTRICTED_FIELDS

    def allowed_fields(
        self,
        task_name: str,
        mode: str,
        explicit_allow: frozenset[str] | None = None,
    ) -> frozenset[str]:
        """Return the set of metadata field names permitted as model inputs.

        Parameters
        ----------
        task_name:
            Registered task name (must exist in TASK_REGISTRY).
        mode:
            One of the ModelInputMode values (string or enum).
        explicit_allow:
            Additional restricted fields to permit (e.g. for ablation studies).
            Every field in this set must be in CANONICAL_METADATA_FIELDS;
            unrecognised fields raise ValueError.

        Returns
        -------
        frozenset[str]
            Names of CanonicalSample metadata fields allowed as model inputs.

        Raises
        ------
        ValueError
            If task_name, mode, or any explicit_allow field is not recognised.
        """
        # Validate mode
        mode_str = mode.value if isinstance(mode, ModelInputMode) else mode
        if mode_str not in _MODE_BASES:
            valid = [m.value for m in ModelInputMode]
            raise ValueError(
                f"Unknown mode {mode_str!r}. Valid modes: {valid}"
            )

        # Validate task (fail closed)
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task {task_name!r} — FeaturePolicy fails closed. "
                f"Register the task in tasks.py first."
            )

        # Validate explicit allows before any early return so config typos fail closed.
        if explicit_allow:
            unknown = explicit_allow - CANONICAL_METADATA_FIELDS
            if unknown:
                raise ValueError(
                    f"explicit_allow contains unrecognised metadata fields: {sorted(unknown)}. "
                    f"Valid metadata fields: {sorted(CANONICAL_METADATA_FIELDS)}"
                )

        # image_only always returns empty after validation.
        if mode_str == ModelInputMode.IMAGE_ONLY.value:
            return frozenset()

        # Start from mode base
        base: frozenset[str] = _MODE_BASES[mode_str]

        # Apply explicit allows for restricted fields
        if explicit_allow:
            # Only restricted fields can be added via explicit_allow;
            # non-restricted fields are already in the base or are label/identifier fields.
            base = base | (explicit_allow & _RESTRICTED_FIELDS)

        # Apply per-task hard blocks (leakage prevention)
        task_blocks = _TASK_BLOCKS.get(task_name, frozenset())
        return base - task_blocks

    def is_field_allowed(
        self,
        field_name: str,
        task_name: str,
        mode: str,
        explicit_allow: frozenset[str] | None = None,
    ) -> bool:
        """Return True if *field_name* is permitted for *task_name* in *mode*."""
        return field_name in self.allowed_fields(task_name, mode, explicit_allow)
