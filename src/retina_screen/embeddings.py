"""
embeddings.py -- Backbone loading, frozen embedding extraction, and cache management.

Owns: backbone loading, one-image verification, frozen embedding extraction,
cache path construction, manifest writing/loading, checksum validation, cache repair.

Must not contain: task losses, fairness metrics, dashboard UI, native dataset parsing,
or any import of concrete adapter classes. Image loading is injected via an
image_loader callback to keep this module adapter-agnostic.

Cache namespace
---------------
    cache_root / backbone_name / dataset_source / preprocessing_hash /

Manifest columns
----------------
    sample_id, patient_id, dataset_source, cache_path, embedding_dim,
    backbone_name, backbone_version, preprocessing_hash, created_at, checksum

Silent cache skipping is FORBIDDEN (see docs/ai_context/04_forbidden_patterns.md).
Missing or corrupt cache entries raise CacheMissError / CacheCorruptError.

overwrite=False contract
------------------------
- Valid cache reuse requires a manifest row, matching namespace metadata, matching
  checksum, and exact one-dimensional shape (embedding_dim,).
- Existing orphan cache files without a manifest row are not trusted; they are
  re-extracted.
- Corrupt, missing, wrong-rank, or wrong-dim manifest-backed files raise
  CacheCorruptError / CacheMissError.

overwrite=True: always re-extract regardless of existing state.

Deferred for later stages
--------------------------
All non-mock model_type values (dinov2, retfound, convnext, resnet) raise
NotImplementedError in Stage 6. Real backbone loading is implemented in Stage 7+.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from retina_screen.preprocessing import (
    PreprocessingConfig,
    build_preprocessing_pipeline,
    get_preprocessing_hash,
)
from retina_screen.schema import CanonicalSample

logger = logging.getLogger(__name__)

MANIFEST_FIELDNAMES: list[str] = [
    "sample_id",
    "patient_id",
    "dataset_source",
    "cache_path",
    "embedding_dim",
    "backbone_name",
    "backbone_version",
    "preprocessing_hash",
    "created_at",
    "checksum",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CacheMissError(FileNotFoundError):
    """Raised when an expected embedding file is absent from the cache."""


class CacheCorruptError(ValueError):
    """Raised when an embedding file fails checksum or dimension validation."""


# ---------------------------------------------------------------------------
# BackboneConfig
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Backbone identity and output specification."""

    name: str            # "mock", "dinov2_large", etc.
    embedding_dim: int   # output dimensionality, e.g. 1024
    model_type: str      # "mock" | "dinov2" | "retfound" | "convnext" | "resnet"
    version: str = ""    # version string; empty for mock


# ---------------------------------------------------------------------------
# MockBackbone
# ---------------------------------------------------------------------------


class MockBackbone(nn.Module):
    """Deterministic frozen backbone for testing without real weights.

    Architecture: AdaptiveAvgPool2d(1) → Flatten → Linear(in_channels, embedding_dim).
    Fixed seed=0 at init. All parameters have requires_grad=False.
    Creating a MockBackbone does NOT disturb the global RNG state.
    """

    def __init__(self, embedding_dim: int = 1024, in_channels: int = 3) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Save and restore RNG state so MockBackbone construction is side-effect free.
        saved_state = torch.get_rng_state()
        torch.manual_seed(0)
        self.proj = nn.Linear(in_channels, embedding_dim, bias=True)
        torch.set_rng_state(saved_state)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) → (B, embedding_dim)."""
        pooled = self.pool(x)                   # (B, C, 1, 1)
        flat = pooled.view(pooled.size(0), -1)  # (B, C)
        return self.proj(flat)                  # (B, embedding_dim)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------


def load_backbone(config: BackboneConfig, device: torch.device) -> nn.Module:
    """Return a ready backbone model on the given device.

    Only model_type='mock' is supported in Stage 6.
    All other model_type values raise NotImplementedError.
    """
    if config.model_type == "mock":
        backbone = MockBackbone(embedding_dim=config.embedding_dim)
        backbone.eval()
        backbone.to(device)
        logger.info(
            "Loaded MockBackbone (embedding_dim=%d, device=%s)",
            config.embedding_dim, device,
        )
        return backbone

    raise NotImplementedError(
        f"Real backbone loading is not implemented until Stage 7. "
        f"Received model_type={config.model_type!r}. "
        f"Use model_type='mock' for Stage 6."
    )


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def get_cache_dir(
    cache_root: Path | str,
    backbone_name: str,
    dataset_source: str,
    preprocessing_hash: str,
) -> Path:
    """Return the canonical cache directory, creating it if needed.

    Structure: cache_root / backbone_name / dataset_source / preprocessing_hash /
    """
    cache_dir = Path(cache_root) / backbone_name / dataset_source / preprocessing_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sample_id_to_filename(sample_id: str) -> str:
    """Convert sample_id to a collision-safe cache filename.

    Keeps a sanitized, readable prefix and appends a 12-char SHA256 hash of
    the original sample_id to prevent collisions from samples that differ only
    in special characters. The manifest stores the original sample_id.
    """
    safe_stem = re.sub(r"[^\w\-.]", "_", sample_id)[:64]
    short_hash = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_stem}_{short_hash}.pt"


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_tensor_checksum(tensor: torch.Tensor) -> str:
    """Return the SHA256 hex digest of the tensor's raw bytes (CPU, contiguous)."""
    data = tensor.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _validate_embedding_shape(
    tensor: torch.Tensor,
    expected_dim: int,
    cache_path: Path | str,
) -> None:
    """Validate that a cached per-sample embedding is exactly 1-D."""
    if tensor.ndim != 1 or tensor.shape != (expected_dim,):
        raise CacheCorruptError(
            f"Embedding dim/shape mismatch for {cache_path}: expected "
            f"({expected_dim},), got {tuple(tensor.shape)}"
        )


# ---------------------------------------------------------------------------
# Save / load individual embeddings
# ---------------------------------------------------------------------------


def save_embedding(
    embedding: torch.Tensor,
    sample_id: str,
    cache_dir: Path | str,
) -> tuple[Path, str]:
    """Save a 1-D embedding tensor to cache.

    Returns
    -------
    (absolute_path, checksum)
    """
    cache_dir = Path(cache_dir)
    filename = _sample_id_to_filename(sample_id)
    path = cache_dir / filename
    torch.save(embedding.cpu(), path)
    checksum = compute_tensor_checksum(embedding)
    logger.debug(
        "Saved embedding sample_id=%s → %s (checksum=%s...)",
        sample_id, path, checksum[:12],
    )
    return path, checksum


def load_embedding(
    cache_path: Path | str,
    expected_checksum: str,
    expected_dim: int,
) -> torch.Tensor:
    """Load and validate an embedding from its manifest cache_path.

    The manifest cache_path is the authoritative source for the file location;
    do not reconstruct from sample_id.

    Raises
    ------
    CacheMissError
        If the file does not exist.
    CacheCorruptError
        If the checksum mismatches OR the last dimension != expected_dim.
    """
    path = Path(cache_path)
    if not path.exists():
        raise CacheMissError(f"Cache file absent: {path}")

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CacheCorruptError(f"Could not load embedding cache file {path}: {exc}") from exc

    if not isinstance(tensor, torch.Tensor):
        raise CacheCorruptError(
            f"Cache file {path} did not contain a torch.Tensor; got "
            f"{type(tensor).__name__}"
        )

    _validate_embedding_shape(tensor, expected_dim, path)

    actual_checksum = compute_tensor_checksum(tensor)
    if actual_checksum != expected_checksum:
        raise CacheCorruptError(
            f"Checksum mismatch for {path}: "
            f"expected {expected_checksum[:12]}..., got {actual_checksum[:12]}..."
        )

    return tensor


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def write_embedding_manifest(records: list[dict], manifest_path: Path | str) -> None:
    """Write embedding manifest CSV.

    All records must contain all MANIFEST_FIELDNAMES columns.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    logger.info(
        "Wrote embedding manifest: %s (%d rows)", manifest_path, len(records)
    )


def load_embedding_manifest(manifest_path: Path | str) -> list[dict]:
    """Load embedding manifest CSV as a list of dicts.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _manifest_index_by_sample_id(manifest_path: Path) -> dict[str, dict]:
    """Load a manifest into a sample_id -> row map, rejecting duplicate rows."""
    if not manifest_path.exists():
        return {}

    rows = load_embedding_manifest(manifest_path)
    index: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in index:
            duplicates.append(sample_id)
            continue
        index[sample_id] = row

    if duplicates:
        raise CacheCorruptError(
            f"Embedding manifest {manifest_path} contains duplicate sample_id "
            f"values, including {sorted(set(duplicates))[:5]}"
        )
    return index


def _validate_manifest_row_matches_request(
    row: dict,
    sample: CanonicalSample,
    backbone_config: BackboneConfig,
    preprocessing_hash: str,
) -> None:
    """Validate manifest metadata before reusing a cached embedding."""
    checks = {
        "dataset_source": sample.dataset_source,
        "backbone_name": backbone_config.name,
        "backbone_version": backbone_config.version,
        "preprocessing_hash": preprocessing_hash,
        "embedding_dim": str(backbone_config.embedding_dim),
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {row.get(field)!r}"
        for field, expected in checks.items()
        if str(row.get(field, "")) != expected
    ]
    if mismatches:
        raise CacheCorruptError(
            f"Manifest row for sample_id={sample.sample_id!r} does not match "
            f"the requested embedding namespace: {mismatches}"
        )


def _validate_unique_sample_ids(samples: list[CanonicalSample]) -> None:
    """Reject duplicate sample IDs before cache paths are derived."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.append(sample.sample_id)
        seen.add(sample.sample_id)

    if duplicates:
        raise ValueError(
            "Duplicate sample_id values are not allowed during embedding "
            f"extraction. Examples: {sorted(set(duplicates))[:5]}"
        )


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    manifest: list[CanonicalSample],
    backbone: nn.Module,
    backbone_config: BackboneConfig,
    preprocessing_config: PreprocessingConfig,
    cache_root: Path | str,
    device: torch.device,
    image_loader: Callable[[CanonicalSample], Image.Image],
    batch_size: int = 32,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    """Extract embeddings for all samples (up to limit) and write manifest CSV.

    Parameters
    ----------
    manifest:
        List of CanonicalSample objects to process.
    backbone:
        Frozen backbone module. Must be in eval mode and on the correct device.
    backbone_config:
        Backbone identity and embedding_dim specification.
    preprocessing_config:
        Preprocessing configuration used to derive the preprocessing_hash.
    cache_root:
        Root directory for the embedding cache.
    device:
        Torch device to run inference on.
    image_loader:
        Callable(CanonicalSample) → PIL.Image.Image. Called once per sample.
        Must not raise on valid samples. Injected by the caller to keep this
        module adapter-agnostic.
    batch_size:
        Reserved for future batching (not used in Stage 6 sequential path).
    overwrite:
        False (default): reuse valid cache entries; raise on corrupt/wrong-dim.
        True: always re-extract regardless of existing state.
    limit:
        Process only the first `limit` samples (None = all).

    Returns
    -------
    Path
        Absolute path to the written manifest.csv.

    Notes
    -----
    - Always calls backbone.eval() and uses torch.no_grad().
    - Always uses mode='extract' (deterministic preprocessing).
    - Manifest is written atomically after all samples are processed.
    - Unreadable images raise immediately; no fake manifest rows are written.
    - overwrite=False reuses cache only through validated manifest rows.
    """
    if not manifest:
        raise ValueError("Cannot extract embeddings from an empty manifest.")

    prep_hash = get_preprocessing_hash(preprocessing_config)
    pipeline = build_preprocessing_pipeline(preprocessing_config, mode="extract")

    samples = list(manifest)
    if limit is not None:
        samples = samples[:limit]
    _validate_unique_sample_ids(samples)

    records: list[dict] = []
    backbone.eval()
    manifest_indexes: dict[Path, dict[str, dict]] = {}

    with torch.no_grad():
        for sample in samples:
            cache_dir = get_cache_dir(
                cache_root, backbone_config.name, sample.dataset_source, prep_hash
            )
            filename = _sample_id_to_filename(sample.sample_id)
            cache_path = cache_dir / filename
            existing_manifest_path = cache_dir / "manifest.csv"
            if existing_manifest_path not in manifest_indexes:
                manifest_indexes[existing_manifest_path] = _manifest_index_by_sample_id(
                    existing_manifest_path
                )
            existing_row = manifest_indexes[existing_manifest_path].get(sample.sample_id)

            if not overwrite and existing_row is not None:
                # Manifest row is the source of truth for validating cache reuse.
                _validate_manifest_row_matches_request(
                    existing_row, sample, backbone_config, prep_hash
                )
                cache_path = Path(existing_row["cache_path"])
                load_embedding(
                    cache_path,
                    existing_row["checksum"],
                    backbone_config.embedding_dim,
                )
                checksum = existing_row["checksum"]
                logger.debug("Validated cache hit: sample_id=%s", sample.sample_id)
            else:
                if not overwrite and cache_path.exists():
                    logger.warning(
                        "Existing orphan cache file has no manifest row; "
                        "re-extracting sample_id=%s path=%s",
                        sample.sample_id, cache_path,
                    )
                # Extract embedding.
                img = image_loader(sample)
                tensor = pipeline(img).unsqueeze(0).to(device)   # (1, C, H, W)
                embedding = backbone(tensor).squeeze(0).cpu()    # (embedding_dim,)
                _validate_embedding_shape(
                    embedding, backbone_config.embedding_dim, cache_path
                )
                cache_path, checksum = save_embedding(
                    embedding, sample.sample_id, cache_dir
                )

            records.append({
                "sample_id": sample.sample_id,
                "patient_id": sample.patient_id,
                "dataset_source": sample.dataset_source,
                "cache_path": str(cache_path),
                "embedding_dim": backbone_config.embedding_dim,
                "backbone_name": backbone_config.name,
                "backbone_version": backbone_config.version,
                "preprocessing_hash": prep_hash,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "checksum": checksum,
            })

    # Derive manifest path from first sample's cache directory.
    first_cache_dir = get_cache_dir(
        cache_root, backbone_config.name, samples[0].dataset_source, prep_hash
    )
    manifest_path = first_cache_dir / "manifest.csv"
    write_embedding_manifest(records, manifest_path)
    logger.info(
        "Extraction complete: %d samples processed, manifest at %s",
        len(records), manifest_path,
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Cache integrity verification
# ---------------------------------------------------------------------------


def verify_cache_integrity(
    manifest_path: Path | str,
    expected_dim: int,
) -> list[str]:
    """Verify all entries in the manifest.

    For each row, validates:
    - File existence (cache_path)
    - Checksum (from manifest)
    - Embedding dimension (expected_dim)
    - backbone_name appears in cache_path directory components (if present)
    - preprocessing_hash appears in cache_path directory components (if present)

    Returns
    -------
    list[str]
        sample_ids of missing or corrupt entries. Empty list means all valid.
    """
    rows = load_embedding_manifest(manifest_path)
    failed: list[str] = []

    for row in rows:
        sample_id = row["sample_id"]
        cache_path = Path(row["cache_path"])
        expected_checksum = row["checksum"]
        row_backbone = row.get("backbone_name", "")
        row_prep_hash = row.get("preprocessing_hash", "")

        try:
            load_embedding(cache_path, expected_checksum, expected_dim)
        except (CacheMissError, CacheCorruptError) as exc:
            logger.warning(
                "Cache integrity failure for sample_id=%s: %s", sample_id, exc
            )
            failed.append(sample_id)
            continue

        # Structural validation: path components should include backbone and hash.
        path_parts = set(cache_path.parts)
        if row_backbone and row_backbone not in path_parts:
            logger.warning(
                "backbone_name %r not found in cache path components for sample_id=%s "
                "(path=%s)",
                row_backbone, sample_id, cache_path,
            )
            failed.append(sample_id)
            continue
        if row_prep_hash and row_prep_hash not in path_parts:
            logger.warning(
                "preprocessing_hash %r not found in cache path components "
                "for sample_id=%s (path=%s)",
                row_prep_hash, sample_id, cache_path,
            )
            failed.append(sample_id)
            continue

    return failed
