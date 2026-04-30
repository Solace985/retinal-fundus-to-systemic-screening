"""
preprocessing.py -- Retinal image preprocessing pipeline.

Owns: retinal crop, resize, optional CLAHE, optional Graham preprocessing,
ImageNet normalization, training augmentations, deterministic extraction transforms,
preprocessing config, stable config hash for cache namespacing.

Must not contain: dataset-native column names, training loops, model task heads,
evaluation metrics.

Deferred for later stages
--------------------------
CLAHE (use_clahe=True) and Graham preprocessing (use_graham=True) raise
NotImplementedError in Stage 6. Implement in Stage 7+ when real images are used.

Mode contract
-------------
mode="extract" | "val" | "test": deterministic — Resize → CenterCrop → ToTensor → Normalize.
    Random transforms (hflip, rotation, color jitter) are NEVER applied in these modes,
    regardless of config values. This guarantees reproducible embedding extraction.
mode="train": augmentation-enabled — config values are applied before the deterministic
    Resize → CenterCrop → ToTensor → Normalize sequence.

Note on horizontal flip
-----------------------
random_hflip_p defaults to 0.0 because fundus images have laterality (left/right eye).
Horizontal flip swaps laterality and must not be used in extraction or by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing config
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Immutable (by convention) preprocessing configuration.

    Fields used in hash computation: all fields. Changing any field changes
    the preprocessing_hash and invalidates the cache namespace.
    """

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)   # ImageNet
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)    # ImageNet
    use_clahe: bool = False
    use_graham: bool = False
    interpolation: str = "bilinear"
    # Train-mode augmentations (never applied in extract/val/test mode).
    random_hflip_p: float = 0.0     # 0.0 = disabled; fundus laterality sensitive
    random_rotation_deg: float = 0.0
    color_jitter: bool = False


def get_preprocessing_hash(config: PreprocessingConfig) -> str:
    """Return first 16 hex chars of SHA256 of stable sorted-JSON serialization.

    Deterministic regardless of Python version, platform, or field insertion order.
    Different configs always produce different hashes (collision probability negligible).
    """
    raw: dict[str, Any] = asdict(config)
    serialized = json.dumps(raw, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _get_interpolation_mode(interpolation: str) -> transforms.InterpolationMode:
    mapping = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    if interpolation not in mapping:
        raise ValueError(
            f"Unsupported interpolation={interpolation!r}. "
            f"Supported: {sorted(mapping)}"
        )
    return mapping[interpolation]


def build_preprocessing_pipeline(
    config: PreprocessingConfig,
    mode: str = "extract",
) -> transforms.Compose:
    """Build a torchvision transform pipeline.

    Parameters
    ----------
    config:
        Preprocessing configuration.
    mode:
        "extract" | "val" | "test" → deterministic; no random transforms applied.
        "train" → augmented; config augmentation fields are applied.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if config.use_clahe:
        raise NotImplementedError(
            "CLAHE preprocessing (use_clahe=True) is deferred to Stage 7+."
        )
    if config.use_graham:
        raise NotImplementedError(
            "Graham preprocessing (use_graham=True) is deferred to Stage 7+."
        )

    interp = _get_interpolation_mode(config.interpolation)
    deterministic_steps: list[Any] = [
        transforms.Resize(config.image_size, interpolation=interp),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(config.mean), std=list(config.std)),
    ]

    if mode in ("extract", "val", "test"):
        logger.debug(
            "Preprocessing pipeline: mode=%s, image_size=%d (deterministic)",
            mode, config.image_size,
        )
        return transforms.Compose(deterministic_steps)

    if mode == "train":
        aug_steps: list[Any] = []
        if config.random_hflip_p > 0.0:
            aug_steps.append(transforms.RandomHorizontalFlip(p=config.random_hflip_p))
        if config.random_rotation_deg > 0.0:
            aug_steps.append(
                transforms.RandomRotation(degrees=config.random_rotation_deg)
            )
        if config.color_jitter:
            aug_steps.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                )
            )
        logger.debug(
            "Preprocessing pipeline: mode=train, image_size=%d, augmentations=%d",
            config.image_size, len(aug_steps),
        )
        return transforms.Compose(aug_steps + deterministic_steps)

    raise ValueError(
        f"Unsupported mode={mode!r}. Supported: 'extract', 'val', 'test', 'train'."
    )


def preprocess_image(
    image: Image.Image,
    config: PreprocessingConfig,
    mode: str = "extract",
) -> torch.Tensor:
    """Apply preprocessing pipeline to a single PIL image.

    Returns
    -------
    torch.Tensor
        Shape (C, H, W), dtype float32.
    """
    pipeline = build_preprocessing_pipeline(config, mode=mode)
    rgb_image = image.convert("RGB")
    tensor: torch.Tensor = pipeline(rgb_image)
    return tensor
