"""
test_cache_manifest.py -- Tier 6 architecture-enforcement tests for the
embedding cache and manifest system.

Verifies:
- save/load round-trip correctness
- checksum computation and sensitivity
- CacheMissError on absent files
- CacheCorruptError on checksum mismatch and wrong dim
- manifest schema completeness and round-trip
- preprocessing hash stability and sensitivity
- cache directory structure
- extract_embeddings integration (mock backbone, synthetic samples)
- overwrite=False / overwrite=True cache behaviour
- MockBackbone determinism and frozen parameters
- verify_cache_integrity detection of missing/corrupt files
- load_backbone factory for mock and real backbone guard
- preprocess_image output shape and dtype
- extract mode determinism (no random transforms regardless of config)
- manifest purity (no checksum='error' entries)

All tests use synthetic torch tensors and tmp_path.
No real images, no real backbones, no internet access required.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from retina_screen.embeddings import (
    MANIFEST_FIELDNAMES,
    BackboneConfig,
    CacheCorruptError,
    CacheMissError,
    MockBackbone,
    compute_tensor_checksum,
    extract_embeddings,
    get_cache_dir,
    load_backbone,
    load_embedding,
    load_embedding_manifest,
    save_embedding,
    verify_cache_integrity,
    write_embedding_manifest,
)
from retina_screen.preprocessing import (
    PreprocessingConfig,
    build_preprocessing_pipeline,
    get_preprocessing_hash,
    preprocess_image,
)
from retina_screen.schema import CanonicalSample


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sample(
    sample_id: str = "s0001",
    patient_id: str = "P0001",
    dataset_source: str = "dummy",
) -> CanonicalSample:
    return CanonicalSample(
        sample_id=sample_id,
        patient_id=patient_id,
        dataset_source=dataset_source,
        image_path=f"test://{sample_id}",
    )


def _dummy_image_loader(sample: CanonicalSample) -> Image.Image:
    """Return a fixed-colour 64×64 RGB image regardless of sample content."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


def _make_backbone_config(embedding_dim: int = 1024) -> BackboneConfig:
    return BackboneConfig(
        name="mock", embedding_dim=embedding_dim, model_type="mock", version=""
    )


def _make_prep_config(image_size: int = 224) -> PreprocessingConfig:
    return PreprocessingConfig(image_size=image_size)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_dim() -> int:
    return 1024


@pytest.fixture
def sample_tensor(embedding_dim: int) -> torch.Tensor:
    torch.manual_seed(99)
    return torch.randn(embedding_dim)


@pytest.fixture
def backbone_config(embedding_dim: int) -> BackboneConfig:
    return _make_backbone_config(embedding_dim)


@pytest.fixture
def prep_config() -> PreprocessingConfig:
    return _make_prep_config()


# ---------------------------------------------------------------------------
# 1. save + reload returns identical tensor
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, sample_tensor: torch.Tensor, embedding_dim: int) -> None:
    path, checksum = save_embedding(sample_tensor, "s0001", tmp_path)
    reloaded = load_embedding(path, checksum, embedding_dim)
    assert torch.allclose(sample_tensor, reloaded), (
        "Round-trip failed: saved and loaded tensors differ"
    )


# ---------------------------------------------------------------------------
# 2. checksum is reproducible
# ---------------------------------------------------------------------------


def test_checksum_reproducible(sample_tensor: torch.Tensor) -> None:
    c1 = compute_tensor_checksum(sample_tensor)
    c2 = compute_tensor_checksum(sample_tensor)
    assert c1 == c2, "Checksum is not reproducible for the same tensor"


# ---------------------------------------------------------------------------
# 3. different tensor → different checksum
# ---------------------------------------------------------------------------


def test_checksum_sensitivity(sample_tensor: torch.Tensor, embedding_dim: int) -> None:
    torch.manual_seed(77)
    other = torch.randn(embedding_dim)
    assert compute_tensor_checksum(sample_tensor) != compute_tensor_checksum(other), (
        "Different tensors produced the same checksum"
    )


# ---------------------------------------------------------------------------
# 4. missing file → CacheMissError
# ---------------------------------------------------------------------------


def test_missing_file_raises_cache_miss(tmp_path: Path, embedding_dim: int) -> None:
    fake_path = tmp_path / "nonexistent_abc123.pt"
    with pytest.raises(CacheMissError, match="absent"):
        load_embedding(fake_path, "aabbcc" * 10, embedding_dim)


# ---------------------------------------------------------------------------
# 5. checksum mismatch → CacheCorruptError
# ---------------------------------------------------------------------------


def test_checksum_mismatch_raises_cache_corrupt(
    tmp_path: Path, sample_tensor: torch.Tensor, embedding_dim: int
) -> None:
    path, _ = save_embedding(sample_tensor, "s0001", tmp_path)
    wrong_checksum = "deadbeef" * 8
    with pytest.raises(CacheCorruptError, match="[Cc]hecksum"):
        load_embedding(path, wrong_checksum, embedding_dim)


# ---------------------------------------------------------------------------
# 6. wrong embedding dim → CacheCorruptError
# ---------------------------------------------------------------------------


def test_wrong_dim_raises_cache_corrupt(tmp_path: Path, sample_tensor: torch.Tensor) -> None:
    path, checksum = save_embedding(sample_tensor, "s0001", tmp_path)
    with pytest.raises(CacheCorruptError, match="dim"):
        load_embedding(path, checksum, expected_dim=512)


def test_load_embedding_accepts_exact_1d_shape(
    tmp_path: Path, sample_tensor: torch.Tensor, embedding_dim: int
) -> None:
    path, checksum = save_embedding(sample_tensor, "s0001", tmp_path)

    loaded = load_embedding(path, checksum, expected_dim=embedding_dim)

    assert loaded.shape == (embedding_dim,)


def test_load_embedding_rejects_batch_shaped_tensor_with_matching_last_dim(
    tmp_path: Path, embedding_dim: int
) -> None:
    bad = torch.zeros((2, embedding_dim))
    path = tmp_path / "batch_shaped.pt"
    torch.save(bad, path)
    checksum = compute_tensor_checksum(bad)

    with pytest.raises(CacheCorruptError, match="shape"):
        load_embedding(path, checksum, expected_dim=embedding_dim)


# ---------------------------------------------------------------------------
# 7. manifest written with all required columns
# ---------------------------------------------------------------------------


def test_manifest_has_required_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    records = [{
        "sample_id": "s0001",
        "patient_id": "P0001",
        "dataset_source": "dummy",
        "cache_path": str(tmp_path / "s0001.pt"),
        "embedding_dim": 1024,
        "backbone_name": "mock",
        "backbone_version": "",
        "preprocessing_hash": "abcd1234abcd1234",
        "created_at": "2026-04-30T00:00:00+00:00",
        "checksum": "abc123",
    }]
    write_embedding_manifest(records, manifest_path)

    with manifest_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        actual_cols = set(reader.fieldnames or [])

    missing = set(MANIFEST_FIELDNAMES) - actual_cols
    assert not missing, f"Manifest is missing required columns: {missing}"


# ---------------------------------------------------------------------------
# 8. manifest round-trips via load_embedding_manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    records = [{
        "sample_id": "s0001",
        "patient_id": "P0001",
        "dataset_source": "dummy",
        "cache_path": "cache/mock/dummy/abcd1234/s0001.pt",
        "embedding_dim": "1024",
        "backbone_name": "mock",
        "backbone_version": "",
        "preprocessing_hash": "abcd1234abcd1234",
        "created_at": "2026-04-30",
        "checksum": "abc123",
    }]
    write_embedding_manifest(records, manifest_path)
    loaded = load_embedding_manifest(manifest_path)

    assert len(loaded) == 1, f"Expected 1 row, got {len(loaded)}"
    assert loaded[0]["sample_id"] == "s0001"
    assert loaded[0]["backbone_name"] == "mock"


# ---------------------------------------------------------------------------
# 9. preprocessing hash is stable (same config → same hash)
# ---------------------------------------------------------------------------


def test_preprocessing_hash_stable() -> None:
    cfg = PreprocessingConfig(image_size=224)
    h1 = get_preprocessing_hash(cfg)
    h2 = get_preprocessing_hash(cfg)
    assert h1 == h2, "Preprocessing hash is not stable for the same config"


# ---------------------------------------------------------------------------
# 10. different PreprocessingConfig → different hash
# ---------------------------------------------------------------------------


def test_preprocessing_hash_sensitivity() -> None:
    cfg1 = PreprocessingConfig(image_size=224)
    cfg2 = PreprocessingConfig(image_size=512)
    assert get_preprocessing_hash(cfg1) != get_preprocessing_hash(cfg2), (
        "Different preprocessing configs produced the same hash"
    )


# ---------------------------------------------------------------------------
# 11. get_cache_dir returns backbone/dataset/hash structure
# ---------------------------------------------------------------------------


def test_cache_dir_structure(tmp_path: Path) -> None:
    prep_hash = "abcd1234abcd1234"
    cache_dir = get_cache_dir(tmp_path, "mock", "dummy", prep_hash)

    rel_parts = cache_dir.relative_to(tmp_path).parts
    assert rel_parts == ("mock", "dummy", prep_hash), (
        f"Unexpected cache dir structure: {rel_parts}. "
        f"Expected ('mock', 'dummy', '{prep_hash}')"
    )
    assert cache_dir.is_dir(), "Cache directory was not created"


# ---------------------------------------------------------------------------
# 12. extract_embeddings integration
# ---------------------------------------------------------------------------


def test_extract_embeddings_integration(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample(f"s{i:04d}", f"P{i:04d}") for i in range(3)]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")

    manifest_path = extract_embeddings(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
        overwrite=False,
    )

    assert manifest_path.exists(), "manifest.csv was not created"

    rows = load_embedding_manifest(manifest_path)
    assert len(rows) == 3, f"Expected 3 manifest rows, got {len(rows)}"

    for row in rows:
        pt_path = Path(row["cache_path"])
        assert pt_path.exists(), f".pt file missing for sample_id={row['sample_id']}: {pt_path}"
        assert int(row["embedding_dim"]) == backbone_config.embedding_dim


# ---------------------------------------------------------------------------
# 13. overwrite=False skips valid cache entries
# ---------------------------------------------------------------------------


def test_overwrite_false_skips_valid_cache(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample("s0001", "P0001")]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")
    kwargs = dict(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    manifest_path = extract_embeddings(**kwargs, overwrite=False)
    rows_first = load_embedding_manifest(manifest_path)
    pt_path = Path(rows_first[0]["cache_path"])
    mtime_before = pt_path.stat().st_mtime

    manifest_path2 = extract_embeddings(**kwargs, overwrite=False)
    rows_second = load_embedding_manifest(manifest_path2)
    mtime_after = Path(rows_second[0]["cache_path"]).stat().st_mtime

    assert mtime_before == mtime_after, (
        "Cache file was re-written with overwrite=False and a valid existing entry"
    )


# ---------------------------------------------------------------------------
# 14. overwrite=False with corrupt (wrong-dim) cache raises CacheCorruptError
# ---------------------------------------------------------------------------


def test_overwrite_false_corrupt_cache_raises(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample("s0001", "P0001")]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")
    kwargs = dict(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    manifest_path = extract_embeddings(**kwargs, overwrite=False)
    rows = load_embedding_manifest(manifest_path)
    cache_path = Path(rows[0]["cache_path"])

    # Corrupt: replace with a wrong-dim tensor.
    torch.save(torch.zeros(512), cache_path)

    with pytest.raises(CacheCorruptError, match="dim"):
        extract_embeddings(**kwargs, overwrite=False)


def test_overwrite_false_same_dim_corrupt_cache_raises(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample("s0001", "P0001")]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")
    kwargs = dict(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    manifest_path = extract_embeddings(**kwargs, overwrite=False)
    rows = load_embedding_manifest(manifest_path)
    cache_path = Path(rows[0]["cache_path"])

    # Corrupt: replace with a different same-dimensional tensor.
    torch.save(torch.full((backbone_config.embedding_dim,), 999.0), cache_path)

    with pytest.raises(CacheCorruptError, match="[Cc]hecksum"):
        extract_embeddings(**kwargs, overwrite=False)


def test_orphan_cache_file_without_manifest_is_reextracted_not_trusted(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    sample = _make_sample("s0001", "P0001")
    prep_hash = get_preprocessing_hash(prep_config)
    cache_dir = get_cache_dir(tmp_path / "cache", "mock", "dummy", prep_hash)
    orphan_tensor = torch.full((backbone_config.embedding_dim,), 999.0)
    _, orphan_checksum = save_embedding(orphan_tensor, sample.sample_id, cache_dir)

    manifest_path = extract_embeddings(
        manifest=[sample],
        backbone=MockBackbone(embedding_dim=backbone_config.embedding_dim),
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=torch.device("cpu"),
        image_loader=_dummy_image_loader,
        batch_size=4,
        overwrite=False,
    )
    rows = load_embedding_manifest(manifest_path)

    assert rows[0]["checksum"] != orphan_checksum, (
        "Orphan cache file without a manifest row was trusted instead of re-extracted"
    )


# ---------------------------------------------------------------------------
# 15. overwrite=True re-extracts existing entries
# ---------------------------------------------------------------------------


def test_overwrite_true_re_extracts(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample("s0001", "P0001")]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")
    kwargs = dict(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    manifest_path = extract_embeddings(**kwargs, overwrite=False)
    rows_first = load_embedding_manifest(manifest_path)
    cache_path = Path(rows_first[0]["cache_path"])

    # Corrupt: replace with a wrong-dim tensor.
    torch.save(torch.zeros(512), cache_path)

    # overwrite=True should replace the corrupt file.
    manifest_path2 = extract_embeddings(**kwargs, overwrite=True)
    rows_second = load_embedding_manifest(manifest_path2)
    new_checksum = rows_second[0]["checksum"]

    # The re-extracted file should now be valid.
    reloaded = load_embedding(
        Path(rows_second[0]["cache_path"]), new_checksum, backbone_config.embedding_dim
    )
    assert reloaded.shape == (backbone_config.embedding_dim,), (
        f"Re-extracted embedding has wrong shape: {reloaded.shape}"
    )


def test_extract_embeddings_rejects_duplicate_sample_ids(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [
        _make_sample("duplicate", "P0001"),
        _make_sample("duplicate", "P0002"),
    ]

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        extract_embeddings(
            manifest=samples,
            backbone=MockBackbone(embedding_dim=backbone_config.embedding_dim),
            backbone_config=backbone_config,
            preprocessing_config=prep_config,
            cache_root=tmp_path / "cache",
            device=torch.device("cpu"),
            image_loader=_dummy_image_loader,
            batch_size=4,
            overwrite=False,
        )


# ---------------------------------------------------------------------------
# 16. MockBackbone deterministic (same input → same output)
# ---------------------------------------------------------------------------


def test_mock_backbone_deterministic(embedding_dim: int) -> None:
    backbone = MockBackbone(embedding_dim=embedding_dim)
    backbone.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out1 = backbone(x)
        out2 = backbone(x)
    assert torch.allclose(out1, out2), (
        "MockBackbone produced different outputs for the same input"
    )


# ---------------------------------------------------------------------------
# 17. MockBackbone parameters have requires_grad=False
# ---------------------------------------------------------------------------


def test_mock_backbone_frozen(embedding_dim: int) -> None:
    backbone = MockBackbone(embedding_dim=embedding_dim)
    for name, param in backbone.named_parameters():
        assert not param.requires_grad, (
            f"MockBackbone parameter {name!r} has requires_grad=True "
            f"(backbone must be fully frozen)"
        )


# ---------------------------------------------------------------------------
# 18. MockBackbone output shape is (B, embedding_dim)
# ---------------------------------------------------------------------------


def test_mock_backbone_output_shape(embedding_dim: int) -> None:
    backbone = MockBackbone(embedding_dim=embedding_dim)
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, embedding_dim), (
        f"MockBackbone output shape {out.shape} != (4, {embedding_dim})"
    )


# ---------------------------------------------------------------------------
# 19. verify_cache_integrity happy path
# ---------------------------------------------------------------------------


def test_verify_cache_integrity_happy_path(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample(f"s{i:04d}", f"P{i:04d}") for i in range(3)]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")

    manifest_path = extract_embeddings(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    failed = verify_cache_integrity(manifest_path, backbone_config.embedding_dim)
    assert failed == [], (
        f"verify_cache_integrity reported failures on a fresh valid cache: {failed}"
    )


# ---------------------------------------------------------------------------
# 20. verify_cache_integrity detects deleted file
# ---------------------------------------------------------------------------


def test_verify_cache_integrity_deleted_file(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [
        _make_sample("s0001", "P0001"),
        _make_sample("s0002", "P0002"),
    ]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")

    manifest_path = extract_embeddings(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    rows = load_embedding_manifest(manifest_path)
    deleted_id = rows[0]["sample_id"]
    Path(rows[0]["cache_path"]).unlink()

    failed = verify_cache_integrity(manifest_path, backbone_config.embedding_dim)
    assert deleted_id in failed, (
        f"Deleted file for sample_id={deleted_id!r} was not detected. "
        f"Failed list: {failed}"
    )


# ---------------------------------------------------------------------------
# 21. load_backbone("mock") returns MockBackbone
# ---------------------------------------------------------------------------


def test_load_backbone_mock_returns_mock_backbone() -> None:
    cfg = BackboneConfig(name="mock", embedding_dim=1024, model_type="mock")
    backbone = load_backbone(cfg, torch.device("cpu"))
    assert isinstance(backbone, MockBackbone), (
        f"load_backbone returned {type(backbone).__name__}, expected MockBackbone"
    )


# ---------------------------------------------------------------------------
# 22. load_backbone("dinov2") → NotImplementedError
# ---------------------------------------------------------------------------


def test_load_backbone_real_raises_not_implemented() -> None:
    cfg = BackboneConfig(name="dinov2_large", embedding_dim=1024, model_type="dinov2")
    with pytest.raises(NotImplementedError, match="Stage 7"):
        load_backbone(cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 23. preprocess_image returns (3, H, W) float32 tensor
# ---------------------------------------------------------------------------


def test_preprocess_image_shape_and_dtype() -> None:
    cfg = PreprocessingConfig(image_size=224)
    img = Image.new("RGB", (512, 512))
    tensor = preprocess_image(img, cfg, mode="extract")
    assert tensor.shape == (3, 224, 224), (
        f"preprocess_image output shape {tensor.shape} != (3, 224, 224)"
    )
    assert tensor.dtype == torch.float32, (
        f"preprocess_image output dtype {tensor.dtype} != float32"
    )


def test_preprocess_image_grayscale_converts_to_rgb() -> None:
    cfg = PreprocessingConfig(image_size=224)
    img = Image.new("L", (512, 512), color=128)

    tensor = preprocess_image(img, cfg, mode="extract")

    assert tensor.shape == (3, 224, 224)


def test_preprocess_image_rgba_converts_to_rgb() -> None:
    cfg = PreprocessingConfig(image_size=224)
    img = Image.new("RGBA", (512, 512), color=(100, 150, 200, 128))

    tensor = preprocess_image(img, cfg, mode="extract")

    assert tensor.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# 24. build_preprocessing_pipeline extract mode is deterministic
# ---------------------------------------------------------------------------


def test_extract_mode_deterministic_despite_aug_config() -> None:
    """Augmentation config values must not affect extract-mode determinism."""
    cfg = PreprocessingConfig(
        image_size=224,
        random_hflip_p=0.5,
        random_rotation_deg=10.0,
        color_jitter=True,
    )
    img = Image.new("RGB", (256, 256), color=(128, 64, 32))
    pipeline = build_preprocessing_pipeline(cfg, mode="extract")
    t1 = pipeline(img)
    t2 = pipeline(img)
    assert torch.allclose(t1, t2), (
        "build_preprocessing_pipeline mode='extract' produced different results "
        "for the same input (augmentation was applied despite extract mode)"
    )


# ---------------------------------------------------------------------------
# 25. manifest has no entries with checksum="error"
# ---------------------------------------------------------------------------


def test_manifest_no_error_checksum_entries(
    tmp_path: Path, backbone_config: BackboneConfig, prep_config: PreprocessingConfig
) -> None:
    samples = [_make_sample(f"s{i:04d}", f"P{i:04d}") for i in range(3)]
    backbone = MockBackbone(embedding_dim=backbone_config.embedding_dim)
    backbone.eval()
    device = torch.device("cpu")

    manifest_path = extract_embeddings(
        manifest=samples,
        backbone=backbone,
        backbone_config=backbone_config,
        preprocessing_config=prep_config,
        cache_root=tmp_path / "cache",
        device=device,
        image_loader=_dummy_image_loader,
        batch_size=4,
    )

    rows = load_embedding_manifest(manifest_path)
    for row in rows:
        assert row["checksum"] != "error", (
            f"sample_id={row['sample_id']!r} has checksum='error'. "
            f"Invalid images must raise, not produce fake manifest rows."
        )


def test_script_03_fails_when_cache_verification_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path("scripts") / "03_extract_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_embeddings_script_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_load_config(path: Path | str) -> dict:
        text = str(path).replace("\\", "/")
        if "configs/backbone" in text:
            return {"name": "mock", "embedding_dim": 1024, "model_type": "mock"}
        if "configs/preprocessing" in text:
            return {"image_size": 224}
        return {
            "seed": 42,
            "backbone": "mock",
            "preprocessing": "default_224",
            "n_patients": 1,
            "cache_root": str(tmp_path / "cache"),
        }

    class FakeAdapter:
        def __init__(self, n_patients: int) -> None:
            self.n_patients = n_patients

        def build_manifest(self) -> list[CanonicalSample]:
            return [_make_sample("s0001", "P0001")]

    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "setup_logging", lambda: None)
    monkeypatch.setattr(module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "load_backbone", lambda config, device: object())
    monkeypatch.setattr(module, "DummyAdapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "extract_embeddings",
        lambda **kwargs: tmp_path / "manifest.csv",
    )
    monkeypatch.setattr(module, "verify_cache_integrity", lambda path, dim: ["s0001"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["03_extract_embeddings.py", "--config", "configs/experiment/smoke_dummy.yaml"],
    )

    with pytest.raises(RuntimeError, match="Cache integrity verification failed"):
        module.main()
