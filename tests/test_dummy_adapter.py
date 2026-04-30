from __future__ import annotations
from collections import Counter
import pytest
from retina_screen.adapters.base import DatasetAdapter
from retina_screen.adapters.dummy import DummyAdapter
from retina_screen.schema import CanonicalSample
from retina_screen.tasks import (
    LabelQuality,
    LossType,
    MetricType,
    TASK_REGISTRY,
    TaskDefinition,
    TaskType,
)


class _ValidationAdapter(DatasetAdapter):
    def __init__(
        self,
        manifest: list[object],
        supported_tasks: list[str] | None = None,
    ) -> None:
        self._manifest = manifest
        self._supported_tasks = supported_tasks or ["glaucoma"]

    def build_manifest(self) -> list[CanonicalSample]:
        return self._manifest  # type: ignore[return-value]

    def load_sample(self, sample_id: str) -> CanonicalSample:
        for sample in self._manifest:
            if isinstance(sample, CanonicalSample) and sample.sample_id == sample_id:
                return sample
        raise KeyError(sample_id)

    def load_image(self, sample_id: str) -> object:
        self.load_sample(sample_id)
        return object()

    def get_supported_tasks(self) -> list[str]:
        return list(self._supported_tasks)

    def get_stratification_columns(self) -> list[str]:
        return ["sex"]

    def get_quality_columns(self) -> list[str]:
        return ["image_quality_score"]

@pytest.fixture(scope="module")
def adapter():
    return DummyAdapter()

@pytest.fixture(scope="module")
def manifest(adapter):
    return adapter.build_manifest()

def test_manifest_nonempty(manifest):
    assert len(manifest) > 0

def test_all_samples_are_canonical(manifest):
    for s in manifest:
        assert isinstance(s, CanonicalSample)

def test_sample_ids_unique(manifest):
    ids = [s.sample_id for s in manifest]
    dupes = [sid for sid, cnt in Counter(ids).items() if cnt > 1]
    assert dupes == [], f"Duplicate sample IDs: {dupes}"

def test_sample_ids_nonempty(manifest):
    assert all(s.sample_id for s in manifest)

def test_patient_ids_nonempty(manifest):
    assert all(s.patient_id for s in manifest)

def test_image_paths_nonempty(manifest):
    assert all(s.image_path for s in manifest)

def test_multiple_patients(manifest):
    pids = {s.patient_id for s in manifest}
    assert len(pids) >= 2

def test_dataset_source_is_dummy(manifest):
    assert all(s.dataset_source == "dummy" for s in manifest)

def test_at_least_one_patient_has_two_eyes(manifest):
    pid_counts = Counter(s.patient_id for s in manifest)
    bilateral = [pid for pid, cnt in pid_counts.items() if cnt >= 2]
    assert bilateral, "No patient has two samples (bilateral structure missing)"

def test_bilateral_patients_have_different_lateralities(manifest):
    from collections import defaultdict
    pid_lats = defaultdict(set)
    for s in manifest:
        if s.eye_laterality is not None:
            pid_lats[s.patient_id].add(s.eye_laterality)
    bilateral = [pid for pid, lats in pid_lats.items() if len(lats) >= 2]
    assert bilateral, "No patient has two distinct lateralities"

def test_supported_tasks_nonempty(adapter):
    assert len(adapter.get_supported_tasks()) > 0

def test_supported_tasks_in_registry(adapter):
    for task in adapter.get_supported_tasks():
        assert task in TASK_REGISTRY, f"Task {task!r} not in TASK_REGISTRY"

def test_supported_task_targets_in_schema(adapter):
    schema_fields = set(CanonicalSample.model_fields.keys())
    for task in adapter.get_supported_tasks():
        target = TASK_REGISTRY[task].target_column
        assert target in schema_fields, f"Task {task!r} target {target!r} not in schema"

def test_each_task_has_at_least_one_observed_label(manifest, adapter):
    for task in adapter.get_supported_tasks():
        col = TASK_REGISTRY[task].target_column
        observed = [s for s in manifest if getattr(s, col) is not None]
        assert observed, f"Task {task!r} column {col!r} has no observed labels"

def test_missing_labels_are_none_not_zero(manifest, adapter):
    found_none = False
    for task in adapter.get_supported_tasks():
        col = TASK_REGISTRY[task].target_column
        if any(getattr(s, col) is None for s in manifest):
            found_none = True
            break
    assert found_none, "No None labels found; missing values must be None not 0"

def test_some_tasks_have_partial_missingness(manifest, adapter):
    for task in adapter.get_supported_tasks():
        col = TASK_REGISTRY[task].target_column
        observed = sum(1 for s in manifest if getattr(s, col) is not None)
        missing = sum(1 for s in manifest if getattr(s, col) is None)
        if observed > 0 and missing > 0:
            return
    pytest.fail("No supported task has both observed and missing labels")

def test_stratification_columns_nonempty(adapter):
    assert len(adapter.get_stratification_columns()) > 0

def test_stratification_columns_in_schema(adapter):
    schema_fields = set(CanonicalSample.model_fields.keys())
    for col in adapter.get_stratification_columns():
        assert col in schema_fields, f"Strat col {col!r} not in schema"

def test_stratification_column_has_variation(manifest, adapter):
    for col in adapter.get_stratification_columns():
        values = {getattr(s, col) for s in manifest if getattr(s, col) is not None}
        if len(values) > 1:
            return
    pytest.fail("No stratification column has more than one distinct value")

def test_quality_columns_in_schema(adapter):
    schema_fields = set(CanonicalSample.model_fields.keys())
    for col in adapter.get_quality_columns():
        assert col in schema_fields, f"Quality col {col!r} not in schema"

def test_load_sample_returns_correct_sample(adapter, manifest):
    s = manifest[0]
    loaded = adapter.load_sample(s.sample_id)
    assert loaded.sample_id == s.sample_id
    assert loaded.patient_id == s.patient_id

def test_load_sample_raises_for_unknown(adapter):
    with pytest.raises(KeyError):
        adapter.load_sample("nonexistent_xyz_000")

def test_get_patient_id_matches_sample(adapter, manifest):
    for s in manifest[:5]:
        pid = adapter.get_patient_id(s.sample_id)
        assert pid == s.patient_id

def test_get_patient_id_raises_for_unknown(adapter):
    with pytest.raises(KeyError):
        adapter.get_patient_id("nonexistent_xyz_000")

def test_validate_passes(adapter):
    adapter.validate()

def test_validate_rejects_non_canonical_manifest_item():
    bad_adapter = _ValidationAdapter([object()])
    with pytest.raises(ValueError, match="index 0.*CanonicalSample.*object"):
        bad_adapter.validate()

def test_validate_rejects_supported_task_with_invalid_target(monkeypatch, manifest):
    monkeypatch.setitem(
        TASK_REGISTRY,
        "fake_bad_target",
        TaskDefinition(
            name="fake_bad_target",
            task_type=TaskType.BINARY,
            target_column="not_a_schema_field",
            loss=LossType.BCE,
            primary_metric=MetricType.AUROC,
            label_quality=LabelQuality.UNKNOWN,
            allowed_as_headline=False,
        ),
    )
    bad_adapter = _ValidationAdapter([manifest[0]], supported_tasks=["fake_bad_target"])
    with pytest.raises(ValueError, match="fake_bad_target.*not_a_schema_field"):
        bad_adapter.validate()

def test_manifest_stable_across_calls(adapter):
    ids1 = [s.sample_id for s in adapter.build_manifest()]
    ids2 = [s.sample_id for s in adapter.build_manifest()]
    assert ids1 == ids2

def test_different_seeds_give_different_labels():
    a1 = DummyAdapter(seed=42)
    a2 = DummyAdapter(seed=99)
    labels1 = [s.glaucoma for s in a1.build_manifest()]
    labels2 = [s.glaucoma for s in a2.build_manifest()]
    assert labels1 != labels2, "Different seeds must give different labels"

def test_small_dummy_config_has_observed_label_for_every_supported_task():
    small_adapter = DummyAdapter(n_patients=4, seed=77)
    small_manifest = small_adapter.build_manifest()
    for task in small_adapter.get_supported_tasks():
        col = TASK_REGISTRY[task].target_column
        observed = [s for s in small_manifest if getattr(s, col) is not None]
        assert observed, f"Task {task!r} column {col!r} has no observed labels"

def test_small_dummy_config_still_contains_missing_label():
    small_adapter = DummyAdapter(n_patients=4, seed=77)
    small_manifest = small_adapter.build_manifest()
    has_missing = any(
        getattr(s, TASK_REGISTRY[task].target_column) is None
        for task in small_adapter.get_supported_tasks()
        for s in small_manifest
    )
    assert has_missing, "DummyAdapter coverage fix removed all missing labels"

def test_load_image_returns_synthetic_image_without_real_file(adapter, manifest):
    image = adapter.load_image(manifest[0].sample_id)
    assert image.mode == "RGB"
    assert image.size == (64, 64)
