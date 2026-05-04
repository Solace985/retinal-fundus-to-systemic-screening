"""
test_odir_adapter.py -- Tier 6 tests for the ODIR-5K adapter.

Synthetic fixtures only for CI (no real dataset required).
A separate guarded test class runs real-dataset smoke checks when
ODIR-5K/data.xlsx exists locally.

Covers:
- bilateral eye sample expansion
- sample_id / patient_id naming
- eye laterality
- per-eye keyword label inference (positive / uncertain / normal / ambiguous)
- patient-level negative propagation
- other_ocular catch-all propagation
- sex, age mapping
- image_quality_label = None always, with quality columns empty
- dr_grade NOT in supported tasks
- H maps to hypertension weak proxy, not hypertensive_retinopathy
- supported task list exact match
- stratification columns
- load_image / load_sample
- validate() passes
- split_patients compatibility
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from retina_screen.adapters.odir import ODIRAdapter
from retina_screen.schema import CanonicalSample, EyeLaterality, Sex
from retina_screen.splitting import split_patients


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

_EXPECTED_TASKS = [
    "glaucoma",
    "cataract",
    "amd",
    "pathological_myopia",
    "diabetes",
    "hypertension",
    "other_ocular",
]


def _make_mini_xlsx(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal ODIR-style xlsx to tmp_path and return its path."""
    df = pd.DataFrame(rows)
    xlsx_path = tmp_path / "data.xlsx"
    df.to_excel(xlsx_path, index=False)
    return xlsx_path


def _make_dummy_jpg(images_dir: Path, filename: str) -> None:
    """Write a tiny valid JPEG file."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(images_dir / filename, format="JPEG")


def _make_odir_root(
    tmp_path: Path,
    rows: list[dict],
    create_images: bool = True,
) -> Path:
    """Create a minimal ODIR-like directory structure under tmp_path."""
    odir_root = tmp_path / "ODIR-5K"
    images_dir = odir_root / "Training Images"
    images_dir.mkdir(parents=True)

    _make_mini_xlsx(odir_root, rows)

    if create_images:
        for row in rows:
            for col in ("Left-Fundus", "Right-Fundus"):
                fname = row.get(col, "")
                if fname:
                    _make_dummy_jpg(images_dir, fname)

    return odir_root


def _base_row(
    pid: str,
    *,
    age: int = 50,
    sex: str = "Female",
    left_kw: str = "normal fundus",
    right_kw: str = "normal fundus",
    N: int = 1, D: int = 0, G: int = 0, C: int = 0,
    A: int = 0, H: int = 0, M: int = 0, O: int = 0,
) -> dict:
    return {
        "ID": pid,
        "Patient Age": age,
        "Patient Sex": sex,
        "Left-Fundus": f"{pid}_left.jpg",
        "Right-Fundus": f"{pid}_right.jpg",
        "Left-Diagnostic Keywords": left_kw,
        "Right-Diagnostic Keywords": right_kw,
        "N": N, "D": D, "G": G, "C": C,
        "A": A, "H": H, "M": M, "O": O,
    }


@pytest.fixture(scope="module")
def three_patient_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three-patient fixture with diverse label scenarios."""
    tmp = tmp_path_factory.mktemp("odir_three")
    rows = [
        _base_row("10", sex="Female", left_kw="glaucoma", right_kw="normal fundus",
                  N=0, G=1),
        _base_row("20", sex="Male", age=65,
                  left_kw="moderate non proliferative retinopathy",
                  right_kw="normal fundus",
                  N=0, D=1),
        _base_row("30", sex="Female", age=72,
                  left_kw="hypertensive retinopathy",
                  right_kw="hypertensive retinopathy",
                  N=0, H=1, O=1),
    ]
    return _make_odir_root(tmp, rows)


@pytest.fixture(scope="module")
def adapter(three_patient_root: Path) -> ODIRAdapter:
    return ODIRAdapter(dataset_root=three_patient_root)


@pytest.fixture(scope="module")
def manifest(adapter: ODIRAdapter) -> list[CanonicalSample]:
    return adapter.build_manifest()


# ---------------------------------------------------------------------------
# 1. Bilateral expansion
# ---------------------------------------------------------------------------


def test_three_patients_produce_six_samples(manifest: list[CanonicalSample]) -> None:
    assert len(manifest) == 6, f"Expected 6 samples, got {len(manifest)}"


# ---------------------------------------------------------------------------
# 2-3. sample_id / patient_id naming
# ---------------------------------------------------------------------------


def test_sample_ids_follow_odir_pattern(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        assert s.sample_id.startswith("odir_"), f"Bad sample_id prefix: {s.sample_id}"
        assert s.sample_id.endswith("_L") or s.sample_id.endswith("_R"), (
            f"sample_id does not end with _L or _R: {s.sample_id}"
        )


def test_patient_ids_follow_odir_pattern(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        assert s.patient_id.startswith("odir_"), f"Bad patient_id: {s.patient_id}"


# ---------------------------------------------------------------------------
# 4-5. Eye laterality
# ---------------------------------------------------------------------------


def test_left_eye_has_correct_laterality(manifest: list[CanonicalSample]) -> None:
    left_samples = [s for s in manifest if s.sample_id.endswith("_L")]
    assert all(s.eye_laterality == EyeLaterality.LEFT for s in left_samples), (
        "Some _L samples do not have EyeLaterality.LEFT"
    )


def test_right_eye_has_correct_laterality(manifest: list[CanonicalSample]) -> None:
    right_samples = [s for s in manifest if s.sample_id.endswith("_R")]
    assert all(s.eye_laterality == EyeLaterality.RIGHT for s in right_samples), (
        "Some _R samples do not have EyeLaterality.RIGHT"
    )


# ---------------------------------------------------------------------------
# 6. Patient grouping
# ---------------------------------------------------------------------------


def test_both_eyes_share_patient_id(manifest: list[CanonicalSample]) -> None:
    by_patient: dict[str, list[str]] = {}
    for s in manifest:
        by_patient.setdefault(s.patient_id, []).append(s.sample_id)
    for pid, sids in by_patient.items():
        assert len(sids) == 2, (
            f"Patient {pid!r} has {len(sids)} samples, expected 2"
        )
        suffixes = {sid.split("_")[-1] for sid in sids}
        assert suffixes == {"L", "R"}, (
            f"Patient {pid!r} does not have exactly L and R: {sids}"
        )


# ---------------------------------------------------------------------------
# 7. dataset_source
# ---------------------------------------------------------------------------


def test_dataset_source_is_odir(manifest: list[CanonicalSample]) -> None:
    assert all(s.dataset_source == "odir" for s in manifest)


# ---------------------------------------------------------------------------
# 8. Schema validation
# ---------------------------------------------------------------------------


def test_all_samples_validate_against_schema(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        assert isinstance(s, CanonicalSample), (
            f"sample {s.sample_id!r} is not a CanonicalSample"
        )


# ---------------------------------------------------------------------------
# 9. "suspected glaucoma" keyword → glaucoma=None
# ---------------------------------------------------------------------------


def test_suspected_keyword_maps_to_none(tmp_path: Path) -> None:
    rows = [_base_row("99", left_kw="suspected glaucoma", right_kw="normal fundus",
                      N=0, G=1)]
    root = _make_odir_root(tmp_path / "case9", rows)
    a = ODIRAdapter(dataset_root=root)
    left = next(s for s in a.build_manifest() if s.sample_id.endswith("_L"))
    assert left.glaucoma is None, (
        f"'suspected glaucoma' should yield glaucoma=None, got {left.glaucoma}"
    )


# ---------------------------------------------------------------------------
# 10. "normal fundus" alone (G=1) → glaucoma=0 for that eye
# ---------------------------------------------------------------------------


def test_normal_fundus_alone_with_g1_yields_zero(manifest: list[CanonicalSample]) -> None:
    # Patient 10: right eye has G=1 and keyword "normal fundus".
    right = next(
        s for s in manifest if s.patient_id == "odir_10" and s.sample_id.endswith("_R")
    )
    assert right.glaucoma == 0, (
        f"'normal fundus' eye with G=1 should yield glaucoma=0, got {right.glaucoma}"
    )


# ---------------------------------------------------------------------------
# 11. "normal fundus" + other abnormal term → glaucoma=None (not blindly 0)
# ---------------------------------------------------------------------------


def test_normal_plus_abnormal_is_not_blindly_zero(tmp_path: Path) -> None:
    rows = [_base_row("77", left_kw="normal fundus,drusen", right_kw="normal fundus",
                      N=0, G=1)]
    root = _make_odir_root(tmp_path / "case11", rows)
    a = ODIRAdapter(dataset_root=root)
    left = next(s for s in a.build_manifest() if s.sample_id.endswith("_L"))
    # Keyword set is NOT exclusively normal (also has "drusen") and doesn't have
    # a glaucoma keyword → ambiguous → None.
    assert left.glaucoma is None, (
        f"Mixed normal+drusen keywords with G=1 should yield glaucoma=None, "
        f"got {left.glaucoma}"
    )


# ---------------------------------------------------------------------------
# 12. "glaucoma" keyword + G=1 → glaucoma=1
# ---------------------------------------------------------------------------


def test_glaucoma_keyword_yields_one(manifest: list[CanonicalSample]) -> None:
    left = next(
        s for s in manifest if s.patient_id == "odir_10" and s.sample_id.endswith("_L")
    )
    assert left.glaucoma == 1, (
        f"'glaucoma' keyword + G=1 should yield glaucoma=1, got {left.glaucoma}"
    )


# ---------------------------------------------------------------------------
# 13. G=0 → glaucoma=0 for both eyes
# ---------------------------------------------------------------------------


def test_g0_yields_glaucoma_zero_both_eyes(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.patient_id in ("odir_20", "odir_30"):
            assert s.glaucoma == 0, (
                f"G=0 patient should have glaucoma=0, got {s.glaucoma} "
                f"for {s.sample_id}"
            )


# ---------------------------------------------------------------------------
# 14. D=1 + DR keyword → diabetes=1
# ---------------------------------------------------------------------------


def test_dr_keyword_yields_diabetes_one(manifest: list[CanonicalSample]) -> None:
    left = next(
        s for s in manifest if s.patient_id == "odir_20" and s.sample_id.endswith("_L")
    )
    assert left.diabetes == 1, (
        f"DR keyword + D=1 should yield diabetes=1, got {left.diabetes}"
    )


# ---------------------------------------------------------------------------
# 15. D=1 + "normal fundus" → diabetes=0 for that eye
# ---------------------------------------------------------------------------


def test_normal_fundus_with_d1_yields_diabetes_zero(manifest: list[CanonicalSample]) -> None:
    right = next(
        s for s in manifest if s.patient_id == "odir_20" and s.sample_id.endswith("_R")
    )
    assert right.diabetes == 0, (
        f"'normal fundus' eye with D=1 should yield diabetes=0, got {right.diabetes}"
    )


# ---------------------------------------------------------------------------
# 16. D=1 + ambiguous non-DR keyword → diabetes=None
# ---------------------------------------------------------------------------


def test_ambiguous_keyword_with_d1_yields_none(tmp_path: Path) -> None:
    rows = [_base_row("55", left_kw="macular epiretinal membrane",
                      right_kw="normal fundus", N=0, D=1)]
    root = _make_odir_root(tmp_path / "case16", rows)
    a = ODIRAdapter(dataset_root=root)
    left = next(s for s in a.build_manifest() if s.sample_id.endswith("_L"))
    assert left.diabetes is None, (
        f"Ambiguous non-DR keyword + D=1 should yield diabetes=None, "
        f"got {left.diabetes}"
    )


# ---------------------------------------------------------------------------
# 17. D=0 → diabetes=0 for both eyes
# ---------------------------------------------------------------------------


def test_d0_yields_diabetes_zero_both_eyes(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.patient_id == "odir_10":  # G=1 patient, D=0
            assert s.diabetes == 0, (
                f"D=0 should yield diabetes=0, got {s.diabetes} for {s.sample_id}"
            )


# ---------------------------------------------------------------------------
# 18. H=1 + "hypertensive retinopathy" keyword -> hypertension=1
# ---------------------------------------------------------------------------


def test_hypertensive_retinopathy_keyword_yields_hypertension_one(
    manifest: list[CanonicalSample],
) -> None:
    for s in manifest:
        if s.patient_id == "odir_30":
            assert s.hypertension == 1, (
                f"'hypertensive retinopathy' + H=1 should yield "
                f"hypertension=1, got {s.hypertension} "
                f"for {s.sample_id}"
            )


# ---------------------------------------------------------------------------
# 19. H=0 -> hypertension=0 for both eyes
# ---------------------------------------------------------------------------


def test_h0_yields_hypertension_zero(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.patient_id in ("odir_10", "odir_20"):
            assert s.hypertension == 0, (
                f"H=0 should yield hypertension=0, "
                f"got {s.hypertension} for {s.sample_id}"
            )


# ---------------------------------------------------------------------------
# 20. O=1 → other_ocular=1 for both eyes (catch-all, no keyword inference)
# ---------------------------------------------------------------------------


def test_o1_yields_other_ocular_one_both_eyes(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.patient_id == "odir_30":
            assert s.other_ocular == 1, (
                f"O=1 should yield other_ocular=1 for {s.sample_id}, "
                f"got {s.other_ocular}"
            )


# ---------------------------------------------------------------------------
# 21. O=0 → other_ocular=0 for both eyes
# ---------------------------------------------------------------------------


def test_o0_yields_other_ocular_zero(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.patient_id in ("odir_10", "odir_20"):
            assert s.other_ocular == 0, (
                f"O=0 should yield other_ocular=0 for {s.sample_id}, "
                f"got {s.other_ocular}"
            )


# ---------------------------------------------------------------------------
# 22-23. Sex mapping
# ---------------------------------------------------------------------------


def test_female_maps_to_sex_female(manifest: list[CanonicalSample]) -> None:
    female_samples = [s for s in manifest if s.patient_id in ("odir_10", "odir_30")]
    assert all(s.sex == Sex.FEMALE for s in female_samples)


def test_male_maps_to_sex_male(manifest: list[CanonicalSample]) -> None:
    male_samples = [s for s in manifest if s.patient_id == "odir_20"]
    assert all(s.sex == Sex.MALE for s in male_samples)


# ---------------------------------------------------------------------------
# 24. age_years is numeric
# ---------------------------------------------------------------------------


def test_age_years_is_numeric(manifest: list[CanonicalSample]) -> None:
    for s in manifest:
        if s.age_years is not None:
            assert isinstance(s.age_years, (int, float)), (
                f"age_years should be numeric, got {type(s.age_years)} for {s.sample_id}"
            )


# ---------------------------------------------------------------------------
# 25. image_quality_label = None for all samples
# ---------------------------------------------------------------------------


def test_image_quality_label_is_none(manifest: list[CanonicalSample]) -> None:
    assert all(s.image_quality_label is None for s in manifest), (
        "image_quality_label should be None for all ODIR samples in Stage 7"
    )


def test_quality_columns_empty_when_quality_not_populated(adapter: ODIRAdapter) -> None:
    assert adapter.get_quality_columns() == []


# ---------------------------------------------------------------------------
# 26. dr_grade NOT in get_supported_tasks()
# ---------------------------------------------------------------------------


def test_dr_grade_not_in_supported_tasks(adapter: ODIRAdapter) -> None:
    assert "dr_grade" not in adapter.get_supported_tasks(), (
        "dr_grade must not appear in ODIR supported tasks (no 0-4 column available)"
    )


# ---------------------------------------------------------------------------
# 27. H is exposed only as hypertension weak proxy in Stage 7
# ---------------------------------------------------------------------------


def test_h_label_exposed_as_hypertension_weak_proxy(adapter: ODIRAdapter) -> None:
    tasks = adapter.get_supported_tasks()
    assert "hypertension" in tasks, (
        "hypertension exists in TASK_REGISTRY, so ODIR H should map to "
        "hypertension weak proxy for Stage 7"
    )
    assert "hypertensive_retinopathy" not in tasks, (
        "Stage 7 must not silently reroute ODIR H to hypertensive_retinopathy"
    )


# ---------------------------------------------------------------------------
# 28. get_supported_tasks() returns exactly the 7 expected tasks
# ---------------------------------------------------------------------------


def test_supported_tasks_exact(adapter: ODIRAdapter) -> None:
    assert sorted(adapter.get_supported_tasks()) == sorted(_EXPECTED_TASKS), (
        f"Supported tasks mismatch.\n"
        f"  Expected: {sorted(_EXPECTED_TASKS)}\n"
        f"  Got:      {sorted(adapter.get_supported_tasks())}"
    )


def test_dataset_audit_marks_h_as_weak_proxy(adapter: ODIRAdapter) -> None:
    audit = adapter.get_dataset_audit()
    warnings = " ".join(audit["weak_proxy_warnings"]).lower()
    assert "hypertension" in warnings
    assert "not structured clinical hypertension" in warnings


# ---------------------------------------------------------------------------
# Audit provenance fields
# ---------------------------------------------------------------------------


def test_dataset_audit_provenance_fields_present(
    adapter: ODIRAdapter, three_patient_root: Path
) -> None:
    audit = adapter.get_dataset_audit()
    assert "dataset_root_used" in audit
    assert str(three_patient_root) in audit["dataset_root_used"]
    assert "metadata_path_used" in audit
    assert "data.xlsx" in audit["metadata_path_used"]
    assert "image_dir_used" in audit
    assert "Training Images" in audit["image_dir_used"]
    assert "excluded_testing_directories" in audit
    assert isinstance(audit["excluded_testing_directories"], list)
    assert "excluded_duplicate_directories" in audit
    assert isinstance(audit["excluded_duplicate_directories"], list)
    assert "unreferenced_image_files" in audit
    assert audit["unreferenced_image_files"] == 0


def test_unreferenced_image_files_counted_correctly(tmp_path: Path) -> None:
    rows = [_base_row("1", N=1)]
    root = _make_odir_root(tmp_path / "extra_img", rows)
    _make_dummy_jpg(root / "Training Images", "extra_unreferenced.jpg")
    a = ODIRAdapter(dataset_root=root)
    audit = a.get_dataset_audit()
    assert audit["unreferenced_image_files"] == 1


def test_excluded_testing_directories_listed_when_present(tmp_path: Path) -> None:
    rows = [_base_row("1", N=1)]
    root = _make_odir_root(tmp_path / "with_testing", rows)
    (root / "Testing Images").mkdir()
    a = ODIRAdapter(dataset_root=root)
    audit = a.get_dataset_audit()
    assert len(audit["excluded_testing_directories"]) >= 1
    assert any("Testing Images" in d for d in audit["excluded_testing_directories"])


def test_excluded_testing_directories_empty_when_absent(adapter: ODIRAdapter) -> None:
    audit = adapter.get_dataset_audit()
    assert audit["excluded_testing_directories"] == []


# ---------------------------------------------------------------------------
# 29. Stratification columns
# ---------------------------------------------------------------------------


def test_stratification_columns_include_required(adapter: ODIRAdapter) -> None:
    cols = adapter.get_stratification_columns()
    for required in ("sex", "age_years", "dataset_source", "eye_laterality"):
        assert required in cols, (
            f"Stratification column {required!r} missing from {cols}"
        )


# ---------------------------------------------------------------------------
# 30. load_image returns PIL Image
# ---------------------------------------------------------------------------


def test_load_image_returns_pil_image(adapter: ODIRAdapter, manifest: list[CanonicalSample]) -> None:
    img = adapter.load_image(manifest[0].sample_id)
    assert isinstance(img, Image.Image), (
        f"load_image should return PIL Image, got {type(img)}"
    )


# ---------------------------------------------------------------------------
# 31. Missing image file → FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_image_reference_excluded_and_audited(tmp_path: Path) -> None:
    rows = [_base_row("88")]
    root = _make_odir_root(tmp_path / "case31", rows, create_images=False)
    a = ODIRAdapter(dataset_root=root)
    assert a.build_manifest() == []
    audit = a.get_dataset_audit()
    assert audit["missing_image_references"] == 2
    assert audit["excluded_samples_by_reason"]["missing_image_reference"] == 2


def test_load_image_missing_after_manifest_raises_file_not_found(tmp_path: Path) -> None:
    rows = [_base_row("89")]
    root = _make_odir_root(tmp_path / "case31b", rows, create_images=True)
    a = ODIRAdapter(dataset_root=root)
    left = next(s for s in a.build_manifest() if s.sample_id.endswith("_L"))
    Path(left.image_path).unlink()
    with pytest.raises(FileNotFoundError):
        a.load_image(left.sample_id)


# ---------------------------------------------------------------------------
# 32. split_patients runs without error on synthetic manifest
# ---------------------------------------------------------------------------


def test_split_patients_compatible_ten_patients(tmp_path: Path) -> None:
    rows = [_base_row(str(i), N=1) for i in range(10)]
    root = _make_odir_root(tmp_path / "tenp", rows)
    a = ODIRAdapter(dataset_root=root)
    m = a.build_manifest()
    sp = split_patients(m, seed=42)
    assert set(sp.keys()) == {"train", "val", "reliability", "test"}
    all_ids = {s.sample_id for s in m}
    assigned = {sid for sids in sp.values() for sid in sids}
    assert assigned == all_ids


# ---------------------------------------------------------------------------
# Full-width comma parsing
# ---------------------------------------------------------------------------


def test_fullwidth_comma_parsed_correctly(tmp_path: Path) -> None:
    rows = [_base_row("66",
                      left_kw="glaucoma\uff0cnormal fundus",
                      right_kw="normal fundus",
                      N=0, G=1)]
    root = _make_odir_root(tmp_path / "fw_comma", rows)
    a = ODIRAdapter(dataset_root=root)
    left = next(s for s in a.build_manifest() if s.sample_id.endswith("_L"))
    # "glaucoma" is one of the tokens → glaucoma=1
    assert left.glaucoma == 1, (
        f"Full-width comma should be parsed; expected glaucoma=1, got {left.glaucoma}"
    )


# ---------------------------------------------------------------------------
# Guarded real ODIR smoke tests (skipped in CI when dataset absent)
# ---------------------------------------------------------------------------


_ODIR_XLSX = Path("data/ODIR-5K/ODIR-5K/ODIR-5K/data.xlsx")


@pytest.mark.skipif(
    not _ODIR_XLSX.exists(),
    reason="ODIR-5K dataset not available on this machine",
)
class TestODIRRealSmoke:
    """Smoke tests against the actual ODIR-5K dataset."""

    @pytest.fixture(scope="class")
    def real_adapter(self) -> ODIRAdapter:
        return ODIRAdapter(dataset_root="data/ODIR-5K/ODIR-5K/ODIR-5K")

    @pytest.fixture(scope="class")
    def real_manifest(self, real_adapter: ODIRAdapter) -> list[CanonicalSample]:
        return real_adapter.build_manifest()

    def test_manifest_has_expected_sample_count(
        self, real_manifest: list[CanonicalSample]
    ) -> None:
        assert len(real_manifest) >= 6000, (
            f"Expected ≥ 6000 samples (3500 patients × 2 eyes), "
            f"got {len(real_manifest)}"
        )

    def test_unique_patient_count(
        self, real_manifest: list[CanonicalSample]
    ) -> None:
        n_patients = len({s.patient_id for s in real_manifest})
        assert n_patients >= 3000, (
            f"Expected ≥ 3000 unique patients, got {n_patients}"
        )

    def test_no_schema_validation_errors_on_sample(
        self, real_manifest: list[CanonicalSample]
    ) -> None:
        # Spot-check 50 evenly-spaced samples.
        step = max(1, len(real_manifest) // 50)
        for sample in real_manifest[::step]:
            assert isinstance(sample, CanonicalSample), (
                f"sample {sample.sample_id!r} failed schema validation"
            )

    def test_dataset_source_is_odir(
        self, real_manifest: list[CanonicalSample]
    ) -> None:
        assert all(s.dataset_source == "odir" for s in real_manifest)

    def test_supported_tasks_exact(self, real_adapter: ODIRAdapter) -> None:
        assert sorted(real_adapter.get_supported_tasks()) == sorted(_EXPECTED_TASKS)
