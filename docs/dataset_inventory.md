# Dataset Inventory

## Purpose

This file records local dataset availability, safe root paths, dataset roles, access/licensing
handling, and implementation status for the retinal screening pipeline.

This file must NOT contain:
- Patient IDs or sample-level identifiers
- Raw metadata row samples or dumps
- Exhaustive file listings (image filenames, etc.)
- Definitive license claims not confirmed from local or official documentation

---

## Global Data-Handling Policy

All datasets under `data/` are local-only and gitignored. None are committed to the repository.

- Raw images, raw metadata, patient-level manifests, split files with sample IDs, embedding
  caches, and run outputs must never be committed.
- This applies to all datasets — public and credentialed alike. Public availability does not
  mean commit-safe (size, licensing, reproducibility-from-source).
- BRSET and mBRSET are credentialed/private (PhysioNet) and require extra care: see Decision 018.
- Licensing terms must be confirmed from local dataset LICENSE files or official dataset
  documentation before any paper submission, model publication, or external sharing.

**Allowed to commit:** adapter code, config templates, documentation, synthetic test fixtures.

**Not allowed to commit:** raw images, raw metadata, patient IDs, generated caches, splits,
runs, or outputs containing private identifiers.

Environment variables for dataset roots:
- ODIR: `RETINA_SCREEN_ODIR_ROOT` (optional; config default used if not set)
- BRSET: `RETINA_SCREEN_BRSET_ROOT` (required when adapter is implemented)
- mBRSET: `RETINA_SCREEN_MBRSET_ROOT` (required when adapter is implemented)

---

## Dataset Table

| Dataset | Local root | Access / licensing | Impl status | Planned role | Entry stage | Raw data committed |
|---------|-----------|-------------------|-------------|-------------|-------------|-------------------|
| ODIR-5K | `data/ODIR-5K/ODIR-5K/ODIR-5K/` | Open-access per project docs (verify from ODIR source) | Stage 7 integrated and accepted | Engineering smoke + auxiliary ocular benchmark | 7 | No |
| BRSET | `data/brset/` | Credentialed via PhysioNet; see `data/brset/LICENSE.txt` — confirm terms before use/sharing | Local only, not integrated | Primary scientific dataset after adapter implementation | 8C | No |
| mBRSET | `data/mbrset/` | Credentialed via PhysioNet; see `data/mbrset/LICENSE.txt` — confirm terms before use/sharing | Local only, not integrated | Cross-device/smartphone validation + continual-learning stream candidate | 8F | No |
| APTOS 2019 | `data/aptos2019/` | See dataset source for license terms | Local only, not integrated | External DR validation | 8G | No |
| IDRiD | `data/idrid/` | CC BY 4.0 observed in subdir license files (verify from IDRiD source before citing) | Local only, not integrated | External DR + DME + lesion segmentation validation | 8G | No |
| Messidor-2 | `data/messidor2/` | See dataset source for license terms | Local only, not integrated | External DR validation | 8G | No |
| EyePACS DR dataset | `data/eyepacs/` | See dataset source (Kaggle/EyePACS) for license | Local only, not integrated; observed layout (Images/ + split label CSVs) suggests a preprocessed DR variant — exact variant to confirm during Stage 8G preflight | Large-scale external DR validation | 8G | No |
| EyePACS-AIROGS-light-V2 | `data/eyepac-light-v2-512-jpg/` | See dataset source for license terms | Local only, not integrated | Glaucoma external benchmark | 8G | No |
| RFMiD | `data/rfmid/` | CC BY 4.0 observed in subdir license files (verify from RFMiD source before citing) | Local only, not integrated | TCAV concept source + secondary external ocular validation (NOT a continual-learning stream — Decision 011) | 8G | No |

---

## ODIR-5K Notes

ODIR-5K is the Stage 7 integration target and ongoing auxiliary ocular benchmark.

**Active root:** `data/ODIR-5K/ODIR-5K/ODIR-5K/`
(Triple-nested archive structure; innermost directory is the actual data root.)

**Used files:**
- `data.xlsx` — patient metadata and one-hot disease labels
- `Training Images/` — 7,000 fundus images (left and right eye per patient)

**Excluded from pipeline use:**
- `Testing Images/` — 1,000 unlabelled test images; no labels available
- `data/ODIR-5K/full_df.csv` — Kaggle-internal metadata with absolute paths; unusable
- `data/ODIR-5K/preprocessed_images/` — partial preprocessed set (6,392/7,000); not used
- Outer wrapper directories `data/ODIR-5K/` and `data/ODIR-5K/ODIR-5K/` — duplicate nesting

**Root-level stray artifacts (gitignored):**
- `full_df.csv` at repo root — extracted to wrong location; gitignored
- `preprocessed_images/` at repo root — extracted to wrong location; gitignored

---

## AIROGS Note

A standalone AIROGS directory was not found in `data/`. Glaucoma lightweight coverage is
provided by `data/eyepac-light-v2-512-jpg/` (EyePACS-AIROGS-light-V2). This will be
confirmed during Stage 8G preflight.

---

## Dataset Availability Summary (as of 2026-05-05)

All currently planned local datasets are present except standalone AIROGS. Glaucoma
lightweight coverage is provided by EyePACS-AIROGS-light-V2 (confirmed present).

Note: Exact dataset variants (EyePACS DR variant, RFMiD metadata root layout) must be
confirmed during each dataset's Stage 8G preflight. Do not treat this summary as a lock
on exact dataset structures — only on physical presence.

See Decision 019 for the locked record of dataset availability.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| ODIR-5K adapter | Implemented and accepted (Stage 7) |
| BRSET adapter | Not yet implemented (planned Stage 8C) |
| mBRSET adapter | Not yet implemented (planned Stage 8F) |
| External dataset adapters (APTOS, IDRiD, Messidor-2, EyePACS, RFMiD) | Not yet implemented (planned Stage 8G) |
| Real foundation backbones (DINOv2, ConvNeXt, ResNet-50, RETFound) | Not yet implemented (mock backbone only; planned Stage 8A) |
| Visual diagnostics | Not yet implemented (planned Stage 8E) |
