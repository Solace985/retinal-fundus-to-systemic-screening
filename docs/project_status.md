# Project Status

Last updated: 2026-05-05

## Current Stage

Stage 7.5 — Documentation, dataset inventory, privacy policy, and planner correction
completed after this documentation patch / ready to commit.

## Most Recently Completed Stage

Stage 7 — ODIR real-dataset smoke. All Stage 7 tests and gates passed in the final Stage 7 report.

## Next Stage

Stage 8A — Real foundation backbone integration (DINOv2-Large, ConvNeXt-Base, ResNet-50; RETFound pending weights).

## Implemented Adapters

- ODIR-5K (`src/retina_screen/adapters/odir.py`) — Stage 7, accepted

## Locally Present, Not Integrated

- BRSET (`data/brset/`) — primary scientific dataset; adapter planned Stage 8C
- mBRSET (`data/mbrset/`) — cross-device validation; adapter planned Stage 8F
- APTOS 2019 (`data/aptos2019/`) — external DR validation; planned Stage 8G
- IDRiD (`data/idrid/`) — external DR + DME + lesion validation; planned Stage 8G
- Messidor-2 (`data/messidor2/`) — external DR validation; planned Stage 8G
- EyePACS DR dataset (`data/eyepacs/`) — large-scale external DR; planned Stage 8G
- EyePACS-AIROGS-light-V2 (`data/eyepac-light-v2-512-jpg/`) — glaucoma benchmark; planned Stage 8G
- RFMiD (`data/rfmid/`) — TCAV concept source + secondary ocular validation; planned Stage 8G

## Implemented Backbones

- Mock backbone only (`configs/backbone/mock.yaml`)

## Pending Backbones

- DINOv2-Large
- ConvNeXt-Base
- ResNet-50
- RETFound (pending weight availability)

## Stage 7 Tests and Gates

Stage 7 tests and gates passed in the final Stage 7 report.

## Current Blockers

None. Stage 8A may begin after Stage 7.5 documentation is accepted and committed.

## Resolved Documentation Risk

`docs/ai_context/03_file_generation_order.md` was explicitly approved for a narrow
protected-file update and now marks the old post-Stage-7 order as superseded by
Decisions 019-022 and `docs/mvp_build_order.md`.
