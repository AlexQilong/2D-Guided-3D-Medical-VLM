# Pipeline Overview

The README covers the method and end-to-end commands. This page maps each stage
to the code that implements it, so you know where to look when adapting the pipeline.

## Track 1 — Pseudo reports (M3D-Cap)

| Stage | What it does | Code |
|-------|--------------|------|
| Slice selection + per-slice report | Extracts one slice per plane (axial/sagittal/coronal) and prompts Qwen3-VL for a FINDINGS/IMPRESSION report. Selection strategy is `middle` (default), `variance`, or `mip`. | `pseudo_labels/reports/ct_report_generator.py` |
| Scan-level summary | Feeds the three per-plane reports back to the same model (text-only) to produce one consolidated pseudo-report. | `pseudo_labels/reports/ct_summary_generator.py` |

Decoding is deterministic (`do_sample=False`) so pseudo-labels are reproducible.
The output `summaries.json` maps `case_id + scan_type` → pseudo-report and is the
supervision signal for caption training.

## Track 2 — Pseudo masks (Duke MRI)

| Stage | What it does | Code |
|-------|--------------|------|
| Per-slice bbox | Prompts Qwen2-VL for one JSON bounding box around the target string; absent targets are skipped. Boxes are validated by area/dimension constraints. | `pseudo_labels/masks/qwen_bbox.py`, `pseudo_labels/masks/utils.py` |
| Mask refinement | Uses each box as a prompt to MedSAM (or SAM ViT-H), stacking slice masks into a 3D volume. | `pseudo_labels/masks/medsam_mask.py` |
| Orchestration | Runs both stages over a folder of volumes; writes per-volume NPZ (masks, boxes, quality codes) + JSON metadata. | `pseudo_labels/masks/process_volumes.py` |

`pseudo_labels/masks/evaluate_pseudo_quality.py` scores pseudo-masks against GT
(Dice / IoU) when you want to sanity-check label quality before training.

### SAM variant

`MedSAMMaskGenerator` wraps `sam_model_registry`. `model_type="vit_b"` + the MedSAM
checkpoint is the paper default and what the tables report; `model_type="vit_h"` + the
SAM ViT-H checkpoint is the higher-quality alternative. Swap via the last two arguments
of `scripts/02_generate_pseudo_masks.sh`.

## 3D student

The student is a vendored fork of [M3D](https://github.com/BAAI-DCAI/M3D)'s LaMed-Phi-3
(`lamed/`). A new `[SEG]` token is added to the tokenizer and randomly initialized; its
hidden state drives the SegVol decoder. Training jointly optimizes:

- **caption**: cross-entropy on pseudo-report tokens — `lamed/train/train.py`
- **segmentation**: cross-entropy + BCE + Dice over the SegVol output, with the pseudo
  (or GT) mask as target — `lamed/train/train_text_guided_refseg.py`

See [training.md](training.md) for hyperparameters and [evaluation.md](evaluation.md)
for metrics.
