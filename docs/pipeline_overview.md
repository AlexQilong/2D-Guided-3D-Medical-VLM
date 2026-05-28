# Pipeline Overview

This document explains how the 2D-guided 3D supervision pipeline works end-to-end.

## High-level idea

3D medical VLMs need annotated 3D volumes for training, but expert annotations are scarce and expensive. We use 2D foundation models — vision–language models for reports and SAM-family models for segmentation — to produce *pseudo* labels per slice. We then aggregate per-slice signals into volume-level supervision and train a 3D VLM on this pseudo-labeled data.

## Track 1 — Pseudo reports (M3D-Cap)

**Inputs**: 3D CT volumes `(D, H, W)` saved as `.npy`.

**Stage 1 — Per-slice reports**: For each volume, we extract one mid-slice from each of three planes (axial, sagittal, coronal). Each slice image is sent to **`Qwen/Qwen3-VL-8B-Instruct`** with a fixed radiology-style prompt to produce a free-text report.

Code: `pseudo_labels/reports/ct_report_generator.py`

**Stage 2 — Volume-level summary**: The three per-plane reports for one volume are concatenated and sent to the same Qwen3-VL (text-only mode) to produce a single scan-level pseudo report.

Code: `pseudo_labels/reports/ct_summary_generator.py`

The resulting JSON (`ct_summaries_qwen3.json`) maps each volume id to its pseudo report. This file is the supervision signal for the 3D VLM caption-training stage.

## Track 2 — Pseudo segmentation masks (Duke MRI)

**Inputs**: 3D MRI volumes `(D, H, W)` saved as `.npy`, plus a *target description* string (e.g. `"breast tissue"` or `"liver tumour"`).

**Stage 1 — Per-slice bbox**: For each slice we prompt **`Qwen/Qwen2-VL-7B-Instruct`** to predict a bounding box around the target structure. Slices where the target is absent return no bbox and are skipped.

Code: `pseudo_labels/masks/qwen_bbox.py`

**Stage 2 — Mask refinement**: Each predicted bbox is used as a prompt to **MedSAM (ViT-B)** which returns a refined binary mask for that slice. The 2D masks are stacked back into a 3D pseudo mask volume aligned with the input.

Code: `pseudo_labels/masks/medsam_mask.py`

The orchestration script that runs both stages over a folder of volumes is `pseudo_labels/masks/process_volumes.py`. Outputs are NPZ files containing per-slice masks, bboxes, scores, and metadata.

### SAM variant choice

The mask generator class `MedSAMMaskGenerator` is a thin wrapper around `segment-anything`'s `sam_model_registry`. It supports:
- `model_type="vit_b"` with the MedSAM checkpoint (paper default, what's reported in the tables)
- `model_type="vit_h"` with the SAM ViT-H checkpoint
- `model_type` adapter combinations for SAM-Med2D

The paper reports results with **MedSAM**, so the default scripts use that. To swap in SAM ViT-H, pass the `vit_h` model type and the corresponding checkpoint to `scripts/02_generate_pseudo_masks.sh`.

## 3D VLM training

The 3D VLM is a vendored fork of [M3D](https://github.com/BAAI-DCAI/M3D)'s LaMed-Phi-3:
- LLM backbone: `microsoft/Phi-3-mini-4k-instruct` (base, not medical-pretrained)
- Vision tower: ViT3D initialized from M3D-CLIP
- Cross-modal projector: SPP projector initialized from M3D-LaMed-Phi-3-4B
- Segmentation head: SegVol

A new `[SEG]` token is added to the LLM tokenizer and randomly initialized. The model is trained jointly to:
- generate reports given an image + question (caption task)
- emit `[SEG]` at the right place + drive the SegVol decoder via the `[SEG]` hidden state (segmentation task)

For caption training, the loss is standard cross-entropy on pseudo-report tokens. For segmentation training, the loss is a sum of LLM cross-entropy + BCE + Dice over the SegVol output, using the pseudo mask (or GT mask) as target.

Training scripts:
- `lamed/train/train.py` (caption + VQA)
- `lamed/train/train_text_guided_refseg.py` (referring-expression segmentation)

Both ship as shell wrappers in `scripts/03_train_caption.sh` and `scripts/04_train_segmentation.sh`.

## Evaluation

- **Captions**: BERTScore F1, ROUGE-1, BLEU, METEOR per sample; Clinical F1 (Micro + Macro) over the full split using a configurable findings ontology (`eval/clinical_f1_labels.yaml`).
- **Segmentation**: BinaryDice over the full test set.

The two evaluators (`eval/eval_caption_standalone.py`, `eval/eval_segmentation.py`) are wrapped by `scripts/05_eval_caption.sh` and `scripts/06_eval_segmentation.sh`.
