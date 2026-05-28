# 2D-Guided 3D Medical VLM

Official code for **"2D-Guided 3D Medical Vision–Language Model"**.

We use off-the-shelf 2D foundation models — **Qwen3-VL-8B** for report generation and **MedSAM** for mask generation — to produce *pseudo* labels from individual CT/MRI slices, then train a 3D medical VLM (LaMed-Phi-3) on those pseudo labels. The result: a 3D VLM that does report generation and referring-expression segmentation **without** any human-annotated 3D supervision (or with very little of it).

This repo lets you reproduce both tracks of the paper:

1. **Report generation on M3D-Cap** — Qwen3-VL-generated reports as pseudo supervision (BERTScore F1, ROUGE-1, Clinical F1)
2. **Segmentation on Duke Breast Cancer MRI** — MedSAM-generated masks combined with limited expert GT (Dice / IoU)

---

## Method overview

```
                  ┌───────────────────────────────┐
                  │   3D volume (CT / MRI .npy)   │
                  └───────────────┬───────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  3 canonical 2D slices      │     │  Per-slice anatomical bbox  │
│  (axial / sagittal / coronal)│    │  via Qwen2-VL prompt        │
│  → Qwen3-VL-8B report each  │     │  → MedSAM mask refinement   │
│  → text summarizer LLM      │     │  → 3D pseudo mask volume    │
└──────────────┬──────────────┘     └──────────────┬──────────────┘
               │                                   │
               ▼                                   ▼
        pseudo report                       pseudo 3D mask
               │                                   │
               └──────────────┬────────────────────┘
                              ▼
              ┌─────────────────────────────────┐
              │   Train 3D LaMed (Phi-3 + ViT3D │
              │   + SegVol) on pseudo labels    │
              └─────────────────────────────────┘
```

---

## Repository layout

```
2D-Guided-3D-Medical-VLM/
├── pseudo_labels/         # ★ Core contribution
│   ├── reports/           # Qwen3-VL slice reports + summarizer
│   └── masks/             # Qwen2-VL bbox + MedSAM mask refinement
├── lamed/                 # 3D VLM: ViT3D + Phi-3 + SegVol
│   ├── model/             # Architecture
│   ├── dataset/           # Multi-task datasets and prompt templates
│   └── train/             # Training scripts (caption + segmentation)
├── eval/                  # BERTScore / ROUGE / Clinical F1 / Dice
├── data_prep/             # Convert datasets into the expected layout
├── scripts/               # End-to-end pipeline runners (numbered 01-06)
├── configs/               # DeepSpeed config
└── docs/                  # Pipeline, data, training, eval guides
```

---

## Installation

```bash
git clone https://github.com/AlexQilong/2D-Guided-3D-Medical-VLM.git
cd 2D-Guided-3D-Medical-VLM

conda create -n med-vlm python=3.10 -y
conda activate med-vlm

# Install PyTorch with a CUDA build matching your driver (example: CUDA 11.8)
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# segment-anything (for SAM / MedSAM mask pipeline)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download pretrained checkpoints into `./checkpoints/`:

| Model | Used for | Source |
|-------|----------|--------|
| `microsoft/Phi-3-mini-4k-instruct` | LLM backbone | HuggingFace (auto-download) |
| `Qwen/Qwen3-VL-8B-Instruct` | 2D pseudo-report VLM | HuggingFace (auto-download) |
| `Qwen/Qwen2-VL-7B-Instruct` | 2D bbox prompter | HuggingFace (auto-download) |
| `GoodBaiBai88/M3D-CLIP` (`pretrained_ViT.bin`) | 3D vision encoder | HuggingFace |
| `GoodBaiBai88/M3D-LaMed-Phi-3-4B` (`mm_projector.bin`) | MM projector init | HuggingFace |
| MedSAM ViT-B (`medsam_vit_b.pth`) | 2D mask refinement | [bowang-lab/MedSAM](https://huggingface.co/wanglab/medsam-vit-base) |

Optional alternative: SAM ViT-H (`sam_vit_h_4b8939.pth`) — slightly higher pseudo-mask quality, see `docs/pipeline_overview.md`.

---

## Data preparation

### M3D-Cap (report generation)

```bash
# Download from HuggingFace
# https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap
# Place .npy volumes at ./data/M3D_Cap_npy/ct_case/
```

### Duke Breast Cancer MRI (segmentation)

Download from TCIA: <https://doi.org/10.7937/TCIA.e3sv-re93>

Place `.npy` volumes at `./data/Duke-Breast-Cancer-MRI/preprocessed/` and GT masks (NRRD) at `./data/Duke-Breast-Cancer-MRI/Segmentation_Masks_NRRD/`.

Then run:

```bash
python data_prep/prepare_duke.py
```

This produces:
- `./finetune_data/duke_volumes/*.npy` (resized to 1×32×256×256)
- `./finetune_data/duke_gt/train.csv` (50 GT samples)
- `./finetune_data/duke_test.csv` (50 held-out test samples)

For GT+pseudo training mixes:

```bash
python data_prep/prepare_train_csv.py --n_pseudo 100  # or 500
```

See [docs/data_preparation.md](docs/data_preparation.md) for full details.

---

## End-to-end usage

### Track 1 — Report generation

```bash
# 1. Generate pseudo reports with Qwen3-VL-8B
bash scripts/01_generate_pseudo_reports.sh \
    ./data/M3D_Cap_npy/ct_case \
    ./outputs/m3d_cap_slice_reports_qwen3.json \
    ./outputs/m3d_cap_summaries_qwen3.json

# 2. Train (one run per condition: 100, 500, 2000, 10000 pseudo samples)
bash scripts/03_train_caption.sh 500 ./outputs/finetune_cap_pseudo500

# 3. Merge LoRA + evaluate
python lamed/utils/merge_lora_weights.py \
    --input ./outputs/finetune_cap_pseudo500 \
    --output ./outputs/finetune_cap_pseudo500-merged

bash scripts/05_eval_caption.sh ./outputs/finetune_cap_pseudo500-merged
```

### Track 2 — Segmentation

```bash
# 1. Generate pseudo masks with MedSAM (Qwen2-VL provides bboxes)
bash scripts/02_generate_pseudo_masks.sh \
    ./data/Duke-Breast-Cancer-MRI/preprocessed \
    ./outputs/duke_pseudo_masks_medsam \
    "breast tissue" \
    ./checkpoints/medsam_vit_b.pth \
    vit_b

# 2. Train (one of four conditions)
bash scripts/04_train_segmentation.sh \
    ./finetune_data/duke_gt_plus_pseudo/train.csv \
    ./outputs/finetune_duke_gt_plus_pseudo

# 3. Evaluate
bash scripts/06_eval_segmentation.sh ./outputs/finetune_duke_gt_plus_pseudo
```

---

## Reproducing paper results

### Table 1 — Report generation (M3D-Cap)

Qwen3-VL pseudo reports on the 500-sample held-out test split:

| Train size | BERTScore F1 ↑ | ROUGE-1 ↑ | Clinical F1 (Micro) ↑ | Clinical F1 (Macro) ↑ |
|------------|----------------|-----------|-----------------------|-----------------------|
| 100   | 0.812 | 0.145 | 0.267 | 0.032 |
| 500   | 0.822 | 0.169 | 0.293 | 0.082 |
| 2000  | 0.829 | 0.186 | 0.266 | 0.083 |
| 10000 | 0.834 | 0.199 | 0.237 | 0.077 |

### Table 5 — Segmentation (Duke Breast Cancer MRI)

See the paper for the full table of conditions (GT-only, Pseudo-only, GT+100 Pseudo, GT+500 Pseudo). All are reproducible with `scripts/04_train_segmentation.sh` + `scripts/06_eval_segmentation.sh`.

---

## Citation

```bibtex
@inproceedings{2dguided3dvlm,
  title={2D-Guided 3D Medical Vision--Language Model},
  author={...},
  year={2026}
}
```

---

## Acknowledgements

This repo builds directly on:
- **[M3D](https://github.com/BAAI-DCAI/M3D)** (Apache 2.0) — base 3D VLM (LaMed-Phi-3, ViT3D, M3D-CLIP)
- **[SegVol](https://github.com/BAAI-DCAI/SegVol)** — 3D segmentation backbone
- **[MedSAM](https://github.com/bowang-lab/MedSAM)** / **[Segment Anything](https://github.com/facebookresearch/segment-anything)** — 2D mask refinement
- **[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)** / **[Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)** — 2D vision–language models

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
