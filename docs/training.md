# Training

All training is driven by `scripts/03_train_caption.sh` (reports) and `scripts/04_train_segmentation.sh` (segmentation). The shell scripts are thin wrappers around the Python entry points.

## Hardware

Single GPU is sufficient for the paper's experiments (one ~80 GB H100/H200 fits the model in bf16 with batch 4 + grad accum 2). DeepSpeed ZeRO-2 (`configs/deepspeed_zero2.json`) is provided if you want to scale to multiple GPUs.

## Reports (caption track)

Train one model per pseudo-sample budget. Paper conditions are 100, 500, 2000, 10000.

```bash
for N in 100 500 2000 10000; do
    bash scripts/03_train_caption.sh $N ./outputs/finetune_cap_pseudo${N}
done
```

Default hyperparameters (matching the paper):

| Hyperparameter | Value |
|----------------|-------|
| LR | 1e-4 |
| Epochs | 3 |
| Batch size | 4 |
| Grad accumulation | 2 (effective batch 8) |
| Warmup ratio | 0.05 |
| Scheduler | cosine |
| Precision | bf16 |
| Tuning | LoRA |

After training, merge LoRA → HF format:

```bash
python lamed/utils/merge_lora_weights.py \
    --input ./outputs/finetune_cap_pseudo500 \
    --output ./outputs/finetune_cap_pseudo500-merged
```

## Segmentation track

Train one model per condition (paper Table 5):

```bash
# GT only
bash scripts/04_train_segmentation.sh \
    ./finetune_data/duke_gt/train.csv \
    ./outputs/finetune_duke_gt

# Pseudo only
bash scripts/04_train_segmentation.sh \
    ./finetune_data/duke_pseudo/train.csv \
    ./outputs/finetune_duke_pseudo

# GT + 100 Pseudo
bash scripts/04_train_segmentation.sh \
    ./finetune_data/duke_gt_plus_pseudo/train.csv \
    ./outputs/finetune_duke_gt_plus_pseudo

# GT + 500 Pseudo
bash scripts/04_train_segmentation.sh \
    ./finetune_data/duke_gt_plus_pseudo_500/train.csv \
    ./outputs/finetune_duke_gt_plus_pseudo_500
```

Default hyperparameters (matching the paper):

| Hyperparameter | Value |
|----------------|-------|
| LR | 5e-6 |
| Epochs | 5 |
| Batch size | 4 |
| Grad accumulation | 2 (effective batch 8) |
| Warmup ratio | 0.03 |
| Scheduler | cosine |
| Precision | bf16 |
| Tuning | Full fine-tuning |

## Starting point

Both training scripts start from **base `microsoft/Phi-3-mini-4k-instruct`** plus three independently-pretrained components:
- Vision tower: M3D-CLIP ViT3D
- MM projector: M3D-LaMed-Phi-3-4B mm_projector
- Seg head: SegVol

This is intentional: the paper's claim is that you can train a 3D medical VLM from a base LLM using only 2D-foundation-model pseudo labels — *without* starting from an existing medical VLM.

## Checkpoints

Trained model checkpoints are not released. To reproduce paper results, run training end-to-end with the scripts above.
