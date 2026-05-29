# Training

The README shows the run commands and conditions. This page lists the
hyperparameters behind the wrapper scripts and the rationale for the starting point.

## Hardware

A single ~80 GB GPU (H100) fits the model in bf16 with batch 4 + grad accum 2.
DeepSpeed ZeRO-2 (`configs/deepspeed_zero2.json`) is provided for multi-GPU scaling.

## Hyperparameters

| | Reports (`03_train_caption.sh`) | Segmentation (`04_train_segmentation.sh`) |
|---|---|---|
| LR | 1e-4 | 5e-6 |
| Epochs | 3 | 5 |
| Batch size | 4 | 4 |
| Grad accumulation | 2 (eff. 8) | 2 (eff. 8) |
| Warmup ratio | 0.05 | 0.03 |
| Scheduler | cosine | cosine |
| Precision | bf16 | bf16 |
| Tuning | LoRA | Full fine-tuning |

Caption conditions sweep the pseudo-sample budget (100 / 500 / 2000 / 10000); only the
training subset size changes. Segmentation conditions vary the supervision mix
(GT-only, pseudo-only, GT+100, GT+500).

For LoRA caption runs, merge the adapter to HF format before evaluation — see the
merge command in the README.

## Starting point

Both tracks start from **base `microsoft/Phi-3-mini-4k-instruct`** plus three
independently-pretrained components: the M3D-CLIP ViT3D vision encoder, the
M3D-LaMed-Phi-3 mm_projector, and the SegVol segmentation head. The paper's claim is that a 3D medical VLM can be trained from a base LLM using only
2D-foundation-model pseudo labels, *without* starting from an existing medical VLM.
