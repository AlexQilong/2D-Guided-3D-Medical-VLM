#!/bin/bash
# Generate pseudo segmentation masks for 3D medical volumes.
# Pipeline: Qwen2-VL bbox prediction + MedSAM (or SAM ViT-H) mask refinement.
#
# Inputs:
#   - 3D volumes (.npy) in $VOLUME_DIR
#   - Pretrained MedSAM checkpoint (default) or SAM ViT-H
# Outputs:
#   - Per-volume NPZ with pseudo masks at $OUTPUT_DIR
#
# Usage:
#   bash scripts/02_generate_pseudo_masks.sh \
#       /path/to/volumes_dir \
#       /path/to/output_dir \
#       "breast tissue"          # target description for bbox prompting
#       /path/to/medsam_vit_b.pth  # optional: SAM checkpoint

set -e

VOLUME_DIR=${1:-./data/Duke-Breast-Cancer-MRI/preprocessed}
OUTPUT_DIR=${2:-./outputs/duke_pseudo_masks_medsam}
TARGET=${3:-"breast tissue"}
SAM_CKPT=${4:-./checkpoints/medsam_vit_b.pth}
SAM_MODEL_TYPE=${5:-vit_b}   # 'vit_b' for MedSAM, 'vit_h' for SAM ViT-H

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "Generating pseudo segmentation masks"
echo "==================================================================="
echo "Volume dir:  $VOLUME_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Target:      $TARGET"
echo "SAM weights: $SAM_CKPT (model_type=$SAM_MODEL_TYPE)"
echo ""

python pseudo_labels/masks/process_volumes.py \
    --volume_dir "$VOLUME_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --model_id "Qwen/Qwen2-VL-7B-Instruct" \
    --medsam_checkpoint "$SAM_CKPT" \
    --sam_model_type "$SAM_MODEL_TYPE"

echo ""
echo "Done. Pseudo masks saved to: $OUTPUT_DIR"
