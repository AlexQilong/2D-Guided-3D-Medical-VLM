#!/bin/bash
# Evaluate a trained segmentation model on the held-out Duke test set.
#
# Produces a per-sample CSV with Dice (BinaryDice metric).
#
# Usage:
#   bash scripts/06_eval_segmentation.sh <model_dir> [test_csv] [output_dir]
#   e.g. bash scripts/06_eval_segmentation.sh ./outputs/finetune_duke_gt

set -e

MODEL_DIR=${1:-./outputs/finetune_duke_gt}
TEST_CSV=${2:-./finetune_data/duke_test.csv}
OUTPUT_DIR=${3:-./outputs/eval/$(basename "$MODEL_DIR")}

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "Segmentation eval: $MODEL_DIR"
echo "==================================================================="

python eval/eval_segmentation.py \
    --model_name_or_path "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seg_data_path "$TEST_CSV" \
    --res True \
    --seg_enable True \
    --proj_out_num 256

echo ""
echo "Done. Per-sample Dice: $OUTPUT_DIR/eval_seg.csv"
