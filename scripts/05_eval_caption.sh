#!/bin/bash
# Evaluate a trained report-generation model on M3D-Cap test split.
#
# Produces a per-sample CSV with BERTScore F1, ROUGE-1, BLEU, METEOR,
# then runs Clinical F1 (Micro + Macro) over the same CSV.
#
# Usage:
#   bash scripts/05_eval_caption.sh <model_dir> [output_csv]
#   e.g. bash scripts/05_eval_caption.sh ./outputs/finetune_cap_pseudo500-merged

set -e

MODEL_DIR=${1:-./outputs/finetune_cap_pseudo500-merged}
OUTPUT_CSV=${2:-./outputs/eval/$(basename "$MODEL_DIR")_caption.csv}
CLINICAL_F1_OUT=${3:-./outputs/eval/$(basename "$MODEL_DIR")_clinical_f1.csv}

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "==================================================================="
echo "Caption eval: $MODEL_DIR"
echo "==================================================================="

# eval_caption_standalone.py uses bert-score + rouge-score directly
# (avoids broken `evaluate` library on HF Hub auth changes).
python eval/eval_caption_standalone.py \
    --model_name_or_path "$MODEL_DIR" \
    --output_dir "$(dirname "$OUTPUT_CSV")" \
    --file_name "$(basename "$OUTPUT_CSV")" \
    --proj_out_num 256

echo ""
echo "Computing Clinical F1..."
python eval/clinical_f1.py \
    --input "$OUTPUT_CSV" \
    --reference-field "Ground Truth" \
    --generated-field "pred" \
    --output-label-metrics "$CLINICAL_F1_OUT"

echo ""
echo "Done. Per-sample metrics: $OUTPUT_CSV"
echo "      Clinical F1 table: $CLINICAL_F1_OUT"
