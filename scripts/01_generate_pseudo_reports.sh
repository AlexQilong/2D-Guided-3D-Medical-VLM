#!/bin/bash
# Generate pseudo reports for 3D CT volumes using Qwen3-VL-8B-Instruct.
#
# Stage 1: 3-view (axial/sagittal/coronal) slice-level reports
# Stage 2: Aggregate the 3 view reports into one scan-level summary
#
# Inputs:
#   - CT volumes as .npy files (shape: D x H x W or 1 x D x H x W)
# Outputs:
#   - Slice-level reports JSON
#   - Scan-level summaries JSON (used to train the 3D VLM)
#
# Usage:
#   bash scripts/01_generate_pseudo_reports.sh \
#       /path/to/ct_volumes_dir \
#       /path/to/output_slice_reports.json \
#       /path/to/output_summaries.json

set -e

CT_DIR=${1:-./data/M3D_Cap_npy/ct_case}
SLICE_REPORTS=${2:-./outputs/m3d_cap_slice_reports_qwen3.json}
SUMMARIES=${3:-./outputs/m3d_cap_summaries_qwen3.json}

mkdir -p "$(dirname "$SLICE_REPORTS")" "$(dirname "$SUMMARIES")"

echo "==================================================================="
echo "Stage 1: Generating 3-view slice reports with Qwen3-VL-8B-Instruct"
echo "==================================================================="
echo "Input CT dir: $CT_DIR"
echo "Output: $SLICE_REPORTS"

python pseudo_labels/reports/ct_report_generator.py \
    --ct_dir "$CT_DIR" \
    --output "$SLICE_REPORTS" \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --selection_strategy middle \
    --slices_per_plane 1

echo ""
echo "==================================================================="
echo "Stage 2: Summarizing 3-view reports into scan-level pseudo-reports"
echo "==================================================================="
echo "Input slice reports: $SLICE_REPORTS"
echo "Output summaries: $SUMMARIES"

python pseudo_labels/reports/ct_summary_generator.py \
    --input "$SLICE_REPORTS" \
    --output "$SUMMARIES" \
    --model "Qwen/Qwen3-VL-8B-Instruct"

echo ""
echo "Done. Pseudo-reports ready at: $SUMMARIES"
