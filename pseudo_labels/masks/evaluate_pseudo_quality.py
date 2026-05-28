#!/usr/bin/env python3
"""
Evaluate pseudo-masks against ground truth segmentation masks.
Computes IoU and Dice metrics for both Duke and M3D_RefSeg datasets.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import nrrd

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_metrics(pred_mask, gt_mask):
    """Compute IoU and Dice between prediction and ground truth."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0

    return {
        'iou': float(iou),
        'dice': float(dice),
        'intersection': int(intersection),
        'union': int(union),
        'pred_pixels': int(pred_sum),
        'gt_pixels': int(gt_sum)
    }


def load_duke_gt_mask(patient_id, gt_masks_dir):
    """Load Duke GT mask from NRRD file."""
    # Find the mask file
    mask_dir = Path(gt_masks_dir)
    breast_mask_pattern = f"*{patient_id}*_Breast.seg.nrrd"

    mask_files = list(mask_dir.rglob(breast_mask_pattern))
    if not mask_files:
        return None

    # Load NRRD mask
    mask_data, header = nrrd.read(str(mask_files[0]))

    # NRRD masks may need transposition to match volume orientation
    # Typically (H, W, D) -> (D, H, W)
    if mask_data.ndim == 3:
        mask_data = np.transpose(mask_data, (2, 0, 1))

    return (mask_data > 0).astype(np.uint8)


def load_m3d_gt_mask(patient_id, gt_dir):
    """Load M3D_RefSeg GT mask from npy file."""
    mask_path = os.path.join(gt_dir, patient_id, 'mask.npy')
    if not os.path.exists(mask_path):
        return None

    mask = np.load(mask_path)
    if mask.ndim == 4:
        mask = mask[0]  # Remove batch dimension

    return (mask > 0).astype(np.uint8)


def evaluate_duke(pseudo_dir, gt_masks_dir, output_file):
    """Evaluate Duke pseudo-masks against GT."""
    results = []

    # Find all pseudo-mask files
    pseudo_files = list(Path(pseudo_dir).glob("*_pseudo_masks.npz"))
    print(f"Found {len(pseudo_files)} pseudo-mask files in Duke output")

    for pseudo_file in sorted(pseudo_files):
        patient_id = pseudo_file.stem.replace('_pseudo_masks', '')

        # Load pseudo-masks
        data = np.load(pseudo_file)
        pseudo_masks = data['masks']  # (D, H, W)

        # Load GT mask
        gt_mask = load_duke_gt_mask(patient_id, gt_masks_dir)

        if gt_mask is None:
            print(f"WARNING: GT not found for {patient_id}")
            continue

        # Handle shape mismatch
        if pseudo_masks.shape != gt_mask.shape:
            print(f"WARNING: Shape mismatch for {patient_id}: pseudo={pseudo_masks.shape}, gt={gt_mask.shape}")
            # Try to align by resizing or cropping
            min_d = min(pseudo_masks.shape[0], gt_mask.shape[0])
            min_h = min(pseudo_masks.shape[1], gt_mask.shape[1])
            min_w = min(pseudo_masks.shape[2], gt_mask.shape[2])
            pseudo_masks = pseudo_masks[:min_d, :min_h, :min_w]
            gt_mask = gt_mask[:min_d, :min_h, :min_w]

        # Compute 3D metrics (over entire volume)
        metrics_3d = compute_metrics(pseudo_masks, gt_mask)

        # Compute 2D metrics per slice (only where GT exists)
        slice_metrics = []
        for d in range(pseudo_masks.shape[0]):
            if gt_mask[d].sum() > 0:  # Only evaluate slices with GT
                m = compute_metrics(pseudo_masks[d], gt_mask[d])
                slice_metrics.append(m)

        avg_slice_iou = np.mean([m['iou'] for m in slice_metrics]) if slice_metrics else 0.0
        avg_slice_dice = np.mean([m['dice'] for m in slice_metrics]) if slice_metrics else 0.0

        result = {
            'patient_id': patient_id,
            'volume_3d': metrics_3d,
            'per_slice_avg': {
                'iou': float(avg_slice_iou),
                'dice': float(avg_slice_dice),
                'num_gt_slices': len(slice_metrics)
            }
        }
        results.append(result)

        print(f"{patient_id}: 3D IoU={metrics_3d['iou']:.3f}, Dice={metrics_3d['dice']:.3f} | "
              f"2D Avg IoU={avg_slice_iou:.3f}, Dice={avg_slice_dice:.3f} ({len(slice_metrics)} slices)")

    # Compute overall statistics
    all_3d_iou = [r['volume_3d']['iou'] for r in results]
    all_3d_dice = [r['volume_3d']['dice'] for r in results]
    all_2d_iou = [r['per_slice_avg']['iou'] for r in results]
    all_2d_dice = [r['per_slice_avg']['dice'] for r in results]

    summary = {
        'dataset': 'Duke-Breast-Cancer-MRI',
        'num_patients': len(results),
        'overall': {
            '3d_iou_mean': float(np.mean(all_3d_iou)) if all_3d_iou else 0,
            '3d_iou_std': float(np.std(all_3d_iou)) if all_3d_iou else 0,
            '3d_dice_mean': float(np.mean(all_3d_dice)) if all_3d_dice else 0,
            '3d_dice_std': float(np.std(all_3d_dice)) if all_3d_dice else 0,
            '2d_iou_mean': float(np.mean(all_2d_iou)) if all_2d_iou else 0,
            '2d_iou_std': float(np.std(all_2d_iou)) if all_2d_iou else 0,
            '2d_dice_mean': float(np.mean(all_2d_dice)) if all_2d_dice else 0,
            '2d_dice_std': float(np.std(all_2d_dice)) if all_2d_dice else 0,
        },
        'per_patient': results,
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Duke Overall Results ===")
    print(f"3D IoU: {summary['overall']['3d_iou_mean']:.3f} ± {summary['overall']['3d_iou_std']:.3f}")
    print(f"3D Dice: {summary['overall']['3d_dice_mean']:.3f} ± {summary['overall']['3d_dice_std']:.3f}")
    print(f"2D IoU: {summary['overall']['2d_iou_mean']:.3f} ± {summary['overall']['2d_iou_std']:.3f}")
    print(f"2D Dice: {summary['overall']['2d_dice_mean']:.3f} ± {summary['overall']['2d_dice_std']:.3f}")
    print(f"Results saved to {output_file}")

    return summary


def evaluate_m3d_refseg(pseudo_dir, gt_dir, output_file):
    """Evaluate M3D_RefSeg pseudo-masks against GT."""
    results = []

    # Find all pseudo-mask files
    pseudo_files = list(Path(pseudo_dir).glob("*_pseudo_masks.npz"))
    print(f"Found {len(pseudo_files)} pseudo-mask files in M3D_RefSeg output")

    for pseudo_file in sorted(pseudo_files):
        patient_id = pseudo_file.stem.replace('_pseudo_masks', '')

        # Load pseudo-masks
        data = np.load(pseudo_file)
        pseudo_masks = data['masks']  # (D, H, W)

        # Load GT mask
        gt_mask = load_m3d_gt_mask(patient_id, gt_dir)

        if gt_mask is None:
            print(f"WARNING: GT not found for {patient_id}")
            continue

        # Handle shape mismatch
        if pseudo_masks.shape != gt_mask.shape:
            print(f"WARNING: Shape mismatch for {patient_id}: pseudo={pseudo_masks.shape}, gt={gt_mask.shape}")
            min_d = min(pseudo_masks.shape[0], gt_mask.shape[0])
            min_h = min(pseudo_masks.shape[1], gt_mask.shape[1])
            min_w = min(pseudo_masks.shape[2], gt_mask.shape[2])
            pseudo_masks = pseudo_masks[:min_d, :min_h, :min_w]
            gt_mask = gt_mask[:min_d, :min_h, :min_w]

        # Compute 3D metrics
        metrics_3d = compute_metrics(pseudo_masks, gt_mask)

        # Compute 2D metrics per slice
        slice_metrics = []
        for d in range(pseudo_masks.shape[0]):
            if gt_mask[d].sum() > 0:
                m = compute_metrics(pseudo_masks[d], gt_mask[d])
                slice_metrics.append(m)

        avg_slice_iou = np.mean([m['iou'] for m in slice_metrics]) if slice_metrics else 0.0
        avg_slice_dice = np.mean([m['dice'] for m in slice_metrics]) if slice_metrics else 0.0

        result = {
            'patient_id': patient_id,
            'volume_3d': metrics_3d,
            'per_slice_avg': {
                'iou': float(avg_slice_iou),
                'dice': float(avg_slice_dice),
                'num_gt_slices': len(slice_metrics)
            }
        }
        results.append(result)

        print(f"{patient_id}: 3D IoU={metrics_3d['iou']:.3f}, Dice={metrics_3d['dice']:.3f} | "
              f"2D Avg IoU={avg_slice_iou:.3f}, Dice={avg_slice_dice:.3f} ({len(slice_metrics)} slices)")

    # Compute overall statistics
    all_3d_iou = [r['volume_3d']['iou'] for r in results]
    all_3d_dice = [r['volume_3d']['dice'] for r in results]
    all_2d_iou = [r['per_slice_avg']['iou'] for r in results]
    all_2d_dice = [r['per_slice_avg']['dice'] for r in results]

    summary = {
        'dataset': 'M3D_RefSeg',
        'num_patients': len(results),
        'overall': {
            '3d_iou_mean': float(np.mean(all_3d_iou)) if all_3d_iou else 0,
            '3d_iou_std': float(np.std(all_3d_iou)) if all_3d_iou else 0,
            '3d_dice_mean': float(np.mean(all_3d_dice)) if all_3d_dice else 0,
            '3d_dice_std': float(np.std(all_3d_dice)) if all_3d_dice else 0,
            '2d_iou_mean': float(np.mean(all_2d_iou)) if all_2d_iou else 0,
            '2d_iou_std': float(np.std(all_2d_iou)) if all_2d_iou else 0,
            '2d_dice_mean': float(np.mean(all_2d_dice)) if all_2d_dice else 0,
            '2d_dice_std': float(np.std(all_2d_dice)) if all_2d_dice else 0,
        },
        'per_patient': results,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== M3D_RefSeg Overall Results ===")
    print(f"3D IoU: {summary['overall']['3d_iou_mean']:.3f} ± {summary['overall']['3d_iou_std']:.3f}")
    print(f"3D Dice: {summary['overall']['3d_dice_mean']:.3f} ± {summary['overall']['3d_dice_std']:.3f}")
    print(f"2D IoU: {summary['overall']['2d_iou_mean']:.3f} ± {summary['overall']['2d_iou_std']:.3f}")
    print(f"2D Dice: {summary['overall']['2d_dice_mean']:.3f} ± {summary['overall']['2d_dice_std']:.3f}")
    print(f"Results saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate pseudo-masks against GT")
    parser.add_argument('--dataset', type=str, required=True, choices=['duke', 'm3d_refseg'],
                        help='Dataset to evaluate')
    parser.add_argument('--pseudo_dir', type=str, required=True,
                        help='Directory with pseudo-mask outputs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for evaluation results')

    # Duke-specific
    parser.add_argument('--duke_gt_dir', type=str,
                        default='./3D-Med-VLM-main/Data/data/Duke-Breast-Cancer-MRI/PKG-Duke-Breast-Cancer-MRI-Supplement-v3',
                        help='Duke GT masks directory')

    # M3D-specific
    parser.add_argument('--m3d_gt_dir', type=str,
                        default='./Data/data/M3D_RefSeg_npy',
                        help='M3D_RefSeg directory with GT masks')

    args = parser.parse_args()

    if args.dataset == 'duke':
        evaluate_duke(args.pseudo_dir, args.duke_gt_dir, args.output_file)
    elif args.dataset == 'm3d_refseg':
        evaluate_m3d_refseg(args.pseudo_dir, args.m3d_gt_dir, args.output_file)


if __name__ == '__main__':
    main()
