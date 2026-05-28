#!/usr/bin/env python3
"""
Process ALL slices of 3D volumes to generate pseudo-labels.
Uses Qwen2-VL for bbox detection and MedSAM for mask generation.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pseudo_masks.qwen_bbox import QwenBBoxGenerator
from pseudo_masks.medsam_mask import MedSAMMaskGenerator
from pseudo_masks.utils import (
    load_volume, detect_modality, preprocess_slice,
    save_results, create_overlay_visualization, QualityCode
)


def process_volume_all_slices(
    volume_path: str,
    output_dir: str,
    target: str,
    qwen: QwenBBoxGenerator,
    medsam: MedSAMMaskGenerator,
    modality_hint: str = None,
    visualize_every: int = 0,
    skip_empty_slices: bool = False,
):
    """
    Process ALL slices of a volume and generate pseudo-masks.

    Args:
        volume_path: Path to .npy volume file
        output_dir: Output directory
        target: Target description for bbox detection
        qwen: QwenBBoxGenerator instance
        medsam: MedSAMMaskGenerator instance
        modality_hint: Optional modality override ('ct', 'mri')
        visualize_every: Create visualization every N slices (0=disabled)
        skip_empty_slices: Skip slices where no bbox is detected

    Returns:
        dict with processing results
    """
    # Load volume
    volume, patient_id = load_volume(volume_path)
    modality = modality_hint or detect_modality(volume)
    D, H, W = volume.shape

    print(f"Processing {patient_id}: shape={volume.shape}, modality={modality}")

    # Initialize arrays for ALL slices
    boxes_array = np.full((D, 4), -1, dtype=np.int32)
    masks_array = np.zeros((D, H, W), dtype=np.uint8)
    quality_array = np.zeros(D, dtype=np.int32)
    slice_indices = np.arange(D)

    box_results = [None] * D
    detected_count = 0

    # Phase 1: Qwen bbox generation for all slices
    print(f"Phase 1: Qwen bbox generation for {D} slices...")
    for idx in tqdm(range(D), desc="Bbox detection"):
        slice_2d = volume[idx]
        slice_rgb = preprocess_slice(slice_2d, modality)

        box, response = qwen.process_slice(slice_rgb, target)
        box_results[idx] = box

        if box and box.visible:
            boxes_array[idx] = box.to_array()
            quality_array[idx] = int(box.quality)
            detected_count += 1

    print(f"  Direct detections: {detected_count}/{D} slices")

    # Phase 2: MedSAM mask generation for detected slices
    print(f"Phase 2: MedSAM mask generation for {detected_count} slices...")
    mask_count = 0
    for idx in tqdm(range(D), desc="Mask generation"):
        box = box_results[idx]
        if box and box.visible:
            slice_2d = volume[idx]
            slice_rgb = preprocess_slice(slice_2d, modality)
            mask, conf = medsam.process_slice(slice_rgb, box)
            masks_array[idx] = mask
            mask_count += 1

    print(f"  Generated masks: {mask_count}/{D} slices")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    metadata = {
        'patient_id': patient_id,
        'volume_shape': list(volume.shape),
        'modality': modality,
        'target': target,
        'all_slices': True,
        'num_slices': int(D),
        'num_detected': int(detected_count),
        'quality_summary': {
            'qwen_direct': int((quality_array == int(QualityCode.QWEN_BOX)).sum()),
            'missing': int((quality_array == int(QualityCode.MISSING)).sum())
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save NPZ with all data
    npz_path = os.path.join(output_dir, f"{patient_id}_pseudo_masks.npz")
    np.savez_compressed(
        npz_path,
        slice_indices=slice_indices,
        boxes=boxes_array,
        masks=masks_array,
        quality=quality_array
    )

    # Save metadata JSON
    json_path = os.path.join(output_dir, f"{patient_id}_pseudo_masks.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {npz_path}")

    # Optional visualizations
    if visualize_every > 0:
        vis_dir = os.path.join(output_dir, 'visualizations', patient_id)
        os.makedirs(vis_dir, exist_ok=True)
        for idx in range(0, D, visualize_every):
            slice_2d = volume[idx]
            slice_rgb = preprocess_slice(slice_2d, modality)
            vis_path = os.path.join(vis_dir, f"slice_{idx:04d}.png")
            create_overlay_visualization(
                slice_rgb, masks_array[idx], box_results[idx],
                vis_path, f"{patient_id} - Slice {idx}"
            )
        print(f"Visualizations saved to {vis_dir}")

    return metadata


def process_duke_dataset(
    preprocessed_dir: str,
    gt_list_file: str,
    output_dir: str,
    target: str,
    qwen: QwenBBoxGenerator,
    medsam: MedSAMMaskGenerator,
    max_patients: int = None,
    visualize_every: int = 0,
):
    """Process Duke dataset patients with GT masks."""

    # Load list of patients with GT
    with open(gt_list_file, 'r') as f:
        patients_with_gt = [line.strip() for line in f if line.strip()]

    print(f"Found {len(patients_with_gt)} Duke patients with GT masks")

    if max_patients:
        patients_with_gt = patients_with_gt[:max_patients]
        print(f"Processing first {max_patients} patients")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    failed = []

    for patient_id in patients_with_gt:
        volume_path = os.path.join(preprocessed_dir, f"{patient_id}.npy")

        # Skip already processed patients
        existing_npz = os.path.join(output_dir, f"{patient_id}_pseudo_masks.npz")
        if os.path.exists(existing_npz):
            print(f"SKIP: {patient_id} (already processed)")
            continue

        if not os.path.exists(volume_path):
            print(f"WARNING: Volume not found: {volume_path}")
            failed.append({'patient_id': patient_id, 'error': 'Volume not found'})
            continue

        try:
            metadata = process_volume_all_slices(
                volume_path=volume_path,
                output_dir=output_dir,
                target=target,
                qwen=qwen,
                medsam=medsam,
                modality_hint='mri',
                visualize_every=visualize_every
            )
            results.append(metadata)
        except Exception as e:
            print(f"ERROR processing {patient_id}: {e}")
            failed.append({'patient_id': patient_id, 'error': str(e)})

    # Save summary
    summary = {
        'dataset': 'Duke-Breast-Cancer-MRI',
        'total_patients': len(patients_with_gt),
        'processed': len(results),
        'failed': len(failed),
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'failed_patients': failed
    }

    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary: {len(results)}/{len(patients_with_gt)} processed, {len(failed)} failed")
    print(f"Saved summary to {summary_path}")

    return summary


def process_m3d_refseg_dataset(
    input_dir: str,
    output_dir: str,
    target: str,
    qwen: QwenBBoxGenerator,
    medsam: MedSAMMaskGenerator,
    max_patients: int = None,
    visualize_every: int = 0,
):
    """Process M3D_RefSeg dataset (all patients with mask.npy)."""

    # Find all patients with GT masks
    patients = []
    for patient_dir in sorted(Path(input_dir).iterdir()):
        if patient_dir.is_dir():
            ct_path = patient_dir / 'ct.npy'
            mask_path = patient_dir / 'mask.npy'
            if ct_path.exists() and mask_path.exists():
                patients.append(patient_dir.name)

    print(f"Found {len(patients)} M3D_RefSeg patients with GT masks")

    if max_patients:
        patients = patients[:max_patients]
        print(f"Processing first {max_patients} patients")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    failed = []

    for patient_id in patients:
        volume_path = os.path.join(input_dir, patient_id, 'ct.npy')

        try:
            metadata = process_volume_all_slices(
                volume_path=volume_path,
                output_dir=output_dir,
                target=target,
                qwen=qwen,
                medsam=medsam,
                modality_hint=None,  # Auto-detect (handles normalized CT)
                visualize_every=visualize_every
            )
            results.append(metadata)
        except Exception as e:
            print(f"ERROR processing {patient_id}: {e}")
            failed.append({'patient_id': patient_id, 'error': str(e)})

    # Save summary
    summary = {
        'dataset': 'M3D_RefSeg',
        'total_patients': len(patients),
        'processed': len(results),
        'failed': len(failed),
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'failed_patients': failed
    }

    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary: {len(results)}/{len(patients)} processed, {len(failed)} failed")
    print(f"Saved summary to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-masks for all slices")
    parser.add_argument('--dataset', type=str, required=True, choices=['duke', 'm3d_refseg'],
                        help='Dataset to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudo-masks')
    parser.add_argument('--target', type=str, default='lesion or mass',
                        help='Target description for bbox detection')
    parser.add_argument('--model_id', type=str,
                        default='./models/Qwen2-VL-7B-Instruct',
                        help='Qwen2-VL model path')
    parser.add_argument('--medsam_checkpoint', type=str,
                        default='./medsam_vit_b.pth',
                        help='MedSAM checkpoint path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='Maximum number of patients to process')
    parser.add_argument('--visualize_every', type=int, default=0,
                        help='Create visualization every N slices (0=disabled)')

    # Dataset-specific paths
    parser.add_argument('--duke_preprocessed', type=str,
                        default='./3D-Med-VLM-main/Data/data/Duke-Breast-Cancer-MRI/preprocessed',
                        help='Duke preprocessed directory')
    parser.add_argument('--duke_gt_list', type=str,
                        default='./pseudo_masks/duke_with_gt.txt',
                        help='List of Duke patients with GT masks')
    parser.add_argument('--m3d_refseg_dir', type=str,
                        default='./Data/data/M3D_RefSeg_npy',
                        help='M3D_RefSeg directory')

    args = parser.parse_args()

    # Initialize models
    print("Loading Qwen2-VL model...")
    qwen = QwenBBoxGenerator(model_id=args.model_id, device=args.device)

    print("Loading MedSAM model...")
    medsam = MedSAMMaskGenerator(checkpoint_path=args.medsam_checkpoint, device=args.device)

    # Process dataset
    if args.dataset == 'duke':
        process_duke_dataset(
            preprocessed_dir=args.duke_preprocessed,
            gt_list_file=args.duke_gt_list,
            output_dir=args.output_dir,
            target=args.target,
            qwen=qwen,
            medsam=medsam,
            max_patients=args.max_patients,
            visualize_every=args.visualize_every
        )
    elif args.dataset == 'm3d_refseg':
        process_m3d_refseg_dataset(
            input_dir=args.m3d_refseg_dir,
            output_dir=args.output_dir,
            target=args.target,
            qwen=qwen,
            medsam=medsam,
            max_patients=args.max_patients,
            visualize_every=args.visualize_every
        )


if __name__ == '__main__':
    main()
