#!/usr/bin/env python3
"""
Prepare finetuning data for 3D VLM segmentation training.

Creates SEPARATE datasets for Duke and M3D:
- Duke: 50 train, 50 test (from 100 samples with NRRD GT masks)
- M3D: 100 train, 108 test (from 208 samples)

Usage:
    python prepare_finetune_data.py \
        --m3d_train_samples 100 \
        --duke_train_samples 50 \
        --seed 42
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

try:
    import nrrd
except ImportError:
    print("Installing pynrrd...")
    os.system("pip install pynrrd")
    import nrrd

# Paths
BASE_DIR = Path(".")

# M3D paths
M3D_PSEUDO_DIR = BASE_DIR / "pseudo_masks_output" / "m3d_refseg_all_slices"
M3D_DATA_DIR = BASE_DIR / "Data" / "data" / "M3D_RefSeg_npy"

# Duke paths
DUKE_PSEUDO_DIR = BASE_DIR / "pseudo_masks_output" / "duke_all_slices"
DUKE_VOLUME_DIR = BASE_DIR / "3D-Med-VLM-main" / "Data" / "data" / "Duke-Breast-Cancer-MRI" / "preprocessed"
DUKE_GT_DIR = BASE_DIR / "3D-Med-VLM-main" / "Data" / "data" / "Duke-Breast-Cancer-MRI" / "PKG-Duke-Breast-Cancer-MRI-Supplement-v3" / "Duke-Breast-Cancer-MRI-Supplement-v3" / "Segmentation_Masks_NRRD"

OUTPUT_DIR = BASE_DIR / "finetune_data"

# Target size for model input
TARGET_SIZE = (32, 256, 256)

# Questions and answers for segmentation task
QUESTIONS = [
    "Please segment the lesion in this scan.",
    "Identify and segment the abnormal region.",
    "Segment the pathological area in this image.",
    "Please locate and segment the lesion.",
    "Segment the region of interest in this scan.",
]

ANSWERS = [
    "The lesion is segmented as shown in [SEG].",
    "The abnormal region is highlighted in [SEG].",
    "The pathological area is marked in [SEG].",
    "The lesion location is shown in [SEG].",
    "The region of interest is segmented in [SEG].",
]


def resize_volume(volume: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize 3D volume to target size using trilinear interpolation."""
    if volume.shape == target_size:
        return volume
    factors = tuple(t / s for t, s in zip(target_size, volume.shape))
    resized = zoom(volume, factors, order=1)
    return resized


def resize_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize 3D mask to target size using nearest neighbor."""
    if mask.shape == target_size:
        return mask
    factors = tuple(t / s for t, s in zip(target_size, mask.shape))
    resized = zoom(mask, factors, order=0)
    return (resized > 0.5).astype(np.int8)


def load_duke_gt_mask(patient_id: str) -> np.ndarray:
    """Load Duke GT mask from NRRD format."""
    mask_dir = DUKE_GT_DIR / patient_id

    # Try to load breast mask first (primary target)
    breast_mask_path = mask_dir / f"Segmentation_{patient_id}_Breast.seg.nrrd"
    dense_mask_path = mask_dir / f"Segmentation_{patient_id}_Dense_and_Vessels.seg.nrrd"

    if breast_mask_path.exists():
        data, header = nrrd.read(str(breast_mask_path))
        if data.ndim == 3:
            # NRRD is (H, W, D) -> transpose to (D, H, W), then rotate 90° CW in H-W plane
            mask = data.transpose(2, 0, 1).astype(np.float32)
            mask = np.rot90(mask, k=-1, axes=(1, 2)).copy()
            mask = (mask > 0).astype(np.int8)
            return mask

    if dense_mask_path.exists():
        data, header = nrrd.read(str(dense_mask_path))
        if data.ndim == 3:
            # NRRD is (H, W, D) -> transpose to (D, H, W), then rotate 90° CW in H-W plane
            mask = data.transpose(2, 0, 1).astype(np.float32)
            mask = np.rot90(mask, k=-1, axes=(1, 2)).copy()
            mask = (mask > 0).astype(np.int8)
            return mask

    raise FileNotFoundError(f"No GT mask found for {patient_id}")


def get_m3d_samples():
    """Get list of M3D patient IDs that have both pseudo masks and GT data."""
    pseudo_files = list(M3D_PSEUDO_DIR.glob("*_pseudo_masks.npz"))
    pseudo_ids = set(f.stem.replace("_pseudo_masks", "") for f in pseudo_files)

    gt_dirs = [d for d in M3D_DATA_DIR.iterdir() if d.is_dir() and (d / "ct.npy").exists()]
    gt_ids = set(d.name for d in gt_dirs)

    common_ids = sorted(pseudo_ids & gt_ids)
    print(f"M3D: {len(pseudo_ids)} pseudo, {len(gt_ids)} GT, {len(common_ids)} common")

    return common_ids


def get_duke_samples():
    """Get list of Duke patient IDs that have pseudo masks and NRRD GT masks."""
    # Get pseudo mask files
    pseudo_files = list(DUKE_PSEUDO_DIR.glob("*_pseudo_masks.npz"))
    pseudo_ids = set(f.stem.replace("_pseudo_masks", "") for f in pseudo_files)

    # Get GT mask directories (NRRD format)
    gt_dirs = [d for d in DUKE_GT_DIR.iterdir() if d.is_dir()]
    gt_ids = set(d.name for d in gt_dirs)

    # Get volume files
    volume_files = list(DUKE_VOLUME_DIR.glob("*.npy"))
    volume_ids = set(f.stem for f in volume_files)

    common_ids = sorted(pseudo_ids & gt_ids & volume_ids)
    print(f"Duke: {len(pseudo_ids)} pseudo, {len(gt_ids)} GT (NRRD), "
          f"{len(volume_ids)} volumes, {len(common_ids)} common")

    return common_ids


def process_m3d_sample(patient_id: str, masks_dir: Path) -> dict:
    """Process M3D sample - convert pseudo mask and prepare paths."""
    npz_path = M3D_PSEUDO_DIR / f"{patient_id}_pseudo_masks.npz"
    pseudo_mask = np.load(npz_path)['masks']

    if pseudo_mask.ndim == 3:
        pseudo_mask = pseudo_mask[np.newaxis, ...]

    pseudo_mask = (pseudo_mask > 0).astype(np.int8)

    pseudo_path = masks_dir / f"{patient_id}_pseudo_mask.npy"
    np.save(pseudo_path, pseudo_mask)

    return {
        "patient_id": patient_id,
        "dataset": "m3d",
        "image_path": f"M3D_RefSeg_npy/{patient_id}/ct.npy",
        "pseudo_mask_path": f"../finetune_data/converted_masks/{patient_id}_pseudo_mask.npy",
        "gt_mask_path": f"M3D_RefSeg_npy/{patient_id}/mask.npy",
    }


def process_duke_sample(patient_id: str, masks_dir: Path, volumes_dir: Path) -> dict:
    """Process Duke sample - resize volume and masks to standard size."""
    # Load volume
    volume_path = DUKE_VOLUME_DIR / f"{patient_id}.npy"
    volume = np.load(volume_path)  # (D, H, W) e.g., (142, 512, 512)

    # Resize volume
    volume_resized = resize_volume(volume, TARGET_SIZE)
    volume_resized = volume_resized[np.newaxis, ...]  # (1, 32, 256, 256)

    volume_out_path = volumes_dir / f"{patient_id}_volume.npy"
    np.save(volume_out_path, volume_resized.astype(np.float32))

    # Load and resize pseudo mask
    pseudo_npz = np.load(DUKE_PSEUDO_DIR / f"{patient_id}_pseudo_masks.npz")
    pseudo_mask = pseudo_npz['masks']
    pseudo_resized = resize_mask(pseudo_mask, TARGET_SIZE)
    pseudo_resized = pseudo_resized[np.newaxis, ...]

    pseudo_path = masks_dir / f"{patient_id}_pseudo_mask.npy"
    np.save(pseudo_path, pseudo_resized.astype(np.int8))

    # Load and resize GT mask from NRRD
    gt_mask = load_duke_gt_mask(patient_id)
    gt_resized = resize_mask(gt_mask, TARGET_SIZE)
    gt_resized = gt_resized[np.newaxis, ...]

    gt_path = masks_dir / f"{patient_id}_gt_mask.npy"
    np.save(gt_path, gt_resized.astype(np.int8))

    return {
        "patient_id": patient_id,
        "dataset": "duke",
        "image_path": f"../finetune_data/duke_volumes/{patient_id}_volume.npy",
        "pseudo_mask_path": f"../finetune_data/converted_masks/{patient_id}_pseudo_mask.npy",
        "gt_mask_path": f"../finetune_data/converted_masks/{patient_id}_gt_mask.npy",
    }


def create_csv_entry(info: dict, mask_type: str, idx: int) -> dict:
    """Create a single CSV entry for training."""
    q_idx = idx % len(QUESTIONS)
    mask_path = info["pseudo_mask_path"] if mask_type == "pseudo" else info["gt_mask_path"]

    return {
        "Image": info["image_path"],
        "Mask": mask_path,
        "Mask_ID": 1,
        "Question_Type": 0,
        "Question": QUESTIONS[q_idx],
        "Answer": ANSWERS[q_idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare finetuning data")
    parser.add_argument("--m3d_train_samples", type=int, default=100,
                        help="Number of M3D samples for training")
    parser.add_argument("--duke_train_samples", type=int, default=50,
                        help="Number of Duke samples for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    duke_pseudo_dir = OUTPUT_DIR / "duke_pseudo"
    duke_gt_dir = OUTPUT_DIR / "duke_gt"
    m3d_pseudo_dir = OUTPUT_DIR / "m3d_pseudo"
    m3d_gt_dir = OUTPUT_DIR / "m3d_gt"
    masks_dir = OUTPUT_DIR / "converted_masks"
    volumes_dir = OUTPUT_DIR / "duke_volumes"

    for d in [duke_pseudo_dir, duke_gt_dir, m3d_pseudo_dir, m3d_gt_dir, masks_dir, volumes_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get available samples
    m3d_samples = get_m3d_samples()
    duke_samples = get_duke_samples()

    # Adjust sample counts
    m3d_train = min(args.m3d_train_samples, len(m3d_samples))
    duke_train = min(args.duke_train_samples, len(duke_samples))

    print(f"\nUsing {m3d_train} M3D samples, {duke_train} Duke samples for training")

    # Shuffle and split
    random.shuffle(m3d_samples)
    random.shuffle(duke_samples)

    m3d_train_samples = m3d_samples[:m3d_train]
    m3d_test_samples = m3d_samples[m3d_train:]
    duke_train_samples = duke_samples[:duke_train]
    duke_test_samples = duke_samples[duke_train:]

    print(f"M3D: {len(m3d_train_samples)} train, {len(m3d_test_samples)} test")
    print(f"Duke: {len(duke_train_samples)} train, {len(duke_test_samples)} test")

    # ==================== Process M3D ====================
    print("\n" + "="*50)
    print("Processing M3D samples...")
    print("="*50)

    m3d_pseudo_entries = []
    m3d_gt_entries = []

    for idx, patient_id in enumerate(tqdm(m3d_train_samples, desc="M3D train")):
        info = process_m3d_sample(patient_id, masks_dir)
        m3d_pseudo_entries.append(create_csv_entry(info, "pseudo", idx))
        m3d_gt_entries.append(create_csv_entry(info, "gt", idx))

    # Save M3D training CSVs
    pd.DataFrame(m3d_pseudo_entries).to_csv(m3d_pseudo_dir / "train.csv", index=False)
    pd.DataFrame(m3d_gt_entries).to_csv(m3d_gt_dir / "train.csv", index=False)
    print(f"Saved M3D pseudo train CSV: {m3d_pseudo_dir / 'train.csv'} ({len(m3d_pseudo_entries)} samples)")
    print(f"Saved M3D GT train CSV: {m3d_gt_dir / 'train.csv'} ({len(m3d_gt_entries)} samples)")

    # M3D test CSV
    m3d_test_entries = []
    for idx, patient_id in enumerate(m3d_test_samples):
        entry = {
            "Image": f"M3D_RefSeg_npy/{patient_id}/ct.npy",
            "Mask": f"M3D_RefSeg_npy/{patient_id}/mask.npy",
            "Mask_ID": 1,
            "Question_Type": 0,
            "Question": QUESTIONS[idx % len(QUESTIONS)],
            "Answer": ANSWERS[idx % len(ANSWERS)],
        }
        m3d_test_entries.append(entry)

    pd.DataFrame(m3d_test_entries).to_csv(OUTPUT_DIR / "m3d_test.csv", index=False)
    print(f"Saved M3D test CSV: {OUTPUT_DIR / 'm3d_test.csv'} ({len(m3d_test_entries)} samples)")

    # ==================== Process Duke ====================
    print("\n" + "="*50)
    print("Processing Duke samples...")
    print("="*50)

    duke_pseudo_entries = []
    duke_gt_entries = []

    for idx, patient_id in enumerate(tqdm(duke_train_samples, desc="Duke train")):
        try:
            info = process_duke_sample(patient_id, masks_dir, volumes_dir)
            duke_pseudo_entries.append(create_csv_entry(info, "pseudo", idx))
            duke_gt_entries.append(create_csv_entry(info, "gt", idx))
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

    # Save Duke training CSVs
    pd.DataFrame(duke_pseudo_entries).to_csv(duke_pseudo_dir / "train.csv", index=False)
    pd.DataFrame(duke_gt_entries).to_csv(duke_gt_dir / "train.csv", index=False)
    print(f"Saved Duke pseudo train CSV: {duke_pseudo_dir / 'train.csv'} ({len(duke_pseudo_entries)} samples)")
    print(f"Saved Duke GT train CSV: {duke_gt_dir / 'train.csv'} ({len(duke_gt_entries)} samples)")

    # Duke test CSV
    duke_test_entries = []
    for idx, patient_id in enumerate(tqdm(duke_test_samples, desc="Duke test")):
        try:
            info = process_duke_sample(patient_id, masks_dir, volumes_dir)
            entry = {
                "Image": info["image_path"],
                "Mask": info["gt_mask_path"],
                "Mask_ID": 1,
                "Question_Type": 0,
                "Question": QUESTIONS[idx % len(QUESTIONS)],
                "Answer": ANSWERS[idx % len(ANSWERS)],
            }
            duke_test_entries.append(entry)
        except Exception as e:
            print(f"Error processing test {patient_id}: {e}")

    pd.DataFrame(duke_test_entries).to_csv(OUTPUT_DIR / "duke_test.csv", index=False)
    print(f"Saved Duke test CSV: {OUTPUT_DIR / 'duke_test.csv'} ({len(duke_test_entries)} samples)")

    # ==================== Verification ====================
    print("\n" + "="*50)
    print("Verification")
    print("="*50)

    if m3d_train_samples:
        sample_id = m3d_train_samples[0]
        pseudo_mask = np.load(masks_dir / f"{sample_id}_pseudo_mask.npy")
        image = np.load(M3D_DATA_DIR / sample_id / "ct.npy")
        print(f"M3D sample: {sample_id}")
        print(f"  Image shape: {image.shape}")
        print(f"  Pseudo mask shape: {pseudo_mask.shape}")

    if duke_train_samples:
        sample_id = duke_train_samples[0]
        volume = np.load(volumes_dir / f"{sample_id}_volume.npy")
        pseudo_mask = np.load(masks_dir / f"{sample_id}_pseudo_mask.npy")
        gt_mask = np.load(masks_dir / f"{sample_id}_gt_mask.npy")
        print(f"Duke sample: {sample_id}")
        print(f"  Resized volume shape: {volume.shape}")
        print(f"  Pseudo mask shape: {pseudo_mask.shape}")
        print(f"  GT mask shape: {gt_mask.shape}")

    # Save summary
    summary = {
        "m3d_train": len(m3d_train_samples),
        "m3d_test": len(m3d_test_samples),
        "duke_train": len(duke_train_samples),
        "duke_test": len(duke_test_samples),
        "m3d_train_ids": m3d_train_samples,
        "m3d_test_ids": m3d_test_samples,
        "duke_train_ids": duke_train_samples,
        "duke_test_ids": duke_test_samples,
        "target_size": TARGET_SIZE,
    }

    with open(OUTPUT_DIR / "data_split.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved data split info: {OUTPUT_DIR / 'data_split.json'}")
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    print(f"  M3D: {len(m3d_train_samples)} train, {len(m3d_test_samples)} test")
    print(f"  Duke: {len(duke_train_samples)} train, {len(duke_test_samples)} test")


if __name__ == "__main__":
    main()
