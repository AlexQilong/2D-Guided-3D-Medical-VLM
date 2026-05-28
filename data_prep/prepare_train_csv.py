#!/usr/bin/env python3
"""
Prepare Duke GT+Pseudo combined training data.

Combines:
- 50 GT entries (reuse existing preprocessed volumes/masks from duke_gt)
- 100 additional pseudo mask entries from patients WITHOUT GT annotations

Usage:
    python finetune_scripts/prepare_duke_gt_plus_pseudo.py
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

BASE_DIR = Path(".")
PSEUDO_DIR = BASE_DIR / "pseudo_masks_output" / "duke_all_slices"
VOLUME_DIR = BASE_DIR / "3D-Med-VLM-main" / "Data" / "data" / "Duke-Breast-Cancer-MRI" / "preprocessed"
OUTPUT_DIR = BASE_DIR / "finetune_data" / "duke_gt_plus_pseudo"
VOLUMES_OUT = BASE_DIR / "finetune_data" / "duke_volumes"
MASKS_OUT = BASE_DIR / "finetune_data" / "converted_masks"

TARGET_SIZE = (32, 256, 256)

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


def resize_volume(volume, target_size):
    if volume.shape == target_size:
        return volume
    factors = tuple(t / s for t, s in zip(target_size, volume.shape))
    return zoom(volume, factors, order=1)


def resize_mask(mask, target_size):
    if mask.shape == target_size:
        return mask
    factors = tuple(t / s for t, s in zip(target_size, mask.shape))
    resized = zoom(mask, factors, order=0)
    return (resized > 0.5).astype(np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pseudo", type=int, default=100,
                        help="Number of additional pseudo mask patients")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output subfolder name (default: duke_gt_plus_pseudo)")
    args = parser.parse_args()

    num_pseudo = args.num_pseudo
    output_name = args.output_name or f"duke_gt_plus_pseudo_{num_pseudo}" if num_pseudo != 100 else "duke_gt_plus_pseudo"
    out_dir = BASE_DIR / "finetune_data" / output_name

    random.seed(42)
    np.random.seed(42)

    out_dir.mkdir(parents=True, exist_ok=True)
    VOLUMES_OUT.mkdir(parents=True, exist_ok=True)
    MASKS_OUT.mkdir(parents=True, exist_ok=True)

    # Load original 100 Duke IDs (50 train + 50 test)
    with open(BASE_DIR / "finetune_data" / "data_split.json") as f:
        split = json.load(f)
    original_ids = set(split["duke_train_ids"] + split["duke_test_ids"])
    gt_train_ids = split["duke_train_ids"]  # 50 GT train IDs

    print(f"Original Duke IDs: {len(original_ids)} (50 train + 50 test)")

    # Find new patients with pseudo masks NOT in original 100
    pseudo_files = list(PSEUDO_DIR.glob("*_pseudo_masks.npz"))
    all_pseudo_ids = set(f.stem.replace("_pseudo_masks", "") for f in pseudo_files)
    new_ids = sorted(all_pseudo_ids - original_ids)

    # Filter to those that also have raw volumes
    volume_files = set(f.stem for f in VOLUME_DIR.glob("*.npy"))
    new_ids = [pid for pid in new_ids if pid in volume_files]

    print(f"Pseudo masks available: {len(all_pseudo_ids)}")
    print(f"New patients (not in original 100, have volume): {len(new_ids)}")

    # Select new patients
    random.shuffle(new_ids)
    selected_new = new_ids[:num_pseudo]
    print(f"Selected {len(selected_new)} new pseudo patients (requested {num_pseudo})")

    # === Part 1: 50 GT entries (reuse existing paths) ===
    gt_entries = []
    for idx, pid in enumerate(gt_train_ids):
        gt_entries.append({
            "Image": f"finetune_data/duke_volumes/{pid}_volume.npy",
            "Mask": f"finetune_data/converted_masks/{pid}_gt_mask.npy",
            "Mask_ID": 1,
            "Question_Type": 0,
            "Question": QUESTIONS[idx % len(QUESTIONS)],
            "Answer": ANSWERS[idx % len(ANSWERS)],
        })
    print(f"GT entries: {len(gt_entries)}")

    # === Part 2: 100 new pseudo entries (preprocess volumes + masks) ===
    pseudo_entries = []
    skipped = 0
    for idx, pid in enumerate(tqdm(selected_new, desc="Processing new pseudo patients")):
        try:
            # Check if volume already preprocessed
            vol_out = VOLUMES_OUT / f"{pid}_volume.npy"
            if not vol_out.exists():
                volume = np.load(VOLUME_DIR / f"{pid}.npy")
                volume_resized = resize_volume(volume, TARGET_SIZE)
                # Normalize to [0, 1]
                vmax = volume_resized.max()
                if vmax > 0:
                    volume_resized = volume_resized / vmax
                volume_resized = volume_resized[np.newaxis, ...].astype(np.float32)  # (1, 32, 256, 256)
                np.save(vol_out, volume_resized)

            # Check if pseudo mask already preprocessed
            mask_out = MASKS_OUT / f"{pid}_pseudo_mask.npy"
            if not mask_out.exists():
                pseudo_data = np.load(PSEUDO_DIR / f"{pid}_pseudo_masks.npz")
                pseudo_mask = pseudo_data["masks"]
                pseudo_resized = resize_mask(pseudo_mask, TARGET_SIZE)
                pseudo_resized = pseudo_resized[np.newaxis, ...].astype(np.int8)  # (1, 32, 256, 256)
                np.save(mask_out, pseudo_resized)

            pseudo_entries.append({
                "Image": f"finetune_data/duke_volumes/{pid}_volume.npy",
                "Mask": f"finetune_data/converted_masks/{pid}_pseudo_mask.npy",
                "Mask_ID": 1,
                "Question_Type": 0,
                "Question": QUESTIONS[(idx + len(gt_entries)) % len(QUESTIONS)],
                "Answer": ANSWERS[(idx + len(gt_entries)) % len(ANSWERS)],
            })
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            skipped += 1

    print(f"Pseudo entries: {len(pseudo_entries)} (skipped: {skipped})")

    # === Combine and save ===
    all_entries = gt_entries + pseudo_entries
    df = pd.DataFrame(all_entries)
    csv_path = out_dir / "train.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved combined CSV: {csv_path}")
    print(f"  Total: {len(all_entries)} entries (50 GT + {len(pseudo_entries)} pseudo)")

    # Save metadata
    meta = {
        "gt_train_ids": gt_train_ids,
        "pseudo_train_ids": [e["Image"].split("/")[-1].replace("_volume.npy", "") for e in pseudo_entries],
        "num_gt": len(gt_entries),
        "num_pseudo": len(pseudo_entries),
        "total": len(all_entries),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved split info: {out_dir / 'split_info.json'}")


if __name__ == "__main__":
    main()
