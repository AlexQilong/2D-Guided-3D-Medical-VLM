"""
Utility functions for pseudo-mask generation pipeline.
Handles I/O, preprocessing, validation, and common operations.
"""

import os
import re
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import IntEnum


class QualityCode(IntEnum):
    """Quality codes for mask quality tracking."""
    MISSING = 0       # No mask generated
    QWEN_BOX = 1      # Direct Qwen bbox -> MedSAM
    INTERPOLATED = 2  # Interpolated from neighbors
    HEURISTIC = 3     # Otsu/full-image fallback


@dataclass
class BoxResult:
    """Result from bbox generation for a single slice."""
    visible: bool
    x1: int
    y1: int
    x2: int
    y2: int
    quality: QualityCode
    low_quality: bool = False

    def to_array(self) -> np.ndarray:
        """Return as [x1, y1, x2, y2] or [-1, -1, -1, -1] if missing."""
        if not self.visible:
            return np.array([-1, -1, -1, -1], dtype=np.int32)
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int32)


def load_volume(path: str) -> Tuple[np.ndarray, str]:
    """
    Load 3D volume from .npy file, handle shape variations.

    Args:
        path: Path to .npy file

    Returns:
        (volume, patient_id) where volume is shape (D, H, W)

    Raises:
        ValueError: If volume shape is invalid
    """
    volume = np.load(path)

    # Handle (1, D, H, W) -> (D, H, W) for M3D_RefSeg format
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    # Extract patient ID from path
    path_obj = Path(path)
    if path_obj.stem in ['ct', 'mask']:
        # M3D_RefSeg format: .../s0000/ct.npy -> patient_id = "s0000"
        patient_id = path_obj.parent.name
    else:
        # Duke format: .../Breast_MRI_001.npy -> patient_id = "Breast_MRI_001"
        patient_id = path_obj.stem

    return volume, patient_id


def detect_modality(volume: np.ndarray) -> str:
    """
    Detect modality based on dtype and value range.

    Args:
        volume: 3D numpy array

    Returns:
        'ct' if likely CT scan, 'ct_normalized' if normalized CT,
        'mri' if likely MRI, 'unknown' otherwise
    """
    if volume.dtype == np.int16:
        return 'mri'
    elif volume.dtype == np.float32:
        if volume.min() >= -0.1 and volume.max() <= 1.1:
            return 'ct_normalized'
        return 'ct'
    elif volume.dtype == np.float64:
        if volume.min() >= -0.1 and volume.max() <= 1.1:
            return 'ct_normalized'
        return 'ct'
    return 'unknown'


def preprocess_slice_ct(
    slice_2d: np.ndarray,
    window_level: int = 40,
    window_width: int = 400
) -> np.ndarray:
    """
    Preprocess CT slice with windowing.

    Args:
        slice_2d: 2D slice (H, W)
        window_level: Window center (default: 40 for soft tissue)
        window_width: Window width (default: 400)

    Returns:
        RGB uint8 image (H, W, 3)
    """
    # Apply windowing
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    windowed = np.clip(slice_2d, lower, upper)

    # Scale to 0-255
    if upper > lower:
        normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)

    # Convert to 3-channel RGB
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return rgb


def preprocess_slice_percentile(
    slice_2d: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Preprocess slice with percentile clipping (for unknown modality or MRI).

    Args:
        slice_2d: 2D slice (H, W)
        lower_percentile: Lower percentile for clipping (default: 1%)
        upper_percentile: Upper percentile for clipping (default: 99%)

    Returns:
        RGB uint8 image (H, W, 3)
    """
    # Compute percentiles
    p_low = np.percentile(slice_2d, lower_percentile)
    p_high = np.percentile(slice_2d, upper_percentile)

    # Clip and normalize
    clipped = np.clip(slice_2d, p_low, p_high)
    if p_high > p_low:
        normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)

    # Convert to 3-channel RGB
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return rgb


def preprocess_slice(
    slice_2d: np.ndarray,
    modality: str = 'unknown',
    window_level: int = 40,
    window_width: int = 400
) -> np.ndarray:
    """
    Preprocess 2D slice to 3-channel uint8 RGB.

    Args:
        slice_2d: 2D slice (H, W)
        modality: 'ct', 'ct_normalized', 'mri', or 'unknown'
        window_level: CT window center
        window_width: CT window width

    Returns:
        RGB uint8 image (H, W, 3)
    """
    if modality == 'ct':
        return preprocess_slice_ct(slice_2d, window_level, window_width)
    elif modality == 'ct_normalized':
        # Already normalized [0, 1], just scale to [0, 255]
        normalized = (np.clip(slice_2d, 0, 1) * 255).astype(np.uint8)
        return np.stack([normalized, normalized, normalized], axis=-1)
    else:
        # MRI or unknown: use percentile clipping
        return preprocess_slice_percentile(slice_2d)


def validate_box(
    box: Dict[str, Any],
    width: int,
    height: int,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.90,
    min_dimension: int = 5
) -> Tuple[bool, Optional[BoxResult]]:
    """
    Validate a bounding box against constraints.

    Args:
        box: Dict with keys 'visible', 'x1', 'y1', 'x2', 'y2'
        width: Image width (W)
        height: Image height (H)
        min_area_ratio: Minimum box area / image area
        max_area_ratio: Maximum box area / image area
        min_dimension: Minimum box width or height in pixels

    Returns:
        (is_valid, box_result) tuple
    """
    if not box.get('visible', False):
        return False, BoxResult(visible=False, x1=0, y1=0, x2=0, y2=0,
                                quality=QualityCode.MISSING)

    try:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
    except (KeyError, ValueError, TypeError):
        return False, None

    # Bounds check
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return False, None

    # Dimension check
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < min_dimension or box_height < min_dimension:
        return False, None

    # Area ratio check
    box_area = box_width * box_height
    image_area = width * height
    area_ratio = box_area / image_area

    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return False, None

    return True, BoxResult(visible=True, x1=x1, y1=y1, x2=x2, y2=y2,
                           quality=QualityCode.QWEN_BOX)


def parse_json_response(response: str) -> Optional[Dict]:
    """
    Robustly parse JSON from Qwen response, stripping code fences.

    Args:
        response: Raw text response from Qwen

    Returns:
        Parsed dict or None if parsing fails
    """
    if not response:
        return None

    # Strip markdown code fences
    text = response.strip()

    # Remove ```json ... ``` or ``` ... ```
    code_fence_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(code_fence_pattern, text)
    if match:
        text = match.group(1)

    # Try to find JSON object
    json_pattern = r'\{[^{}]*\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def otsu_body_region_box(slice_rgb: np.ndarray) -> BoxResult:
    """
    Fallback: compute bounding box using Otsu thresholding.

    Args:
        slice_rgb: RGB image (H, W, 3)

    Returns:
        BoxResult with heuristic quality flag
    """
    H, W = slice_rgb.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback to full image box
        return BoxResult(visible=True, x1=0, y1=0, x2=W, y2=H,
                        quality=QualityCode.HEURISTIC, low_quality=True)

    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Ensure minimum size
    if w < 5 or h < 5:
        return BoxResult(visible=True, x1=0, y1=0, x2=W, y2=H,
                        quality=QualityCode.HEURISTIC, low_quality=True)

    return BoxResult(visible=True, x1=x, y1=y, x2=x+w, y2=y+h,
                    quality=QualityCode.HEURISTIC, low_quality=True)


def interpolate_box(
    boxes: List[Optional[BoxResult]],
    index: int,
    width: int,
    height: int
) -> Optional[BoxResult]:
    """
    Interpolate a missing box from valid neighbors.

    Args:
        boxes: List of BoxResult (None for missing)
        index: Index to interpolate
        width: Image width
        height: Image height

    Returns:
        Interpolated BoxResult or None if no valid neighbors
    """
    # Find nearest valid neighbors
    prev_idx, prev_box = None, None
    next_idx, next_box = None, None

    for i in range(index - 1, -1, -1):
        if boxes[i] is not None and boxes[i].visible:
            prev_idx, prev_box = i, boxes[i]
            break

    for i in range(index + 1, len(boxes)):
        if boxes[i] is not None and boxes[i].visible:
            next_idx, next_box = i, boxes[i]
            break

    if prev_box is None and next_box is None:
        return None

    if prev_box is None:
        # Use next box directly
        return BoxResult(visible=True, x1=next_box.x1, y1=next_box.y1,
                        x2=next_box.x2, y2=next_box.y2,
                        quality=QualityCode.INTERPOLATED)

    if next_box is None:
        # Use prev box directly
        return BoxResult(visible=True, x1=prev_box.x1, y1=prev_box.y1,
                        x2=prev_box.x2, y2=prev_box.y2,
                        quality=QualityCode.INTERPOLATED)

    # Linear interpolation
    t = (index - prev_idx) / (next_idx - prev_idx)
    x1 = int(round(prev_box.x1 + t * (next_box.x1 - prev_box.x1)))
    y1 = int(round(prev_box.y1 + t * (next_box.y1 - prev_box.y1)))
    x2 = int(round(prev_box.x2 + t * (next_box.x2 - prev_box.x2)))
    y2 = int(round(prev_box.y2 + t * (next_box.y2 - prev_box.y2)))

    # Clamp to image bounds
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return BoxResult(visible=True, x1=x1, y1=y1, x2=x2, y2=y2,
                    quality=QualityCode.INTERPOLATED)


def save_results(
    output_dir: str,
    patient_id: str,
    k: int,
    slice_indices: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    quality: np.ndarray,
    metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Save results to NPZ and JSON files.

    Args:
        output_dir: Output directory
        patient_id: Patient identifier
        k: Number of slices
        slice_indices: Array of slice indices (k,)
        boxes: Array of boxes (k, 4), -1 for missing
        masks: Array of masks (k, H, W) uint8
        quality: Array of quality codes (k,)
        metadata: Additional metadata dict

    Returns:
        (npz_path, json_path) tuple
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save NPZ
    npz_path = os.path.join(output_dir, f"{patient_id}_masks_k{k}.npz")
    np.savez_compressed(
        npz_path,
        patient_id=patient_id,
        slice_indices=slice_indices,
        boxes=boxes,
        masks=masks,
        quality=quality
    )

    # Save JSON metadata
    json_path = os.path.join(output_dir, f"{patient_id}_masks_k{k}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return npz_path, json_path


def create_overlay_visualization(
    slice_rgb: np.ndarray,
    mask: np.ndarray,
    box: Optional[BoxResult],
    output_path: str,
    title: str = ""
) -> None:
    """
    Create and save overlay visualization for dry run mode.

    Args:
        slice_rgb: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        box: BoxResult or None
        output_path: Path to save visualization
        title: Title for the plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Original image
    axes[0].imshow(slice_rgb)
    axes[0].set_title("Original Slice")
    axes[0].axis('off')

    # Image with box
    axes[1].imshow(slice_rgb)
    if box is not None and box.visible:
        rect = patches.Rectangle(
            (box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[1].add_patch(rect)
        quality_name = QualityCode(box.quality).name
        axes[1].set_title(f"Bounding Box (quality={quality_name})")
    else:
        axes[1].set_title("No Bounding Box")
    axes[1].axis('off')

    # Mask overlay
    overlay = slice_rgb.copy()
    mask_colored = np.zeros_like(slice_rgb)
    mask_colored[mask > 0] = [255, 0, 0]  # Red for mask
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Mask Overlay ({mask.sum()} pixels)")
    axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
