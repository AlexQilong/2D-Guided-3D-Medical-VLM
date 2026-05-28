"""
SAM ViT-H mask generation from bounding box prompts.

Uses SAM ViT-H (largest SAM variant, 1024x1024) for best mask quality.
Also supports MedSAM ViT-B and SAM-Med2D via model_type/use_adapter params.
"""

import torch
import numpy as np
from types import SimpleNamespace
from typing import Optional, Tuple

from segment_anything import sam_model_registry, SamPredictor

from .utils import BoxResult


class MedSAMMaskGenerator:
    """
    Generates binary masks from bounding boxes using SAM.
    Default: SAM ViT-H (best quality on Duke MRI benchmark).
    """

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str = "cuda",
        use_adapter: bool = False,
        image_size: int = 1024,
    ):
        """
        Initialize SAM model.

        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: SAM model type ('vit_h', 'vit_b', etc.)
            device: Device to run on
            use_adapter: If True, load with SAM-Med2D adapter layers.
            image_size: Input image size (1024 for SAM ViT-H/MedSAM, 256 for SAM-Med2D).
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM model from {checkpoint_path} (type={model_type}, adapter={use_adapter})...")

        # SAM-Med2D's segment_anything requires args namespace for all models
        args = SimpleNamespace(
            image_size=image_size,
            encoder_adapter=use_adapter,
            sam_checkpoint=checkpoint_path,
        )
        self.sam = sam_model_registry[model_type](args)
        self.sam.to(device=self.device)
        self.sam.eval()

        # Create predictor
        self.predictor = SamPredictor(self.sam)
        print(f"SAM {model_type} loaded on device: {self.device}")

    def generate_mask(
        self,
        image_rgb: np.ndarray,
        box: BoxResult
    ) -> np.ndarray:
        """
        Generate binary mask from bounding box.

        Args:
            image_rgb: RGB image (H, W, 3) uint8
            box: BoxResult with bounding box coordinates

        Returns:
            Binary mask (H, W) uint8 {0, 1}
        """
        H, W = image_rgb.shape[:2]

        if not box.visible:
            return np.zeros((H, W), dtype=np.uint8)

        # Set image in predictor
        self.predictor.set_image(image_rgb)

        # Prepare box prompt (xyxy format)
        box_xyxy = np.array([box.x1, box.y1, box.x2, box.y2])

        # Run prediction
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy,
            multimask_output=False
        )

        # Extract mask (shape: 1, H, W)
        mask = masks[0]

        # Convert to binary uint8
        binary_mask = (mask > 0).astype(np.uint8)

        return binary_mask

    def process_slice(
        self,
        image_rgb: np.ndarray,
        box: Optional[BoxResult]
    ) -> Tuple[np.ndarray, float]:
        """
        Process a slice with box prompt and return mask with confidence.

        Args:
            image_rgb: RGB image (H, W, 3) uint8
            box: BoxResult or None

        Returns:
            (mask, confidence) where mask is (H, W) uint8
        """
        H, W = image_rgb.shape[:2]

        if box is None or not box.visible:
            return np.zeros((H, W), dtype=np.uint8), 0.0

        try:
            # Set image
            self.predictor.set_image(image_rgb)

            # Prepare box
            box_xyxy = np.array([box.x1, box.y1, box.x2, box.y2])

            # Predict
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy,
                multimask_output=False
            )

            mask = (masks[0] > 0).astype(np.uint8)
            confidence = float(scores[0])

            return mask, confidence

        except Exception as e:
            print(f"    Warning: SAM prediction failed: {e}")
            return np.zeros((H, W), dtype=np.uint8), 0.0
