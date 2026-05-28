"""
Qwen2-VL-based bounding box generation for medical image slices.
Uses zero-shot prompting to detect anatomical structures.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, Any
import tempfile

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    _QwenModelClass = Qwen2VLForConditionalGeneration
    _QwenProcessorClass = Qwen2VLProcessor
    QWEN_NATIVE = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoProcessor
    _QwenModelClass = AutoModelForCausalLM
    _QwenProcessorClass = AutoProcessor
    QWEN_NATIVE = False

from .utils import parse_json_response, BoxResult, QualityCode, validate_box


# Prompt template from requirements - exact format
BBOX_PROMPT_TEMPLATE = (
    "You are given a medical image slice of size {W}x{H}. "
    "Return exactly one bounding box for {TARGET} if it is visible in this slice. "
    "Output JSON only with keys: visible, x1, y1, x2, y2. "
    'If {TARGET} is not visible, output {{"visible": false}}. '
    "If visible=true, x1<x2, y1<y2, integers. "
    "Coordinates must satisfy 0<=x1<x2<{W} and 0<=y1<y2<{H}. "
    "Do not include any other text."
)


class QwenBBoxGenerator:
    """
    Generates bounding boxes for anatomical targets using Qwen2-VL.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto"
    ):
        """
        Initialize the Qwen VL model for bbox generation.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to run on ('cuda', 'cpu', 'auto')
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16')
        """
        self.device = device
        print(f"Loading Qwen VL model from {model_id}...")

        # Load model
        self.model = _QwenModelClass.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device if device == "auto" else None,
            trust_remote_code=True
        )

        if device != "auto" and device != "cpu":
            self.model = self.model.to(device)

        # Load processor
        self.processor = _QwenProcessorClass.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.actual_device = next(self.model.parameters()).device
        print(f"Qwen VL loaded on device: {self.actual_device}")

    def generate_bbox(
        self,
        image_rgb: np.ndarray,
        target: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 128
    ) -> Tuple[Optional[Dict], str]:
        """
        Generate bounding box for target in image.

        Args:
            image_rgb: RGB image (H, W, 3) uint8
            target: Target to detect (e.g., "breast tissue")
            temperature: Sampling temperature (0 for deterministic)
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate

        Returns:
            (parsed_dict, raw_response) tuple
        """
        H, W = image_rgb.shape[:2]

        # Format prompt
        prompt = BBOX_PROMPT_TEMPLATE.format(W=W, H=H, TARGET=target)

        # Save image to temporary file (Qwen expects file path or PIL Image)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(image_rgb).save(tmp_path)

        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Load and process image
            pil_image = Image.open(tmp_path).convert("RGB")
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.actual_device)

            # Generate with deterministic decoding
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Extract generated tokens (exclude input)
            input_ids = inputs.get("input_ids", None)
            input_length = int(input_ids.shape[1]) if input_ids is not None else 0

            sequences = generated_ids
            if hasattr(sequences, "dim") and sequences.dim() == 2:
                sequences = sequences[0]

            if hasattr(sequences, "shape") and sequences.shape[0] > input_length:
                continuation_tokens = sequences[input_length:]
            else:
                continuation_tokens = sequences

            raw_response = self.processor.tokenizer.decode(
                continuation_tokens, skip_special_tokens=True
            ).strip()

            # Parse JSON response
            parsed = parse_json_response(raw_response)
            return parsed, raw_response

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def process_slice(
        self,
        image_rgb: np.ndarray,
        target: str,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.90,
        min_dimension: int = 5
    ) -> Tuple[Optional[BoxResult], str]:
        """
        Process a single slice and return validated BoxResult.

        Args:
            image_rgb: RGB image (H, W, 3) uint8
            target: Target to detect
            min_area_ratio: Minimum box area ratio
            max_area_ratio: Maximum box area ratio
            min_dimension: Minimum box dimension

        Returns:
            (BoxResult or None, raw_response)
        """
        H, W = image_rgb.shape[:2]

        # Generate bbox
        try:
            parsed, raw_response = self.generate_bbox(image_rgb, target)
        except Exception as e:
            print(f"    Warning: Qwen generation failed: {e}")
            return None, f"ERROR: {e}"

        if parsed is None:
            return None, raw_response

        # Validate box
        is_valid, box_result = validate_box(
            parsed, W, H,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_dimension=min_dimension
        )

        if is_valid:
            return box_result, raw_response

        # Check if explicitly not visible
        if parsed.get('visible') == False:
            return BoxResult(
                visible=False, x1=0, y1=0, x2=0, y2=0,
                quality=QualityCode.MISSING
            ), raw_response

        return None, raw_response
