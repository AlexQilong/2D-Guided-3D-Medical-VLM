import numpy as np
import torch
try:
    from transformers import Qwen3VLForConditionalGeneration as _QwenModelClass
    from transformers import AutoProcessor as _QwenProcessorClass
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _QwenModelClass
        from transformers import AutoProcessor as _QwenProcessorClass
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as _QwenModelClass
            from transformers import Qwen2VLProcessor as _QwenProcessorClass
        except ImportError:
            from transformers import AutoModelForCausalLM as _QwenModelClass
            from transformers import AutoProcessor as _QwenProcessorClass
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # Fallback if qwen_vl_utils is not available
    def process_vision_info(messages):
        image_inputs = []
        video_inputs = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    # Load the image from file path
                    from PIL import Image
                    image = Image.open(content["image"]).convert('RGB')
                    image_inputs.append(image)
        return image_inputs, video_inputs
import random
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import argparse

class CTSliceReportGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        target_image_size: int = 512,
        selection_strategy: str = "middle",  # middle | variance | mip
        slices_per_plane: int = 1,
        variance_window: int = 10,
        mip_window: int = 5,
    ):
        """
        Initialize the CT slice report generator with Qwen2-VL model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on ("auto", "cuda", "cpu")
        """
        print(f"Loading model: {model_name}")
        
        # Load model and processor
        self.model = _QwenModelClass.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map=device,
            trust_remote_code=True
        )

        self.processor = _QwenProcessorClass.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")
        self.target_image_size = target_image_size
        self.selection_strategy = selection_strategy
        self.slices_per_plane = max(1, int(slices_per_plane))
        self.variance_window = max(1, int(variance_window))
        self.mip_window = max(1, int(mip_window))
        
        # Base prompt template
        self.system_prompt = (
            "You are a board-certified radiologist. Be concise and clinically accurate."
        )
        self.report_prompt_template = (
            "You are given a single CT slice. {context}\n\n"
            "Write a brief report with sections:\n\n"
            "FINDINGS:\n- ...\n\n"
            "IMPRESSION:\n- ...\n\n"
            "Use standard radiology terminology; avoid speculation; no patient identifiers."
        )

    def load_ct_volume(self, ct_path: str) -> np.ndarray:
        """
        Load CT volume from .npy file.
        
        Args:
            ct_path: Path to the .npy CT file
            
        Returns:
            CT volume as numpy array
        """
        try:
            ct_volume = np.load(ct_path)
            # print(f"Loaded CT volume with shape: {ct_volume.shape}")
            return ct_volume
        except Exception as e:
            print(f"Error loading CT volume from {ct_path}: {e}")
            return None

    def normalize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Normalize CT slice for better visualization.
        
        Args:
            slice_2d: 2D CT slice
            
        Returns:
            Normalized slice
        """
        # Handle different bit depths and normalize to 0-255
        slice_min, slice_max = slice_2d.min(), slice_2d.max()
        if slice_max > slice_min:
            normalized = (slice_2d - slice_min) / (slice_max - slice_min) * 255
        else:
            normalized = np.zeros_like(slice_2d)
        
        return normalized.astype(np.uint8)

    def _normalize_to_3d(self, ct_volume: np.ndarray) -> np.ndarray:
        if len(ct_volume.shape) == 4:
            ct_volume = ct_volume[0]
        elif len(ct_volume.shape) != 3:
            raise ValueError(f"Expected 3D or 4D volume, got shape {ct_volume.shape}")
        return ct_volume

    def _select_middle(self, ct_volume: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
        ct_volume = self._normalize_to_3d(ct_volume)
        depth, height, width = ct_volume.shape
        axial_middle = depth // 2
        sagittal_middle = width // 2
        coronal_middle = height // 2
        return [
            ("axial", axial_middle, ct_volume[axial_middle, :, :]),
            ("sagittal", sagittal_middle, ct_volume[:, :, sagittal_middle]),
            ("coronal", coronal_middle, ct_volume[:, coronal_middle, :]),
        ]

    def _select_variance(self, ct_volume: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
        ct_volume = self._normalize_to_3d(ct_volume)
        depth, height, width = ct_volume.shape
        axial_middle = depth // 2
        sagittal_middle = width // 2
        coronal_middle = height // 2

        def top_k_indices(values: np.ndarray, k: int) -> List[int]:
            k = min(k, len(values))
            if k <= 0:
                return []
            return list(np.argsort(values)[-k:][::-1])

        # Axial window and variances
        a_start = max(0, axial_middle - self.variance_window)
        a_end = min(depth, axial_middle + self.variance_window + 1)
        axial_vars = [ct_volume[i].var() for i in range(a_start, a_end)]
        axial_indices = [a_start + i for i in top_k_indices(np.array(axial_vars), self.slices_per_plane)]

        # Sagittal window and variances
        s_start = max(0, sagittal_middle - self.variance_window)
        s_end = min(width, sagittal_middle + self.variance_window + 1)
        sagittal_vars = [ct_volume[:, :, i].var() for i in range(s_start, s_end)]
        sagittal_indices = [s_start + i for i in top_k_indices(np.array(sagittal_vars), self.slices_per_plane)]

        # Coronal window and variances
        c_start = max(0, coronal_middle - self.variance_window)
        c_end = min(height, coronal_middle + self.variance_window + 1)
        coronal_vars = [ct_volume[:, i, :].var() for i in range(c_start, c_end)]
        coronal_indices = [c_start + i for i in top_k_indices(np.array(coronal_vars), self.slices_per_plane)]

        selected: List[Tuple[str, int, np.ndarray]] = []
        for i in axial_indices:
            selected.append(("axial", i, ct_volume[i, :, :]))
        for i in sagittal_indices:
            selected.append(("sagittal", i, ct_volume[:, :, i]))
        for i in coronal_indices:
            selected.append(("coronal", i, ct_volume[:, i, :]))
        return selected

    def _select_mip(self, ct_volume: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
        ct_volume = self._normalize_to_3d(ct_volume)
        depth, height, width = ct_volume.shape
        axial_middle = depth // 2
        sagittal_middle = width // 2
        coronal_middle = height // 2

        # Axial MIP slab
        a_start = max(0, axial_middle - self.mip_window)
        a_end = min(depth, axial_middle + self.mip_window + 1)
        axial_mip = ct_volume[a_start:a_end].max(axis=0)

        # Sagittal MIP slab
        s_start = max(0, sagittal_middle - self.mip_window)
        s_end = min(width, sagittal_middle + self.mip_window + 1)
        sagittal_mip = ct_volume[:, :, s_start:s_end].max(axis=2)

        # Coronal MIP slab
        c_start = max(0, coronal_middle - self.mip_window)
        c_end = min(height, coronal_middle + self.mip_window + 1)
        coronal_mip = ct_volume[:, c_start:c_end, :].max(axis=1)

        return [
            ("axial", axial_middle, axial_mip),
            ("sagittal", sagittal_middle, sagittal_mip),
            ("coronal", coronal_middle, coronal_mip),
        ]

    def select_slices(self, ct_volume: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
        if self.selection_strategy == "variance":
            return self._select_variance(ct_volume)
        if self.selection_strategy == "mip":
            return self._select_mip(ct_volume)
        return self._select_middle(ct_volume)

    def slice_to_image(self, slice_2d: np.ndarray, slice_idx: int, ct_path: str, 
                      plane_name: str = "axial", save_dir: str = "temp_slices", 
                      target_size: Optional[int] = None) -> str:
        """
        Convert CT slice to optimized image file for VLM processing.
        
        Args:
            slice_2d: 2D CT slice
            slice_idx: Slice index
            ct_path: Original CT path for naming
            plane_name: Plane name (axial, sagittal, coronal)
            save_dir: Directory to save temporary images
            target_size: Target image size (default: 512 for high-res VLM)
            
        Returns:
            Path to saved image
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Normalize slice
        normalized_slice = self.normalize_slice(slice_2d)
        
        # Convert to PIL Image (much faster than matplotlib)
        from PIL import Image
        image = Image.fromarray(normalized_slice, mode='L')  # Grayscale
        
        # Determine target size (fallback to class default)
        if target_size is None:
            target_size = self.target_image_size
        # Resize to target size (maintains aspect ratio with high quality)
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create filename
        case_id = Path(ct_path).parent.name
        scan_type = Path(ct_path).stem
        image_path = os.path.join(save_dir, f"{case_id}_{scan_type}_{plane_name}_slice_{slice_idx:03d}.png")
        
        # Save optimized PNG (much smaller file size)
        image.save(image_path, 'PNG', optimize=True, compress_level=6)
        
        return image_path

    def generate_report(
        self,
        image_path: str,
        plane_name: Optional[str] = None,
        slice_idx: Optional[int] = None,
        volume_shape: Optional[tuple] = None,
    ) -> str:
        """
        Generate medical report for a CT slice using Qwen2-VL.
        
        Args:
            image_path: Path to the CT slice image
            
        Returns:
            Generated medical report
        """
        try:
            # Prepare messages in the correct format for Qwen2VL
            if plane_name is not None and slice_idx is not None and volume_shape is not None:
                # volume_shape is (D, H, W). Choose axis length by plane
                if plane_name == "axial":
                    axis_len = volume_shape[0]
                elif plane_name == "coronal":
                    axis_len = volume_shape[1]
                elif plane_name == "sagittal":
                    axis_len = volume_shape[2]
                else:
                    axis_len = volume_shape[0]
                context = f"(PLANE: {plane_name}, SLICE: {slice_idx} of {axis_len})"
            elif plane_name is not None and slice_idx is not None:
                context = f"(PLANE: {plane_name}, SLICE: {slice_idx})"
            else:
                context = ""

            user_text = self.report_prompt_template.format(context=context)
            combined_text = f"{self.system_prompt}\n\n{user_text}" if self.system_prompt else user_text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": combined_text}
                    ]
                }
            ]
            
            # Build model inputs using the recommended vision/text processing
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Load image directly to ensure availability regardless of utils version
            from PIL import Image as _PILImage
            pil_image = _PILImage.open(image_path).convert("RGB")
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device)
            
            # Generate report (deterministic decoding for stable pseudo-labels)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Extract only the generated continuation (exclude input prompt)
            # Compute continuation safely across HF variants
            input_ids = inputs.get("input_ids", None)
            input_length = int(input_ids.shape[1]) if input_ids is not None else 0

            sequences = generated_ids
            # sequences may be (batch, seq) tensor
            if hasattr(sequences, "dim") and sequences.dim() == 2:
                sequences = sequences[0]
            # Guard against short outputs
            if hasattr(sequences, "shape") and sequences.shape[0] > input_length:
                continuation_tokens = sequences[input_length:]
            else:
                continuation_tokens = sequences

            decoded = self.processor.tokenizer.decode(continuation_tokens, skip_special_tokens=True)
            return decoded.strip()
            
        except Exception as e:
            print(f"Error generating report for {image_path}: {e}")
            return f"Error generating report: {str(e)}"

    def process_single_ct(self, ct_path: str, save_images: bool = False) -> Dict:
        """
        Process a single CT volume and generate reports for middle slices of all three planes.
        
        Args:
            ct_path: Path to CT .npy file
            save_images: Whether to keep the generated slice images
            
        Returns:
            Dictionary containing results
        """
        # print(f"\nProcessing CT: {ct_path}")
        
        # Load CT volume
        ct_volume = self.load_ct_volume(ct_path)
        if ct_volume is None:
            return {"error": f"Failed to load CT from {ct_path}"}
        
        # Select slices from all planes per configured strategy
        try:
            selected_slices = self.select_slices(ct_volume)
            # print(f"Selected slices: {[(plane, idx) for plane, idx, _ in selected_slices]}")
        except Exception as e:
            return {"error": f"Failed to select slices: {str(e)}"}
        
        # Generate reports for each slice
        results = {
            "ct_path": ct_path,
            "case_id": Path(ct_path).parent.name,
            "scan_type": Path(ct_path).stem,
            "volume_shape": ct_volume.shape,
            "slice_reports": []
        }
        
        temp_images = []
        
        for plane_name, slice_idx, slice_2d in selected_slices:
            # print(f"Processing {plane_name} slice {slice_idx}...")
            
            # Convert slice to image
            image_path = self.slice_to_image(slice_2d, slice_idx, ct_path, plane_name)
            temp_images.append(image_path)
            
            # Generate report
            report = self.generate_report(
                image_path,
                plane_name=plane_name,
                slice_idx=slice_idx,
                volume_shape=ct_volume.shape,
            )
            
            slice_result = {
                "plane": plane_name,
                "slice_index": slice_idx,
                "image_path": image_path if save_images else None,
                "report": report
            }
            
            results["slice_reports"].append(slice_result)
            # print(f"Generated report for {plane_name} slice {slice_idx}")
        
        # Clean up temporary images if not saving
        if not save_images:
            for img_path in temp_images:
                try:
                    os.remove(img_path)
                except:
                    pass
        
        return results

    def process_multiple_cts(self, ct_paths: List[str], 
                           output_file: str = "ct_reports.json", 
                           save_images: bool = False) -> List[Dict]:
        """
        Process multiple CT volumes.
        
        Args:
            ct_paths: List of paths to CT .npy files
            output_file: Output JSON file to save results
            save_images: Whether to keep generated slice images
            
        Returns:
            List of results for each CT
        """
        all_results = []
        
        for i, ct_path in enumerate(ct_paths):
            print(f"\n{'='*50}")
            print(f"Processing CT {i+1}/{len(ct_paths)}")
            print(f"{'='*50}")
            
            result = self.process_single_ct(ct_path, save_images)
            all_results.append(result)
            
            # Save intermediate results
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                # print(f"Saved intermediate results to {output_file}")
        
        print(f"\nCompleted processing {len(ct_paths)} CT volumes")
        return all_results

    def preprocess_ct_slices(self, ct_path: str, output_dir: str = "preprocessed_slices") -> Dict:
        """
        Preprocess CT volume by extracting and saving the 3 plane slices.
        
        Args:
            ct_path: Path to CT .npy file
            output_dir: Directory to save preprocessed slices
            
        Returns:
            Dictionary containing preprocessing results
        """
        # Load CT volume
        ct_volume = self.load_ct_volume(ct_path)
        if ct_volume is None:
            return {"error": f"Failed to load CT from {ct_path}"}
        
        # Create output directory
        case_id = Path(ct_path).parent.name
        scan_type = Path(ct_path).stem
        case_dir = os.path.join(output_dir, case_id, scan_type)
        os.makedirs(case_dir, exist_ok=True)
        
        # Extract slices per configured strategy
        selected_slices = self.select_slices(ct_volume)
        
        slice_paths = {}
        for plane_name, slice_idx, slice_2d in selected_slices:
            # Use optimized image processing
            image_path = self.slice_to_image(
                slice_2d, slice_idx, ct_path, plane_name, 
                save_dir=case_dir
            )
            
            slice_paths[plane_name] = {
                "path": image_path,
                "index": int(slice_idx),
                "plane": plane_name
            }
        
        return {
            "ct_path": ct_path,
            "case_id": case_id,
            "scan_type": scan_type,
            "volume_shape": tuple(int(dim) for dim in ct_volume.shape),
            "slice_paths": slice_paths,
            "preprocessed_dir": case_dir
        }

    def preprocess_multiple_cts(self, ct_paths: List[str], 
                               output_dir: str = "preprocessed_slices") -> List[Dict]:
        """
        Preprocess multiple CT volumes by extracting and saving slices.
        
        Args:
            ct_paths: List of paths to CT .npy files
            output_dir: Directory to save preprocessed slices
            
        Returns:
            List of preprocessing results for each CT
        """
        all_results = []
        
        for i, ct_path in enumerate(ct_paths):
            print(f"\n{'='*50}")
            print(f"Preprocessing CT {i+1}/{len(ct_paths)}")
            print(f"{'='*50}")
            
            result = self.preprocess_ct_slices(ct_path, output_dir)
            all_results.append(result)
            
            if "error" not in result:
                print(f"✓ Preprocessed {len(result['slice_paths'])} slices for {result['case_id']}/{result['scan_type']}")
            else:
                print(f"✗ Error: {result['error']}")
        
        print(f"\nCompleted preprocessing {len(ct_paths)} CT volumes")
        return all_results

    def process_preprocessed_ct(self, preprocessed_result: Dict, save_images: bool = False) -> Dict:
        """
        Process a preprocessed CT by generating reports for the saved slices.
        
        Args:
            preprocessed_result: Result from preprocess_ct_slices
            save_images: Whether to keep the generated slice images (not needed since slices are already saved)
            
        Returns:
            Dictionary containing results
        """
        if "error" in preprocessed_result:
            return preprocessed_result
        
        results = {
            "ct_path": preprocessed_result["ct_path"],
            "case_id": preprocessed_result["case_id"],
            "scan_type": preprocessed_result["scan_type"],
            "volume_shape": preprocessed_result["volume_shape"],
            "slice_reports": []
        }
        
        # Generate reports for each preprocessed slice
        for plane_name, slice_info in preprocessed_result["slice_paths"].items():
            image_path = slice_info["path"]
            
            # Generate report
            report = self.generate_report(
                image_path,
                plane_name=plane_name,
                slice_idx=slice_info["index"],
                volume_shape=preprocessed_result["volume_shape"],
            )
            
            slice_result = {
                "plane": plane_name,
                "slice_index": int(slice_info["index"]),
                "image_path": image_path,  # Always keep path since slices are preprocessed
                "report": report
            }
            
            results["slice_reports"].append(slice_result)
        
        return results

    def process_multiple_preprocessed_cts(self, preprocessed_results: List[Dict], 
                                        output_file: str = "ct_reports.json") -> List[Dict]:
        """
        Process multiple preprocessed CT volumes.
        
        Args:
            preprocessed_results: List of preprocessing results
            output_file: Output JSON file to save results
            
        Returns:
            List of results for each CT
        """
        all_results = []
        
        for i, preprocessed_result in enumerate(preprocessed_results):
            print(f"\n{'='*50}")
            print(f"Processing preprocessed CT {i+1}/{len(preprocessed_results)}")
            print(f"{'='*50}")
            
            result = self.process_preprocessed_ct(preprocessed_result)
            all_results.append(result)
            
            # Save intermediate results
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
        
        print(f"\nCompleted processing {len(preprocessed_results)} preprocessed CT volumes")
        return all_results


def find_ct_files(base_dir: str) -> List[str]:
    """
    Find all .npy CT files in the directory structure.
    
    Args:
        base_dir: Base directory to search (e.g., "/path/to/ct_case")
        
    Returns:
        List of paths to .npy files
    """
    ct_paths = []
    
    # Pattern 1: Direct recursive search
    pattern1 = os.path.join(base_dir, "**", "*.npy")
    files1 = glob.glob(pattern1, recursive=True)
    ct_paths.extend(files1)
    
    # Pattern 2: Specific case structure search
    pattern2 = os.path.join(base_dir, "*", "*.npy")
    files2 = glob.glob(pattern2)
    ct_paths.extend(files2)
    
    # Remove duplicates and sort
    ct_paths = sorted(list(set(ct_paths)))

    # Filter out obvious non-CT volumes (e.g., masks)
    filtered_paths = []
    for path in ct_paths:
        name = os.path.basename(path).lower()
        if name.endswith('.npy') and 'mask' in name:
            continue
        filtered_paths.append(path)
    ct_paths = filtered_paths
    
    print(f"Search patterns used:")
    print(f"  Pattern 1: {pattern1}")
    print(f"  Pattern 2: {pattern2}")
    print(f"Found {len(ct_paths)} total .npy files")
    
    if ct_paths:
        print(f"Example files found:")
        for i, path in enumerate(ct_paths[:5]):
            print(f"  {i+1}. {path}")
        if len(ct_paths) > 5:
            print(f"  ... and {len(ct_paths) - 5} more files")
    
    return ct_paths


def main():
    parser = argparse.ArgumentParser(description="Generate medical reports for CT slices using Qwen2-VL")
    parser.add_argument("--ct_dir", type=str, required=True, 
                       help="Directory containing CT case folders (e.g., /path/to/ct_case)")
    parser.add_argument("--output", type=str, default="ct_reports.json",
                       help="Output JSON file (default: ct_reports.json)")
    parser.add_argument("--preprocessed_dir", type=str, default="preprocessed_slices",
                       help="Directory for preprocessed slices (default: preprocessed_slices)")
    parser.add_argument("--preprocess_only", action="store_true",
                       help="Only preprocess CT slices, don't generate reports")
    parser.add_argument("--process_preprocessed", type=str, default=None,
                       help="Process preprocessed slices from specified directory")
    parser.add_argument("--save_images", action="store_true",
                       help="Save generated slice images")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Qwen model to use")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Target image size for VLM processing (default: 512)")
    parser.add_argument("--max_cts", type=int, default=None,
                       help="Maximum number of CTs to process (for testing)")
    parser.add_argument("--selection_strategy", type=str, default="middle", choices=["middle", "variance", "mip"],
                       help="Slice selection strategy per plane")
    parser.add_argument("--slices_per_plane", type=int, default=1,
                       help="Number of slices to select per plane (used in variance strategy)")
    parser.add_argument("--variance_window", type=int, default=10,
                       help="Half-window size around mid-slice to search by variance")
    parser.add_argument("--mip_window", type=int, default=5,
                       help="Half-window size around mid-slice for MIP slabs")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.ct_dir):
        print(f"Error: Directory {args.ct_dir} does not exist!")
        return
    
    print(f"Searching for CT files in: {args.ct_dir}")
    
    # Find all CT files
    ct_paths = find_ct_files(args.ct_dir)
    
    if not ct_paths:
        print(f"No .npy files found in {args.ct_dir}")
        print("Please check that:")
        print("1. The directory path is correct")
        print("2. The directory contains subdirectories with .npy files")
        print("3. The file structure matches: ct_case/case_id/scan_type.npy")
        return
    
    if args.max_cts:
        ct_paths = ct_paths[:args.max_cts]
        print(f"Limited to first {args.max_cts} files for testing")
    
    print(f"\nWill process {len(ct_paths)} CT files")
    
    # Initialize generator (single initialization)
    try:
        generator = CTSliceReportGenerator(
            model_name=args.model,
            target_image_size=args.image_size,
            selection_strategy=args.selection_strategy,
            slices_per_plane=args.slices_per_plane,
            variance_window=args.variance_window,
            mip_window=args.mip_window,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.process_preprocessed:
        # Process preprocessed slices from existing directory
        print(f"Processing preprocessed slices from: {args.process_preprocessed}")
        
        # Load preprocessed results from directory
        preprocessed_results = []
        for ct_path in ct_paths:
            case_id = Path(ct_path).parent.name
            scan_type = Path(ct_path).stem
            case_dir = os.path.join(args.process_preprocessed, case_id, scan_type)
            
            if os.path.exists(case_dir):
                slice_paths = {}
                for plane in ["axial", "sagittal", "coronal"]:
                    # Match filenames saved by preprocess: {case_id}_{scan_type}_{plane}_slice_XXX.png
                    slice_files = glob.glob(os.path.join(case_dir, f"*_{plane}_slice_*.png"))
                    if slice_files:
                        # Extract index from filename
                        filename = os.path.basename(slice_files[0])
                        index_str = filename.split("_slice_")[1].split(".")[0]
                        slice_idx = int(index_str)
                        
                        slice_paths[plane] = {
                            "path": slice_files[0],
                            "index": slice_idx,
                            "plane": plane
                        }
                
                if len(slice_paths) == 3:  # All 3 planes found
                    preprocessed_results.append({
                        "ct_path": ct_path,
                        "case_id": case_id,
                        "scan_type": scan_type,
                        "volume_shape": None,  # Not needed for processing
                        "slice_paths": slice_paths,
                        "preprocessed_dir": case_dir
                    })
        
        if not preprocessed_results:
            print("No preprocessed slices found!")
            return
        
        final_results = generator.process_multiple_preprocessed_cts(
            preprocessed_results=preprocessed_results,
            output_file=args.output
        )
        
    elif args.preprocess_only:
        # Only preprocess CT slices
        print("Preprocessing CT slices only...")
        preprocessed_results = generator.preprocess_multiple_cts(
            ct_paths=ct_paths,
            output_dir=args.preprocessed_dir
        )
        print(f"Preprocessing completed. Slices saved to: {args.preprocessed_dir}")
        
    else:
        # Full pipeline: preprocess + process
        print("Running full pipeline: preprocessing + processing...")
        
        # Preprocess CTs
        preprocessed_results = generator.preprocess_multiple_cts(
            ct_paths=ct_paths,
            output_dir=args.preprocessed_dir
        )
        
        # Process preprocessed CTs
        final_results = generator.process_multiple_preprocessed_cts(
            preprocessed_results=preprocessed_results,
            output_file=args.output
        )
    
    if not args.preprocess_only:
        print(f"\nFinal results saved to {args.output}")
        print(f"Successfully processed {len([r for r in final_results if 'error' not in r])} out of {len(final_results)} CT files")


if __name__ == "__main__":
    # Example usage for testing with your specific directory structure
    print("CT Report Generator for M3D Dataset")
    print("="*50)
    
    # Test if running directly - you can modify this for quick testing
    if len(os.sys.argv) == 1:  # No command line arguments provided
        print("No arguments provided. Here are usage examples:")
        print()
        print("1. Full pipeline (preprocess + process):")
        print('   python ct_report_generator.py --ct_dir "./Data/data/M3D_Cap_npy/ct_case"')
        print()
        print("2. Preprocess only (extract slices):")
        print('   python ct_report_generator.py --ct_dir "./Data/data/M3D_Cap_npy/ct_case" --preprocess_only')
        print()
        print("3. Process preprocessed slices only:")
        print('   python ct_report_generator.py --ct_dir "./Data/data/M3D_Cap_npy/ct_case" --process_preprocessed preprocessed_slices')
        print()
        print("4. Test with limited files:")
        print('   python ct_report_generator.py --ct_dir "./Data/data/M3D_Cap_npy/ct_case" --max_cts 5')
        print()
        
        # Quick test to find files
        test_dir = "./Data/data/M3D_Cap_npy/ct_case"
        if os.path.exists(test_dir):
            print(f"Testing file discovery in: {test_dir}")
            ct_files = find_ct_files(test_dir)
            if ct_files:
                print(f"✓ Found {len(ct_files)} CT files - ready to process!")
            else:
                print("✗ No CT files found - please check directory structure")
        else:
            print(f"Directory {test_dir} not found - please update the path")
    else:
        # Run main function with command line arguments
        main()
