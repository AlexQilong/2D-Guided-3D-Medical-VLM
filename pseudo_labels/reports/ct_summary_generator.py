import os
import json
import argparse
from typing import Dict, List, Optional

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


class CTReportSummarizer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 512,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        disable_hf_progress: bool = True,
    ) -> None:
        """Initialize a text-only summarizer using the same 2D VLM.

        The model is multimodal but supports pure text chat. We compose a
        conservative radiology prompt and decode deterministically.
        """
        if disable_hf_progress:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        if cache_dir:
            os.environ.setdefault("HF_HOME", cache_dir)

        print(
            f"Loading model for summarization: {model_name} | device_map={device} | "
            f"local_files_only={local_files_only} | cache_dir={cache_dir or 'default'}"
        )
        load_kwargs = dict(
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if hf_token:
            load_kwargs["token"] = hf_token

        self.model = _QwenModelClass.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.processor = _QwenProcessorClass.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            token=hf_token,
        )
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens

        self.system_prompt = (
            "You are a board-certified radiologist. Produce a conservative, scan-level summary by "
            "integrating three slice-level CT reports (axial, sagittal, coronal). Use standard radiology "
            "terminology, avoid speculation, and do not introduce findings that are not present in the inputs. "
            "Resolve any conflicts conservatively and prefer consensus across planes. Deduplicate repeated "
            "content and keep the result concise and clinically accurate."
        )

    def _build_messages(
        self,
        axial_report: Optional[str],
        sagittal_report: Optional[str],
        coronal_report: Optional[str],
        case_id: Optional[str] = None,
        scan_type: Optional[str] = None,
    ) -> List[Dict]:
        context_lines: List[str] = []
        if case_id is not None:
            context_lines.append(f"CASE: {case_id}")
        if scan_type is not None:
            context_lines.append(f"SCAN: {scan_type}")
        context_str = " | ".join(context_lines)

        user_text_parts: List[str] = []
        if context_str:
            user_text_parts.append(context_str)
        user_text_parts.append(
            "Summarize the following three slice-level reports into a single scan-level summary."
        )
        if axial_report:
            user_text_parts.append("AXIAL REPORT:\n" + axial_report.strip())
        if sagittal_report:
            user_text_parts.append("SAGITTAL REPORT:\n" + sagittal_report.strip())
        if coronal_report:
            user_text_parts.append("CORONAL REPORT:\n" + coronal_report.strip())

        user_text_parts.append(
            "Requirements:\n"
            "- Include all key abnormalities mentioned in any plane when they are not contradictory.\n"
            "- If planes disagree, prefer consensus; otherwise mark uncertainty briefly.\n"
            "- Do not copy slice-only indexing or plane tags; produce a scan-level view.\n"
            "- Do not invent new findings; avoid patient identifiers.\n"
            "- Deduplicate repeated content; keep language concise.\n\n"
            "Return only these sections in order:\n\n"
            "FINDINGS:\n- ...\n\n"
            "IMPRESSION:\n- ...\n"
        )
        combined_text = self.system_prompt + "\n\n" + "\n\n".join(user_text_parts)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text},
                ],
            }
        ]
        return messages

    def summarize_three_reports(
        self,
        plane_to_report: Dict[str, str],
        case_id: Optional[str] = None,
        scan_type: Optional[str] = None,
    ) -> str:
        """Summarize the three orthogonal reports into a scan-level summary."""
        axial_report = plane_to_report.get("axial")
        sagittal_report = plane_to_report.get("sagittal")
        coronal_report = plane_to_report.get("coronal")

        messages = self._build_messages(
            axial_report=axial_report,
            sagittal_report=sagittal_report,
            coronal_report=coronal_report,
            case_id=case_id,
            scan_type=scan_type,
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode continuation safely
        input_ids = inputs.get("input_ids", None)
        prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0
        sequences = generated_ids
        if hasattr(sequences, "dim") and sequences.dim() == 2:
            sequences = sequences[0]
        if hasattr(sequences, "shape") and sequences.shape[0] > prompt_len:
            continuation = sequences[prompt_len:]
        else:
            continuation = sequences
        decoded = self.processor.tokenizer.decode(continuation, skip_special_tokens=True)
        return decoded.strip()


def process_ct_reports(
    input_json_path: str,
    output_json_path: str,
    model_name: str,
    max_cts: Optional[int] = None,
    device: str = "auto",
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[Dict]:
    """Read ct_reports.json and write ct_summaries.json with scan-level summaries."""
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    with open(input_json_path, "r") as f:
        ct_items = json.load(f)

    if not isinstance(ct_items, list):
        raise ValueError("Input JSON must be a list of CT entries")

    if max_cts is not None:
        ct_items = ct_items[: int(max_cts)]

    summarizer = CTReportSummarizer(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    summaries: List[Dict] = []
    for idx, item in enumerate(ct_items):
        print("=" * 50)
        print(f"Summarizing CT {idx + 1}/{len(ct_items)}")
        print("=" * 50)

        case_id = item.get("case_id")
        scan_type = item.get("scan_type")
        slice_reports = item.get("slice_reports", [])

        plane_to_report: Dict[str, str] = {}
        for sr in slice_reports:
            plane = (sr.get("plane") or "").lower()
            report_text = sr.get("report")
            if plane and isinstance(report_text, str):
                # Prefer first occurrence per plane
                if plane not in plane_to_report:
                    plane_to_report[plane] = report_text

        summary_text = summarizer.summarize_three_reports(
            plane_to_report=plane_to_report,
            case_id=case_id,
            scan_type=scan_type,
        )

        summaries.append(
            {
                "ct_path": item.get("ct_path"),
                "case_id": case_id,
                "scan_type": scan_type,
                "summary": summary_text,
                "planes_present": sorted(list(plane_to_report.keys())),
            }
        )

        if output_json_path:
            with open(output_json_path, "w") as f:
                json.dump(summaries, f, indent=2)

    print(f"\nCompleted summarizing {len(ct_items)} CT entries")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize three orthogonal 2D reports per CT into a scan-level summary using Qwen2-VL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ct_reports.json",
        help="Path to ct_reports.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ct_summaries.json",
        help="Path to output summaries JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name to use for summarization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device map for loading (e.g., "auto", "cpu", "cuda")',
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force loading from local cache only (no network)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face cache directory (also sets HF_HOME)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private or rate-limited downloads",
    )
    parser.add_argument(
        "--max_cts",
        type=int,
        default=None,
        help="Optional limit for number of CT entries to summarize",
    )

    args = parser.parse_args()

    process_ct_reports(
        input_json_path=args.input,
        output_json_path=args.output,
        model_name=args.model,
        max_cts=args.max_cts,
        device=args.device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
