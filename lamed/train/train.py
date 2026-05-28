import os
import os

# Sanitize any distributed env to avoid accidental init
for _var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
    # Keep user-specified unique ports if explicitly set by launcher
    if _var in ("MASTER_ADDR", "MASTER_PORT"):
        continue
    os.environ.pop(_var, None)
# Ensure LOCAL_RANK is -1 for single process
os.environ.setdefault("LOCAL_RANK", "-1")

# Safe shims for torch.distributed when not initialized
import torch
import torch.distributed as _dist
if hasattr(_dist, "get_world_size"):
    _orig_get_world_size = _dist.get_world_size
    def _safe_get_world_size(group=None):
        try:
            return _orig_get_world_size(group)
        except Exception:
            return 1
    _dist.get_world_size = _safe_get_world_size
if hasattr(_dist, "get_rank"):
    _orig_get_rank = _dist.get_rank
    def _safe_get_rank(group=None):
        try:
            return _orig_get_rank(group)
        except Exception:
            return 0
    _dist.get_rank = _safe_get_rank

# SLURM compatibility - force single process mode
if 'SLURM_JOB_ID' in os.environ:
    # Running under SLURM; do not set WORLD_SIZE/RANK to avoid DDP
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    # Force single GPU binding if SLURM provides local id
    if 'SLURM_LOCALID' in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SLURM_LOCALID']
else:
    # Local development
    os.environ["LOCAL_RANK"] = "-1"
    os.environ["RANK"] = "-1" 
    os.environ["WORLD_SIZE"] = "1"

# Optional: print CUDA version without invoking collectives
print("CUDA:", torch.version.cuda)

import logging
import random
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field

import sys
# Add the parent directory of LaMed to sys.path
sys.path.append(os.path.abspath('.'))

# from lamed.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
from lamed.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset, TextGuidanceDatasets, TextOnlyCapDatasetSum, ImageTextCapDatasetSum
from lamed.dataset.flexible_cap_dataset import FlexibleCapDataset
from lamed.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM, LamedPhi3ForCausalLMSum
from lamed.train.lamed_trainer import LaMedTrainer

# Use plain torch device; avoid Accelerate to prevent implicit process-group init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})

    # summarizer
    sum_enable: bool = field(default=False)


@dataclass
class DataArguments:
    data_root: str = field(default="./Data/data/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="Data/data/M3D_Cap_npy/filtered_train_all.json", metadata={"help": "Path to caption data."})
    sample_limit: str = field(default=None, metadata={"help": "3D sample cap (backward-compat). Prefer sample_limit_3d."})
    sample_limit_3d: Optional[int] = field(default=None, metadata={"help": "Limit number of 3D GT samples used."})
    sample_limit_2d: Optional[int] = field(default=0, metadata={"help": "Number of 2D-only samples to add (make-up)."})

    # VQA data
    vqa_data_train_path: str = field(default="./Data/data/M3D-VQA/filtered_M3D_VQA_train.csv", metadata={"help": "Path to training VQA data."})
    vqa_data_val_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_val.csv", metadata={"help": "Path to validation VQA data."})
    vqa_data_test_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_test.csv", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="./Data/data/M3D-VQA/filtered_M3D_VQA_yn_train.csv", metadata={"help": "Path to training VQA Yes or No data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="./Data/data/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
    refseg_data_train_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})
    # Optional: path to JSON summaries (case_id/scan_type keyed) to use as 3D targets
    summaries_json_path: Optional[str] = field(
        default="./ct_summaries.json",
        metadata={"help": "Path to ct_summaries.json for scan-level targets (if exists)"}
    )
    # Unified training with configurable weights (replaces training_mode)
    weight_2d: float = field(
        default=0.4, 
        metadata={"help": "Weight for 2D summary loss (0.0=3D mode, 1.0=2D mode, 0.4=mixed)"}
    )
    weight_3d: float = field(
        default=0.6, 
        metadata={"help": "Weight for 3D GT report loss (0.0=2D mode, 1.0=3D mode, 0.6=mixed)"}
    )

    # Control how many 2D reports to include per sample (guidance within a sample)
    two_d_reports_per_sample: Optional[int] = field(
        default=None,
        metadata={"help": "Max 2D reports concatenated per sample (None/<=0 = all)"}
    )
    # Alias for 2D sample limit (kept for CLI compatibility): if set and sample_limit_2d is None, will be used as 2D make-up sample count
    num_2d_guidance: Optional[int] = field(default=None, metadata={"help": "Alias for sample_limit_2d (2D make-up sample count)."})

    # Overlap control between 3D-limited pool and 2D make-up pool
    two_d_overlap_mode: str = field(default="disjoint", metadata={"help": "Overlap mode for 2D pool vs 3D pool: same|disjoint|mixed"})
    two_d_overlap_ratio: float = field(default=0.5, metadata={"help": "If mixed, fraction of 2D pool drawn from 3D pool (0-1)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "no"  # Disable eval for faster training
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 100 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]

    if len(filtered_labels) > 0:
        acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)
    else:
        acc_score = 0  # or some other appropriate value

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):  # If logits is a tuple, extract the first element
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def maybe_zero_3(param, ignore_status=False, name=None):
    # from deepspeed import zero
    # from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    # if hasattr(param, "ds_id"):
    #     if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
    #         if not ignore_status:
    #             logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
    #     with zero.GatheredParameters([param]):
    #         param = param.data.detach().cpu().clone()
    # else:
    #     param = param.detach().cpu().clone()
    # return param
    return None

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable, sum_enable):
        self.seg_enable = seg_enable
        self.sum_enable = sum_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        elif self.sum_enable:
            # Image + text mode
            images, input_ids, labels, attention_mask, guidance_tokens, guidance_attention_mask = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'guidance_tokens', 'guidance_attention_mask')
            )
        
            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
            guidance_tokens = torch.stack(guidance_tokens, dim=0)
            guidance_attention_mask = torch.stack(guidance_attention_mask, dim=0)
            
            return dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                guidance_tokens=guidance_tokens,
                guidance_attention_mask=guidance_attention_mask,
            )
        else:
            # Remove bad samples
            batch = [b for b in batch if b is not None]

            if len(batch) == 0:
                return None  # Or raise error / skip this batch gracefully

            # Required fields
            images = torch.stack([b["image"] for b in batch], dim=0)
            input_ids = torch.stack([b["input_id"] for b in batch], dim=0)
            labels = torch.stack([b["label"] for b in batch], dim=0)
            attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
            question_type = [str(b["question_type"]) for b in batch]
            print(f"Question types: {question_type}")
            
            # Extract weights and target sources for debugging
            weights_2d = [b.get("weight_2d", 0.4) for b in batch]
            weights_3d = [b.get("weight_3d", 0.6) for b in batch]
            target_sources = [b.get("target_source", "UNKNOWN") for b in batch]
            print(f"2D weights: {weights_2d} | 3D weights: {weights_3d}")
            print(f"Target sources: {target_sources}")

            # Initialize return dict
            output = {
                "images": images,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "question_type": question_type,
                "weights_2d": weights_2d,  # Pass to model
                "weights_3d": weights_3d,  # Pass to model
                "target_sources": target_sources,  # For debugging
            }

            # Optional fields
            optional_keys = [
                "guidance_tokens",
                "guidance_attention_mask",
                "summarization_tokens",
                "summarization_attention_mask",
            ]

            for key in optional_keys:
                if all(key in b for b in batch):
                    output[key] = torch.stack([b[key] for b in batch], dim=0)
            
            # Handle mixed mode dual targets
            mixed_mode_keys = [
                "label_2d", "label_3d", 
                "input_id_2d", "input_id_3d",
                "attention_mask_2d", "attention_mask_3d"
            ]
            
            for key in mixed_mode_keys:
                if all(key in b for b in batch):
                    output[key] = torch.stack([b[key] for b in batch], dim=0)
                    print(f"Added {key} to batch with shape: {output[key].shape}")

            return output


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print("Model args:", model_args)
    # print("Data args:", data_args)
    # print("Training args:", training_args)

    local_rank = training_args.local_rank

    # Set a global seed once for deterministic shuffling and reproducibility
    try:
        transformers.set_seed(training_args.seed)
        print(f"[SEED] Global seed set to {training_args.seed}")
    except Exception as e:
        print(f"[SEED] Failed to set seed: {e}")

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif 'phi3' in model_args.model_type:
            if model_args.sum_enable:
                model = LamedPhi3ForCausalLMSum.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir
                    )
            else:
                model = LamedPhi3ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir
                    )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.tokenizer = tokenizer
    
    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    # ckpt = torch.load("PATH/model_with_lora.bin", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")

    # disable when training without SEG masks
    data_args.seg_enable = False

    if model_args.tune_mm_mlp_adapter:
        train_dataset = TextDatasets(data_args, tokenizer, mode='train')
    elif model_args.sum_enable:
        # Pass sample_limit to the dataset
        if hasattr(data_args, 'sample_limit') and data_args.sample_limit is not None:
            print(f"Using sample_limit (legacy): {data_args.sample_limit}")
        if getattr(data_args, 'sample_limit_3d', None) is not None:
            print(f"Using sample_limit_3d: {data_args.sample_limit_3d}")
        
        # DEBUG: Print which dataset is being used
        print(f"\n=== DEBUG: Summarizer Training Configuration ===")
        print(f"sum_enable: {model_args.sum_enable}")
        print(f"sample_limit: {data_args.sample_limit}")
        print(f"model_max_length: {training_args.model_max_length}")
        print(f"Using ImageTextCapDatasetSum (with images)")
        print("=" * 50)
        
        # train_dataset = TextOnlyCapDatasetSum(data_args, tokenizer, mode='train')
        train_dataset = ImageTextCapDatasetSum(data_args, tokenizer, mode='train')
    else:  # train VLM - NEW: Use FlexibleCapDataset for 3 training modes
        # Resolve 3D limit
        sample_limit_int = int(data_args.sample_limit) if data_args.sample_limit else None
        if getattr(data_args, 'sample_limit_3d', None) is not None:
            sample_limit_int = int(data_args.sample_limit_3d)
        
        print(f"\n=== DEBUG: Flexible Training Configuration ===")
        print(f"weight_2d: {data_args.weight_2d} | weight_3d: {data_args.weight_3d}")
        if data_args.weight_2d == 0.0:
            print("MODE: 3D Only (weight_2d=0)")
        elif data_args.weight_3d == 0.0:
            print("MODE: 2D Only (weight_3d=0)")
        else:
            print("MODE: Mixed (both weights > 0)")
        print(f"sample_limit: {sample_limit_int}")
        print(f"summaries_json_path: {data_args.summaries_json_path}")
        print(f"model_max_length: {training_args.model_max_length}")
        print("=" * 50)
        
        # First, create the 3D-limited subset (primary pool)
        primary_dataset = FlexibleCapDataset(
            data_args,
            tokenizer,
            mode='train',
            sample_limit=sample_limit_int,
            weight_2d=data_args.weight_2d,
            weight_3d=data_args.weight_3d
        )

        # Determine if we need a 2D make-up pool
        two_d_makeup = getattr(data_args, 'sample_limit_2d', 0)
        if (two_d_makeup is None) and (getattr(data_args, 'num_2d_guidance', None) is not None):
            # Backward compat: use num_2d_guidance as 2D sample count if provided
            try:
                two_d_makeup = int(data_args.num_2d_guidance)
            except Exception:
                two_d_makeup = 0

        # Build a second dataset without 3D cap to draw 2D-only samples from
        # We reuse the same dataset class but we will pass subset indices after computing overlap selection
        full_dataset = FlexibleCapDataset(
            data_args,
            tokenizer,
            mode='train',
            sample_limit=None,
            weight_2d=data_args.weight_2d,
            weight_3d=data_args.weight_3d
        )

        # Compute overlap indices according to mode
        overlap_mode = getattr(data_args, 'two_d_overlap_mode', 'disjoint')
        overlap_ratio = float(getattr(data_args, 'two_d_overlap_ratio', 0.5))
        primary_indices = set(primary_dataset.valid_indices)
        all_indices = set(full_dataset.valid_indices)

        selected_two_d_indices = []
        if two_d_makeup and two_d_makeup > 0:
            if overlap_mode == 'same':
                pool = list(primary_indices)
                random.shuffle(pool)
                selected_two_d_indices = pool[:two_d_makeup]
            elif overlap_mode == 'disjoint':
                disjoint_pool = list(all_indices - primary_indices)
                random.shuffle(disjoint_pool)
                selected_two_d_indices = disjoint_pool[:two_d_makeup]
            else:  # mixed
                k_same = int(round(two_d_makeup * max(0.0, min(1.0, overlap_ratio))))
                k_disjoint = two_d_makeup - k_same
                pool_same = list(primary_indices)
                random.shuffle(pool_same)
                chosen_same = pool_same[:k_same]
                pool_disjoint = list(all_indices - primary_indices)
                random.shuffle(pool_disjoint)
                chosen_disjoint = pool_disjoint[:k_disjoint]
                selected_two_d_indices = chosen_same + chosen_disjoint

        # Construct the 2D-only dataset if needed
        if selected_two_d_indices:
            two_d_only_dataset = FlexibleCapDataset(
                data_args,
                tokenizer,
                mode='train',
                sample_limit=None,
                weight_2d=data_args.weight_2d,
                weight_3d=0.0,  # 2D only target for make-up pool
                subset_indices=selected_two_d_indices
            )
            # Simple concatenation of pools: primary (3D-limited) + 2D-only make-up samples
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([primary_dataset, two_d_only_dataset])
            print(f"DEBUG: Built train dataset with 3D-limited={len(primary_dataset)} and 2D-makeup={len(two_d_only_dataset)} => total={len(train_dataset)}")
        else:
            train_dataset = primary_dataset
            print(f"DEBUG: Built train dataset with 3D-limited only => total={len(train_dataset)}")

    # Use same small dataset for eval to match training size
    eval_dataset = FlexibleCapDataset(
        data_args, 
        tokenizer, 
        mode='train', 
        sample_limit=sample_limit_int,
        weight_2d=data_args.weight_2d,
        weight_3d=data_args.weight_3d
    )
    data_collator = DataCollator(data_args.seg_enable, model_args.sum_enable)

    rank0_print("="*20 + " Training " + "="*20)
    
    # DEBUG: Print dataset info before training starts
    print("\n" + "="*60)
    print("DEBUG: DATASET INFORMATION")
    print("="*60)
    print(f"Dataset type: {type(train_dataset)}")
    print(f"Dataset length: {len(train_dataset)}")
    print(f"Expected sample_limit: {sample_limit_int}")
    print(f"Training args batch_size: {training_args.per_device_train_batch_size}")
    print(f"Expected total steps: {len(train_dataset) // training_args.per_device_train_batch_size}")
    print(f"Model max length: {training_args.model_max_length}")
    
    # DEBUG: Test a few samples to see the prompts
    print("\nDEBUG: TESTING FIRST 3 SAMPLES")
    print("="*60)
    for i in range(min(3, len(train_dataset))):
        try:
            sample = train_dataset[i]
            print(f"\nSample {i}:")
            print(f"  Keys: {list(sample.keys())}")
            if 'question' in sample:
                print(f"  Question: {sample['question']}")
            if 'answer' in sample:
                print(f"  Answer preview: {sample['answer'][:100]}...")
            if 'input_id' in sample:
                print(f"  Input ID shape: {sample['input_id'].shape}")
            if 'label' in sample:
                print(f"  Label shape: {sample['label'].shape}")
        except Exception as e:
            print(f"Error getting sample {i}: {e}")
    
    print("\n" + "="*60)
    print("DEBUG: STARTING TRAINING")
    print("="*60)
    
    model.to(device)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                            sample_limit=data_args.sample_limit
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = False

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
