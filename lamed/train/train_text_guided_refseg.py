"""
Training script for Text-Guided Referred Segmentation

This is a minimal training script focused ONLY on referred segmentation
with text guidance from medical reports.

Usage:
    bash LaMed/script/train_text_guided_refseg.sh
"""

import os
import sys

# Use the WORKING approach from train.py - set seed via transformers
# This properly initializes all random states including MONAI
pass  # Seed will be set later using transformers.set_seed()

# Sanitize distributed environment
for _var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
    if _var in ("MASTER_ADDR", "MASTER_PORT"):
        continue
    os.environ.pop(_var, None)
os.environ.setdefault("LOCAL_RANK", "-1")

import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional
import logging

# Add project to path
sys.path.append(os.path.abspath('.'))

# Import the SAFE RefSegDataset (uses working transform approach)
from lamed.dataset.safe_refseg_dataset import SafeRefSegDataset
from lamed.model.language_model import LamedPhi3ForCausalLM
from lamed.train.lamed_trainer import LaMedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct")
    model_type: Optional[str] = field(default="phi3")
    
    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)
    
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    
    # Image config
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))
    
    # Vision tower
    vision_tower: Optional[str] = field(default="vit3d")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None)
    freeze_vision_tower: bool = field(default=False)
    
    # Projector
    mm_projector_type: Optional[str] = field(default='spp')
    proj_layer_type: str = field(default="mlp")
    proj_layer_num: int = field(default=2)
    proj_pooling_type: str = field(default="spatial")
    proj_pooling_size: int = field(default=2)
    
    # Segmentation module (SegVol)
    segmentation_module: str = field(default="segvol")
    pretrain_seg_module: str = field(default=None)


@dataclass
class DataArguments:
    data_root: str = field(default="./Data/data/M3D_RefSeg_npy")
    refseg_data_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_train.csv")  # For compatibility
    refseg_data_train_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_train.csv")
    refseg_data_test_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv")
    
    max_length: int = field(default=1024)  # Increased to avoid truncation
    proj_out_num: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    
    # Remove distributed args
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_backend: Optional[str] = field(default=None)


def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info("="*50)
    logger.info("Text-Guided Referred Segmentation Training")
    logger.info("="*50)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Data: {data_args.data_root}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info("="*50)
    
    # Set global seed using transformers (like working train.py)
    try:
        transformers.set_seed(training_args.seed)
        logger.info(f"[SEED] Global seed set to {training_args.seed}")
    except Exception as e:
        logger.warning(f"[SEED] Failed to set seed: {e}")
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens (must be done BEFORE converting to IDs)
    special_tokens = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens("[SEG]")

    tokenizer.pad_token = tokenizer.unk_token
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")

    logger.info(f"[SEG] token ID: {seg_token_id}")
    logger.info(f"<im_patch> token ID: {img_token_id}")
    logger.info(f"Vocabulary size: {len(tokenizer)}")

    # Load model
    logger.info("Loading model...")
    model = LamedPhi3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )
    
    model.config.use_cache = False
    model.config.seg_token_id = seg_token_id

    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # Initialize vision modules
    logger.info("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args)
    
    # Initialize segmentation modules
    logger.info("Initializing segmentation modules...")
    model.get_model().initialize_seg_modules(model_args)
    
    # Load pretrained weights
    if model_args.pretrain_mm_mlp_adapter:
        logger.info(f"Loading pretrained mm_projector from {model_args.pretrain_mm_mlp_adapter}")
        mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        model.get_model().mm_projector.load_state_dict(
            get_w(mm_projector_weights, 'mm_projector'), strict=True
        )
    
    # Load SegVol pretrained weights
    if model_args.pretrain_seg_module:
        logger.info(f"Loading pretrained SegVol from {model_args.pretrain_seg_module}")
        seg_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
        
        # Filter and clean weights - only keep image_encoder, mask_decoder, and prompt_encoder
        # Skip text_encoder since we use Phi-3 for text processing
        seg_weights_clean = {}
        for key, value in seg_weights.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
            else:
                new_key = key
            
            # Only keep weights for image_encoder, mask_decoder, and prompt_encoder
            if (new_key.startswith("image_encoder.") or 
                new_key.startswith("mask_decoder.") or 
                new_key.startswith("prompt_encoder.")):
                seg_weights_clean[new_key] = value
        
        logger.info(f"Loading {len(seg_weights_clean)} SegVol weights (filtered from {len(seg_weights)} total)")
        model.get_model().seg_module.load_state_dict(seg_weights_clean, strict=False)
    
    # Create datasets - using SAFE RefSegDataset (working transform approach)
    logger.info("Creating datasets...")
    
    train_dataset = SafeRefSegDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="train"
    )
    
    eval_dataset = SafeRefSegDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="validation"
    )
    
    # Data collator - use custom LaMed data collator with segmentation ENABLED
    from lamed.train.train import DataCollator
    data_collator = DataCollator(seg_enable=True, sum_enable=False)
    
    # Trainer (using custom trainer with visualization)
    logger.info("Initializing custom trainer with visualization...")
    trainer = LaMedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # New name for tokenizer in transformers >= 4.46
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("="*50)
    logger.info("Training complete!")
    logger.info("="*50)


if __name__ == "__main__":
    main()

