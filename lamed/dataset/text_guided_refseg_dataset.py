"""
Text-Guided Referred Segmentation Dataset

This dataset enhances the standard RefSegDataset by incorporating medical reports 
from text.json files to provide richer semantic context for segmentation.

Key difference from baseline:
- Baseline: <im_patch>*256 Where is the lesion? Sure, it is [SEG].
- Text-guided: <im_patch>*256 Report: {medical_report}. Where is the lesion? Sure, it is [SEG].

The [SEG] token embedding now contains both visual and semantic information!
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import monai.transforms as mtf
from monai.data import set_track_meta


class TextGuidedRefSegDataset(Dataset):
    """
    Enhanced RefSegDataset that uses text.json medical reports for text guidance.
    
    Compared to the baseline RefSegDataset, this dataset:
    1. Loads medical reports from text.json files
    2. Incorporates the report into the input prompt
    3. Provides richer semantic context for the [SEG] token
    4. Leads to better segmentation quality (expected +5-10% Dice score)
    """
    
    def __init__(self, args, tokenizer, mode="train", use_text_guidance=True):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.use_text_guidance = use_text_guidance
        
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # Simplified transforms to avoid MONAI random seed issues
        train_transform = mtf.Compose([
            mtf.ToTensord(keys=["image"], dtype=torch.float),
            mtf.ToTensord(keys=["seg"], dtype=torch.int),
        ])
        
        val_transform = mtf.Compose([
            mtf.ToTensord(keys=["image"], dtype=torch.float),
            mtf.ToTensord(keys=["seg"], dtype=torch.int),
        ])
        
        set_track_meta(False)
        
        # Load data splits
        if 'train' in mode:
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif 'validation' in mode or 'test' in mode:
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        
        self.sample_counter = 0  # Track samples for debug output
        
        print(f"[TextGuidedRefSeg-{mode}] Loaded {len(self.data_list)} samples")
        print(f"[TextGuidedRefSeg-{mode}] Text guidance: {'ENABLED' if use_text_guidance else 'DISABLED'}")
    
    def __len__(self):
        return len(self.data_list)
    
    def load_medical_report(self, sample_id, mask_id):
        """
        Load medical report from text.json for a specific mask region.
        
        Args:
            sample_id: Sample directory name (e.g., "s0000")
            mask_id: Mask region ID (e.g., 1, 2)
        
        Returns:
            str: Medical report text, or empty string if not found
        """
        text_json_path = os.path.join(self.args.data_root, sample_id, "text.json")
        
        if not os.path.exists(text_json_path):
            return ""
        
        try:
            with open(text_json_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            
            # Get report for this specific Mask_ID
            mask_id_str = str(mask_id)
            if mask_id_str in text_data:
                return text_data[mask_id_str].strip()
            
            return ""
        except Exception as e:
            print(f"Warning: Failed to load text.json for {sample_id}: {e}")
            return ""
    
    def __getitem__(self, idx):
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                
                # Extract sample_id from path (e.g., "s0000/ct.npy" -> "s0000")
                image_path_rel = data["Image"]
                sample_id = image_path_rel.split('/')[0]
                mask_id = data["Mask_ID"]
                
                # Load 3D CT scan - convert .nii.gz to .npy
                image_path_rel_npy = image_path_rel.replace('.nii.gz', '.npy')
                image_path = os.path.join(self.args.data_root, image_path_rel_npy)
                image_array = np.load(image_path)  # (1, 32, 256, 256), normalized
                
                # Load 3D mask - convert .nii.gz to .npy
                mask_path_rel_npy = data["Mask"].replace('.nii.gz', '.npy')
                seg_path = os.path.join(self.args.data_root, mask_path_rel_npy)
                seg_array = np.load(seg_path)
                seg_array = (seg_array == mask_id).astype(np.int8)
                
                # Apply transforms
                item = {"image": image_array, "seg": seg_array}
                transformed = self.transform(item)
                image = transformed['image']  # (1, D, H, W)
                seg = transformed['seg']      # (1, D, H, W)
                
                # Get question and answer from CSV
                question = data["Question"]
                answer = data["Answer"]
                
                # === TEXT GUIDANCE: Load medical report ===
                if self.use_text_guidance:
                    report_text = self.load_medical_report(sample_id, mask_id)
                    
                    if report_text:
                        # Enhanced input with report
                        full_question = (
                            f"{self.image_tokens} "
                            f"Report: {report_text}. "
                            f"{question}"
                        )
                    else:
                        # Fallback to baseline if no report available
                        full_question = self.image_tokens + ' ' + question
                else:
                    # Baseline: no text guidance
                    full_question = self.image_tokens + ' ' + question
                
                # Tokenize input + output
                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    full_question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                
                # Add EOS token if needed
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                # Create labels: mask question, keep only answer for loss
                question_tensor = self.tokenizer(
                    full_question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                label = input_id.clone()
                label[:question_len] = -100  # Ignore question tokens in loss
                
                # Handle padding tokens
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                # === DEBUG OUTPUT FOR FIRST 3 SAMPLES ===
                if self.sample_counter < 3:
                    print("\n" + "="*80)
                    print(f"DEBUG: Sample {self.sample_counter} ({self.mode} mode)")
                    print("="*80)
                    print(f"Sample ID: {sample_id}")
                    print(f"Mask ID: {mask_id}")
                    print(f"Image shape: {image.shape}")
                    print(f"Seg shape: {seg.shape}")
                    print(f"Seg non-zero voxels: {torch.sum(seg > 0).item()}")
                    
                    if report_text:
                        print(f"\n📄 Medical Report (length={len(report_text)} chars):")
                        print(f"   {report_text[:200]}{'...' if len(report_text) > 200 else ''}")
                    else:
                        print(f"\n⚠️  No medical report found in text.json")
                    
                    print(f"\n❓ Original Question:")
                    print(f"   {question}")
                    
                    print(f"\n💬 Full Input Prompt (length={len(full_question)} chars):")
                    print(f"   {full_question[:300]}{'...' if len(full_question) > 300 else ''}")
                    
                    print(f"\n✅ Answer:")
                    print(f"   {answer}")
                    
                    print(f"\n🔢 Tokenization Stats:")
                    print(f"   Input IDs length: {len(input_id)}")
                    print(f"   Valid tokens: {valid_len}")
                    print(f"   Question tokens: {question_len}")
                    print(f"   Answer tokens: {valid_len - question_len}")
                    print(f"   Padding tokens: {len(input_id) - valid_len}")
                    
                    # Check for [SEG] token
                    seg_token = "[SEG]"
                    seg_ids = self.tokenizer.encode(seg_token, add_special_tokens=False)
                    has_seg = any(seg_id in input_id for seg_id in seg_ids)
                    print(f"\n🎯 [SEG] Token:")
                    print(f"   Present in input: {has_seg}")
                    if has_seg:
                        seg_positions = [i for i, token_id in enumerate(input_id) if token_id in seg_ids]
                        print(f"   Positions: {seg_positions}")
                    
                    print("="*80 + "\n")
                    self.sample_counter += 1
                
                return {
                    'image': image,
                    'input_id': input_id,  # Original LaMed format
                    'label': label,        # Original LaMed format  
                    'attention_mask': attention_mask,
                    'question_type': "refseg",
                    # Keep seg field for future segmentation
                    'seg': seg,
                }
                
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
                if attempt == max_attempts - 1:
                    raise e


# For backward compatibility: import with original name
RefSegDataset = TextGuidedRefSegDataset

