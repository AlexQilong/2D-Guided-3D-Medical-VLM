import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List
import json
import monai.transforms as mtf
from monai.data import set_track_meta

# Import existing helpers  
from .multi_dataset import load_summaries_json
from .prompt_templates import Caption_templates


class FlexibleCapDataset(Dataset):
    """
    Flexible dataset for 3 training modes:
    1. only_2d_summaries: Uses 2D summaries as targets (no GT files needed)
    2. only_gt_reports: Uses GT reports as targets (no 2D summaries needed)
    3. mixed: Uses both 2D summaries and GT reports as targets
    """
    def __init__(self, args, tokenizer, mode="train", sample_limit=None, weight_2d=0.4, weight_3d=0.6, subset_indices: Optional[List[int]] = None):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.sample_limit = sample_limit
        self.weight_2d = weight_2d
        self.weight_3d = weight_3d
        
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        # Load CT volume paths
        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]
        
        # DEBUG: Print dataset info
        print(f"DEBUG: Loaded {len(self.data_list)} samples from {args.cap_data_path} [{mode}]")
        
        # Same caption prompts as CapDataset
        self.caption_prompts = Caption_templates
        
        # No complex patient-level limiting - keep it simple
        
        # Always load summaries for unified training approach
        summaries_path = getattr(args, 'summaries_json_path', "./ct_summaries.json")
        self.summaries_map = load_summaries_json(summaries_path)
        if not self.summaries_map:
            print(f"Warning: weight_2d={self.weight_2d} but no summaries loaded from {summaries_path}!")
        
        # Transforms
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
        set_track_meta(False)
        
        if 'train' in mode:
            self.transform = train_transform
        else:
            self.transform = val_transform
        
        # Validate file existence and build valid indices to avoid crashes
        valid_indices_all = []
        missing_count = 0
        for idx_item, item in enumerate(self.data_list):
            img_rel = item.get('image')
            txt_rel = item.get('text')
            if not img_rel or not txt_rel:
                missing_count += 1
                continue
            img_abs = os.path.join(self.data_root, img_rel)
            txt_abs = os.path.join(self.data_root, txt_rel)
            if os.path.exists(img_abs) and os.path.exists(txt_abs):
                valid_indices_all.append(idx_item)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"WARNING: Skipping {missing_count} samples with missing files. Valid={len(valid_indices_all)}")

        # Apply subset selection or sample limit after validation
        if subset_indices is not None:
            # Keep only indices that are valid
            valid_set = set(valid_indices_all)
            filtered = [i for i in subset_indices if i in valid_set]
            if len(filtered) < len(subset_indices):
                print(f"WARNING: {len(subset_indices) - len(filtered)} requested indices were invalid and were dropped.")
            self.valid_indices = filtered
            print(f"DEBUG: FlexibleCapDataset: Using subset of {len(self.valid_indices)} samples (requested={len(subset_indices)}, available_valid={len(valid_indices_all)})")
        else:
            if self.sample_limit is not None:
                dataset_size = min(self.sample_limit, len(valid_indices_all))
                self.valid_indices = valid_indices_all[:dataset_size]
                print(f"DEBUG: FlexibleCapDataset: Using first {dataset_size} VALID samples (sample_limit={self.sample_limit}, available_valid={len(valid_indices_all)})")
            else:
                self.valid_indices = valid_indices_all
                print(f"DEBUG: FlexibleCapDataset: Using all {len(self.valid_indices)} VALID samples")
        
        print(f"DEBUG: FlexibleCapDataset: weights_2d={self.weight_2d}, weights_3d={self.weight_3d}, final_samples={len(self.valid_indices)}")
        
        # Check first few sample paths
        if len(self.valid_indices) > 0:
            for i in range(min(3, len(self.valid_indices))):
                sample = self.data_list[i]
                print(f"DEBUG: Sample {i}: image={sample.get('image', 'MISSING')}, text={sample.get('text', 'MISSING')}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Debug first few samples
        if idx < 3:
            print(f"DEBUG: __getitem__({idx}) -> actual_idx={actual_idx}, total_valid={len(self.valid_indices)}")
        
        for attempt in range(5):
            try:
                data = self.data_list[actual_idx]
                
                # Load 3D CT volume
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                image = np.load(image_abs_path)
                image = self.transform(image)
                
                # Determine target text based on training mode ONLY - input is always the same
                filename = os.path.basename(image_path)
                position = os.path.splitext(filename)[0]
                image_id = os.path.basename(os.path.dirname(image_path))
                key = f"{image_id}{position}"
                
                target_source = "UNKNOWN"
                # Always load BOTH targets for unified training
                answer_2d = self.summaries_map.get(key, "")
                
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as f:
                    answer_3d = f.read()
                
                # Use 3D as primary answer (backward compatibility)
                answer = answer_3d
                target_source = f"UNIFIED_2D({self.weight_2d})+3D({self.weight_3d})"
                
                # EXACT SAME INPUT as CapDataset - only target differs
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question
                
                # Tokenize primary target (used for single-target modes)
                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                
                # Loss masking for primary target
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])
                
                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100
                
                # Always create separate labels for 2D and 3D targets (unified approach)
                # Tokenize 2D summary target
                text_2d_tensor = self.tokenizer(
                    question + ' ' + answer_2d,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_id_2d = text_2d_tensor["input_ids"][0]
                attention_mask_2d = text_2d_tensor["attention_mask"][0]
                valid_len_2d = torch.sum(attention_mask_2d)
                if valid_len_2d < len(input_id_2d):
                    input_id_2d[valid_len_2d] = self.tokenizer.eos_token_id
                
                label_2d = input_id_2d.clone()
                label_2d[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label_2d[label_2d == self.tokenizer.pad_token_id] = -100
                    if valid_len_2d < len(label_2d):
                        label_2d[valid_len_2d] = self.tokenizer.eos_token_id
                else:
                    label_2d[label_2d == self.tokenizer.pad_token_id] = -100
                
                # Tokenize 3D GT target  
                text_3d_tensor = self.tokenizer(
                    question + ' ' + answer_3d,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_id_3d = text_3d_tensor["input_ids"][0]
                attention_mask_3d = text_3d_tensor["attention_mask"][0]
                valid_len_3d = torch.sum(attention_mask_3d)
                if valid_len_3d < len(input_id_3d):
                    input_id_3d[valid_len_3d] = self.tokenizer.eos_token_id
                
                label_3d = input_id_3d.clone()
                label_3d[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label_3d[label_3d == self.tokenizer.pad_token_id] = -100
                    if valid_len_3d < len(label_3d):
                        label_3d[valid_len_3d] = self.tokenizer.eos_token_id
                else:
                    label_3d[label_3d == self.tokenizer.pad_token_id] = -100
                
                # Debug info
                if idx < 3:
                    print(f"=== DEBUG FlexibleCapDataset UNIFIED ===")
                    print(f"Sample {idx}: key={key} | Target: {target_source}")
                    print(f"Primary tokens used: {int(valid_len)}/{len(input_id)} | question_len: {int(question_len)}")
                    print(f"2D answer length: {len(answer_2d)} chars | 3D answer length: {len(answer_3d)} chars")
                    print(f"2D tokens: {int(valid_len_2d) - int(question_len)} | 3D tokens: {int(valid_len_3d) - int(question_len)}")
                
                # Unified return dictionary - always include both targets and weights
                result = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    'weight_2d': self.weight_2d,
                    'weight_3d': self.weight_3d,
                    'target_source': target_source,
                    # Always include dual targets
                    'answer_2d': answer_2d,
                    'answer_3d': answer_3d,
                    'label_2d': label_2d,
                    'label_3d': label_3d,
                    'input_id_2d': input_id_2d,
                    'input_id_3d': input_id_3d,
                    'attention_mask_2d': attention_mask_2d,
                    'attention_mask_3d': attention_mask_3d,
                }
                
                return result
                
            except FileNotFoundError as e:
                # Return None so the collator can drop this sample gracefully
                print(f"Warning: Missing file for index {actual_idx} ({e}). Returning None to skip.")
                return None
            except Exception as e:
                print(f"Error in __getitem__ at index {actual_idx}, attempt {attempt + 1}: {e}")
                if attempt < 4:
                    actual_idx = self.valid_indices[(idx + 1) % len(self.valid_indices)]
                else:
                    # On persistent error, return None so the collator can drop the batch item
                    print(f"Giving up on index {actual_idx} after 5 attempts. Returning None to skip.")
                    return None
