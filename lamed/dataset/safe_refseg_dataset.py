"""
Safe RefSegDataset for segmentation training.

This is a simplified RefSegDataset that avoids MONAI random seed issues
and works with the finetuning pipeline for pseudo/GT mask comparison.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import monai.transforms as mtf
from monai.data import set_track_meta


class SafeRefSegDataset(Dataset):
    """
    Safe RefSegDataset for referred segmentation training.

    Simplified from the original RefSegDataset to avoid MONAI random seed issues.
    Supports custom CSV paths for pseudo/GT mask training.
    """

    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        # Get proj_out_num from args or default
        proj_out_num = getattr(args, 'proj_out_num', 256)
        self.image_tokens = "<im_patch>" * proj_out_num

        # Get max_length from args or default
        self.max_length = getattr(args, 'max_length', 512)

        # Simplified transforms (avoid random transforms that cause seed issues)
        train_transform = mtf.Compose([
            mtf.ToTensord(keys=["image"], dtype=torch.float),
            mtf.ToTensord(keys=["seg"], dtype=torch.int),
        ])

        val_transform = mtf.Compose([
            mtf.ToTensord(keys=["image"], dtype=torch.float),
            mtf.ToTensord(keys=["seg"], dtype=torch.int),
        ])

        set_track_meta(False)

        # Load data based on mode
        if 'train' in mode:
            csv_path = getattr(args, 'refseg_data_train_path', None)
            if csv_path is None:
                csv_path = getattr(args, 'refseg_data_path', None)
            self.data_list = pd.read_csv(csv_path, engine='python')
            self.transform = train_transform
        else:
            csv_path = getattr(args, 'refseg_data_test_path', None)
            if csv_path is None:
                csv_path = getattr(args, 'refseg_data_path', None)
            self.data_list = pd.read_csv(csv_path, engine='python')
            self.transform = val_transform

        print(f"[SafeRefSeg-{mode}] Loaded {len(self.data_list)} samples from {csv_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100

        for attempt in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]

                # Get data root - handle both absolute and relative paths
                data_root = getattr(self.args, 'data_root', './Data/data/')

                # Load image
                image_path = data["Image"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(data_root, image_path)

                image_array = np.load(image_path)  # Expected: (1, D, H, W) or (D, H, W)

                # Ensure 4D: (C, D, H, W)
                if image_array.ndim == 3:
                    image_array = image_array[np.newaxis, ...]

                # Load mask
                mask_path = data["Mask"]
                if not os.path.isabs(mask_path):
                    mask_path = os.path.join(data_root, mask_path)

                seg_array = np.load(mask_path)  # Expected: (1, D, H, W) or (D, H, W)

                # Ensure 4D: (C, D, H, W)
                if seg_array.ndim == 3:
                    seg_array = seg_array[np.newaxis, ...]

                # Handle Mask_ID if present (select specific mask region)
                if "Mask_ID" in data:
                    mask_id = data["Mask_ID"]
                    if mask_id > 0:
                        seg_array = (seg_array == mask_id).astype(np.int8)
                    else:
                        seg_array = (seg_array > 0).astype(np.int8)
                else:
                    seg_array = (seg_array > 0).astype(np.int8)

                # Apply transforms
                item = {
                    "image": image_array,
                    "seg": seg_array,
                }
                item = self.transform(item)

                image = item['image']  # (C, D, H, W)
                seg = item['seg']      # (C, D, H, W)

                # Get question and answer
                question = data.get("Question", "Please segment the lesion in this scan.")
                answer = data.get("Answer", "The lesion is segmented as shown in [SEG].")

                # Prepend image tokens to question
                question = self.image_tokens + ' ' + question

                # Tokenize
                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                # Add EOS token at end of valid sequence
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # Create labels (mask question tokens)
                question_tensor = self.tokenizer(
                    question,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100  # Mask question tokens

                # Handle padding tokens
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                return {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Error in __getitem__ at index {idx} after {max_attempts} attempts: {e}")
                    raise
                idx = random.randint(0, len(self.data_list) - 1)
