import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

import ast
from scipy import sparse
import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates
from .term_dictionary import term_dict


def load_guidance(path_to_guidance="./ct_reports_20250806_010532_patient2k.json"):
    """
    Load guidance reports from the new JSON format.
    
    New format structure:
    [
        {
            "case_id": "000006",
            "scan_type": "Axial_non_contrast", 
            "slice_reports": [
                {
                    "plane": "axial",
                    "slice_index": 16,
                    "report": "..."
                },
                ...
            ]
        },
        ...
    ]
    
    Returns:
    dict: Keys are "{case_id}{scan_type}", values are lists of reports
    """
    import json
    from collections import defaultdict
    
    grouped_reports = defaultdict(list)
    
    with open(path_to_guidance, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load the entire JSON array
    
    for case in data:
        case_id = case["case_id"]
        scan_type = case["scan_type"]
        slice_reports = case["slice_reports"]
        
        # Create the key in the same format as expected: case_id + scan_type
        key = f"{case_id}{scan_type}"
        
        # Add all slice reports for this case/scan_type combination
        for slice_report in slice_reports:
            report_text = slice_report["report"]
            plane = slice_report.get("plane", "unknown")
            slice_index = slice_report["slice_index"]
            
            # Format report with plane information
            formatted_report = f"[{plane.upper()}] {report_text}"
            
            grouped_reports[key].append(formatted_report)
    
    return grouped_reports

def load_summaries_json(path_to_summaries="./ct_summaries_5k.json"):
    """Load scan-level summaries keyed by case_id+scan_type.

    Expected format:
    [
      {"case_id": "000006", "scan_type": "Axial_non_contrast", "summary": "..."},
      ...
    ]
    """
    summaries_map = {}
    try:
        if path_to_summaries and os.path.exists(path_to_summaries):
            with open(path_to_summaries, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                key = f"{item.get('case_id')}{item.get('scan_type')}"
                summaries_map[key] = item.get("summary", "")
            print(f"Loaded {len(summaries_map)} summaries from {path_to_summaries}")
    except Exception as e:
        print(f"Warning: failed to load summaries JSON: {e}")
    return summaries_map


class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)




class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class CapDatasetSum(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates
        self.grouped_reports = load_guidance()

        # Pre-validate dataset and keep only valid indices
        self.valid_indices = []
        self._validate_dataset()

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def _validate_dataset(self):
        """Pre-validate all samples and keep only valid indices"""
        print(f"Validating {self.mode} dataset...")
        
        for idx in range(len(self.data_list)):
            if self._is_valid_sample(idx):
                self.valid_indices.append(idx)
            # else:
            #     print(f"Skipping invalid sample at index {idx}")
        
        print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.data_list)} valid samples")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid samples found in {self.mode} dataset!")

    def _is_valid_sample(self, idx):
        """Check if a sample is valid without loading heavy data"""
        try:
            data = self.data_list[idx]
            
            # Check if image file exists
            image_path = data["image"]
            image_abs_path = os.path.join(self.data_root, image_path)
            if not os.path.exists(image_abs_path):
                return False
            
            # Check if text file exists
            text_path = data["text"]
            text_abs_path = os.path.join(self.data_root, text_path)
            if not os.path.exists(text_abs_path):
                return False
            
            # Check if guidance reports exist
            filename = os.path.basename(image_path)
            position = os.path.splitext(filename)[0]
            image_id = os.path.basename(os.path.dirname(image_path))
            image_name = f"{image_id}{position}"
            
            if image_name not in self.grouped_reports:
                return False
            
            # Check if guidance reports are not empty
            concat_reports = "\n".join(self.grouped_reports[image_name])
            if not concat_reports.strip():
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error for sample {idx}: {e}")
            return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual data index from valid indices
        actual_idx = self.valid_indices[idx]
        
        max_attempts = 5  # Reduced since we pre-validated
        for attempt in range(max_attempts):
            try:
                data = self.data_list[actual_idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                
                # Extract parts
                filename = os.path.basename(image_path)
                position = os.path.splitext(filename)[0]
                image_id = os.path.basename(os.path.dirname(image_path))
                image_name = f"{image_id}{position}"
                
                # Load and transform image
                image = np.load(image_abs_path)
                image = self.transform(image)

                # Load text
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                # Get guidance reports (we know it exists from validation)
                concat_reports = "\n".join(self.grouped_reports[image_name])

                # Tokenize guidance
                guidance_tensor = self.tokenizer(
                    concat_reports, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                guidance_tokens = guidance_tensor["input_ids"][0]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]
                
                # Double check guidance tokens (shouldn't be empty after validation)
                if len(guidance_tokens) == 0:
                    print(f"Warning: Empty guidance tokens for validated sample {actual_idx}")
                    # Try a different valid sample
                    if attempt < max_attempts - 1:
                        if idx + 1 < len(self.valid_indices):
                            actual_idx = self.valid_indices[idx + 1]
                            continue
                        else:
                            actual_idx = self.valid_indices[0]  # Wrap around
                            continue
                    else:
                        raise ValueError("Unable to get valid guidance tokens")

                # Create question and process text
                prompt_question = "Summarize the 2D reports into one 3D radiology report."
                question = self.image_tokens + "\n" + prompt_question + "\n" + concat_reports
                # question = prompt_question + "\n" + concat_reports  # dont use it cause image matters!!!

                text_tensor = self.tokenizer(
                    question + ' ' + answer, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,
                    'guidance_attention_mask': guidance_attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {actual_idx}, attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Try next valid sample
                    if idx + 1 < len(self.valid_indices):
                        actual_idx = self.valid_indices[idx + 1]
                    else:
                        actual_idx = self.valid_indices[0]  # Wrap around
                else:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load sample after {max_attempts} attempts. Last error: {e}")
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Unable to load valid sample for index {idx}")
    
# class CapDatasetSum(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode

#         self.image_tokens = "<im_patch>" * args.proj_out_num

#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]

#         self.caption_prompts = Caption_templates

#         self.grouped_reports = load_guidance()

#         train_transform = mtf.Compose(
#             [
#                 mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#                 mtf.RandFlip(prob=0.10, spatial_axis=0),
#                 mtf.RandFlip(prob=0.10, spatial_axis=1),
#                 mtf.RandFlip(prob=0.10, spatial_axis=2),
#                 mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#                 mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

#                 mtf.ToTensor(dtype=torch.float),
#             ]
#         )

#         val_transform = mtf.Compose(
#                 [
#                     mtf.ToTensor(dtype=torch.float),
#                 ]
#             )
#         set_track_meta(False)

#         if 'train' in mode:
#             self.transform = train_transform
#         elif 'validation' in mode:
#             self.transform = val_transform
#         elif 'test' in mode:
#             self.transform = val_transform

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):

#             data = self.data_list[idx]
#             image_path = data["image"]
#             image_abs_path = os.path.join(self.data_root, image_path)
#             image_id = image_path.split("/")[2]
            
#             # Extract parts
#             filename = os.path.basename(image_path)                # "Sagittal_C__portal_venous_phase.npy"
#             position = os.path.splitext(filename)[0]         # "Sagittal_C__portal_venous_phase"
#             image_id = os.path.basename(os.path.dirname(image_path))  # "000259"
#             # Combine
#             image_name = f"{image_id}{position}"
            
#             if not os.path.exists(image_abs_path):
#                 print(f"Warning: Image {image_abs_path} not found, skipping.")
#                 return None  # Skip this sample
            
#             try:

#                 image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
#                 # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
#                 image = self.transform(image)

#                 text_path = data["text"]
#                 text_abs_path = os.path.join(self.data_root, text_path)
#                 with open(text_abs_path, 'r') as text_file:
#                     raw_text = text_file.read()
#                 answer = raw_text

#                 if image_name in self.grouped_reports:
#                     concat_reports = "\n".join(self.grouped_reports[image_name])
#                 else:
#                     concat_reports = ""

#                 guidance_tensor = self.tokenizer(
#                     concat_reports, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 )

#                 guidance_tokens = guidance_tensor["input_ids"][0]  # Shape: [seq_len]
#                 if len(guidance_tokens) == 0:
#                     print(f"Skipping sample {idx}: empty guidance tokens")
#                     return None
#                 guidance_attention_mask = guidance_tensor["attention_mask"][0]  # Ensures padding is handled

#                 prompt_question = "Summarize the 2D reports into one 3D radiology report."
#                 question = self.image_tokens + "\n" + prompt_question + "\n" + concat_reports

#                 text_tensor = self.tokenizer(
#                     question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 )

#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]

#                 valid_len = torch.sum(attention_mask)
#                 if valid_len < len(input_id):
#                     input_id[valid_len] = self.tokenizer.eos_token_id

#                 question_tensor = self.tokenizer(
#                     question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 )

#                 question_len = torch.sum(question_tensor["attention_mask"][0])

#                 label = input_id.clone()
#                 label[:question_len] = -100
#                 if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#                     label[label == self.tokenizer.pad_token_id] = -100
#                     if valid_len < len(label):
#                         label[valid_len] = self.tokenizer.eos_token_id
#                 else:
#                     label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'guidance_tokens': guidance_tokens,  # Multiple 2D report tokens
#                     'guidance_attention_mask': guidance_attention_mask,
#                     'question': question,
#                     'answer': answer,
#                     'question_type': "Caption",
#                 }
#                 return ret

#             except Exception as e:
#                 print(f"Error in __getitem__ at index {idx}: {e}")
#                 idx = random.randint(0, len(self.data_list) - 1)


class CapDatasetGenSum(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates
        self.grouped_reports = load_guidance()

        # Pre-validate dataset and keep only valid indices
        self.valid_indices = []
        self._validate_dataset()

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def _validate_dataset(self):
        """Pre-validate all samples and keep only valid indices"""
        print(f"Validating {self.mode} dataset...")
        
        for idx in range(len(self.data_list)):
            if self._is_valid_sample(idx):
                self.valid_indices.append(idx)
        
        print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.data_list)} valid samples")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid samples found in {self.mode} dataset!")

    def _is_valid_sample(self, idx):
        """Check if a sample is valid without loading heavy data"""
        try:
            data = self.data_list[idx]
            
            # Check if image file exists
            image_path = data["image"]
            image_abs_path = os.path.join(self.data_root, image_path)
            if not os.path.exists(image_abs_path):
                return False
            
            # Check if text file exists
            text_path = data["text"]
            text_abs_path = os.path.join(self.data_root, text_path)
            if not os.path.exists(text_abs_path):
                return False
            
            # Extract image_id and construct the key like in CapDatasetSum
            image_id = image_path.split("/")[2]
            filename = os.path.basename(image_path)
            position = os.path.splitext(filename)[0]
            image_name = f"{image_id}{position}"
            
            # Check if guidance reports exist
            if image_name not in self.grouped_reports:
                return False
            
            # Check if guidance reports are not empty
            concat_reports = "\n".join(self.grouped_reports[image_name])
            if not concat_reports.strip():
                return False
            
            return True
            
        except Exception as e:
            return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual data index from valid indices
        actual_idx = self.valid_indices[idx]
        
        max_attempts = 5  # Reduced since we pre-validated
        for attempt in range(max_attempts):
            try:
                data = self.data_list[actual_idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                image_id = image_path.split("/")[2]
                
                # Extract filename and construct the key like in validation
                filename = os.path.basename(image_path)
                position = os.path.splitext(filename)[0]
                image_name = f"{image_id}{position}"
                
                # Load and transform image
                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = self.transform(image)

                # Load text
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                # Get guidance reports (we know it exists from validation)
                concat_reports = "\n".join(self.grouped_reports[image_name])

                # Tokenize guidance
                guidance_tensor = self.tokenizer(
                    concat_reports, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                guidance_tokens = guidance_tensor["input_ids"][0]  # Shape: [seq_len]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]  # Ensures padding is handled
                
                # Double check guidance tokens (shouldn't be empty after validation)
                if len(guidance_tokens) == 0:
                    print(f"Warning: Empty guidance tokens for validated sample {actual_idx}")
                    # Try a different valid sample
                    if attempt < max_attempts - 1:
                        if idx + 1 < len(self.valid_indices):
                            actual_idx = self.valid_indices[idx + 1]
                            continue
                        else:
                            actual_idx = self.valid_indices[0]  # Wrap around
                            continue
                    else:
                        raise ValueError("Unable to get valid guidance tokens")

                # Create question and process text (KEEP ORIGINAL DIFFERENCE: includes image_tokens)
                prompt_question = "Summarize the 2D reports into one 3D radiology report."
                question = self.image_tokens + "\n" + prompt_question + "\n" + concat_reports
                # question = prompt_question + "\n" + concat_reports  # dont use it cause image matters!!!

                text_tensor = self.tokenizer(
                    question + ' ' + answer, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

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

                ret = {
                    'image': image,
                    'image_id': image_id,  # KEEP: Original includes image_id
                    'image_name': image_name,  # ADD: Expected by evaluation script
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,  # Multiple 2D report tokens
                    'guidance_attention_mask': guidance_attention_mask,
                    'question': question,
                    'answer': answer,
                    'concat_reports': concat_reports,  # KEEP: Original includes concat_reports
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {actual_idx}, attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Try next valid sample
                    if idx + 1 < len(self.valid_indices):
                        actual_idx = self.valid_indices[idx + 1]
                    else:
                        actual_idx = self.valid_indices[0]  # Wrap around
                else:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load sample after {max_attempts} attempts. Last error: {e}")
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Unable to load valid sample for index {idx}")


class CapGuidanceDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", sample_limit=None):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.sample_limit = sample_limit

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        # Create set of limited image_ids if sample_limit is specified
        if self.sample_limit is not None:
            # Extract unique image_ids and take first sample_limit
            unique_image_ids = []
            seen_ids = set()
            for data in self.data_list:
                image_path = data["image"]
                image_id = image_path.split("/")[2]  # Extract image_id from path
                if image_id not in seen_ids:
                    unique_image_ids.append(image_id)
                    seen_ids.add(image_id)
                    if len(unique_image_ids) >= self.sample_limit:
                        break
            self.limited_image_ids = set(unique_image_ids)
        else:
            self.limited_image_ids = None

        self.caption_prompts = Caption_templates
        
        # Load 2D guidance and optional 3D summaries JSON
        self.grouped_reports = load_guidance()
        self.summaries_map = load_summaries_json(getattr(args, 'summaries_json_path', None))
        if not self.summaries_map:
            # Fallback to previous CSV path if JSON not provided
            summarized_reports_path = f"LaMed/output/LaMed-finetune-0000/eval_caption/summarization.csv"
            print("summarized_reports_path:", summarized_reports_path)
            df = pd.read_csv(summarized_reports_path, dtype={"image_id": str})
            self.summarized_reports = df

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):

            data = self.data_list[idx]
            image_path = data["image"]
            image_abs_path = os.path.join(self.data_root, image_path)
            image_id = image_path.split("/")[2]
            
            # Extract parts
            filename = os.path.basename(image_path)                # "Sagittal_C__portal_venous_phase.npy"
            position = os.path.splitext(filename)[0]         # "Sagittal_C__portal_venous_phase"
            image_id = os.path.basename(os.path.dirname(image_path))  # "000259"
            # Combine
            image_name = f"{image_id}{position}"

            if not os.path.exists(image_abs_path):
                print(f"Warning: Image {image_abs_path} not found, skipping.")
                return None  # Skip this sample
            
            try:

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                # Target answer: prefer summaries JSON mapping if available
                if self.summaries_map and image_name in self.summaries_map:
                    answer = self.summaries_map[image_name]
                else:
                    # Fallback to raw text file as before (if within sample limit)
                    if self.limited_image_ids is None or image_id in self.limited_image_ids:
                        text_path = data["text"]
                        text_abs_path = os.path.join(self.data_root, text_path)
                        with open(text_abs_path, 'r') as text_file:
                            answer = text_file.read()
                    else:
                        answer = ""

                # Build guidance from grouped 2D reports
                concat_reports = "\n".join(self.grouped_reports.get(image_name, []))
                prompt_question = random.choice(self.caption_prompts)
                question = self.image_tokens + prompt_question

                # Tokenize Q+A
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                # Tokenize guidance for optional inputs
                guidance_tensor = self.tokenizer(
                    concat_reports, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                guidance_tokens = guidance_tensor["input_ids"][0]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]

                # Tokenize summarization (target) for optional inputs
                summarization_tensor = self.tokenizer(
                    answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                summarization_tokens = summarization_tensor["input_ids"][0]
                summarization_attention_mask = summarization_tensor["attention_mask"][0]

                # Loss masking
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id
                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                # Debug truncation
                if idx < 3:
                    print("=== DEBUG CapGuidanceDataset ===")
                    print(f"max_length: {self.args.max_length} | used tokens: {int(valid_len)}/{len(input_id)} | question_len: {int(question_len)}")
                    print(f"concat_reports chars: {len(concat_reports)} | answer chars: {len(answer)}")

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,
                    'guidance_attention_mask': guidance_attention_mask,
                    'summarization_tokens': summarization_tokens,
                    'summarization_attention_mask': summarization_attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': torch.tensor(0, dtype=torch.long),
                }
                return ret

            except Exception as e:
                # print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class CapDatasetEval2D(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates
        
        self.grouped_reports = load_guidance()

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):

            data = self.data_list[idx]
            image_path = data["image"]
            image_abs_path = os.path.join(self.data_root, image_path)
            image_id = image_path.split("/")[2]
            
            # Extract parts
            filename = os.path.basename(image_path)                # "Sagittal_C__portal_venous_phase.npy"
            position = os.path.splitext(filename)[0]         # "Sagittal_C__portal_venous_phase"
            image_id = os.path.basename(os.path.dirname(image_path))  # "000259"
            # Combine
            image_name = f"{image_id}{position}"
            
            if not os.path.exists(image_abs_path):
                print(f"Warning: Image {image_abs_path} not found, skipping.")
                return None  # Skip this sample
            
            try:
                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                if image_name in self.grouped_reports:
                    concat_reports = "\n".join(self.grouped_reports[image_name])
                else:
                    concat_reports = ""

                guidance_tensor = self.tokenizer(
                    concat_reports, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                guidance_tokens = guidance_tensor["input_ids"][0]  # Shape: [seq_len]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]  # Ensures padding is handled

                prompt_question = "Summarize the 2D reports into one 3D radiology report."
                question = self.image_tokens + "\n" + prompt_question + "\n" + concat_reports

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,  # Multiple 2D report tokens
                    'guidance_attention_mask': guidance_attention_mask,
                    'question': question,
                    'answer': concat_reports,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", sample_limit=None):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path, nrows=int(sample_limit)*30)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif mode == "test":
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    question = question + ' ' + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])


                question = self.image_tokens + ' ' + question
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'answer_choice': data["Answer Choice"],
                    'question_type': torch.tensor(data["Question Type"], dtype=torch.long),
                }
                return ret

            except Exception as e:
                # print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + ' ' + question
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'answer_choice': data["Answer Choice"],
                    'question_type': data["Question Type"],
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                # print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosRECDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="training",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

        self.cls_questions = PosREC_templates["cls_questions"]
        self.des_questions = PosREC_templates["des_questions"]
        self.cls_answers = PosREC_templates["cls_answers"]
        self.des_answers = PosREC_templates["des_answers"]
        self.cls_no_answers = PosREC_templates["cls_no_answers"]
        self.des_no_answers = PosREC_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.cls_answers).format(box_text)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], box_text)
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REC",
                    'tag': self.tag,
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosREGDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="training",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

        self.cls_questions = PosREG_templates["cls_questions"]
        self.des_questions = PosREG_templates["des_questions"]
        self.cls_answers = PosREG_templates["cls_answers"]
        self.des_answers = PosREG_templates["des_answers"]

        self.cls_no_questions = PosREC_templates["cls_questions"]
        self.des_no_questions = PosREC_templates["des_questions"]

        self.cls_no_answers = PosREG_templates["cls_no_answers"]
        self.des_no_answers = PosREG_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])


            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)
                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers).format(cls_list[cls_id])
                    else:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]]))
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_no_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_no_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REG",
                    'tag': self.tag,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        target_size = (32, 256, 256)
        train_transform = mtf.Compose(
            [
                mtf.Resized(keys=["image", "seg"], spatial_size=target_size, mode=["trilinear", "nearest"]),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
                mtf.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),  # Normalize
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.Resized(keys=["image", "seg"], spatial_size=target_size, mode=["trilinear", "nearest"]),
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                    mtf.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),  # Normalize
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.transform = train_transform
        elif 'validation' in mode:
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']

            image_array = np.load(image_path) #1*32*256*256, normalized
            # following data_load_demo.py:
            seg_array= sparse.load_npz(seg_path)
            s = seg_path.split('.')[-2].split('_')[-1]
            gt_shape = ast.literal_eval(s)
            seg_array = seg_array.toarray().reshape(gt_shape)
            # Randomly select a class
            num_cls = eval(s)[0]
            cls_id = random.choice(range(num_cls))  
            seg_array = seg_array[cls_id]
            seg_array = np.expand_dims(seg_array, axis=0)

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "seg",
                    'tag': self.tag,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        self.data_list = pd.read_csv(args.refseg_data_path, engine='python')

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if 'train' in mode:
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif 'validation' in mode:
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif 'test' in mode:
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        dataset_info_path = os.path.join(args.seg_data_path, "dataset_info.json")
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            # self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        # self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]




class PosRECTestDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosRECTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ds_list = []
        dataset_code = '0003'
        self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
        self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class PosREGTestDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosREGTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ds_list = []
        dataset_code = '0003'
        self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
        self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiPosDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        dataset_info_path = os.path.join(args.seg_data_path, "dataset_info.json")
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDatasetSum(args, tokenizer, mode),
            # VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextOnlyCapDatasetSum(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.grouped_reports = load_guidance()

        # Pre-validate dataset and keep only valid indices
        self.valid_indices = []
        self._validate_dataset()
        
        # Apply sample limit if specified
        if hasattr(args, 'sample_limit') and args.sample_limit is not None:
            sample_limit = int(args.sample_limit)
            if len(self.valid_indices) > sample_limit:
                print(f"Limiting dataset from {len(self.valid_indices)} to {sample_limit} samples")
                self.valid_indices = self.valid_indices[:sample_limit]

    def _validate_dataset(self):
        """Pre-validate all samples and keep only valid indices"""
        print(f"Validating {self.mode} dataset...")
        
        for idx in range(len(self.data_list)):
            if self._is_valid_sample(idx):
                self.valid_indices.append(idx)
        
        print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.data_list)} valid samples")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid samples found in {self.mode} dataset!")

    def _is_valid_sample(self, idx):
        """Check if a sample is valid without loading heavy data"""
        try:
            data = self.data_list[idx]
            
            # Check if text file exists
            text_path = data["text"]
            text_abs_path = os.path.join(self.data_root, text_path)
            if not os.path.exists(text_abs_path):
                return False
            
            # Extract image_id and construct the key
            image_path = data["image"]
            filename = os.path.basename(image_path)
            position = os.path.splitext(filename)[0]
            image_id = os.path.basename(os.path.dirname(image_path))
            image_name = f"{image_id}{position}"
            
            # Check if guidance reports exist
            if image_name not in self.grouped_reports:
                return False
            
            # Check if guidance reports are not empty
            concat_reports = "\n".join(self.grouped_reports[image_name])
            if not concat_reports.strip():
                return False
            
            return True
            
        except Exception as e:
            return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual data index from valid indices
        actual_idx = self.valid_indices[idx]
        
        max_attempts = 5  # Reduced since we pre-validated
        for attempt in range(max_attempts):
            try:
                data = self.data_list[actual_idx]
                
                # Load text (3D summary)
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                # Extract image_id and construct the key
                image_path = data["image"]
                filename = os.path.basename(image_path)
                position = os.path.splitext(filename)[0]
                image_id = os.path.basename(os.path.dirname(image_path))
                image_name = f"{image_id}{position}"
                
                # Get guidance reports (2D reports) and optionally limit their count
                reports_list = self.grouped_reports[image_name]
                # Prefer new arg two_d_reports_per_sample; fall back to num_2d_guidance for backward-compat
                max_guidance = getattr(self.args, 'two_d_reports_per_sample', None)
                if (max_guidance is None) or (isinstance(max_guidance, int) and max_guidance <= 0):
                    max_guidance = getattr(self.args, 'num_2d_guidance', None)
                if isinstance(max_guidance, int) and max_guidance is not None and max_guidance > 0:
                    reports_list = reports_list[:max_guidance]
                concat_reports = "\n".join(reports_list)

                # Tokenize guidance (2D reports)
                guidance_tensor = self.tokenizer(
                    concat_reports, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                guidance_tokens = guidance_tensor["input_ids"][0]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]
                
                # Create the input sequence: 2D reports + prompt + 3D summary
                prompt_question = "Summarize the 2D reports into one 3D radiology report."
                full_text = concat_reports + "\n\n" + prompt_question + "\n\n" + answer

                # DEBUG: Print the input prompt structure (TEXT-ONLY MODE)
                if idx < 3:  # Only print first 3 samples to avoid spam
                    print(f"\n=== DEBUG: Text-Only Sample {idx} Input Prompt ===")
                    print(f"2D Reports length: {len(concat_reports)} characters")
                    print(f"2D Reports preview: {concat_reports[:200]}...")
                    print(f"Prompt question: {prompt_question}")
                    print(f"3D Summary length: {len(answer)} characters")
                    print(f"3D Summary preview: {answer[:200]}...")
                    print(f"Full text length: {len(full_text)} characters")
                    print("=" * 50)

                # Tokenize the full sequence
                text_tensor = self.tokenizer(
                    full_text, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # Create labels (only compute loss on the 3D summary part)
                # Find where the 3D summary starts
                summary_start = len(concat_reports) + len(prompt_question) + 4  # +4 for "\n\n"
                
                # Tokenize just the guidance + prompt to find where summary starts
                guidance_prompt_tensor = self.tokenizer(
                    concat_reports + "\n\n" + prompt_question + "\n\n", 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                
                question_len = torch.sum(guidance_prompt_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100  # Don't compute loss on guidance + prompt
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,
                    'guidance_attention_mask': guidance_attention_mask,
                    'question': prompt_question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {actual_idx}, attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Try next valid sample
                    if idx + 1 < len(self.valid_indices):
                        actual_idx = self.valid_indices[idx + 1]
                    else:
                        actual_idx = self.valid_indices[0]  # Wrap around
                else:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load sample after {max_attempts} attempts. Last error: {e}")
        
        # This should never be reached, but just in case
        raise RuntimeError(f"Unable to load valid sample for index {idx}")


class ImageTextCapDatasetSum(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.grouped_reports = load_guidance()

        # Add image tokens for prompt
        self.image_tokens = "<im_patch>" * args.proj_out_num

        # Pre-validate dataset and keep only valid indices
        self.valid_indices = []
        self._validate_dataset()
        
        # Apply sample limit if specified
        if hasattr(args, 'sample_limit') and args.sample_limit is not None:
            sample_limit = int(args.sample_limit)
            if len(self.valid_indices) > sample_limit:
                print(f"Limiting dataset from {len(self.valid_indices)} to {sample_limit} samples")
                self.valid_indices = self.valid_indices[:sample_limit]

    def _validate_dataset(self):
        """Pre-validate all samples and keep only valid indices"""
        print(f"Validating {self.mode} dataset...")
        
        for idx in range(len(self.data_list)):
            if self._is_valid_sample(idx):
                self.valid_indices.append(idx)
        
        print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.data_list)} valid samples")
        
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid samples found in {self.mode} dataset!")

    def _is_valid_sample(self, idx):
        """Check if a sample is valid without loading heavy data"""
        try:
            data = self.data_list[idx]
            
            # Check if text file exists
            text_path = data["text"]
            text_abs_path = os.path.join(self.data_root, text_path)
            if not os.path.exists(text_abs_path):
                return False
            
            # Extract image_id and construct the key
            image_path = data["image"]
            filename = os.path.basename(image_path)
            position = os.path.splitext(filename)[0]
            image_id = os.path.basename(os.path.dirname(image_path))
            image_name = f"{image_id}{position}"
            
            # Check if guidance reports exist
            if image_name not in self.grouped_reports:
                return False
            
            # Check if guidance reports are not empty
            concat_reports = "\n".join(self.grouped_reports[image_name])
            if not concat_reports.strip():
                return False
            
            return True
            
        except Exception as e:
            return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual data index from valid indices
        actual_idx = self.valid_indices[idx]
        
        # DEBUG: Print when first few samples are accessed
        if idx < 3:
            print(f"DEBUG: ImageTextCapDatasetSum.__getitem__({idx}) called")
        
        max_attempts = 5  # Reduced since we pre-validated
        for attempt in range(max_attempts):
            try:
                data = self.data_list[actual_idx]
                
                # Load text (3D summary)
                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                # Extract image_id and construct the key
                image_path = data["image"]
                filename = os.path.basename(image_path)
                position = os.path.splitext(filename)[0]
                image_id = os.path.basename(os.path.dirname(image_path))
                image_name = f"{image_id}{position}"
                
                # Load the actual 3D image
                image_abs_path = os.path.join(self.data_root, image_path)
                image = np.load(image_abs_path)  # normalized 0-1, C,D,H,W
                image = torch.from_numpy(image).float()
                
                # Get guidance reports (2D reports) and optionally limit their count
                reports_list = self.grouped_reports[image_name]
                # Prefer new arg two_d_reports_per_sample; fall back to num_2d_guidance for backward-compat
                max_guidance = getattr(self.args, 'two_d_reports_per_sample', None)
                if (max_guidance is None) or (isinstance(max_guidance, int) and max_guidance <= 0):
                    max_guidance = getattr(self.args, 'num_2d_guidance', None)
                if isinstance(max_guidance, int) and max_guidance is not None and max_guidance > 0:
                    reports_list = reports_list[:max_guidance]
                concat_reports = "\n".join(reports_list)

                # Tokenize guidance (2D reports)
                guidance_tensor = self.tokenizer(
                    concat_reports, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                guidance_tokens = guidance_tensor["input_ids"][0]
                guidance_attention_mask = guidance_tensor["attention_mask"][0]
                
                # Create the input sequence: image tokens + 2D reports + prompt + 3D summary
                prompt_question = "Summarize the 2D reports into one 3D radiology report."
                full_text = self.image_tokens + "\n" + concat_reports + "\n\n" + prompt_question + "\n\n" + answer

                # DEBUG: Print the input prompt structure
                if idx < 3:  # Only print first 3 samples to avoid spam
                    print(f"\n=== DEBUG: Sample {idx} Input Prompt ===")
                    print(f"Image tokens: {self.image_tokens}")
                    print(f"2D Reports length: {len(concat_reports)} characters")
                    print(f"2D Reports preview: {concat_reports[:200]}...")
                    print(f"Prompt question: {prompt_question}")
                    print(f"3D Summary length: {len(answer)} characters")
                    print(f"3D Summary preview: {answer[:200]}...")
                    print(f"Full text length: {len(full_text)} characters")
                    print("=" * 50)

                # Tokenize the full sequence
                text_tensor = self.tokenizer(
                    full_text, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # Create labels (only compute loss on the 3D summary part)
                # Find where the 3D summary starts
                # Tokenize just the guidance + prompt to find where summary starts
                guidance_prompt_tensor = self.tokenizer(
                    self.image_tokens + "\n" + concat_reports + "\n\n" + prompt_question + "\n\n", 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                question_len = torch.sum(guidance_prompt_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100  # Don't compute loss on guidance + prompt
                
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'guidance_tokens': guidance_tokens,
                    'guidance_attention_mask': guidance_attention_mask,
                    'question': prompt_question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {actual_idx}, attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Try next valid sample
                    if idx + 1 < len(self.valid_indices):
                        actual_idx = self.valid_indices[idx + 1]
                    else:
                        actual_idx = self.valid_indices[0]  # Wrap around
                else:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load sample after {max_attempts} attempts. Last error: {e}")
        # This should never be reached, but just in case
        raise RuntimeError(f"Unable to load valid sample for index {idx}")


class TextGuidanceDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train', sample_limit=None):
        super(TextGuidanceDatasets, self).__init__()
        self.ds_list = [
            CapGuidanceDataset(args, tokenizer, mode, sample_limit),
            VQADataset(args, tokenizer, close_ended=True, mode=mode, sample_limit=sample_limit),
            VQADataset(args, tokenizer, close_ended=False, mode=mode, sample_limit=sample_limit),
            # VQAYNDataset(args, tokenizer, mode=mode, sample_limit=sample_limit),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            # VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
            # MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



