import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

import sys
# Add the parent directory of LaMed to sys.path
sys.path.append(os.path.abspath('.'))

from lamed.dataset.multi_dataset import CapDataset, CapDatasetEval2D
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from lamed.model.language_model import *
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="LaMed/output/LaMed-Phi3-4B-merge-0000")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--cap_data_path', type=str, default="./Data/data/M3D_Cap_npy/M3D_Cap.json")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-finetune-0000/")
    parser.add_argument('--file_name', type=str, default="eval_caption.csv")

    parser.add_argument('--proj_out_num', type=int, default=256)
    parser.add_argument('--eval_sample_limit', type=int, default=None, help="Limit number of samples for evaluation (None for all samples)")

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        

def custom_collate(batch):
    batch = [sample for sample in batch if sample is not None]  # Remove None samples
    if len(batch) == 0:
        return None  # Return None if all samples are invalid (prevents empty batches)
    return default_collate(batch)  # Process valid samples only


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)
    print("All Args:", args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device=device)

    test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test500')

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=custom_collate
    )  

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, args.file_name)

    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor", "bert_f1"])
        
        # Limit samples if specified
        sample_count = 0
        max_samples = args.eval_sample_limit if args.eval_sample_limit is not None else float('inf')
        print(f"Evaluating up to {max_samples if max_samples != float('inf') else 'all'} samples")
        
        for sample in tqdm(test_dataloader):
            if sample_count >= max_samples:
                print(f"Reached sample limit of {args.eval_sample_limit}, stopping evaluation")
                break

            if sample is None:  # Skip empty batches
                continue  
            
            question = sample["question"]
            answer = sample['answer']

            input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)
            image = sample["image"].to(device=device, dtype=torch.bfloat16)

            generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            result = dict()
            try:
                decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                writer.writerow([question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])
            
            except Exception as e:
                print(f"[Warning] Skipping one sample due to error: {e}")
                writer.writerow([question[0], answer[0], generated_texts[0]])
            
            sample_count += 1


if __name__ == "__main__":
    main()
       