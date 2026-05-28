"""
Standalone caption evaluation that avoids the broken `evaluate` library.
Computes BERTScore F1, ROUGE-1, and writes predictions to CSV.
"""

import os
import sys
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

sys.path.append(os.path.abspath("."))
from lamed.dataset.multi_dataset import CapDataset

# Use rouge-score and bert-score packages directly
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def custom_collate(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--cap_data_path', type=str, default="./Data/data/M3D_Cap_npy/M3D_Cap.json")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, default="eval_caption.csv")
    parser.add_argument('--proj_out_num', type=int, default=256)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device=device)

    test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test500')
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=0,
        pin_memory=True, shuffle=False, drop_last=False,
        collate_fn=custom_collate,
    )

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name)

    all_preds = []
    all_refs = []
    rows = []

    for sample in tqdm(test_dataloader, desc="Generating"):
        if sample is None:
            continue

        question = sample["question"]
        answer = sample["answer"]
        input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(device=device)
        image = sample["image"].to(device=device, dtype=torch.bfloat16)

        generation = model.generate(
            image, input_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

        pred = generated_texts[0].strip()
        ref = answer[0].strip()
        all_preds.append(pred)
        all_refs.append(ref)

        # Per-sample ROUGE-1
        r1 = scorer.score(ref, pred)['rouge1'].fmeasure

        rows.append({
            "Question": question[0],
            "Ground Truth": ref,
            "pred": pred,
            "rouge1": r1,
        })

    # Batch BERTScore
    print("Computing BERTScore (batch)...")
    P, R, F1 = bert_score_fn(all_preds, all_refs, lang="en", verbose=True)
    for i, row in enumerate(rows):
        row["bert_f1"] = F1[i].item()

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "Ground Truth", "pred", "rouge1", "bert_f1"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    rouge1_scores = [r["rouge1"] for r in rows]
    bert_scores = [r["bert_f1"] for r in rows]
    print(f"\n=== Results ({len(rows)} samples) ===")
    print(f"  ROUGE-1:      {np.mean(rouge1_scores):.4f}")
    print(f"  BERTScore F1: {np.mean(bert_scores):.4f}")
    print(f"  Saved to:     {output_path}")


if __name__ == "__main__":
    main()
