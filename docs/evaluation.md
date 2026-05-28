# Evaluation

## Reports (caption track)

Two-step evaluation: per-sample lexical metrics, then aggregate Clinical F1.

```bash
bash scripts/05_eval_caption.sh ./outputs/finetune_cap_pseudo500-merged
```

This produces:
- `outputs/eval/finetune_cap_pseudo500-merged_caption.csv` — per-sample BERTScore F1, ROUGE-1, and the model's generated text
- `outputs/eval/finetune_cap_pseudo500-merged_clinical_f1.csv` — per-label precision / recall / F1 (printed Micro + Macro aggregates to stdout)

### Metrics

| Metric | Definition |
|--------|------------|
| **BERTScore F1** | Semantic similarity between predicted and reference report using contextual embeddings. Computed in batch with the `bert-score` package, default model `roberta-large`. |
| **ROUGE-1** | Unigram lexical overlap F1 (`rouge-score` package). |
| **Clinical F1 (Micro)** | Multi-label finding extraction → micro-averaged F1 over all (label, sample) pairs. Sensitive to label frequency. |
| **Clinical F1 (Macro)** | Same as above but averaged per-label, then over labels. Equal weight to all findings. |

### Customising Clinical F1 labels

The default label ontology lives in `eval/clinical_f1_labels.yaml`. To adapt to a different label set, pass `--config /path/to/your_labels.yaml` to `eval/clinical_f1.py`.

The YAML schema is:

```yaml
labels:
  pneumonia:
    synonyms: ["pneumonia", "consolidation", "infiltrate"]
  effusion:
    synonyms: ["pleural effusion", "fluid in pleura"]
```

The extractor handles simple negation cues (`"no consolidation"`) and uncertainty (`"possible pneumonia"`) — see `eval/clinical_f1.py` for details.

## Segmentation track

```bash
bash scripts/06_eval_segmentation.sh ./outputs/finetune_duke_gt_plus_pseudo
```

This runs the model on every row of the test CSV, decodes the `[SEG]`-conditioned mask logits, computes BinaryDice against the GT mask, and writes:

- `outputs/eval/finetune_duke_gt_plus_pseudo/eval_seg.csv` — per-sample Dice + predicted text

### Metric

BinaryDice (from `eval/metrics.py`):

```
Dice = 2 * |P ∩ G| / (|P| + |G|)
```

where `P = sigmoid(logits) > 0.5` and `G` is the binarized GT mask. Both are resized to `(32, 256, 256)` before computing.

### IoU

The same script can be extended to also report Intersection-over-Union (`IoU = |P ∩ G| / |P ∪ G|`) by adding a `BinaryIoU` metric — left as a one-liner extension to keep the default eval lightweight.

## Reproducing the paper tables

After running training for all conditions, run the corresponding eval script for each output dir and aggregate the resulting CSVs into the table format used in the paper. A simple `pandas` one-liner over the per-sample CSVs gives the mean values reported.
