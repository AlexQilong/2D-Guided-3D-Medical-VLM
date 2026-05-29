# Evaluation

The README shows the eval commands. This page defines the metrics and the outputs
each evaluator writes.

## Reports (caption track)

`scripts/05_eval_caption.sh <model_dir>` runs in two steps and writes:
- `<model_dir>_caption.csv` — per-sample ROUGE-1, BERTScore F1, and the generated text
- `<model_dir>_clinical_f1.csv` — per-label precision / recall / F1 (Micro + Macro printed to stdout)

| Metric | Definition |
|--------|------------|
| **ROUGE-1** | Unigram lexical-overlap F1 (`rouge-score`). |
| **BERTScore F1** | Semantic similarity via contextual embeddings (`bert-score`, default `roberta-large`). |
| **Clinical F1 (Micro)** | Multi-label finding extraction → micro-averaged F1 over all (label, sample) pairs; frequency-sensitive. |
| **Clinical F1 (Macro)** | Same, averaged per-label then over labels; equal weight to all findings. |

`eval_caption_standalone.py` computes ROUGE-1 + BERTScore directly (it avoids the
`evaluate` library, which breaks on HF Hub auth changes).

### Customising Clinical F1 labels

The default ontology is `eval/clinical_f1_labels.yaml`; pass `--config your_labels.yaml`
to `eval/clinical_f1.py` to swap it. Schema:

```yaml
labels:
  pneumonia:
    synonyms: ["pneumonia", "consolidation", "infiltrate"]
  effusion:
    synonyms: ["pleural effusion", "fluid in pleura"]
```

The extractor handles negation (`"no consolidation"`) and uncertainty
(`"possible pneumonia"`) cues — see `eval/clinical_f1.py`.

## Segmentation track

`scripts/06_eval_segmentation.sh <model_dir>` runs the model on each test row, decodes
the `[SEG]`-conditioned mask logits, computes Dice against GT, and writes per-sample
results to `eval_seg.csv`.

**BinaryDice** (`eval/metrics.py`): `Dice = 2·|P∩G| / (|P|+|G|)`, where
`P = sigmoid(logits) > 0.5` and `G` is the binarized GT mask, both at `(32, 256, 256)`.
IoU (`|P∩G| / |P∪G|`) is the second metric reported in the paper.
