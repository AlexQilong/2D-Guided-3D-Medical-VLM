"""Clinical finding F1 for report generation.

This module extracts simple multi-label clinical findings from free text and
computes micro/macro precision/recall/F1 between generated and reference
reports. It uses a configurable label map (YAML/JSON) with synonyms.

Default config path: Bench/eval/clinical_f1_labels.yaml
To replace it, pass --config /path/to/your_labels.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).with_name("clinical_f1_labels.yaml")

NEGATION_CUES = [
    "no",
    "not",
    "without",
    "denies",
    "negative for",
    "free of",
    "absence of",
    "rule out",
    "ruled out",
    "no evidence of",
    "without evidence of",
]

UNCERTAIN_CUES = [
    "cannot exclude",
    "can't exclude",
    "cannot rule out",
    "can't rule out",
    "possible",
    "possibly",
    "probable",
    "likely",
    "suggestive of",
    "may represent",
    "could represent",
]


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _match_token(token: str, target: str) -> bool:
    if token == target:
        return True
    if token == target + "s":
        return True
    if token == target + "es":
        return True
    if target.endswith("y") and token == target[:-1] + "ies":
        return True
    return False


def _contains_cue(tokens: Sequence[str], cue_tokens: Sequence[str]) -> bool:
    if not cue_tokens:
        return False
    n = len(cue_tokens)
    for i in range(len(tokens) - n + 1):
        if list(tokens[i : i + n]) == list(cue_tokens):
            return True
    return False


def _load_config(path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    if path is None:
        path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
    if path is None:
        raise FileNotFoundError(
            "No config path provided and default config not found."
        )
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML configs.")
        data = yaml.safe_load(raw)
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError("Config must contain a top-level 'labels' mapping.")
    return data


class FindingExtractor:
    """Extract a multi-label set of clinical findings from text."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Dict[str, List[str]]]] = None,
        treat_uncertain_as_positive: bool = False,
        negation_window: int = 5,
    ) -> None:
        if config is None:
            config = _load_config(config_path)
        labels_cfg = config.get("labels", {})
        if not isinstance(labels_cfg, dict) or not labels_cfg:
            raise ValueError("Config labels must be a non-empty mapping.")

        self.labels: List[str] = []
        self._label_phrases: List[List[List[str]]] = []
        for label, info in labels_cfg.items():
            if isinstance(info, dict):
                phrases = info.get("synonyms", [])
            else:
                phrases = info
            if not isinstance(phrases, list) or not phrases:
                continue
            phrase_tokens = [tokenize(p) for p in phrases if isinstance(p, str)]
            phrase_tokens = [p for p in phrase_tokens if p]
            if not phrase_tokens:
                continue
            self.labels.append(str(label))
            self._label_phrases.append(phrase_tokens)

        if not self.labels:
            raise ValueError("No valid labels found in config.")

        self.treat_uncertain_as_positive = treat_uncertain_as_positive
        self.negation_window = int(negation_window)
        self._negation_cues = [tokenize(cue) for cue in NEGATION_CUES]
        self._uncertain_cues = [tokenize(cue) for cue in UNCERTAIN_CUES]

    def extract(self, report: str) -> np.ndarray:
        """Return multi-hot vector of findings for a report."""
        tokens = tokenize(report or "")
        vec = np.zeros(len(self.labels), dtype=np.int32)
        if not tokens:
            return vec

        for label_idx, phrase_list in enumerate(self._label_phrases):
            found = False
            for phrase_tokens in phrase_list:
                n = len(phrase_tokens)
                if n == 0 or n > len(tokens):
                    continue
                for i in range(len(tokens) - n + 1):
                    if not self._phrase_matches(tokens, phrase_tokens, i):
                        continue
                    window = tokens[max(0, i - self.negation_window) : i]
                    if self._is_negated(window):
                        continue
                    if self._is_uncertain(window) and not self.treat_uncertain_as_positive:
                        continue
                    found = True
                    break
                if found:
                    break
            vec[label_idx] = 1 if found else 0
        return vec

    def _phrase_matches(
        self, tokens: Sequence[str], phrase_tokens: Sequence[str], start: int
    ) -> bool:
        for offset, target in enumerate(phrase_tokens):
            token = tokens[start + offset]
            if not _match_token(token, target):
                return False
        return True

    def _is_negated(self, window_tokens: Sequence[str]) -> bool:
        return any(_contains_cue(window_tokens, cue) for cue in self._negation_cues)

    def _is_uncertain(self, window_tokens: Sequence[str]) -> bool:
        return any(_contains_cue(window_tokens, cue) for cue in self._uncertain_cues)


def _read_input_rows(path: Path, fmt: str) -> Iterable[Dict[str, str]]:
    if fmt == "auto":
        if path.suffix.lower() == ".csv":
            fmt = "csv"
        elif path.suffix.lower() in {".jsonl", ".json"}:
            fmt = "jsonl"
        else:
            raise ValueError(f"Unable to infer input format from {path.name}.")
    if fmt == "csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    elif fmt == "jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        raise ValueError(f"Unsupported input format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clinical finding F1.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more CSV/JSONL file paths.",
    )
    parser.add_argument("--input-format", default="auto", choices=["auto", "csv", "jsonl"])
    parser.add_argument("--config", default=None, help="YAML/JSON label config path.")
    parser.add_argument("--reference-field", default="reference_report")
    parser.add_argument("--generated-field", default="generated_report")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--output-label-metrics", default="clinical_f1_label_metrics.csv")
    parser.add_argument("--treat-uncertain-as-positive", action="store_true")
    parser.add_argument("--negation-window", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    extractor = FindingExtractor(
        config_path=config_path,
        treat_uncertain_as_positive=args.treat_uncertain_as_positive,
        negation_window=args.negation_window,
    )

    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    for input_item in args.input:
        input_path = Path(input_item)
        for row in _read_input_rows(input_path, args.input_format):
            ref = row.get(args.reference_field, "")
            gen = row.get(args.generated_field, "")
            y_true.append(extractor.extract(ref))
            y_pred.append(extractor.extract(gen))

    if not y_true:
        print("No valid rows found. Exiting.")
        return

    y_true_arr = np.vstack(y_true)
    y_pred_arr = np.vstack(y_pred)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    print(f"micro_precision={micro_p:.4f} micro_recall={micro_r:.4f} micro_f1={micro_f1:.4f}")
    print(f"macro_precision={macro_p:.4f} macro_recall={macro_r:.4f} macro_f1={macro_f1:.4f}")

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average=None, zero_division=0
    )

    out_path = Path(args.output_label_metrics)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        for label, p, r, f1, s in zip(
            extractor.labels, per_p, per_r, per_f1, per_support
        ):
            writer.writerow(
                {
                    "label": label,
                    "precision": f"{p:.6f}",
                    "recall": f"{r:.6f}",
                    "f1": f"{f1:.6f}",
                    "support": int(s),
                }
            )


if __name__ == "__main__":
    main()
