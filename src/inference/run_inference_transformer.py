from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from metrics_utils import evaluate_predictions, load_split, resolve_text_column

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Transformers is required. Install with: pip install transformers torch"
    ) from exc


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, list], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Hugging Face transformer model")

    default_data_dir = Path(__file__).resolve().parents[2] / "output" / "split"
    default_model_dir = Path(__file__).resolve().parents[2] / "output" / "transformer"
    default_output_file = Path(__file__).resolve().parents[2] / "output" / "inference" / "transformer_test.json"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-file", type=Path, default=default_output_file)

    return parser.parse_args()


def load_training_meta(model_dir: Path) -> Dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_subdir(model_dir: Path) -> Path:
    model_subdir = model_dir / "model"
    if model_subdir.exists():
        return model_subdir
    return model_dir


def to_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]


def main() -> None:
    args = parse_args()

    training_meta = load_training_meta(args.model_dir)
    hyper = training_meta.get("hyperparameters", {})

    threshold = args.threshold
    if threshold is None:
        threshold = hyper.get("threshold_used", 0.5)
    threshold = float(threshold)

    max_len = args.max_len
    if max_len is None:
        max_len = int(hyper.get("max_len", 512))

    model_load_dir = resolve_model_subdir(args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_load_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_load_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = load_split(args.data_dir, args.split)
    text_col = resolve_text_column(df, args.text_col)

    x_text = df[text_col].fillna("").astype(str).tolist()
    y_true = df[args.label_col].astype(int).to_numpy()

    encodings = tokenizer(x_text, truncation=True, padding="max_length", max_length=max_len)
    dataset = TextDataset(encodings, y_true)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    logits_list = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = labels  # retained for dataset consistency; not needed in forward.
            outputs = model(**batch)
            logits_list.append(outputs.logits.detach().cpu().numpy())

    logits = np.vstack(logits_list)
    y_prob = to_probs(logits)

    metrics = evaluate_predictions(y_true, y_prob, threshold)

    result = {
        "model_name": str(training_meta.get("model", "HFTransformer")),
        "model_kind": "transformer",
        "model_dir": str(args.model_dir),
        "model_load_dir": str(model_load_dir),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "text_col": text_col,
        "label_col": args.label_col,
        "threshold_used": threshold,
        "max_len_used": max_len,
        "num_samples": int(len(df)),
        "metrics": metrics,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Transformer inference complete.")
    print(f"Split: {args.split}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
