from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
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
    parser = argparse.ArgumentParser(description="Train and evaluate a Hugging Face transformer classifier")

    default_data_dir = Path(__file__).resolve().parents[2] / "output" / "split"
    default_output_dir = Path(__file__).resolve().parents[2] / "output" / "transformer"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)

    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")

    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-len", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. If omitted, picks best validation-F1 threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def resolve_split_dir(data_dir: Path) -> Path:
    direct = data_dir / "train.csv"
    nested = data_dir / "split" / "train.csv"

    if direct.exists():
        return data_dir
    if nested.exists():
        return data_dir / "split"

    raise FileNotFoundError(
        f"Could not find split files under {data_dir}. "
        "Expected train.csv/val.csv/test.csv directly, or inside a split/ subfolder."
    )


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected split file not found: {path}")
    return pd.read_csv(path)


def resolve_text_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "clean_text" in df.columns:
        return "clean_text"
    if "text" in df.columns:
        return "text"
    raise ValueError("No usable text column found. Expected one of: clean_text, text")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.1, 0.901, 0.01):
        pred = (y_prob >= thr).astype(int)
        curr_f1 = f1_score(y_true, pred, zero_division=0)
        if curr_f1 > best_f1:
            best_f1 = float(curr_f1)
            best_threshold = float(thr)
    return best_threshold, best_f1


def leakage_audit(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
    def split_stats(df: pd.DataFrame) -> Dict[str, int]:
        texts = df[text_col].fillna("").astype(str)
        uniq = int(texts.nunique())
        return {
            "rows": int(len(df)),
            "unique_texts": uniq,
            "duplicate_within_split": int(len(df) - uniq),
        }

    train_set = set(train_df[text_col].fillna("").astype(str).tolist())
    val_set = set(val_df[text_col].fillna("").astype(str).tolist())
    test_set = set(test_df[text_col].fillna("").astype(str).tolist())

    return {
        "split_stats": {
            "train": split_stats(train_df),
            "val": split_stats(val_df),
            "test": split_stats(test_df),
        },
        "cross_split_overlap": {
            "train_val": int(len(train_set.intersection(val_set))),
            "train_test": int(len(train_set.intersection(test_set))),
            "val_test": int(len(val_set.intersection(test_set))),
            "triple_overlap": int(len(train_set.intersection(val_set).intersection(test_set))),
        },
    }


def to_probs(logits: np.ndarray) -> np.ndarray:
    # Binary probs from logits with stable softmax.
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]


def compute_trainer_metrics(eval_pred):
    logits, labels = eval_pred
    probs = to_probs(np.asarray(logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    split_dir = resolve_split_dir(args.data_dir)
    train_df = load_split(split_dir / "train.csv")
    val_df = load_split(split_dir / "val.csv")
    test_df = load_split(split_dir / "test.csv")

    text_col = resolve_text_column(train_df, args.text_col)

    x_train = train_df[text_col].fillna("").astype(str).tolist()
    y_train = train_df[args.label_col].astype(int).to_numpy()

    x_val = val_df[text_col].fillna("").astype(str).tolist()
    y_val = val_df[args.label_col].astype(int).to_numpy()

    x_test = test_df[text_col].fillna("").astype(str).tolist()
    y_test = test_df[args.label_col].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_enc = tokenizer(x_train, truncation=True, padding="max_length", max_length=args.max_len)
    val_enc = tokenizer(x_val, truncation=True, padding="max_length", max_length=args.max_len)
    test_enc = tokenizer(x_test, truncation=True, padding="max_length", max_length=args.max_len)

    train_ds = TextDataset(train_enc, y_train)
    val_ds = TextDataset(val_enc, y_val)
    test_ds = TextDataset(test_enc, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_trainer_metrics,
    }

    # Compatible across transformers versions where `tokenizer` may be removed.
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "tokenizer" not in str(exc):
            raise
        trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Save best model/tokenizer for inference.
    model_save_dir = args.output_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    val_logits = trainer.predict(val_ds).predictions
    test_logits = trainer.predict(test_ds).predictions

    val_prob = to_probs(np.asarray(val_logits))
    test_prob = to_probs(np.asarray(test_logits))

    if args.threshold is None:
        threshold, best_val_f1 = find_best_f1_threshold(y_val, val_prob)
    else:
        threshold = float(args.threshold)
        best_val_f1 = None

    val_metrics = evaluate_predictions(y_val, val_prob, threshold)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)

    log_history_path = args.output_dir / "training_log_history.json"
    with log_history_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    summary = {
        "model": f"HF:{args.model_name}",
        "model_kind": "transformer",
        "data_dir": str(args.data_dir),
        "resolved_split_dir": str(split_dir),
        "text_col": text_col,
        "label_col": args.label_col,
        "hyperparameters": {
            "model_name": args.model_name,
            "max_len": args.max_len,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "threshold_used": threshold,
            "seed": args.seed,
        },
        "val_threshold_search": {
            "enabled": args.threshold is None,
            "best_f1_at_threshold": best_val_f1,
        },
        "leakage_audit": leakage_audit(train_df, val_df, test_df, text_col),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_dir": str(model_save_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "training_log_history": str(log_history_path),
        },
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Transformer training complete.")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
