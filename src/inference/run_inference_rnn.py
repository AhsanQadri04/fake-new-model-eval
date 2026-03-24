from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from metrics_utils import evaluate_predictions, load_split, resolve_text_column

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for RNN inference. Install with: pip install tensorflow"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for RNN-family model")

    default_data_dir = Path(__file__).resolve().parents[2] / "output" / "split"
    default_model_dir = Path(__file__).resolve().parents[2] / "output" / "rnn"
    default_output_file = Path(__file__).resolve().parents[2] / "output" / "inference" / "rnn_test.json"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--truncate-strategy",
        type=str,
        default=None,
        choices=["post", "head_tail"],
    )
    parser.add_argument("--output-file", type=Path, default=default_output_file)

    return parser.parse_args()


def load_training_config(model_dir: Path) -> Dict[str, Any]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate_head_tail(sequences: List[List[int]], max_len: int) -> List[List[int]]:
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if max_len == 1:
        return [seq[:1] for seq in sequences]

    head_len = max_len // 2
    tail_len = max_len - head_len

    out: List[List[int]] = []
    for seq in sequences:
        if len(seq) <= max_len:
            out.append(seq)
        else:
            out.append(seq[:head_len] + seq[-tail_len:])
    return out


def main() -> None:
    args = parse_args()

    model_path = args.model_dir / "rnn_model.keras"
    tokenizer_path = args.model_dir / "tokenizer.json"

    if not model_path.exists():
        best_path = args.model_dir / "best_rnn.keras"
        if best_path.exists():
            model_path = best_path

    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            "Expected RNN artifacts not found. Need rnn_model.keras (or best_rnn.keras) and tokenizer.json"
        )

    training_meta = load_training_config(args.model_dir)
    hyper = training_meta.get("hyperparameters", {})

    threshold = args.threshold
    if threshold is None:
        threshold = hyper.get("threshold_used", hyper.get("threshold", 0.5))
    threshold = float(threshold)

    max_len = args.max_len
    if max_len is None:
        max_len = int(hyper.get("max_len", 500))

    truncate_strategy = args.truncate_strategy
    if truncate_strategy is None:
        truncate_strategy = str(hyper.get("truncate_strategy", "post"))

    df = load_split(args.data_dir, args.split)
    text_col = resolve_text_column(df, args.text_col)

    x_text = df[text_col].fillna("").astype(str).tolist()
    y = df[args.label_col].astype(int).to_numpy()

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    model = tf.keras.models.load_model(model_path)

    seq = tokenizer.texts_to_sequences(x_text)
    if truncate_strategy == "head_tail":
        seq = truncate_head_tail(seq, max_len)
        x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    else:
        x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    y_prob = model.predict(x, verbose=0).ravel()
    metrics = evaluate_predictions(y, y_prob, threshold)

    result: Dict[str, Any] = {
        "model_name": str(training_meta.get("model", "RNN")),
        "model_kind": "rnn_family",
        "model_dir": str(args.model_dir),
        "model_file": str(model_path),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "text_col": text_col,
        "label_col": args.label_col,
        "threshold_used": threshold,
        "max_len_used": max_len,
        "truncate_strategy": truncate_strategy,
        "num_samples": int(len(df)),
        "metrics": metrics,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("RNN inference complete.")
    print(f"Split: {args.split}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
