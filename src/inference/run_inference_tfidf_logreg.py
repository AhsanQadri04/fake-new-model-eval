from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from metrics_utils import evaluate_predictions, load_split, resolve_text_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for TF-IDF + Logistic Regression model")

    default_data_dir = Path(__file__).resolve().parents[2] / "output" / "split"
    default_model_dir = Path(__file__).resolve().parents[2] / "output" / "tfidf_logreg"
    default_output_file = Path(__file__).resolve().parents[2] / "output" / "inference" / "tfidf_logreg_test.json"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-file", type=Path, default=default_output_file)

    return parser.parse_args()


def load_model_threshold(model_dir: Path) -> float:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return 0.5

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    hyper = data.get("hyperparameters", {})
    threshold = hyper.get("threshold", 0.5)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.5


def main() -> None:
    args = parse_args()

    vectorizer_path = args.model_dir / "tfidf_vectorizer.joblib"
    model_path = args.model_dir / "logreg_model.joblib"

    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Expected TF-IDF artifacts not found in model directory. "
            "Need tfidf_vectorizer.joblib and logreg_model.joblib"
        )

    df = load_split(args.data_dir, args.split)
    text_col = resolve_text_column(df, args.text_col)

    x = df[text_col].fillna("").astype(str).tolist()
    y = df[args.label_col].astype(int).to_numpy()

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    x_vec = vectorizer.transform(x)
    y_prob = model.predict_proba(x_vec)[:, 1]

    threshold = float(args.threshold) if args.threshold is not None else load_model_threshold(args.model_dir)
    metrics = evaluate_predictions(y, y_prob, threshold)

    result: Dict[str, Any] = {
        "model_name": "TFIDF+LogReg",
        "model_kind": "tfidf_logreg",
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "text_col": text_col,
        "label_col": args.label_col,
        "threshold_used": threshold,
        "num_samples": int(len(df)),
        "metrics": metrics,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("TF-IDF/LogReg inference complete.")
    print(f"Split: {args.split}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
