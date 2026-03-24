from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline")

    default_data_dir = Path(__file__).resolve().parents[1] / "output" / "split"
    default_output_dir = Path(__file__).resolve().parents[1] / "output" / "tfidf_logreg"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)

    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")

    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)

    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--drop-train-overlap-from-eval",
        action="store_true",
        help="Drop val/test rows whose text also appears in training split",
    )

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


def leakage_audit(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str) -> Dict[str, object]:
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


def drop_train_overlap_from_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
):
    train_texts = set(train_df[text_col].fillna("").astype(str).tolist())

    val_mask = ~val_df[text_col].fillna("").astype(str).isin(train_texts)
    test_mask = ~test_df[text_col].fillna("").astype(str).isin(train_texts)

    filtered_val = val_df[val_mask].reset_index(drop=True)
    filtered_test = test_df[test_mask].reset_index(drop=True)

    dropped = {
        "val_dropped": int(len(val_df) - len(filtered_val)),
        "test_dropped": int(len(test_df) - len(filtered_test)),
    }
    return filtered_val, filtered_test, dropped


def evaluate_predictions(y_true, y_prob, threshold: float) -> Dict[str, object]:
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


def main() -> None:
    args = parse_args()

    split_dir = resolve_split_dir(args.data_dir)
    train_df = load_split(split_dir / "train.csv")
    val_df = load_split(split_dir / "val.csv")
    test_df = load_split(split_dir / "test.csv")

    text_col = resolve_text_column(train_df, args.text_col)

    before_audit = leakage_audit(train_df, val_df, test_df, text_col)

    dropped_eval_rows = {"val_dropped": 0, "test_dropped": 0}
    if args.drop_train_overlap_from_eval:
        val_df, test_df, dropped_eval_rows = drop_train_overlap_from_eval(
            train_df, val_df, test_df, text_col
        )

    after_audit = leakage_audit(train_df, val_df, test_df, text_col)

    x_train = train_df[text_col].fillna("").astype(str).tolist()
    y_train = train_df[args.label_col].astype(int).to_numpy()

    x_val = val_df[text_col].fillna("").astype(str).tolist()
    y_val = val_df[args.label_col].astype(int).to_numpy()

    x_test = test_df[text_col].fillna("").astype(str).tolist()
    y_test = test_df[args.label_col].astype(int).to_numpy()

    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train_tfidf, y_train)

    val_prob = model.predict_proba(x_val_tfidf)[:, 1]
    test_prob = model.predict_proba(x_test_tfidf)[:, 1]

    val_metrics = evaluate_predictions(y_val, val_prob, args.threshold)
    test_metrics = evaluate_predictions(y_test, test_prob, args.threshold)

    summary = {
        "model": "TFIDF+LogReg",
        "data_dir": str(args.data_dir),
        "resolved_split_dir": str(split_dir),
        "text_col": text_col,
        "label_col": args.label_col,
        "drop_train_overlap_from_eval": args.drop_train_overlap_from_eval,
        "dropped_eval_rows": dropped_eval_rows,
        "leakage_audit": {
            "before": before_audit,
            "after": after_audit,
        },
        "hyperparameters": {
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "C": args.c,
            "max_iter": args.max_iter,
            "threshold": args.threshold,
        },
        "train_shape": {
            "rows": int(x_train_tfidf.shape[0]),
            "features": int(x_train_tfidf.shape[1]),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, args.output_dir / "logreg_model.joblib")

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("TF-IDF + Logistic Regression training complete.")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
