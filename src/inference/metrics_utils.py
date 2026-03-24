from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    split_dir = resolve_split_dir(data_dir)
    file_path = split_dir / f"{split}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    return pd.read_csv(file_path)


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
