from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU, LSTM, SimpleRNN, SpatialDropout1D
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for run_rnn.py. Install with: pip install tensorflow"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an RNN model for fake-news detection")

    default_data_dir = Path(__file__).resolve().parents[1] / "output" / "split"
    default_output_dir = Path(__file__).resolve().parents[1] / "output" / "rnn"

    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)

    parser.add_argument("--text-col", type=str, default="clean_text")
    parser.add_argument("--label-col", type=str, default="label")

    parser.add_argument(
        "--model-type",
        type=str,
        default="bigru",
        choices=["simple_rnn", "gru", "lstm", "bigru", "bilstm"],
        help="Recurrent architecture to train",
    )

    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--rnn-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--recurrent-dropout", type=float, default=0.2)

    parser.add_argument(
        "--truncate-strategy",
        type=str,
        default="head_tail",
        choices=["post", "head_tail"],
        help="How to truncate long sequences",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. If omitted, uses best validation-F1 threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-train-overlap-from-eval",
        action="store_true",
        help="Drop val/test rows whose text also appears in training split",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected split file not found: {path}")
    return pd.read_csv(path)


def resolve_split_dir(data_dir: Path) -> Path:
    # Support both: data_dir containing split files directly, or data_dir/split/.
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


def resolve_text_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "clean_text" in df.columns:
        return "clean_text"
    if "text" in df.columns:
        return "text"
    raise ValueError("No usable text column found. Expected one of: clean_text, text")


def _unique_labels(values: pd.Series) -> set[int]:
    return set(pd.to_numeric(values, errors="coerce").dropna().astype(int).tolist())


def audit_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> Dict[str, object]:
    def stats(df: pd.DataFrame) -> Dict[str, int]:
        texts = df[text_col].fillna("").astype(str)
        unique_count = int(texts.nunique())
        return {
            "rows": int(len(df)),
            "unique_texts": unique_count,
            "duplicate_within_split": int(len(df) - unique_count),
        }

    train_text = train_df[text_col].fillna("").astype(str)
    val_text = val_df[text_col].fillna("").astype(str)
    test_text = test_df[text_col].fillna("").astype(str)

    train_set = set(train_text.tolist())
    val_set = set(val_text.tolist())
    test_set = set(test_text.tolist())

    train_map = train_df.groupby(text_col)[label_col].apply(_unique_labels)
    val_map = val_df.groupby(text_col)[label_col].apply(_unique_labels)
    test_map = test_df.groupby(text_col)[label_col].apply(_unique_labels)

    def conflict_count(a: pd.Series, b: pd.Series) -> int:
        overlap_keys = set(a.index).intersection(set(b.index))
        return int(sum(1 for k in overlap_keys if a[k] != b[k]))

    return {
        "split_stats": {
            "train": stats(train_df),
            "val": stats(val_df),
            "test": stats(test_df),
        },
        "cross_split_overlap": {
            "train_val": int(len(train_set.intersection(val_set))),
            "train_test": int(len(train_set.intersection(test_set))),
            "val_test": int(len(val_set.intersection(test_set))),
            "triple_overlap": int(len(train_set.intersection(val_set).intersection(test_set))),
        },
        "overlap_label_conflicts": {
            "train_val": conflict_count(train_map, val_map),
            "train_test": conflict_count(train_map, test_map),
            "val_test": conflict_count(val_map, test_map),
        },
    }


def drop_train_overlap_from_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
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


def truncate_head_tail(sequences: list[list[int]], max_len: int) -> list[list[int]]:
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if max_len == 1:
        return [seq[:1] for seq in sequences]

    head_len = max_len // 2
    tail_len = max_len - head_len

    truncated: list[list[int]] = []
    for seq in sequences:
        if len(seq) <= max_len:
            truncated.append(seq)
        else:
            truncated.append(seq[:head_len] + seq[-tail_len:])
    return truncated


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    vocab_size: int,
    max_len: int,
    truncate_strategy: str,
) -> Tuple[Dict[str, np.ndarray], Tokenizer]:
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if label_col not in split_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in {split_name} split")

    x_train_text = train_df[text_col].fillna("").astype(str).tolist()
    x_val_text = val_df[text_col].fillna("").astype(str).tolist()
    x_test_text = test_df[text_col].fillna("").astype(str).tolist()

    y_train = train_df[label_col].astype(int).to_numpy()
    y_val = val_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_text)

    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_val_seq = tokenizer.texts_to_sequences(x_val_text)
    x_test_seq = tokenizer.texts_to_sequences(x_test_text)

    if truncate_strategy == "head_tail":
        x_train_seq = truncate_head_tail(x_train_seq, max_len)
        x_val_seq = truncate_head_tail(x_val_seq, max_len)
        x_test_seq = truncate_head_tail(x_test_seq, max_len)
        x_train = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
        x_val = pad_sequences(x_val_seq, maxlen=max_len, padding="post", truncating="post")
        x_test = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")
    else:
        x_train = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
        x_val = pad_sequences(x_val_seq, maxlen=max_len, padding="post", truncating="post")
        x_test = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")

    arrays = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    return arrays, tokenizer


def build_rnn_model(
    model_type: str,
    vocab_size: int,
    max_len: int,
    embedding_dim: int,
    rnn_units: int,
    dropout: float,
    recurrent_dropout: float,
    lr: float,
) -> Sequential:
    if model_type == "simple_rnn":
        recurrent_layer = SimpleRNN(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
    elif model_type == "gru":
        recurrent_layer = GRU(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
    elif model_type == "lstm":
        recurrent_layer = LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
    elif model_type == "bigru":
        recurrent_layer = Bidirectional(
            GRU(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
        )
    elif model_type == "bilstm":
        recurrent_layer = Bidirectional(
            LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            SpatialDropout1D(dropout),
            recurrent_layer,
            Dropout(dropout),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, object] = {
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
    return metrics


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.1, 0.901, 0.01):
        pred = (y_prob >= thr).astype(int)
        curr_f1 = f1_score(y_true, pred, zero_division=0)
        if curr_f1 > best_f1:
            best_f1 = float(curr_f1)
            best_threshold = float(thr)
    return best_threshold, best_f1


def save_artifacts(
    output_dir: Path,
    model: Sequential,
    tokenizer: Tokenizer,
    history: tf.keras.callbacks.History,
    summary: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "rnn_model.keras")

    with (output_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    split_dir = resolve_split_dir(args.data_dir)
    train_df = load_split(split_dir / "train.csv")
    val_df = load_split(split_dir / "val.csv")
    test_df = load_split(split_dir / "test.csv")

    text_col = resolve_text_column(train_df, args.text_col)

    leakage_report_before = audit_data_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_col=text_col,
        label_col=args.label_col,
    )

    print("Leakage audit (before optional filtering):")
    print(json.dumps(leakage_report_before["cross_split_overlap"], indent=2))
    print(json.dumps(leakage_report_before["overlap_label_conflicts"], indent=2))

    overlap_drop_info = {"val_dropped": 0, "test_dropped": 0}
    if args.drop_train_overlap_from_eval:
        val_df, test_df, overlap_drop_info = drop_train_overlap_from_eval(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            text_col=text_col,
        )
        print(
            "Dropped eval rows that overlap with train: "
            f"val={overlap_drop_info['val_dropped']}, test={overlap_drop_info['test_dropped']}"
        )

    leakage_report_after = audit_data_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_col=text_col,
        label_col=args.label_col,
    )

    arrays, tokenizer = prepare_xy(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_col=text_col,
        label_col=args.label_col,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        truncate_strategy=args.truncate_strategy,
    )

    model = build_rnn_model(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        lr=args.lr,
    )

    class_weights = get_class_weights(arrays["y_train"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(args.output_dir / "best_rnn.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        arrays["x_train"],
        arrays["y_train"],
        validation_data=(arrays["x_val"], arrays["y_val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_prob = model.predict(arrays["x_val"], verbose=0).ravel()
    test_prob = model.predict(arrays["x_test"], verbose=0).ravel()

    if args.threshold is None:
        threshold, best_val_f1 = find_best_f1_threshold(arrays["y_val"], val_prob)
    else:
        threshold = float(args.threshold)
        best_val_f1 = None

    val_metrics = evaluate_predictions(arrays["y_val"], val_prob, threshold)
    test_metrics = evaluate_predictions(arrays["y_test"], test_prob, threshold)

    summary = {
        "model": args.model_type,
        "data_dir": str(args.data_dir),
        "resolved_split_dir": str(split_dir),
        "text_col": text_col,
        "label_col": args.label_col,
        "drop_train_overlap_from_eval": args.drop_train_overlap_from_eval,
        "dropped_eval_rows": overlap_drop_info,
        "leakage_audit": {
            "before": leakage_report_before,
            "after": leakage_report_after,
        },
        "hyperparameters": {
            "model_type": args.model_type,
            "vocab_size": args.vocab_size,
            "max_len": args.max_len,
            "embedding_dim": args.embedding_dim,
            "rnn_units": args.rnn_units,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "truncate_strategy": args.truncate_strategy,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "threshold_used": threshold,
            "seed": args.seed,
        },
        "val_threshold_search": {
            "enabled": args.threshold is None,
            "best_f1_at_threshold": best_val_f1,
        },
        "class_weights": class_weights,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    save_artifacts(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        history=history,
        summary=summary,
    )

    print("RNN training complete.")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
