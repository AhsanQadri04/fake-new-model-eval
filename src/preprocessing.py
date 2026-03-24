from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s\.!\?,'\-]")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True
    remove_html: bool = True
    remove_non_alnum: bool = True
    keep_sentence_punctuation: bool = True
    min_text_length: int = 5
    deduplicate_by_clean_text: bool = True
    drop_conflicting_duplicates: bool = True
    random_state: int = 42
    test_size: float = 0.2
    val_size_from_train: float = 0.1


def deduplicate_for_split(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    deduped = df.copy()
    stats = {
        "rows_before": int(len(deduped)),
        "conflicting_text_rows_dropped": 0,
        "exact_text_duplicates_dropped": 0,
        "rows_after": 0,
    }

    if cfg.drop_conflicting_duplicates:
        label_counts = deduped.groupby("clean_text")["label"].nunique()
        conflicting_texts = set(label_counts[label_counts > 1].index.tolist())
        if conflicting_texts:
            conflict_mask = deduped["clean_text"].isin(conflicting_texts)
            stats["conflicting_text_rows_dropped"] = int(conflict_mask.sum())
            deduped = deduped[~conflict_mask].copy()

    if cfg.deduplicate_by_clean_text:
        before = len(deduped)
        deduped = deduped.drop_duplicates(subset=["clean_text"], keep="first").copy()
        stats["exact_text_duplicates_dropped"] = int(before - len(deduped))

    deduped = deduped.reset_index(drop=True)
    stats["rows_after"] = int(len(deduped))
    return deduped, stats


def split_overlap_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, int]:
    train_set = set(train_df["clean_text"].fillna("").astype(str).tolist())
    val_set = set(val_df["clean_text"].fillna("").astype(str).tolist())
    test_set = set(test_df["clean_text"].fillna("").astype(str).tolist())

    return {
        "train_val": int(len(train_set.intersection(val_set))),
        "train_test": int(len(train_set.intersection(test_set))),
        "val_test": int(len(val_set.intersection(test_set))),
        "triple_overlap": int(len(train_set.intersection(val_set).intersection(test_set))),
    }


def clean_text(text: str, cfg: PreprocessConfig) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    cleaned = text

    if cfg.normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)

    if cfg.remove_html:
        cleaned = HTML_PATTERN.sub(" ", cleaned)

    if cfg.remove_urls:
        cleaned = URL_PATTERN.sub(" ", cleaned)

    if cfg.remove_emails:
        cleaned = EMAIL_PATTERN.sub(" ", cleaned)

    if cfg.remove_mentions:
        cleaned = MENTION_PATTERN.sub(" ", cleaned)

    if cfg.lowercase:
        cleaned = cleaned.lower()

    if cfg.remove_non_alnum:
        if cfg.keep_sentence_punctuation:
            cleaned = NON_ALNUM_PATTERN.sub(" ", cleaned)
        else:
            cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    if csv_path.is_dir():
        candidate = csv_path / "fake_news.csv"
        if candidate.exists():
            csv_path = candidate
        else:
            raise ValueError(
                "Input path points to a directory. Please pass a CSV file path, "
                "or include fake_news.csv inside that directory."
            )

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Input must be a .csv file, got: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset must contain columns {required_columns}. Missing: {sorted(missing)}"
        )

    return df


def preprocess_dataframe(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    processed = df.copy()

    processed["text"] = processed["text"].fillna("").astype(str)
    processed["label"] = pd.to_numeric(processed["label"], errors="coerce")

    before_rows = len(processed)
    processed = processed.dropna(subset=["label"])
    processed["label"] = processed["label"].astype(int)

    # Keep binary labels only (0/1) for fake-news classification.
    processed = processed[processed["label"].isin([0, 1])].copy()

    processed["clean_text"] = processed["text"].apply(lambda t: clean_text(t, cfg))
    processed["text_len"] = processed["clean_text"].str.len()

    processed = processed[processed["text_len"] >= cfg.min_text_length].copy()
    processed = processed.drop(columns=["text_len"])

    processed, dedup_stats = deduplicate_for_split(processed, cfg)

    if cfg.deduplicate_by_clean_text or cfg.drop_conflicting_duplicates:
        print("Deduplication step:")
        print(
            "- conflicting text rows dropped: "
            f"{dedup_stats['conflicting_text_rows_dropped']}"
        )
        print(
            "- exact text duplicates dropped: "
            f"{dedup_stats['exact_text_duplicates_dropped']}"
        )
        print(f"- rows after deduplication:      {dedup_stats['rows_after']}")

    dropped_rows = before_rows - len(processed)
    print(f"Rows before preprocessing: {before_rows}")
    print(f"Rows after preprocessing:  {len(processed)}")
    print(f"Rows dropped:             {dropped_rows}")

    return processed


def split_dataset(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size_from_train,
        random_state=cfg.random_state,
        stratify=train_df["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> Dict[str, float]:
    counts = df["label"].value_counts(normalize=True).sort_index()
    return {str(k): float(v) for k, v in counts.items()}


def build_report(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Dict[str, object]:
    overlap = split_overlap_stats(train_df, val_df, test_df)
    return {
        "config": asdict(cfg),
        "num_samples": {
            "full": int(len(full_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_distribution": {
            "full": label_distribution(full_df),
            "train": label_distribution(train_df),
            "val": label_distribution(val_df),
            "test": label_distribution(test_df),
        },
        "text_length_stats": {
            "full_mean": float(full_df["clean_text"].str.len().mean()),
            "train_mean": float(train_df["clean_text"].str.len().mean()),
            "val_mean": float(val_df["clean_text"].str.len().mean()),
            "test_mean": float(test_df["clean_text"].str.len().mean()),
        },
        "split_overlap": overlap,
    }


def save_splits_and_report(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
