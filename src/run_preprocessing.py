from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing import (
    PreprocessConfig,
    build_report,
    load_and_validate,
    preprocess_dataframe,
    save_splits_and_report,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess fake-news dataset")

    default_input = Path(__file__).resolve().parents[1] / "data" / "fake_news.csv"
    default_output = Path(__file__).resolve().parents[1] / "output" / "split"

    parser.add_argument("--input", type=Path, default=default_input, help="Path to input CSV")
    parser.add_argument(
        "--output-dir",
        "--output",
        type=Path,
        default=default_output,
        help="Directory where processed splits and report are saved",
    )
    parser.add_argument("--min-text-length", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PreprocessConfig(
        min_text_length=args.min_text_length,
        test_size=args.test_size,
        val_size_from_train=args.val_size,
        random_state=args.random_state,
    )

    print(f"Loading data from: {args.input}")
    df = load_and_validate(args.input)

    print("Preprocessing text...")
    processed_df = preprocess_dataframe(df, cfg)

    print("Creating train/val/test split...")
    train_df, val_df, test_df = split_dataset(processed_df, cfg)

    report = build_report(processed_df, train_df, val_df, test_df, cfg)

    print(f"Saving outputs to: {args.output_dir}")
    save_splits_and_report(args.output_dir, train_df, val_df, test_df, report)

    print("Done. Files created:")
    print(f"- {args.output_dir / 'train.csv'}")
    print(f"- {args.output_dir / 'val.csv'}")
    print(f"- {args.output_dir / 'test.csv'}")
    print(f"- {args.output_dir / 'preprocessing_report.json'}")


if __name__ == "__main__":
    main()
