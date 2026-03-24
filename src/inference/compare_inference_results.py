from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare inference outputs from different models")

    default_base = Path(__file__).resolve().parents[2] / "output" / "inference"

    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to inference JSON files",
    )
    parser.add_argument("--output-json", type=Path, default=default_base / "comparison.json")
    parser.add_argument("--output-csv", type=Path, default=default_base / "comparison.csv")

    return parser.parse_args()


def read_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Inference result file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    for p in args.inputs:
        result = read_result(p)
        full_results.append(result)
        m = result.get("metrics", {})
        rows.append(
            {
                "model_name": result.get("model_name"),
                "model_kind": result.get("model_kind"),
                "split": result.get("split"),
                "num_samples": result.get("num_samples"),
                "threshold_used": result.get("threshold_used"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "source_file": str(p),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["f1", "roc_auc"], ascending=False).reset_index(drop=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": rows,
                "sorted_summary": df.to_dict(orient="records"),
                "full_results": full_results,
            },
            f,
            indent=2,
        )

    df.to_csv(args.output_csv, index=False)

    print("Inference comparison complete.")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")
    if not df.empty:
        best = df.iloc[0]
        print(
            "Best by F1: "
            f"{best['model_name']} (F1={best['f1']:.4f}, ROC-AUC={best['roc_auc']:.4f})"
        )


if __name__ == "__main__":
    main()
