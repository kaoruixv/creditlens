"""Score a CSV of raw UCI-format records with the trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import model_matrix


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate probability of default for credit records.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("scored_credit_records.csv"))
    args = parser.parse_args()

    bundle = joblib.load(ROOT / "artifacts" / "credit_risk_model.joblib")
    raw = pd.read_csv(args.input_csv)
    features = model_matrix(raw)[bundle["feature_columns"]]
    probability = bundle["model"].predict_proba(features)[:, 1]
    result = raw.copy()
    result["predicted_probability_of_default"] = probability
    result["manual_review_flag"] = probability >= bundle["threshold"]
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} records -> {args.output}")


if __name__ == "__main__":
    main()
