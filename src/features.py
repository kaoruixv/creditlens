"""Feature engineering for the credit-default model."""
from __future__ import annotations

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "X1": "limit_bal", "X2": "sex", "X3": "education", "X4": "marriage", "X5": "age",
    "X6": "pay_0", "X7": "pay_2", "X8": "pay_3", "X9": "pay_4", "X10": "pay_5", "X11": "pay_6",
    "X12": "bill_amt1", "X13": "bill_amt2", "X14": "bill_amt3", "X15": "bill_amt4", "X16": "bill_amt5", "X17": "bill_amt6",
    "X18": "pay_amt1", "X19": "pay_amt2", "X20": "pay_amt3", "X21": "pay_amt4", "X22": "pay_amt5", "X23": "pay_amt6",
}

PAY_STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILLS = [f"bill_amt{i}" for i in range(1, 7)]
PAYMENTS = [f"pay_amt{i}" for i in range(1, 7)]
EXCLUDED_FROM_MODEL = ["sex"]


def rename_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with human-readable feature names."""
    missing = set(COLUMN_MAP) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    return frame.rename(columns=COLUMN_MAP).copy()


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create financially interpretable behavioural features.

    The raw monthly values are preserved. Derived features summarize utilization,
    repayment capacity, and delinquency persistence.
    """
    data = rename_raw_columns(frame) if "X1" in frame.columns else frame.copy()
    limit = data["limit_bal"].clip(lower=1)
    bill_matrix = data[BILLS]
    payment_matrix = data[PAYMENTS]
    status_matrix = data[PAY_STATUS]

    data["avg_bill"] = bill_matrix.mean(axis=1)
    data["avg_payment"] = payment_matrix.mean(axis=1)
    data["mean_utilization"] = (bill_matrix.div(limit, axis=0)).mean(axis=1)
    data["max_utilization"] = (bill_matrix.div(limit, axis=0)).max(axis=1)
    data["payment_to_bill"] = data["avg_payment"] / data["avg_bill"].abs().clip(lower=1)
    data["max_delinquency"] = status_matrix.max(axis=1)
    data["delinquency_months"] = (status_matrix > 0).sum(axis=1)
    data["severe_delinquency_months"] = (status_matrix >= 2).sum(axis=1)
    data["bill_trend"] = data["bill_amt1"] - data["bill_amt6"]
    data["payment_trend"] = data["pay_amt1"] - data["pay_amt6"]
    return data.replace([np.inf, -np.inf], np.nan).fillna(0)


def model_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Produce model inputs while withholding sex from training."""
    return engineer_features(raw).drop(columns=EXCLUDED_FROM_MODEL)


def apply_repayment_shock(raw: pd.DataFrame) -> pd.DataFrame:
    """Stress scenario: one-step worse delinquency, 10% higher bills, 20% lower payments."""
    stressed = rename_raw_columns(raw) if "X1" in raw.columns else raw.copy()
    stressed[PAY_STATUS] = (stressed[PAY_STATUS] + 1).clip(upper=8)
    stressed[BILLS] = stressed[BILLS] * 1.10
    stressed[PAYMENTS] = stressed[PAYMENTS] * 0.80
    return stressed
