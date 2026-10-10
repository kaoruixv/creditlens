"""Train, evaluate, stress-test, and package the credit-default model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from features import apply_repayment_shock, engineer_features, model_matrix, rename_raw_columns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
MISSED_DEFAULT_COST = 5.0
FALSE_ALARM_COST = 1.0
DECISION_THRESHOLD = FALSE_ALARM_COST / (MISSED_DEFAULT_COST + FALSE_ALARM_COST)
LGD = 0.60


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "credit_default.csv"
    if cache.exists():
        frame = pd.read_csv(cache)
        return frame.drop(columns="default"), frame["default"].astype(int)

    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].astype(int).rename("default")
    pd.concat([X, y], axis=1).to_csv(cache, index=False)
    return X, y


def candidate_models() -> dict[str, object]:
    return {
        "logistic_scorecard": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", C=0.5)),
        ]),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def metric_row(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "recall_default": float(tp / (tp + fn)),
        "precision_default": float(tp / (tp + fp)),
    }


def save_evaluation_plots(y_test, probabilities, chosen_name, model, X_test):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, probability):.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y_test, probability):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes: ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "model_discrimination.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    CalibrationDisplay.from_predictions(y_test, probabilities[chosen_name], n_bins=10, strategy="quantile", ax=ax)
    ax.set_title("Probability calibration — held-out test set"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=180); plt.close(fig)

    importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=4, random_state=RANDOM_STATE)
    top = pd.Series(importance.importances_mean, index=X_test.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 5.8))
    top.plot.barh(ax=ax, color="#164e63")
    ax.set(title="What the model relies on", xlabel="Decrease in ROC-AUC when feature is shuffled")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=180); plt.close(fig)
    return top.sort_values(ascending=False)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    X_raw, y = load_data()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X = model_matrix(X_raw)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    fitted = {}
    probabilities = {}
    candidate_metrics = {}
    for name, estimator in candidate_models().items():
        cv_scores[name] = cross_val_score(estimator, X_train, y_train, scoring="roc_auc", cv=cv).tolist()
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)
        fitted[name] = calibrated
        probabilities[name] = calibrated.predict_proba(X_test)[:, 1]
        candidate_metrics[name] = metric_row(y_test, probabilities[name], DECISION_THRESHOLD)

    chosen_name = max(candidate_metrics, key=lambda n: candidate_metrics[n]["roc_auc"])
    chosen_model = fitted[chosen_name]
    chosen_probability = probabilities[chosen_name]
    top_features = save_evaluation_plots(y_test, probabilities, chosen_name, chosen_model, X_test)

    raw_test = X_raw.iloc[test_idx].copy()
    readable_test = rename_raw_columns(raw_test)
    stressed_raw = apply_repayment_shock(raw_test)
    stressed_probability = chosen_model.predict_proba(model_matrix(stressed_raw))[:, 1]
    exposure = readable_test["limit_bal"].to_numpy()
    base_expected_loss = chosen_probability * LGD * exposure
    stress_expected_loss = stressed_probability * LGD * exposure

    audit = engineer_features(raw_test)[["sex", "age"]].copy()
    audit["actual"] = y_test.to_numpy()
    audit["predicted_pd"] = chosen_probability
    audit["flagged"] = chosen_probability >= DECISION_THRESHOLD
    audit["sex_group"] = audit["sex"].map({1: "male", 2: "female"}).fillna("other")
    fairness = audit.groupby("sex_group", observed=True).agg(
        records=("actual", "size"), actual_default_rate=("actual", "mean"),
        mean_predicted_pd=("predicted_pd", "mean"), review_rate=("flagged", "mean")
    ).reset_index()

    predictions = pd.DataFrame({
        "actual_default": y_test.to_numpy(),
        "predicted_pd": chosen_probability,
        "stress_pd": stressed_probability,
        "credit_limit_ntd": exposure,
        "base_expected_loss_ntd": base_expected_loss,
        "stress_expected_loss_ntd": stress_expected_loss,
        "review_flag": chosen_probability >= DECISION_THRESHOLD,
    })
    predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    fairness.to_csv(ARTIFACTS / "fairness_audit.csv", index=False)
    top_features.rename("roc_auc_importance").to_csv(ARTIFACTS / "feature_importance.csv")

    metrics = {
        "dataset": {"rows": len(y), "features_raw": X_raw.shape[1], "default_rate": float(y.mean()), "test_rows": len(y_test)},
        "decision": {"missed_default_cost": MISSED_DEFAULT_COST, "false_alarm_cost": FALSE_ALARM_COST, "review_threshold": DECISION_THRESHOLD, "loss_given_default": LGD},
        "models": candidate_metrics,
        "cross_validation_roc_auc": {name: {"folds": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals))} for name, vals in cv_scores.items()},
        "selected_model": chosen_name,
        "portfolio": {
            "base_mean_pd": float(chosen_probability.mean()),
            "stress_mean_pd": float(stressed_probability.mean()),
            "base_expected_loss_ntd": float(base_expected_loss.sum()),
            "stress_expected_loss_ntd": float(stress_expected_loss.sum()),
            "expected_loss_increase_pct": float((stress_expected_loss.sum() / base_expected_loss.sum() - 1) * 100),
        },
        "protected_attribute_policy": "Sex is excluded from model inputs and used only for post-model auditing.",
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": chosen_model, "feature_columns": X.columns.tolist(), "threshold": DECISION_THRESHOLD, "lgd": LGD}, ARTIFACTS / "credit_risk_model.joblib")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Observed portfolio", "Repayment shock"]
    values = [metrics["portfolio"]["base_expected_loss_ntd"], metrics["portfolio"]["stress_expected_loss_ntd"]]
    bars = ax.bar(labels, values, color=["#164e63", "#c2410c"])
    ax.bar_label(bars, labels=[f"NT${v/1e6:.1f}m" for v in values], padding=4)
    ax.set(title="Portfolio expected loss under stress", ylabel="Expected loss (NT$)")
    ax.set_ylim(0, max(values) * 1.16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:.0f}m"))
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "stress_test.png", dpi=180); plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
