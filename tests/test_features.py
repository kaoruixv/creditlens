import sys
import unittest
from pathlib import Path

import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import apply_repayment_shock, model_matrix


def sample():
    return pd.DataFrame({f"X{i}": [100] for i in range(1, 24)}).assign(
        X1=1000, X2=1, X6=0, X7=0, X8=1, X9=0, X10=0, X11=0,
        X12=500, X13=500, X14=500, X15=500, X16=500, X17=500,
        X18=100, X19=100, X20=100, X21=100, X22=100, X23=100,
    )


class FeatureTests(unittest.TestCase):
    def test_protected_sex_is_not_a_model_feature(self):
        self.assertNotIn("sex", model_matrix(sample()).columns)

    def test_utilization_and_delinquency_features_are_correct(self):
        features = model_matrix(sample()).iloc[0]
        self.assertEqual(features["mean_utilization"], 0.5)
        self.assertEqual(features["delinquency_months"], 1)

    def test_stress_worsens_payment_behaviour(self):
        original = sample()
        stressed = apply_repayment_shock(original)
        self.assertEqual(stressed["pay_0"].iloc[0], 1)
        self.assertEqual(stressed["bill_amt1"].iloc[0], 550)
        self.assertEqual(stressed["pay_amt1"].iloc[0], 80)

    def test_packaged_model_scores_valid_probabilities(self):
        root = Path(__file__).resolve().parents[1]
        bundle = joblib.load(root / "artifacts" / "credit_risk_model.joblib")
        features = model_matrix(sample())[bundle["feature_columns"]]
        probability = bundle["model"].predict_proba(features)[0, 1]
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)


if __name__ == "__main__":
    unittest.main()
