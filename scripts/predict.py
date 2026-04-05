"""
FinRisk AI — Production Inference Module
=========================================
Loads the saved optimised pipeline and threshold.
Accepts raw input (same format as training data) and returns:
  - probability of default
  - binary prediction (at tuned threshold)
  - risk category (Low / Medium / High)
  - top contributing features (if SHAP is available)

Usage:
    from predict import LoanRiskPredictor
    predictor = LoanRiskPredictor()
    result = predictor.predict(applicant_dict)
    print(result)
"""

import numpy as np
import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PIPELINE_PATH  = os.path.join(MODEL_DIR, "optimised_xgb_pipeline.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "optimised_xgb_threshold.pkl")

# Risk category boundaries (probability of default)
RISK_THRESHOLDS = {"low": 0.20, "high": 0.50}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply identical feature engineering as training."""
    df = df.copy()
    df["Income_safe"]         = df["Income"].replace(0, np.nan).fillna(df["Income"].median())
    df["LoanTerm_safe"]       = df["LoanTerm"].replace(0, 1)
    df["NumCreditLines_safe"] = df["NumCreditLines"].fillna(0)

    df["Loan_to_Income"]  = df["LoanAmount"] / df["Income_safe"]
    df["EMI_to_Income"]   = (df["LoanAmount"] / df["LoanTerm_safe"]) / df["Income_safe"]
    df["Credit_per_Line"] = df["CreditScore"] / (df["NumCreditLines_safe"] + 1)

    # Drop helper columns
    df.drop(columns=["Income_safe", "LoanTerm_safe", "NumCreditLines_safe",
                     "InterestRate", "LoanID"], inplace=True, errors="ignore")
    return df


def _risk_category(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low"]:
        return "Low"
    elif prob < RISK_THRESHOLDS["high"]:
        return "Medium"
    return "High"


class LoanRiskPredictor:
    """
    Production-ready predictor. Loads pipeline once on init.

    Example single applicant dict (all training fields except LoanID):
        {
            "Age": 35, "Income": 60000, "LoanAmount": 20000,
            "CreditScore": 680, "MonthsEmployed": 48,
            "NumCreditLines": 3, "LoanTerm": 36, "DTIRatio": 0.35,
            "Education": "Bachelor's", "EmploymentType": "Full-time",
            "MaritalStatus": "Single", "HasMortgage": "No",
            "HasDependents": "No", "LoanPurpose": "Auto",
            "HasCoSigner": "No"
        }
    Note: InterestRate is intentionally excluded (leakage feature).
    """

    def __init__(self):
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Pipeline not found at {PIPELINE_PATH}\n"
                "Run train_models.py first."
            )
        self.pipeline  = joblib.load(PIPELINE_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)
        self._preprocessor = self.pipeline.named_steps["preprocessor"]
        self._classifier   = self.pipeline.named_steps["classifier"]
        self._feature_names = self._preprocessor.get_feature_names_out()

        # Load SHAP explainer lazily
        self._explainer = None

        print(f"[Predictor] Pipeline loaded. Threshold = {self.threshold:.4f}")

    def _get_shap_explainer(self):
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self._classifier)
            except ImportError:
                pass
        return self._explainer

    def predict(self, raw_input: dict, top_n_features: int = 5) -> dict:
        """
        Predict default risk for a single applicant.

        Parameters
        ----------
        raw_input : dict
            Raw applicant data (field names as in training CSV).
        top_n_features : int
            Number of top contributing features to return.

        Returns
        -------
        dict with keys:
            probability   - float, probability of default
            prediction    - int, 0 = safe, 1 = default
            risk_category - str, "Low" / "Medium" / "High"
            top_features  - list of (feature, shap_value) if shap available
        """
        df = pd.DataFrame([raw_input])
        df = _engineer_features(df)

        prob = float(self.pipeline.predict_proba(df)[:, 1][0])
        pred = int(prob >= self.threshold)
        risk = _risk_category(prob)

        top_features = []
        explainer = self._get_shap_explainer()
        if explainer is not None:
            X_transformed = self._preprocessor.transform(df)
            shap_vals = explainer.shap_values(X_transformed)[0]
            indices   = np.argsort(np.abs(shap_vals))[::-1][:top_n_features]
            top_features = [
                {"feature": str(self._feature_names[i]),
                 "shap_value": round(float(shap_vals[i]), 4)}
                for i in indices
            ]

        return {
            "probability":   round(prob, 4),
            "prediction":    pred,
            "risk_category": risk,
            "top_features":  top_features,
        }

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a DataFrame of applicants.
        Returns original DataFrame with added columns:
            default_probability, prediction, risk_category
        """
        df = _engineer_features(raw_df.copy())
        probs = self.pipeline.predict_proba(df)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        risks = [_risk_category(p) for p in probs]

        result = raw_df.copy()
        result["default_probability"] = probs.round(4)
        result["prediction"]          = preds
        result["risk_category"]       = risks
        return result


# ─────────────────────────────────────────────
# Quick sanity-check when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    predictor = LoanRiskPredictor()

    sample = {
        "Age": 35, "Income": 60000, "LoanAmount": 20000,
        "CreditScore": 680, "MonthsEmployed": 48,
        "NumCreditLines": 3, "LoanTerm": 36, "DTIRatio": 0.35,
        "Education": "Bachelor's", "EmploymentType": "Full-time",
        "MaritalStatus": "Single", "HasMortgage": "No",
        "HasDependents": "No", "LoanPurpose": "Auto",
        "HasCoSigner": "No",
    }

    result = predictor.predict(sample)
    print("\n── Prediction result ──")
    for k, v in result.items():
        print(f"  {k}: {v}")
