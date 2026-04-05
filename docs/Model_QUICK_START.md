# Quick Start Guide - Using Trained Models

## Training Results Summary

### Production Model: **Optimised XGBoost**
- **ROC-AUC**: 0.7338
- **Recall (default class)**: 46.9%
- **F1 (default class)**: 0.345
- **Decision threshold** (saved): 0.6387
- **CV ROC-AUC** (best config): 0.726 ± 0.0025

### Top features (SHAP — global)
1. **Age**
2. **MonthsEmployed**
3. **Loan_to_Income**
4. **HasDependents**
5. **HasCoSigner**
6. **HasMortgage**

---

## Files Created

### Models (`models/` folder)
- `optimised_xgb_pipeline.pkl` - **Full Pipeline** (ColumnTransformer + SMOTE + XGBoost)
- `optimised_xgb_threshold.pkl` - Tuned probability threshold (0.6387)

### Results (`results/` folder)
- `optimised_results.csv` - Metrics, CV summary, best hyperparameters
- `optimised_xgb_feature_importance.csv` - Gain-based importances
- `shap_feature_importance.csv` - Mean |SHAP| importances

---

## How to Use Trained Models

### Example: Making a Prediction

```python
import sys
sys.path.insert(0, "scripts")

import pandas as pd
from predict import LoanRiskPredictor

predictor = LoanRiskPredictor()

# Single applicant (raw fields as in training CSV; InterestRate optional — excluded internally)
applicant_dict = {
    "Age": 35,
    "Income": 75000,
    "LoanAmount": 50000,
    "CreditScore": 720,
    "MonthsEmployed": 60,
    "NumCreditLines": 3,
    "LoanTerm": 36,
    "DTIRatio": 0.35,
    "Education": "Bachelor",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Auto",
    "HasCoSigner": "No",
}

result = predictor.predict(applicant_dict)
# result["probability"]     — P(default)
# result["prediction"]      — 0 or 1 (vs threshold 0.6387)
# result["risk_category"]   — "Low" / "Medium" / "High"
# result["top_features"]    — SHAP contributors if shap is installed

# Batch
df = pd.DataFrame([applicant_dict])
out_df = predictor.predict_batch(df)
```

---

## Model Selection Guide

The repo ships **one** production model: the optimised XGBoost Pipeline. Use **LoanRiskPredictor** for inference so preprocessing, SMOTE-free prediction path, thresholding, risk bands, and optional SHAP stay consistent with training.

---

## Integration with Backend API

### Step 1: Copy models to backend
```bash
# Copy models folder to backend directory
cp -r models/ backend/models/
```

### Step 2: Create prediction endpoint
```python
# In backend — import LoanRiskPredictor from predict.py (ensure scripts on PYTHONPATH
# or vendor predict.py); return probability, prediction, risk_category
```

### Step 3: Connect to frontend
- Frontend sends loan application data
- Backend uses **LoanRiskPredictor** to return probability, binary outcome, and risk category

---

## Retraining Models

To retrain with new data or different parameters:

1. Update `scripts/train_models.py` if you change search space or data path
2. Run: `cd scripts && python train_models.py`
3. New pipeline and threshold overwrite `models/`; refresh `results/`

---

## Notes

- **Class Imbalance**: Handled with **SMOTE** (training) + **scale_pos_weight** (~7.61)
- **Metrics**: ROC-AUC is the primary ranking metric; recall/F1 quoted for the **default** class at the **saved threshold**
- **InterestRate** is **not** used as a feature (leakage)
- **SHAP**: Global table in `shap_feature_importance.csv`; per prediction via `predict.py` when `shap` is installed

---

*Models trained successfully and ready for deployment!*
