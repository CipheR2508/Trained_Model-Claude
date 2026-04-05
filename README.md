# Loan Default Prediction Model

Production-grade ML models for loan default prediction. Part of the FinRisk AI system.

## Directory Structure

```
├── data/                    # Raw training data
│   └── raw/
│       └── Loan_default.csv          # Original dataset (255,347 records)
│
├── models/                  # Production artefacts
│   ├── optimised_xgb_pipeline.pkl    # Full sklearn Pipeline (ColumnTransformer + SMOTE + XGBoost)
│   └── optimised_xgb_threshold.pkl     # Tuned decision threshold (0.6387)
│
├── results/                 # Training results and metrics
│   ├── optimised_results.csv                    # Metrics, CV summary, best hyperparameters
│   ├── optimised_xgb_feature_importance.csv     # XGBoost gain-based importances
│   └── shap_feature_importance.csv              # Mean |SHAP| importances
│
├── scripts/                 # Training and inference
│   ├── train_models.py               # Training / retraining
│   └── predict.py                  # LoanRiskPredictor inference API
│
├── docs/                    # Documentation
│   ├── TRAINING_GUIDE.md             # Complete training guide
│   └── Model_QUICK_START.md          # Quick reference guide
│
└── README.md                # This file
```

## Quick Reference

### For Backend Integration
- **Use**: `models/` folder — pipeline (`optimised_xgb_pipeline.pkl`) and threshold (`optimised_xgb_threshold.pkl`)
- **Primary Model**: Optimised XGBoost pipeline (single production model)

### For Frontend Display
- **Use**: `results/` folder — metrics (`optimised_results.csv`), feature and SHAP importances

### For Retraining
- **Use**: `scripts/train_models.py` — run this to retrain with new data/parameters

### For Documentation
- **See**: `docs/` folder — complete guides and quick references

## Model Performance

| Model | ROC-AUC | Recall (default) | F1 (default) | CV AUC (mean ± std) |
|-------|---------|------------------|--------------|---------------------|
| XGBoost (optimised) | 0.7338 | 46.9% | 0.345 | 0.726 ± 0.0025 |

## Usage

### Making Predictions (Backend)
```python
import sys
sys.path.insert(0, "scripts")

from predict import LoanRiskPredictor

predictor = LoanRiskPredictor()
result = predictor.predict(applicant_dict)
# result: probability, prediction, risk_category (Low/Medium/High), top_features (SHAP if installed)

batch_df = predictor.predict_batch(df)
```

### Retraining Models
```bash
cd scripts
python train_models.py
```

## Notes

- **models/** folder is the production folder - used by backend API
- **results/** folder is for reference - can be used to display metrics in frontend
- **data/** folder contains original dataset - not needed for predictions
- **scripts/** folder contains training code - only needed for retraining
- **docs/** folder contains documentation - for reference only

---

*Last Updated: After model training completion*
