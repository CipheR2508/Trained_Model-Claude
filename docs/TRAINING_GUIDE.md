# Loan Default Prediction - Model Training Guide

## Overview
This guide explains the complete process of training machine learning models to predict loan defaults using the Loan Default Prediction dataset.

**Dataset**: Loan_default.csv  
**Model Trained**: Optimised XGBoost (single production model; full sklearn Pipeline)  
**Target Variable**: Default (0 = No Default, 1 = Default)

---

## Dataset Information

### Dataset Statistics
- **Total Records**: 255,347
- **Features**: 17 (excluding LoanID and Default)
- **Target Distribution**:
  - No Default (0): 225,694 (88.4%)
  - Default (1): 29,653 (11.6%)
- **Missing Values**: None

### Features
1. **LoanID** - Unique identifier (excluded from training)
2. **Age** - Borrower age
3. **Income** - Annual income
4. **LoanAmount** - Loan amount requested
5. **CreditScore** - Credit score
6. **MonthsEmployed** - Employment duration in months
7. **NumCreditLines** - Number of credit lines
8. **InterestRate** - Interest rate (**excluded from modelling** — post-decision leakage signal; rate is set after risk assessment)
9. **LoanTerm** - Loan term in months
10. **DTIRatio** - Debt-to-income ratio
11. **Education** - Education level (categorical)
12. **EmploymentType** - Employment type (categorical)
13. **MaritalStatus** - Marital status (categorical)
14. **HasMortgage** - Has mortgage (Yes/No)
15. **HasDependents** - Has dependents (Yes/No)
16. **LoanPurpose** - Purpose of loan (categorical)
17. **HasCoSigner** - Has co-signer (Yes/No)
18. **Default** - Target variable (0/1)

**Engineered features** (computed before split, same logic as inference): **Loan_to_Income**, **EMI_to_Income**, **Credit_per_Line**.

---

## Training Process Steps

### Step 1: Data Loading
- Load the CSV file using pandas
- Check dataset shape and basic information
- Verify data types

### Step 2: Data Exploration
- Check for missing values
- Analyze target variable distribution
- Identify class imbalance (88.4% vs 11.6%)
- Examine feature distributions

### Step 3: Data Preprocessing

#### 3.1 Feature Engineering and Selection
- Engineer **Loan_to_Income**, **EMI_to_Income**, and **Credit_per_Line** (with safe handling for zero/division edge cases)
- Separate features (X) from target (y)
- Drop **LoanID** and **InterestRate** (and temporary helper columns used only for engineering)
- **InterestRate** is excluded because it behaves as a post-decision signal and would inflate metrics (leakage)

#### 3.2 Categorical Encoding
- Identify categorical columns (object dtype): Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner
- Use **OneHotEncoder** (`handle_unknown="ignore"`) inside a **ColumnTransformer** — nominal categories are not ordinally encoded

#### 3.3 Pipeline and Scaling
- **ColumnTransformer**: OneHotEncoder on categoricals; numeric features pass through (remainder `"passthrough"`) — tree models do not require scaling
- **SMOTE** (training data only, inside the imblearn pipeline) for oversampling the minority class
- **XGBoost** classifier with **scale_pos_weight** set from training class counts (~7.61 for neg/pos ratio)

### Step 4: Train-Test Split
- Split ratio: 80% training, 20% testing
- Use stratified split to maintain class distribution
- Random seed: 42 (for reproducibility)
- Training set: ~204,277 samples
- Test set: ~51,070 samples

### Step 5: Model Training

#### 5.1 XGBoost (optimised)
- **Algorithm**: XGBoost binary classifier inside a **Pipeline**: preprocessor → SMOTE → `XGBClassifier`
- **Hyperparameter tuning**: **RandomizedSearchCV** with **5-fold stratified CV**, scoring **ROC-AUC** (`n_iter` search iterations over the space below)
- **Fixed / structural settings**: `objective="binary:logistic"`, `eval_metric="aucpr"`, `scale_pos_weight` from class ratio, `random_state=42`, `n_jobs=-1`
- **Search space** (best values found in the last run are shown — yours may differ slightly after retraining):

| Parameter | Search values (best run) |
|-----------|---------------------------|
| n_estimators | 100, 200, 300 → **300** |
| max_depth | 3, 5, 7 → **3** |
| learning_rate | 0.03, 0.05, 0.1 → **0.05** |
| subsample | 0.8, 1.0 → **0.8** |
| colsample_bytree | 0.8, 1.0 → **1.0** |
| gamma | 0, 1, 5 → **1** |
| min_child_weight | 1, 5, 10 → **5** |
| reg_alpha | 0, 0.1, 1 → **1** |
| reg_lambda | 1, 2, 5 → **1** |

- **Decision threshold**: optimised on the validation/test probability outputs via precision–recall / F1 on the default class; stored separately (**optimised_xgb_threshold.pkl**, value **0.6387** in the current run)

### Step 6: Model Evaluation

#### Metrics Used
1. **Accuracy**: Overall correctness (secondary; can mislead on imbalance)
2. **Precision / Recall / F1**: Reported for the default class where relevant
3. **ROC-AUC**: Primary ranking metric on held-out test probabilities
4. **Cross-validation**: Mean and std of ROC-AUC across folds for the best configuration (stability ~ **±0.0025** in the current run)

#### Evaluation Process
- Fit **RandomizedSearchCV**, refit best estimator on training data
- Evaluate on the stratified test set using **predicted probabilities** and the **saved threshold** (not only 0.5)
- **Current test-level summary**: ROC-AUC **0.7338**, recall (default) **46.9%**, F1 (default) **0.345**

### Step 7: Artefacts and Explainability
- Persist the **full fitted Pipeline** and **threshold** for leakage-free inference
- Export **gain-based** feature importances and optional **SHAP** summary (`shap_feature_importance.csv`)

### Step 8: Model Saving
- Save **optimised_xgb_pipeline.pkl** (preprocessing + SMOTE + classifier)
- Save **optimised_xgb_threshold.pkl**
- Save **optimised_results.csv** (metrics, CV summary, serialised best params)

### Step 9: Feature Importance and SHAP
- Feature importances from the fitted XGBoost step
- **SHAP** (optional): TreeExplainer on transformed features; per-row explanations available in `scripts/predict.py` when `shap` is installed

---

## How to Run Training

### Prerequisites
Install required Python packages:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap joblib
```

### Running the Training Script

1. **Navigate to the project `scripts` folder** (from repo root):
   ```bash
   cd scripts
   ```

2. **Run the training script**:
   ```bash
   python train_models.py
   ```

3. **Wait for completion** (may take several minutes depending on your system)

4. **Check outputs**:
   - Models saved in `models/` folder
   - Results saved in `results/` folder

---

## Output Files

### Models Directory (`models/`)
- `optimised_xgb_pipeline.pkl` - Full sklearn / imblearn Pipeline (ColumnTransformer + SMOTE + XGBoost)
- `optimised_xgb_threshold.pkl` - Optimal probability threshold for binary prediction

### Results Directory (`results/`)
- `optimised_results.csv` - ROC-AUC, threshold, recall, F1, CV mean/std, best hyperparameters
- `optimised_xgb_feature_importance.csv` - Gain-based feature importances
- `shap_feature_importance.csv` - Mean absolute SHAP values (when SHAP is run during training)

---

## Model Selection Criteria

### Primary Metric: ROC-AUC
- Strong choice for imbalanced datasets
- Measures ranking quality of default vs non-default
- Higher is better (range: 0-1)

### Secondary Metrics
- **Precision**: Important if false positives are costly
- **Recall**: Important if missing defaults is costly
- **F1-Score**: Balanced measure on the default class at the chosen threshold
- **Accuracy**: Overall correctness (can be misleading with imbalance)

### Current Benchmark (test set, tuned threshold)
- **ROC-AUC**: 0.7338  
- **Recall (default)**: ~46.9%  
- **F1 (default)**: 0.345  

---

## Handling Class Imbalance

The dataset has significant class imbalance (88.4% vs 11.6%). This pipeline addresses it by:

1. **SMOTE** on the training set only (inside the Pipeline, after preprocessing)
2. **scale_pos_weight** on XGBoost set to the negative/positive count ratio (~**7.61**)
3. **Stratified** train/test split and CV folds
4. **Metrics**: ROC-AUC and default-class recall/F1 at the **optimised threshold** rather than accuracy alone

---

## Using Trained Models

### Loading and predicting via `LoanRiskPredictor`
```python
import sys
sys.path.insert(0, "scripts")

from predict import LoanRiskPredictor

predictor = LoanRiskPredictor()
result = predictor.predict(applicant_dict)
# Keys: probability, prediction, risk_category, top_features (SHAP if available)

batch = predictor.predict_batch(raw_dataframe)
```

### Making Predictions
1. Instantiate **LoanRiskPredictor** (loads pipeline + threshold).
2. Pass **raw** applicant row(s) consistent with the training CSV (InterestRate may be omitted; it is stripped during feature engineering).
3. Read **probability**, **binary prediction** (vs threshold **0.6387**), and **risk_category** (Low / Medium / High).

---

## Next Steps

After training:

1. **Review Results**: Open `results/optimised_results.csv` for metrics and stability

2. **Feature Analysis**: Review `optimised_xgb_feature_importance.csv` and `shap_feature_importance.csv`

3. **Model Deployment**: 
   - Integrate **LoanRiskPredictor** into the backend API
   - Expose probability, binary outcome, and risk band

4. **Model Monitoring**: 
   - Track performance over time
   - Retrain periodically with new data
   - Monitor for drift

---

## Troubleshooting

### Common Issues

1. **Memory Error**: 
   - Reduce dataset size or use chunking
   - Reduce `n_estimators` or `n_iter` in search

2. **Slow Training**: 
   - Reduce RandomizedSearchCV `n_iter`
   - Reduce `n_estimators` upper bound

3. **Poor Performance**: 
   - Review feature leakage (keep InterestRate out)
   - Adjust threshold business-side if precision/recall trade-off changes

4. **Import Errors**: 
   - Install missing packages: `pip install imbalanced-learn shap xgboost`

---

## Model Performance Expectations

### Current production model (test set)

| Model | ROC-AUC | Recall (default) | F1 (default) | CV ROC-AUC (mean ± std) |
|-------|---------|------------------|--------------|-------------------------|
| Optimised XGBoost | 0.7338 | 46.9% | 0.345 | 0.726 ± 0.0025 |

*Figures correspond to the saved pipeline and threshold; retraining may shift them slightly.*

---

## Best Practices

1. **Always split data before any preprocessing** (to avoid data leakage)
2. **Use stratified splitting** for imbalanced datasets
3. **Save the full Pipeline** and threshold together for reproducible inference
4. **Document hyperparameters** (search space and best params in `optimised_results.csv`)
5. **Validate on the held-out test set** once, and use CV for model selection during tuning
6. **Use appropriate metrics** for imbalanced data (ROC-AUC, recall/F1 on default at operating threshold)
7. **Exclude leaky signals** such as InterestRate from features

---

## Summary

This training process:
- ✅ Loads and explores the dataset
- ✅ Engineers ratios, drops leakage features, and builds a ColumnTransformer + SMOTE + XGBoost Pipeline
- ✅ Tunes hyperparameters with RandomizedSearchCV (5-fold stratified)
- ✅ Evaluates with ROC-AUC, recall, F1, and saves an optimal probability threshold
- ✅ Persists pipeline, threshold, metrics, and importances (including optional SHAP)

**Result**: One production-ready XGBoost pipeline with explicit thresholding and risk bands.

---

*Last Updated: Training script generates this automatically*
