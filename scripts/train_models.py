"""
FinRisk AI — Optimised XGBoost Loan Default Prediction
=======================================================
Improvements over baseline:
  - Feature engineering (Loan_to_Income, EMI_to_Income, Credit_per_Line)
  - InterestRate dropped (post-decision leakage signal)
  - OneHotEncoder for all nominal categoricals (no false ordinal signals)
  - scale_pos_weight for class imbalance + optional SMOTE
  - RandomizedSearchCV (5-fold stratified CV) for hyperparameter tuning
  - Optimal decision threshold via Precision-Recall curve (F1 maximisation)
  - Full sklearn Pipeline saved for leakage-free inference
  - SHAP explainability (if shap is installed)
  - Risk category output: Low / Medium / High

Requirements (install before running):
  pip install xgboost scikit-learn pandas numpy imbalanced-learn shap joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score, recall_score
)

import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "../data/raw/Loan_default.csv"
OUTPUT_DIR  = "../models"
RESULTS_DIR = "../results"
USE_SMOTE   = True          # set False to rely purely on scale_pos_weight
N_ITER_SEARCH = 30          # number of RandomizedSearchCV iterations (increase for better tuning)
CV_FOLDS    = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("FinRisk AI — Optimised XGBoost Training Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Target distribution:\n{df['Default'].value_counts(normalize=True).mul(100).round(2).to_string()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (before any split)
# ─────────────────────────────────────────────
print("\n[2] Engineering features...")

# Guard against zero-division
df["Income_safe"]       = df["Income"].replace(0, np.nan).fillna(df["Income"].median())
df["LoanTerm_safe"]     = df["LoanTerm"].replace(0, 1)
df["NumCreditLines_safe"] = df["NumCreditLines"].replace(np.nan, 0)

# High-signal ratios
df["Loan_to_Income"]  = df["LoanAmount"] / df["Income_safe"]
df["EMI_to_Income"]   = (df["LoanAmount"] / df["LoanTerm_safe"]) / df["Income_safe"]
df["Credit_per_Line"] = df["CreditScore"] / (df["NumCreditLines_safe"] + 1)

print("    Created: Loan_to_Income, EMI_to_Income, Credit_per_Line")

# ─────────────────────────────────────────────
# 3. FEATURE SELECTION  (drop leakage + helpers)
# ─────────────────────────────────────────────
# InterestRate is a POST-risk-decision signal (bank sets rate AFTER assessing risk),
# so it would cause data leakage and inflate metrics artificially.
LEAKAGE_FEATURES = ["InterestRate", "LoanID",
                    "Income_safe", "LoanTerm_safe", "NumCreditLines_safe"]

X = df.drop(columns=["Default"] + [c for c in LEAKAGE_FEATURES if c in df.columns])
y = df["Default"]

print(f"    Dropped leakage/helper cols: {[c for c in LEAKAGE_FEATURES if c in df.columns]}")
print(f"    Final feature count: {X.shape[1]}")

# ─────────────────────────────────────────────
# 4. IDENTIFY COLUMN TYPES
# ─────────────────────────────────────────────
CATEGORICAL_COLS = X.select_dtypes(include="object").columns.tolist()
NUMERICAL_COLS   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"    Categorical: {CATEGORICAL_COLS}")
print(f"    Numerical  : {NUMERICAL_COLS}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
print("\n[3] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 6. OPTIONAL SMOTE  (training set only — no leakage)
# ─────────────────────────────────────────────
if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        print("\n[4] Applying SMOTE on training set only...")
        # We'll use imblearn Pipeline later; for now just note it's enabled.
        # SMOTE is woven into the Pipeline below if available.
        SMOTE_AVAILABLE = True
        print("    SMOTE will be applied inside Pipeline (training only).")
    except ImportError:
        SMOTE_AVAILABLE = False
        print("    WARNING: imbalanced-learn not installed. Skipping SMOTE.")
        print("    Falling back to scale_pos_weight only.")
else:
    SMOTE_AVAILABLE = False
    print("\n[4] SMOTE disabled — relying on scale_pos_weight.")

# ─────────────────────────────────────────────
# 7. CLASS WEIGHT CALCULATION
# ─────────────────────────────────────────────
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight_value = n_neg / n_pos
print(f"\n[5] scale_pos_weight = {scale_pos_weight_value:.2f}  "
      f"(neg={n_neg}, pos={n_pos})")

# ─────────────────────────────────────────────
# 8. PREPROCESSOR
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ],
    remainder="passthrough",   # numeric columns pass through unchanged (trees don't need scaling)
    verbose_feature_names_out=False,
)

# ─────────────────────────────────────────────
# 9. BUILD PIPELINE
# ─────────────────────────────────────────────
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",           # AUC-PR is better than logloss for imbalanced data
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

if SMOTE_AVAILABLE:
    from imblearn.pipeline import Pipeline as ImbPipeline
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=42, k_neighbors=5)),
        ("classifier",   xgb_clf),
    ])
else:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   xgb_clf),
    ])

# ─────────────────────────────────────────────
# 10. HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────
print(f"\n[6] RandomizedSearchCV — {N_ITER_SEARCH} iterations, {CV_FOLDS}-fold CV...")
print("    (This may take several minutes depending on hardware)")

param_dist = {
    "classifier__n_estimators":    [100, 200, 300],
    "classifier__max_depth":       [3, 5, 7],
    "classifier__learning_rate":   [0.03, 0.05, 0.1],
    "classifier__subsample":       [0.8, 1.0],
    "classifier__colsample_bytree":[0.8, 1.0],
    "classifier__gamma":           [0, 1, 5],
    "classifier__min_child_weight":[1, 5, 10],
    "classifier__reg_alpha":       [0, 0.1, 1],      # L1 regularisation
    "classifier__reg_lambda":      [1, 2, 5],         # L2 regularisation
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER_SEARCH,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,           # retrain best params on full training set
    return_train_score=True,
)

search.fit(X_train, y_train)

print(f"\n    Best CV ROC-AUC : {search.best_score_:.4f}")
print(f"    Best params     : {search.best_params_}")

best_pipeline = search.best_estimator_

# ─────────────────────────────────────────────
# 11. THRESHOLD OPTIMISATION (F1 on default class)
# ─────────────────────────────────────────────
print("\n[7] Optimising decision threshold via Precision-Recall curve...")
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print(f"    Optimal threshold : {best_threshold:.4f}  "
      f"(F1={f1_scores[best_idx]:.4f}, "
      f"P={precisions[best_idx]:.3f}, "
      f"R={recalls[best_idx]:.3f})")

# ─────────────────────────────────────────────
# 12. FINAL EVALUATION
# ─────────────────────────────────────────────
print("\n[8] Evaluating on test set...")
y_pred_opt  = (y_prob >= best_threshold).astype(int)
y_pred_def  = best_pipeline.predict(X_test)   # default 0.5 threshold

roc_auc  = roc_auc_score(y_test, y_prob)
recall_opt   = recall_score(y_test, y_pred_opt)
f1_opt       = f1_score(y_test, y_pred_opt)
recall_def   = recall_score(y_test, y_pred_def)

print(f"\n    ── At default threshold (0.50) ──")
print(classification_report(y_test, y_pred_def, target_names=["Non-Default", "Default"]))

print(f"    ── At optimal threshold ({best_threshold:.4f}) ──")
print(classification_report(y_test, y_pred_opt, target_names=["Non-Default", "Default"]))

print(f"    ROC-AUC          : {roc_auc:.4f}")
print(f"    Recall (default) : {recall_opt:.4f}  (was {recall_def:.4f} at 0.5)")
print(f"    F1 (default)     : {f1_opt:.4f}")

cm = confusion_matrix(y_test, y_pred_opt)
print(f"\n    Confusion matrix (optimal threshold):\n{cm}")
print(f"    FN (missed defaults): {cm[1,0]}  |  FP (false alarms): {cm[0,1]}")

# ─────────────────────────────────────────────
# 13. CROSS-VALIDATION SUMMARY
# ─────────────────────────────────────────────
cv_results = pd.DataFrame(search.cv_results_)
best_row   = cv_results.loc[search.best_index_]
print(f"\n[9] Cross-validation summary (best configuration):")
print(f"    Mean CV AUC : {best_row['mean_test_score']:.4f} "
      f"± {best_row['std_test_score']:.4f}")
print(f"    Mean train AUC : {best_row['mean_train_score']:.4f}  "
      f"(gap={best_row['mean_train_score']-best_row['mean_test_score']:.4f})")

# ─────────────────────────────────────────────
# 14. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[10] Feature importances (gain)...")

xgb_step = best_pipeline.named_steps["classifier"]
pre_step  = best_pipeline.named_steps["preprocessor"]
feature_names = pre_step.get_feature_names_out()

importances = xgb_step.feature_importances_
imp_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print(f"\n    Top 10 features:")
print(imp_df.head(10).to_string(index=False))

imp_df.to_csv(f"{RESULTS_DIR}/optimised_xgb_feature_importance.csv", index=False)

# ─────────────────────────────────────────────
# 15. OPTIONAL SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
try:
    import shap
    print("\n[11] Generating SHAP summary (sample of 2000 rows)...")
    X_test_transformed = pre_step.transform(X_test)
    explainer   = shap.TreeExplainer(xgb_step)
    sample_idx  = np.random.choice(len(X_test_transformed), min(2000, len(X_test_transformed)), replace=False)
    shap_values = explainer.shap_values(X_test_transformed[sample_idx])

    shap_df = pd.DataFrame(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
        columns=["mean_abs_shap"]
    ).sort_values("mean_abs_shap", ascending=False)

    print(f"\n    Top 10 by mean |SHAP|:")
    print(shap_df.head(10).to_string())
    shap_df.to_csv(f"{RESULTS_DIR}/shap_feature_importance.csv")
    print("    SHAP CSV saved.")
except ImportError:
    print("\n[11] shap not installed — skipping SHAP. Install with: pip install shap")

# ─────────────────────────────────────────────
# 16. SAVE ARTEFACTS
# ─────────────────────────────────────────────
print("\n[12] Saving pipeline and metadata...")
joblib.dump(best_pipeline,  f"{OUTPUT_DIR}/optimised_xgb_pipeline.pkl")
joblib.dump(best_threshold, f"{OUTPUT_DIR}/optimised_xgb_threshold.pkl")

# Save results summary
results_summary = {
    "roc_auc":          round(roc_auc, 4),
    "optimal_threshold":round(best_threshold, 4),
    "recall_default":   round(recall_opt, 4),
    "f1_default":       round(f1_opt, 4),
    "cv_mean_auc":      round(best_row["mean_test_score"], 4),
    "cv_std_auc":       round(best_row["std_test_score"], 4),
    "best_params":      search.best_params_,
}
pd.DataFrame([results_summary]).to_csv(f"{RESULTS_DIR}/optimised_results.csv", index=False)

print(f"\n    Saved: {OUTPUT_DIR}/optimised_xgb_pipeline.pkl")
print(f"    Saved: {OUTPUT_DIR}/optimised_xgb_threshold.pkl")
print(f"    Saved: {RESULTS_DIR}/optimised_results.csv")
print(f"    Saved: {RESULTS_DIR}/optimised_xgb_feature_importance.csv")

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print(f"  ROC-AUC          : {roc_auc:.4f}")
print(f"  Optimal threshold: {best_threshold:.4f}")
print(f"  Recall (default) : {recall_opt:.4f}")
print(f"  F1 (default)     : {f1_opt:.4f}")
