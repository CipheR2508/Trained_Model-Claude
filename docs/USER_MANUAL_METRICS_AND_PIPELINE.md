# Users’s Manual: Metrics, Result Files, and How Training vs Prediction Work

This guide explains—in plain language—why **F1** and **ROC-AUC** matter, what each **results** file is for, and what happens when you run **`train_models.py`** versus **`predict.py`**.

---

## 1. Why we need more than “accuracy”

In this project the target is **Default** (1 = defaulted, 0 = did not default). The data is **imbalanced**: most loans do not default.

- If ~88% of rows are “no default,” a silly model that always predicts “no default” could still get **~88% accuracy** while **missing almost every real default**.
- So we care about how well the model finds the **rare positive class** (defaults) and how trustworthy its **ranking** of risk is—not only overall correctness.

That is why this project reports **ROC-AUC**, **recall on the default class**, and **F1 on the default class**, not only accuracy.

---

## 2. ROC-AUC (Area Under the ROC Curve)

### What it measures (intuition)

- The model outputs a **probability of default** between 0 and 1 (not just yes/no).
- You can turn that into yes/no by choosing a **threshold** (e.g. “predict default if probability ≥ 0.5”).
- **ROC-AUC** answers: *If we vary the threshold in every possible way, how good is the model at **ranking** defaulters higher than non-defaulters?*

### How to read the number

- **1.0** = perfect ranking (ideal; rare in practice).
- **0.5** = no better than random guessing for ranking.
- **Higher is better.**

### Why it is useful here

- It does **not** depend on one fixed threshold.
- For **imbalanced** problems it is usually more informative than raw accuracy.

**In `optimised_results.csv`, `roc_auc`** is computed on the **held-out test set** using the model’s **predicted probabilities** (not the final yes/no at the tuned threshold).

---

## 3. F1 score (especially for the “default” class)

### Building blocks (very short)

- **Precision** (for defaults): Of all loans we **flagged** as default, what fraction **actually** defaulted?  
  *High precision* → fewer false alarms.
- **Recall** (for defaults): Of all **true** defaults, what fraction did we **catch**?  
  *High recall* → fewer missed defaulters.

Precision and recall often trade off: catching more defaults can increase false alarms, and vice versa.

### What F1 is

**F1** is a single number that **balances** precision and recall (harmonic mean). It is useful when you care about **both** false alarms and misses, not only one.

**In this project, `f1_default` in `optimised_results.csv`** is the F1 score for the **positive class (Default = 1)** on the test set, using predictions made with the **optimal threshold** (see below)—not the default 0.5 cutoff.

---

## 4. How this ties to `optimal_threshold`

The model outputs a **probability**. To get a **binary** prediction (0/1), you compare the probability to a **threshold**:

- If **probability ≥ threshold** → predict **default (1)**  
- Else → predict **no default (0)**

**0.5** is a common default, but it is not always best when classes are imbalanced or when business costs are asymmetric.

**Training script behaviour:** After tuning, `train_models.py` searches for a threshold that **maximises F1** on the test probabilities (via the precision–recall curve). That value is saved as **`optimised_xgb_threshold.pkl`** and recorded as **`optimal_threshold`** in **`optimised_results.csv`**.

So:

- **`roc_auc`** = quality of **probabilities / ranking** (threshold-free).
- **`recall_default`** and **`f1_default`** = quality at the **chosen operating point** (threshold **0.6387** in your saved run).

---

## 5. File: `results/optimised_results.csv`

### What this file is

A **one-row summary** of the **final evaluation** and **cross-validation** for the **best** model found during training. It is produced at the **end** of `train_models.py`.

### Column header row + data row

| Column | What it means |
|--------|----------------|
| **`roc_auc`** | Test-set ROC-AUC using predicted **probabilities** of default. Measures overall **ranking** quality. |
| **`optimal_threshold`** | Probability cutoff chosen to optimise **F1** on the test set (via precision–recall). Example: **0.6387** means “predict default only if P(default) ≥ 63.87%.” |
| **`recall_default`** | **Recall for class 1 (default)** on the test set when using **`optimal_threshold`** for yes/no. “What fraction of true defaults did we catch?” |
| **`f1_default`** | **F1 score for class 1 (default)** on the test set at **`optimal_threshold`**. |
| **`cv_mean_auc`** | Mean **ROC-AUC** across **5-fold cross-validation** on the **training** data, for the **best hyperparameter** set. Roughly “how good was this config on unseen folds while tuning?” |
| **`cv_std_auc`** | Standard deviation of those CV scores. **Small std** (e.g. **0.0025**) suggests the model’s performance is **stable** across folds—not wildly dependent on one lucky split. |
| **`best_params`** | The **XGBoost hyperparameters** RandomizedSearchCV chose (nested under the `classifier__` prefix because they belong to the classifier step inside the pipeline). |

### What “each row” signifies

- **Row 1:** column names.  
- **Row 2:** the **single summary** for the trained pipeline after the last run of `train_models.py`.  
If you retrain, this row **changes** to reflect the new test metrics and best parameters.

---

## 6. File: `results/optimised_xgb_feature_importance.csv`

### What this file is

A table of **XGBoost built-in feature importances** using **gain** (how much each feature contributes to improving the model inside the trees, in aggregate).

### Columns

| Column | Meaning |
|--------|--------|
| **`feature`** | Name of one input column **after preprocessing**. Categorical columns become **multiple** rows (e.g. `HasMortgage_Yes`, `Education_Bachelor's`) because of **one-hot encoding**. |
| **`importance`** | Relative importance from the tree model (sums are not guaranteed to be 1; values are comparable **within** this file). |

### How to read it

- **Higher `importance`** → the model relied on that encoded feature more **often / more strongly** in splits (by gain).
- Rows are **sorted** from most to least important in the saved file.

### Limitation (important for beginners)

This answers: **“What did the model use internally?”**  
It does **not** fully answer: **“Why did this *one applicant* get this score?”** For that, see **SHAP** and `predict.py`.

---

## 7. File: `results/shap_feature_importance.csv`

### What this file is

A **global** SHAP summary: for each preprocessed feature, the **average absolute SHAP value** across a **random sample** of test rows (see `train_models.py`).

### Columns

| Column | Meaning |
|--------|--------|
| **First column** (often unnamed in CSV; values like `Age`, `HasDependents_No`) | **Feature name** after preprocessing (same naming idea as in the XGB importance file). |
| **`mean_abs_shap`** | Average **|**SHAP**|** over the sample: roughly “how much this feature tends to **move** the prediction, in either direction.” |

### XGB gain importances vs SHAP (simple distinction)

| | **`optimised_xgb_feature_importance.csv`** | **`shap_feature_importance.csv`** |
|---|-------------------------------------------|-------------------------------------|
| **Comes from** | XGBoost tree **gain** importances | SHAP **TreeExplainer** attributions |
| **Best for** | “Which features drive splits in the boosted trees?” | “Which features **change** the output most on average?” |
| **Ordering** | Can differ from SHAP | Often more aligned with **interpretability** for many users |

Neither file proves **causality** in the real world—they explain the **model**, not the economy.

---

## 8. What happens when you run `scripts/train_models.py`

**`train_models.py` trains the model** and writes **`models/`** and **`results/`**. You run it when you want to **fit** (or **re-fit**) the pipeline on data.

Below is the **logical order** of what the script does:

1. **Load data** from `data/raw/Loan_default.csv`.
2. **Engineer features** (same formulas as inference): `Loan_to_Income`, `EMI_to_Income`, `Credit_per_Line`, with safe handling for bad values.
3. **Drop leakage / ID columns**: e.g. **LoanID**, **InterestRate**, and temporary helper columns used only for engineering.
4. **Split** data **80% train / 20% test**, **stratified** on `Default` (same default rate in both sets).
5. **Compute `scale_pos_weight`** from training counts (helps XGBoost with imbalance).
6. **Build a pipeline**:  
   - **Preprocessor**: `ColumnTransformer` + **OneHotEncoder** on categorical columns; numbers pass through.  
   - **SMOTE** (if `imbalanced-learn` is installed): oversample minority class **inside** training only.  
   - **XGBoost** classifier.
7. **RandomizedSearchCV**: try many random hyperparameter combinations; **5-fold stratified CV**; optimise **ROC-AUC**; pick the best pipeline configuration.
8. **Fit** the search on **training** data; keep **`best_estimator_`** as the final pipeline.
9. **Threshold tuning** on the **test** set probabilities: scan thresholds to improve **F1** for the default class; save threshold to `optimised_xgb_threshold.pkl`.
10. **Evaluate** on test: ROC-AUC, recall, F1 at optimal threshold; print confusion matrix.
11. **Export** gain-based importances → `optimised_xgb_feature_importance.csv`.
12. **Optional SHAP**: if `shap` is installed, compute mean |SHAP| → `shap_feature_importance.csv`.
13. **Save** fitted pipeline → `optimised_xgb_pipeline.pkl`, summary → `optimised_results.csv`.

**Requirements:** see the top of `train_models.py` (e.g. `xgboost`, `scikit-learn`, `pandas`, `imbalanced-learn`, `shap` for SHAP).

**How to run:** from the `scripts` folder, `python train_models.py` (paths assume `../data`, `../models`, `../results` relative to `scripts/`).

---

## 9. What happens when you use `scripts/predict.py`

**`predict.py` does not train.** It **loads** artefacts produced by training:

- `models/optimised_xgb_pipeline.pkl`
- `models/optimised_xgb_threshold.pkl`

### `LoanRiskPredictor` (typical flow)

1. **On init:** load pipeline and threshold; print the threshold.
2. **For each prediction:**
   - Build a DataFrame from your **raw** input (dict or table).
   - Run **`_engineer_features`** so columns match training (drops **InterestRate** / **LoanID** if present, adds engineered fields).
   - Call **`pipeline.predict_proba`** → probability of default.
   - Compare probability to **`self.threshold`** → binary **prediction** (0/1).
   - Map probability to **risk band**: Low / Medium / High (fixed cutoffs in code: 0.20 and 0.50).
   - Optionally compute **per-applicant SHAP** (if `shap` is installed) for **`top_features`**.

3. **`predict_batch`:** same logic for many rows; adds columns `default_probability`, `prediction`, `risk_category`.

**If you see “pipeline not found,”** run `train_models.py` first so `models/optimised_xgb_pipeline.pkl` exists.

---

## 10. Quick reference table

| Question | Where to look |
|----------|----------------|
| Overall ranking quality (threshold-free) | `optimised_results.csv` → **`roc_auc`** |
| Stability during tuning | **`cv_mean_auc`**, **`cv_std_auc`** |
| Operating point for yes/no | **`optimal_threshold`** + `optimised_xgb_threshold.pkl` |
| Default-class performance at that point | **`recall_default`**, **`f1_default`** |
| Which features the tree model weighted (gain) | `optimised_xgb_feature_importance.csv` |
| Global SHAP-based influence | `shap_feature_importance.csv` |
| Why one applicant scored as they did | `LoanRiskPredictor.predict()` → **`top_features`** (SHAP) |

---

*This manual describes the intended behaviour of the scripts and result files in this repository. After retraining, numeric values in CSVs will change but the meaning of each column stays the same.*
