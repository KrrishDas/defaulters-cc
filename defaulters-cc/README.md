# Credit Card Delinquency – Exploratory Analysis & Logistic Regression

This repository contains an end-to-end workflow using Federal Reserve Economic Data (FRED) series to model US credit card delinquency (DRCCLACBS) as a binary outcome and estimate default risk probabilities.

---

## Project Goals
- Explore macroeconomic relationships with the delinquency rate (plots, correlations, descriptive statistics).
- Build a reproducible binary classifier (Logistic Regression) to predict “high delinquency” periods.
- Generate probabilities (risk scores), not just hard classifications.
- Keep the workflow modular and version-controlled.

---

## Repository Structure

defaulters-cc/
├─ data/
│  ├─ raw/                 # Original CSVs from FRED
│  └─ processed/
│     └─ .gitkeep
├─ models/                 # Saved models and evaluation metrics
│  └─ .gitkeep
├─ reports/
│  └─ figures/             # Exported plots
│     └─ .gitkeep
├─ src/
│  ├─ python/
│  │  ├─ config.py
│  │  ├─ data_prep.py
│  │  ├─ plots.py
│  │  ├─ train.py
│  │  └─ DefaultersLogisticRegression.py  # optional entry script
│  └─ r/
│     └─ DefaultersOnCreditCards.Rmd      # exploratory R Markdown
├─ .gitignore
├─ requirements.txt
└─ README.md

---

## Setup

1) Create and activate a virtual environment
'''bash'''
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

2) Install dependencies
'''bash'''
pip install -r requirements.txt

---

## Data
	•	Target: DRCCLACBS – Delinquency Rate on Credit Card Loans (all commercial banks).
	•	Predictors: Macro series such as unemployment rate (UNRATE), consumer price index (CPIAUCSL), real money supply (M2REAL), federal funds rate (DFF), and others, all from FRED.

Place CSV files in data/raw/. Expected columns:
	•	observation_date (or DATE)
	•	<SERIES_NAME> (e.g., UNRATE, M2REAL)

The pipeline aligns series to a quarterly index, handling different frequencies (monthly, weekly, quarterly).

---

## Reproducing the Pipeline

1) Create target variable

Inside train.py, the binary label is created by thresholding the target:
	•	DefaultFlag = 1 if delinquency is above the median (or another cutoff).
	•	DefaultFlag = 0 otherwise.

This behavior is configurable in src/python/config.py.

2) Train and evaluate

The script:
	•	Loads all CSVs in data/raw/
	•	Merges and aligns time indices
	•	Splits data into training and test sets (stratified)
	•	Builds a Pipeline with:
	•	SimpleImputer(strategy="median")
	•	StandardScaler()
	•	LogisticRegression(penalty="l2", C=1.0, max_iter=1000, class_weight="balanced")
	•	Saves:
	•	models/metrics.json
	•	reports/figures/roc_train_test.png
	•	reports/figures/pr_train_test.png
	•	reports/figures/confusion_matrix.png

---

## Outputs
	•	ROC–AUC: Area under the ROC curve (0.5 = random, 1.0 = perfect).
	•	Precision–Recall: More meaningful with imbalanced classes.
	•	Confusion Matrix: True/false positives and negatives at the default 0.5 cutoff.
	•	Coefficients: Magnitude and sign show feature influence (after scaling).

---

## Configuration (src/python/config.py)

Important parameters:
	•	THRESHOLD_STRATEGY: "median" (default) or numeric cutoff.
	•	TEST_SIZE: Fraction of data used for test (default 0.2).
	•	RANDOM_STATE: Ensures reproducibility.
	•	MODEL_KWARGS: Logistic Regression parameters (penalty, C, solver).
	•	Paths for saving metrics and plots.

---

## Exploratory Analysis (R)

src/r/DefaultersOnCreditCards.Rmd contains exploratory plots and narrative (CPI, GDP, rates, sentiment, savings, etc.).
You can render it with RStudio. 

---

## Methodology Notes
	•	Pipeline guarantees preprocessing steps (impute + scale) are consistently applied at training and inference, preventing data leakage.
	•	StandardScaler normalizes features to z-scores, so all variables are comparable.
	•	Stratified split ensures train/test have similar class balance.
	•	Class weights mitigate imbalance when defaults are rare.
	•	Performance should be tracked over time since macro relationships evolve.

---

## Housekeeping
	•	Large datasets live in data/raw/ (not tracked if very large).
	•	Generated artifacts live in data/processed/, models/, and reports/figures/. .gitkeep keeps empty folders in version control.
	•	Virtual environments are excluded via .gitignore.

---

## Contact

Maintainer: Krrish Das
Open issues or submit pull requests if you have suggestions.