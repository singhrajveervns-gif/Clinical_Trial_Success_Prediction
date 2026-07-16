# Clinical Trial Success Prediction for Pharmaceutical Decision Support

A healthcare / pharmaceutical analytics machine learning project predicting whether a
clinical trial is likely to reach operational completion, using only information
available at or near the trial's design stage.

## Project Overview

This project builds an end-to-end classical ML pipeline — from raw ClinicalTrials.gov
data through cleaning, feature engineering, model tuning, explainability, and business
recommendations — to support portfolio-risk decisions in pharmaceutical R&D and CRO
operations. It is designed to run entirely on Google Colab Free, using only classical
ML (no deep learning, no heavy NLP) to keep it accessible and fast to reproduce.

## Business Motivation

Clinical trials are the largest cost driver in pharmaceutical R&D, and a substantial
share of registered trials terminate before completion — for funding, recruitment, or
strategic reasons. Every trial that fails after resources are already committed
represents sunk cost, delayed patient benefit, and portfolio risk. This project
explores whether design-time trial characteristics carry enough signal to flag at-risk
trials early enough for portfolio and operations teams to act on.

## Dataset

Source: [ClinicalTrials.gov Clinical Trials Dataset (Kaggle)](https://www.kaggle.com/datasets/danielansted/clinicaltrials-gov-clinical-trials-dataset)

The target (`trial_success`) is an **operational** definition — `Completed` = success;
`Terminated` / `Withdrawn` / `Suspended` = failure; unresolved and Expanded Access
Program statuses are excluded. This is explicitly *not* a clinical efficacy label — see
the notebook's Assumptions & Limitations section for the full reasoning.

## Methodology

1. **Dataset Inspection** — schema verified programmatically rather than assumed
2. **Data Cleaning & Target Engineering** — case-normalized status mapping, leakage
   audit, duplicate handling
3. **Feature Engineering** — 13 features built strictly from what the data supports
   (condition/intervention counts, eligibility flags, sponsor type, complexity index)
4. **Exploratory Data Analysis** — business-question-driven, hypothesis-testing EDA
5. **Model Building** — Logistic Regression, Decision Tree, Random Forest, XGBoost
   compared under one shared preprocessing pipeline
6. **Hyperparameter Tuning & Validation** — `RandomizedSearchCV`, calibration analysis,
   decision threshold analysis
7. **Explainability** — SHAP global and local explanations, translated into business
   language and checked against domain hypotheses
8. **Business Insights** — a validated risk-tiering workflow and stakeholder-specific
   recommendations

## Key Results

See the notebook's Phase 5A/5B comparison tables for full metrics. Model selection was
based on `PR-AUC (Failure=0)` and calibration quality — not raw accuracy — since
accuracy is dominated by the ~86% majority (success) class in this dataset.

## Explainability

SHAP (`TreeExplainer`) was used to identify the design-time features most associated
with predicted success/failure, with global summary/dependence plots and three local
explanations (high-confidence success, high-confidence failure risk, and a borderline
case). Findings were explicitly checked against earlier EDA hypotheses, with any
contradictions documented rather than smoothed over.

## Business Recommendations

A three-tier risk workflow (High / Medium / Low, based on predicted success
probability) is proposed for portfolio review prioritization, validated against actual
held-out completion rates. See the notebook's Phase 7 for the full stakeholder-specific
recommendations and an explicit statement of what this analysis does *not* claim
(no causality, no clinical efficacy claim, no measured financial ROI).

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn — all run on
Google Colab Free (CPU only, no paid GPU required).

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. Open `notebooks/Clinical_Trial_Success_Prediction.ipynb` in Google Colab
2. Upload the dataset to Google Drive at:
   `/content/drive/MyDrive/Clinical_Trial_Success_Prediction/Data/clin_trials.csv`
3. Run all cells top to bottom — no manual edits required (see the Reproducibility
   Checklist in the notebook's Phase 8 for details)

## Repository Structure

See `Phase 8.2` in the notebook, or the tree in this repository's root.

## Future Improvements

- Modularize notebook logic into a reusable `src/` package
- Re-run against a richer ClinicalTrials.gov export including enrollment, location,
  and completion-date fields to build the duration/enrollment/site-count features
  originally scoped but unavailable in this export
- Periodic re-validation of feature importance and risk-tier boundaries as new trial
  data accumulates
- Formal SMOTE vs. class-weighting comparison via cross-validation (deferred per
  the project's original phased plan)

## Author

Rajveer Singh
