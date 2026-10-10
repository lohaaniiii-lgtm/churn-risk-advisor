# Churn Risk Advisor

[![Live app](https://img.shields.io/badge/Live%20app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://churn-risk-advisor-luqmankhanlohani.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-green)
![Version](https://img.shields.io/badge/Version-1.0-blue)

A web app that ranks telecom customers by churn risk, shows what could lower that risk, and scores whole customer files in one click. Built as Project 1 of **Introduction to Applied AI** (Weeks 1 to 4).

**Live app:** https://churn-risk-advisor-luqmankhanlohani.streamlit.app/

## Contents
[The problem](#the-problem) | [What the app does](#what-the-app-does) | [Screenshots](#screenshots) | [Results](#results) | [The bug I fixed](#the-bug-i-fixed-training-serving-skew) | [How it works](#how-it-works) | [Model card](#model-card-churn-risk-advisor-v10) | [Known limitations](#known-limitations) | [Files](#files) | [Run locally](#run-locally) | [Notebooks](#notebooks)

## The problem
Losing a customer costs far more than trying to keep one. In this project a missed churner (false negative) is valued at **PKR 6,000** and an unnecessary retention offer (false positive) at **PKR 1,000**, so the decision threshold is chosen to minimise total cost, not to maximise accuracy. The app lets a retention team see who to call first.

## What the app does
- **One customer:** churn probability, risk band and recommended action, with an adjustable contact threshold.
- **Risk bands:** HIGH (probability at or above the threshold, "contact now"), WATCH (at or above half the threshold), LOW (below that).
- **What-if analysis:** shows how the probability changes if the contract, payment method or tech support changes. Labelled as associations, not guaranteed effects.
- **Batch scoring:** upload a CSV with the original Telco columns, see how many customers are above the threshold, and download the scored file. A missing column produces a clear error instead of a crash.
- **About tab:** model metadata and limitations.

## Screenshots
One customer:

![One customer](screenshot_customer.png)

Batch scoring:

![Batch scoring](screenshot_batch.png)

## Results
5-fold stratified cross-validation on the training set, reported as mean +/- standard deviation:

| Model | CV AUC |
|---|---|
| Logistic Regression (tuned C) | 0.846 +/- 0.013 |
| Random Forest (random search) | 0.844 +/- 0.011 |
| **XGBoost (random search)** | **0.850 +/- 0.012** |

The three models are statistically tied, since the gaps are smaller than one standard deviation. XGBoost was chosen by CV, and its **test AUC is 0.848** (the test set was used exactly once). The same model scored between 78.0% and 82.8% accuracy just by changing the random split, which is why every number here comes from cross-validation.

## The bug I fixed: training-serving skew
Scoring a single customer with `pd.get_dummies(drop_first=True)` drops the only category present, so both Contract columns become 0 and the model reads the customer as Month-to-month. The same Two-year-contract customer scored **0.136 instead of the correct 0.012**, and no error was raised.

**Fix:** a separate serving encoder (`get_dummies` without `drop_first`, reindexed to the exact 30 training columns).
**Proof:** a parity test compares notebook predictions with the app's encoder on all 7,043 customers. The largest difference is **0.0**.

## How it works
1. `prepare_input` encodes raw customer fields into exactly the 30 columns the model was trained on.
2. `TotalCharges` is derived as tenure x monthly charges, so the user cannot enter inconsistent values.
3. The model is loaded once per server with `st.cache_resource`.
4. The threshold, library versions, feature columns and metrics are read from `model_meta.json`, so the app and the notebook cannot drift apart.
5. The model is stored in XGBoost's native JSON format. A pickle saved on Linux (Kaggle) failed to load on Windows ("input stream corrupted"), and the JSON file loads on both with identical predictions.

## Model card: Churn Risk Advisor v1.0
**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBoost (XGBClassifier), chosen by 5-fold CV. Features: 30 encoded columns.

**Performance:** CV AUC 0.850 +/- 0.012; test AUC 0.848 (test set used once).

**Threshold:** 0.20, from a cost analysis on out-of-fold training predictions (FN = PKR 6,000, FP = PKR 1,000).

**Limitations:** one US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change.

**Fairness check (Week 15):** compare recall across gender and seniors.

**Owner and version:** Muhammad Luqman Khan, v1.0, October 10, 2026.

## Known limitations
- **Unseen categories:** a category the model never saw (for example a misspelled Contract value in a batch file) gets all-zero dummy columns and is scored as the baseline category without an error. Validating allowed values per column would fix this.
- **What-if is not causal:** a Two-year contract lowers the score for a high-risk customer, but customers who choose longer contracts may already be more loyal. A randomized A/B test would be needed to know whether offering one retains the customer.
- **No per-customer explanation:** the "Why this score?" panel is available for Logistic Regression only, and the final model is XGBoost.
- **Threshold:** tuned on out-of-fold training predictions with assumed costs; a real deployment would need real cost figures.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `churn_xgb.json` | Trained XGBoost model (native format) |
| `model_meta.json` | Feature columns, threshold, library versions, metrics |
| `requirements.txt` | Pinned dependencies (scikit-learn 1.6.1, xgboost 3.4.1) |
| `sample_customers.csv` | 50 customers for the batch demo |

## Run locally
Requires Python 3.12 or newer (xgboost 3.4.1 does not support older versions).
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
On Mac or Linux, activate with `source .venv/bin/activate`.

## Notebooks
Weeks 1 to 3 (EDA, ML models, optimization and segmentation): https://github.com/lohaaniiii-lgtm/applied-ai-project1-churn

## Author
Muhammad Luqman Khan | https://github.com/lohaaniiii-lgtm
