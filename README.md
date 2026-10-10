# Churn Risk Advisor

A Streamlit app that scores telecom customers by churn risk and shows what could lower it. Project 1, Introduction to Applied AI (Week 4).

**Live app:** https://churn-risk-advisor-luqmankhanlohani.streamlit.app/

## What the app does
- **One customer:** churn probability, risk band (HIGH / WATCH / LOW) and action, with an adjustable contact threshold.
- **What-if:** how the probability changes if the contract, payment method or tech support changes (associations, not guaranteed effects).
- **Batch scoring:** upload a CSV with the original Telco columns, get scores, count above the threshold, and download the result. A missing column gives a clear error.
- **About tab:** model metadata and limitations.

## Screenshots
One customer:

![One customer](screenshot_customer.png)

Batch scoring:

![Batch scoring](screenshot_batch.png)

## Results
| Model | 5-fold CV AUC |
|---|---|
| Logistic Regression | 0.846 +/- 0.013 |
| Random Forest | 0.844 +/- 0.011 |
| XGBoost (final) | 0.850 +/- 0.012 |

The three models are statistically tied. XGBoost was chosen by CV; its test AUC is 0.848 (test set used once).

## Training-serving skew bug (fixed)
Scoring one customer with `get_dummies(drop_first=True)` drops the only Contract category present, so both Contract columns become 0 and the model reads Month-to-month. The same Two-year customer scored 0.136 instead of the correct 0.012, with no error raised. The app uses a separate serving encoder (`get_dummies` without `drop_first`, reindexed to the training columns). Parity test: largest difference between notebook and app predictions is 0.0 across all 7,043 customers.

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

## Files
- `app.py`: the Streamlit app
- `churn_xgb.json`: trained XGBoost model (XGBoost native format)
- `model_meta.json`: feature columns, threshold, library versions, metrics
- `requirements.txt`: pinned dependencies (scikit-learn 1.6.1, xgboost 3.4.1)
- `sample_customers.csv`: 50 customers for the batch demo

## Run locally
Requires Python 3.12 or newer (xgboost 3.4.1 does not support older versions).
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notebooks
Week 1 to 3 notebooks: https://github.com/lohaaniiii-lgtm/applied-ai-project1-churn
