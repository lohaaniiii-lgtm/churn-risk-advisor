# Churn Risk Advisor

[![Live app](https://img.shields.io/badge/Live%20app-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://churn-risk-advisor-luqmankhanlohani.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-green)
![Version](https://img.shields.io/badge/Version-1.0-blue)

A **Streamlit web app** that ranks telecom customers by **churn risk**, shows **what-if** changes that could lower it, and scores whole customer files in **batch**. It is **Project 1** of *Introduction to Applied AI* (Weeks 1 to 4), and the step where the Week 3 model becomes a product.

**Live app:** https://churn-risk-advisor-luqmankhanlohani.streamlit.app/

## The problem
Losing a customer costs more than trying to keep one. In this project a **missed churner (false negative) costs PKR 6,000** and an **unnecessary retention offer (false positive) costs PKR 1,000**. The decision **threshold** is therefore chosen to **minimise total cost**, not to maximise accuracy. The app tells a retention team **who to call first**.

## What the app does
- **One customer:** **churn probability**, **risk band** and **recommended action**, with an adjustable contact **threshold**.
- **Risk bands:** **HIGH** (probability at or above the threshold, action "Contact now"), **WATCH** (at or above half the threshold) and **LOW** (below that).
- **What-if analysis:** shows how the probability changes if the **contract**, **payment method** or **tech support** changes. It is labelled as an **association, not a guaranteed effect**.
- **Batch scoring:** upload a CSV with the original Telco columns, see **how many customers are above the threshold**, and **download** the scored file. A **missing column** gives a clear error instead of a crash.
- **About tab:** model metadata and limitations.

## Model card: Churn Risk Advisor v1.0
**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. **Decision support, not automation.**

**Not for:** **credit, pricing, or any decision that denies a service.**

**Data:** IBM Telco Customer Churn, **7,043 customers**, **26.5% churn**.

**Model:** **XGBoost** (XGBClassifier), chosen by **5-fold cross-validation**. Features: **30 encoded columns**.

**Performance:** **CV AUC 0.850 +/- 0.012**; **test AUC 0.848** (test set used **once**).

**Threshold:** **0.20**, from a cost analysis on out-of-fold training predictions (**FN = PKR 6,000, FP = PKR 1,000**).

**Limitations:** one US dataset from one period; no Pakistan data; **associations, not causes**; performance may **drift** as plans change.

**Fairness check (Week 15):** compare **recall** across **gender** and **seniors**.

**Owner and version:** Muhammad Luqman Khan, **v1.0**, October 10, 2026.

## Live app
https://churn-risk-advisor-luqmankhanlohani.streamlit.app/

## Screenshots
**One customer:**

![One customer](screenshot_customer.png)

**Batch scoring:**

![Batch scoring](screenshot_batch.png)

## Results
**5-fold stratified cross-validation** on the training set, reported as **mean +/- standard deviation**:

| Model | CV AUC |
|---|---|
| Logistic Regression (tuned C) | 0.846 +/- 0.013 |
| Random Forest (random search) | 0.844 +/- 0.011 |
| **XGBoost (random search)** | **0.850 +/- 0.012** |

The three models are **statistically tied**, because the gaps are smaller than one standard deviation. XGBoost was chosen by CV, and its **test AUC is 0.848**. The same model scored between **78.0% and 82.8% accuracy** just by changing the random split, which is why every number here comes from **cross-validation** and not from a single split.

## The bug I fixed: training-serving skew
Scoring **one customer** with `pd.get_dummies(drop_first=True)` drops the only category present, so **both Contract columns become 0** and the model reads the customer as **Month-to-month**. The same **Two-year** contract customer scored **0.136 instead of the correct 0.012**, and **no error was raised**.

- **Fix:** a separate **serving encoder** (`get_dummies` without `drop_first`, reindexed to the exact 30 training columns).
- **Proof:** a **parity test** compares notebook predictions with the app's encoder on **all 7,043 customers**. The **largest difference is 0.0**.

## What-if example
For the **riskiest customer** in the data (baseline **P(churn) = 0.935**):

| Change | New probability | Difference |
|---|---|---|
| Contract: One year | 0.862 | -0.073 |
| Contract: Two year | 0.765 | -0.170 |
| Payment: Credit card (automatic) | 0.913 | -0.022 |

A **Two-year contract lowers risk the most**, but the customer stays **far above the 0.20 threshold**. These are **associations**, not proof that the change would make the customer stay.

## Customer segments (from Week 3)
**K-means** (k = 4) on tenure, monthly charges, total charges and number of services found these segments, and **churn was used only to profile them**:

| Segment | Churn rate |
|---|---|
| Mid-tenure, high spend (**at risk**) | **43%** |
| New, low spend | 32% |
| Loyal, high spend, bundled | 14% |
| Long-tenure, low spend | 5% |

## How it works
1. `prepare_input` encodes raw customer fields into **exactly the 30 columns** the model was trained on.
2. **TotalCharges is derived** as tenure x monthly charges, so the user cannot enter inconsistent values.
3. The model is loaded **once per server** with `st.cache_resource`.
4. The **threshold, library versions, feature columns and metrics** are read from `model_meta.json`, so the app and the notebook cannot drift apart.
5. The model is saved in **XGBoost's native JSON format**. A pickle saved on Linux (Kaggle) failed to load on Windows (`input stream corrupted`), and the JSON file loads everywhere with **identical predictions** (largest difference 0.0).

## Known limitations
- **Unseen categories:** a value the model never saw (for example a misspelled Contract in a batch file) gets **all-zero dummy columns** and is scored as the **baseline category with no error**. Validating the allowed values of each column would fix this.
- **What-if is not causal:** customers who choose longer contracts may already be more loyal. A **randomized A/B test** would be needed to know whether offering one retains the customer.
- **No per-customer explanation:** the "Why this score?" panel exists for **Logistic Regression only**, and the final model is XGBoost.
- **Threshold:** tuned with **assumed** costs; a real deployment needs real cost figures.

## Run locally
Requires **Python 3.12 or newer** (xgboost 3.4.1 does not support Python 3.11).

```
python -m venv .venv
.venv\Scripts\activate          # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `churn_xgb.json` | Trained XGBoost model (native format) |
| `model_meta.json` | Feature columns, threshold, library versions, metrics |
| `requirements.txt` | **Pinned** dependencies (scikit-learn 1.6.1, xgboost 3.4.1) |
| `sample_customers.csv` | 50 customers for the batch demo |

## Notebooks
**Weeks 1 to 3** (EDA, ML models, optimization and segmentation): https://github.com/lohaaniiii-lgtm/applied-ai-project1-churn

## Check Your Understanding

**1. Why does `get_dummies(drop_first=True)` break predictions when scoring a single customer?**
With **one row**, only one category of each feature is present, so it is treated as the **first category and dropped**, and all its dummy columns become **0**. The model reads that as the **baseline** (Month-to-month for Contract) and **no error is raised**. The same Two-year customer scored **0.136 instead of 0.012**.

**2. What does the parity test compare, and why must it cover every training row rather than a few?**
It compares the model's predictions on the **training-time features** with its predictions on features built by the **app's serving encoder**. It must cover **all 7,043 rows** because skew can affect **only some categories or rare combinations**, and a few rows could miss it. My **largest difference was 0.0**.

**3. Should the model be loaded with `st.cache_resource` or `st.cache_data`? Why?**
**`st.cache_resource`**, because the model is an **object** that should be loaded **once** and shared. `st.cache_data` **copies** its return value on every call, which is slow and wrong for a model.

**4. Why does the app derive TotalCharges instead of asking the user to type it?**
TotalCharges is **tenure x monthly charges**, so asking for it invites **inconsistent inputs**, for example a total smaller than one month's charge. Deriving it keeps the input **valid**.

**5. Why must requirements.txt pin the exact scikit-learn version used in training?**
A saved model depends on the library's **internals**, so another version can **fail to load** or **silently change predictions**. Pinning the training version (**scikit-learn 1.6.1, xgboost 3.4.1**) makes the server match the training environment.

**6. The what-if shows a two-year contract lowers risk from 0.79 to 0.27. Would offering one make the customer stay? How could a company find out?**
**Not necessarily:** the model learned an **association**, and customers who choose two-year contracts may already be more loyal. A company could run a **randomized A/B test**, offering the contract to a random half of at-risk customers and comparing churn between the two groups.

## Author
Muhammad Luqman Khan | https://github.com/lohaaniiii-lgtm
