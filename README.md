# Customer Segmentation & Retention Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.0+-006ACC" />
  <img src="https://img.shields.io/badge/Plotly-5.15+-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white" />
</p>

<p align="center">
  <a href="https://customer-segmentation-retention-analysis-d3kucmyvxdd3lz3v5rlmt.streamlit.app/">🚀 Explore the Live Dashboard</a>
</p>

<p align="center">
  <b>An end-to-end customer analytics project that segments a real retailer's customer base, predicts churn, estimates lifetime value, and translates model outputs into concrete retention strategies — deployed as an interactive Streamlit dashboard.</b>
</p>

---

## The Problem

Every subscription and retail business faces the same questions: *Which customers are about to leave? Who are the most valuable? Where should we spend our retention budget?*

This project answers those questions using real, messy transactional data from a UK-based online retailer. The pipeline spans three ML paradigms (unsupervised clustering, classification, and regression) and produces a deployable business tool, not just model metrics.

---

## What I Built

**Phase 1 - Data Wrangling & Exploratory Analysis**

Started with ~1M raw transaction records containing missing values, duplicates, cancelled orders, and data entry errors. Built a systematic cleaning pipeline and uncovered key business patterns: extreme revenue concentration (Pareto distribution), strong seasonal trends, B2B purchasing behavior, and a large one-time buyer segment representing untapped retention potential.

A key decision here was preserving cancelled orders as behavioral signals rather than dropping them, this turned out to be one of the most interesting findings in the entire project.

**Phase 2 - Customer Segmentation**

Applied RFM (Recency, Frequency, Monetary) analysis to quantify customer behavior, then segmented customers using both rule-based scoring and K-Means clustering. Validated the segmentation with statistical tests (ANOVA, Welch's t-test, Chi-square) to confirm that segments are meaningfully different, not just artifacts of arbitrary cutoffs.

The two independent methods (rule-based and algorithmic) showed strong agreement, building confidence that the behavioral patterns are real and detectable.

**Phase 3 - Churn Prediction & Lifetime Value**

This phase involved catching and fixing a critical **data leakage** issue. The initial modeling approach produced suspiciously perfect results (AUC = 1.0), which I traced to the churn definition being encoded directly in the feature set. I restructured the entire pipeline with a temporal validation strategy, using historical behavior to predict future outcomes, producing honest, realistic results.

Multiple classification and regression models were compared using GridSearchCV with cross-validation. The final churn model achieves strong discriminative performance, and the CLV model reliably ranks customers by future value despite the inherent difficulty of predicting exact spend from transactional data alone.

**Phase 4 - Business Action Matrix & Deployment**

Combined churn risk and predicted CLV into a 2×2 action framework that tells the business exactly what to do with each customer - from high-touch personal outreach for valuable at-risk customers to automated emails for low-value departing ones. Deployed as an interactive Streamlit dashboard with customer lookup, segment exploration, and dynamic threshold adjustment.

---

## Key Results

| Finding | Detail |
|---|---|
| Revenue concentration | A small fraction of customers drives the vast majority of revenue |
| Best churn model | AUC > 0.80 with strong recall on churners (Random Forest + GridSearchCV) |
| Cancellation paradox | Customers who cancel orders are significantly *less* likely to churn — cancellation signals engagement, not dissatisfaction |
| Data leakage caught | Initial perfect scores were identified as leakage and corrected with temporal validation |
| Actionable output | Customers classified into four business actions with specific retention strategies |
| CLV prediction | Reliable customer value ranking despite limited feature set (transactional data only) |

---

## Interactive Dashboard

**[🚀 Live App](https://customer-segmentation-retention-analysis-d3kucmyvxdd3lz3v5rlmt.streamlit.app/)**

| Tab | What It Does |
|---|---|
| 🔍 Customer Lookup | Search any customer — see their segment, churn risk, predicted CLV, and recommended action |
| 👥 Segment Explorer | Compare segments side-by-side or drill into any group's distribution and customer list |
| ⚠️ Churn Risk Dashboard | Adjust thresholds with sliders — watch action buckets and at-risk revenue update in real time |
| 📈 Business Overview | Revenue trends, geographic breakdown, Pareto curve, purchase timing patterns |

---

## Project Structure

```
├── app/
│   └── streamlit_app.py               # Interactive dashboard
├── data/
│   ├── raw/                            # Original dataset (gitignored)
│   └── processed/                      # Cleaned + feature-engineered data
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── rfm_clustering.ipynb     
│   └── churn_clv.ipynb                
├── src/
│   └── utils.py                        # Helper functions
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| ML & Modeling | Scikit-learn, XGBoost, SHAP |
| Statistics | SciPy (t-tests, ANOVA, Chi-square, effect sizes) |
| Deployment | Streamlit, Streamlit Cloud |

---

## Dataset

[Online Retail II — UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) · Real transactions from a UK-based retailer · ~1M rows · Dec 2009 – Dec 2011 · 43 countries

---

<p align="center">
  Built by <b>Siddharth Patondikar</b>
</p>