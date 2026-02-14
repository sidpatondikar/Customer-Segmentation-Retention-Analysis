# 🛒 Customer Segmentation & Retention Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

End-to-end customer analytics project analyzing **1M+ real e-commerce transactions** from a UK-based online retailer (2009–2011) to segment customers, predict churn, and estimate customer lifetime value — delivering actionable retention strategies backed by data.

## Business Problem

Every subscription and retail business cares about three things: **churn**, **lifetime value**, and **customer behavior**. This project answers the critical business questions:

- **Who are our most valuable customers?** → RFM segmentation + K-Means clustering
- **Who is about to leave?** → Churn prediction with XGBoost
- **How much is each customer worth?** → Customer Lifetime Value estimation
- **Where should we invest retention spend?** → High CLV × High churn risk = priority targets

## Key Results

| Metric | Value |
|--------|-------|
| Customer Segments Identified | _TBD_ |
| Churn Prediction AUC-ROC | _TBD_ |
| Best Model | _TBD_ |
| High-Risk High-Value Customers | _TBD_ |

## Project Structure

```
├── data/
│   ├── raw/                     # Original transaction data (gitignored)
│   └── processed/               # Cleaned + customer-level features
├── notebooks/
│   └── customer_analysis.ipynb  # Full analysis notebook
├── app/
│   └── streamlit_app.py         # Interactive customer lookup app
├── dashboards/
│   └── screenshots/             # Tableau/Power BI dashboard captures
├── models/
│   ├── churn_model.pkl          # Trained churn classifier (gitignored)
│   └── scaler.pkl               # Feature scaler
├── src/
│   ├── __init__.py
│   └── utils.py                 # Helper functions
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Wrangling & EDA
- Cleaned 1M+ transactions: handled cancellations, missing CustomerIDs, negative quantities, duplicates
- Analyzed revenue trends, seasonal patterns, top products, geographic distribution
- Interactive Plotly visualizations

### 2. RFM Analysis & Customer Segmentation
- Engineered Recency, Frequency, Monetary features from raw transactions
- Applied K-Means clustering with Elbow Method + Silhouette Score validation
- Identified customer personas: Champions, Loyal, At-Risk, Lost
- Statistical testing: ANOVA across segments, chi-square for segment distributions

### 3. Churn Prediction
- Defined churn using data-driven inactivity threshold
- Models: Logistic Regression, Random Forest, XGBoost
- Optimized for recall (missing a churning customer is costly)
- SHAP analysis for feature importance and model interpretability

### 4. Customer Lifetime Value (CLV) Estimation
- Regression-based CLV prediction
- Combined CLV + churn risk for retention prioritization
- Business recommendations: who to save, who to upsell, who to let go

### 5. Deployment
- **Streamlit App:** Input customer profile → segment, churn probability, CLV estimate
- **Tableau Dashboard:** Segment overview, revenue breakdown, churn risk distribution

## Tech Stack

**Analysis:** Python, Pandas, NumPy, SciPy
**Visualization:** Plotly, Matplotlib, Seaborn
**ML:** Scikit-learn, XGBoost, SHAP
**Clustering:** K-Means, Silhouette Analysis
**Deployment:** Streamlit
**BI:** Tableau / Power BI

## Dataset

**Source:** [Online Retail II — UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) via [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

| Property | Value |
|----------|-------|
| Transactions | ~1,067,000 |
| Time Period | Dec 2009 – Dec 2011 |
| Unique Customers | ~5,900 |
| Countries | 38+ |
| Features | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Customer-Segmentation-Retention-Analysis.git
cd Customer-Segmentation-Retention-Analysis

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook notebooks/customer_analysis.ipynb

# Run Streamlit app
streamlit run app/streamlit_app.py
```

## License

This project is licensed under the MIT License.
