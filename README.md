# 🛒 AI-Powered E-Commerce Sales Forecasting & Profit Optimization Dashboard

**IBM SkillsBuild Data Analytics with AI — Academic Internship Project**
**Author: Yashaswi**

---

## Overview

This project is an end-to-end Business Intelligence and Machine Learning project that transforms raw e-commerce transaction data into actionable executive insights and a quantitative sales forecast.

It follows the IBM Business Intelligence framework:

```
DATA → INFORMATION → INSIGHT → DECISION → ACTION
```

| BI Level | Question |
|:---------|:---------|
| L1 — KPIs    | What is happening? |
| L2 — Trends  | How is it changing? |
| L3 — Drivers | Why? |
| L4 — Risk    | What could go wrong? |
| L5 — Action  | What should management investigate or test? |

---

## Business Problem

The system helps management answer:

1. How is the business performing in revenue, profit, orders, and customers?
2. How are these metrics changing over time?
3. Which products, categories, regions, and segments drive performance?
4. Where are the business risks?
5. Where are the strongest opportunities?
6. What monthly sales should management expect over the next 3–12 months?
7. What actions should management investigate or test?

---

## Objectives

1. Build a modular Python analytics pipeline for e-commerce transaction data.
2. Compute and present all core business KPIs dynamically from the loaded dataset.
3. Conduct focused EDA across five analytical dimensions: time, product, region, segment, discount.
4. Train, evaluate, and compare three supervised regression models for monthly sales forecasting.
5. Implement chronological train/test splitting to prevent future data leakage.
6. Refit the selected model on all available historical data before forecasting future periods.
7. Generate a configurable 3–12 month sales forecast with clear uncertainty disclaimers.
8. Build a three-page interactive Streamlit executive BI dashboard.
9. Identify evidence-based business risks and opportunities using defined detection rules.
10. Generate dynamic, data-driven recommended actions for management investigation.

---

## Dataset

| Field | Details |
|:------|:--------|
| **Name** | Sample Superstore |
| **File** | `data/Superstore.csv` |
| **Source** | Tableau Sample Data / Kaggle: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final |
| **Type** | Retail order transactions |
| **Key Columns** | Order Date, Ship Date, Customer ID, Segment, Region, State, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit |

> **Dataset setup:** Download `Superstore.csv` from the Kaggle link above and place it at `data/Superstore.csv` before running the application.

---

## Technologies

| Library | Purpose |
|:--------|:--------|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Notebook visualisations |
| `plotly` | Interactive dashboard charts |
| `scikit-learn` | ML models, evaluation, pipelines |
| `streamlit` | Interactive executive dashboard |
| `joblib` | Model persistence |
| `openpyxl` | Excel export support |
| `jupyter` / `ipykernel` | Notebook environment |
| `python-docx` | Word report generation |
| `nbformat` | Notebook generation |

---

## Project Architecture

```
Raw CSV (Superstore.csv)
     ↓ load_data()
Raw DataFrame
     ↓ clean_data()
Clean DataFrame + Cleaning Log
     ↓ create_features()
Feature-Engineered DataFrame
     ↓ calculate_kpis()              → KPI Dictionary
     ↓ create_monthly_series()       → Monthly Aggregated Series
     ↓ create_forecast_features()    → Supervised ML Feature Set
     ↓ chronological_split()         → Train Set (80%) / Test Set (20%)
     ↓ train_forecast_models()       → 3 Candidate Models (on train set)
     ↓ evaluate_models()             → MAE / RMSE / R² / MAPE (on test set)
     ↓ select_best_model()           → Selected Model (lowest MAE)
     ↓ refit_selected_model()        → Final Model (refit on ALL historical data)
     ↓ save_model()                  → models/sales_forecast_model.joblib
     ↓ generate_backtest()           → Test-period predictions only
     ↓ generate_future_forecast()    → Future months forecast
     ↓ generate_business_insights()  → Findings / Risks / Opportunities
     ↓ identify_risks()              → Evidence-Based Risk List
     ↓ identify_opportunities()      → Evidence-Based Opportunity List
     ↓ analyze_profitability()       → Illustrative Discount Scenarios
     ↓
Streamlit Dashboard (3 pages)
```

---

## BI Workflow

```
PAGE 1: Executive Overview
        "How healthy is the business?"
        KPIs → Monthly Trends → Category Performance → Executive Insights

PAGE 2: Sales & Product Analysis
        "What is driving sales and profit?"
        Sub-Category → Products → Regions → Segments → Discounts → Shipping

PAGE 3: Forecast, Risk & Action
        "What might happen next?"
        Forecast → Risk → Opportunity → Scenario Analysis → Recommended Actions
```

---

## Data Cleaning Process

Implemented in `clean_data()`:

1. Detect and remove exact duplicate rows
2. Drop rows with null `Order Date`
3. Drop rows where `Sales ≤ 0` (invalid revenue records)
4. Impute remaining numeric nulls with column median
5. Clip `Discount` to valid range [0, 1]
6. Strip whitespace from all categorical columns

A **cleaning log** records all transformation counts dynamically.

---

## KPI Definitions

| KPI | Definition |
|:----|:-----------|
| Total Revenue | `sum(Sales)` |
| Total Profit | `sum(Profit)` |
| Total Orders | `count(unique Order IDs)` — not row count |
| Total Customers | `count(unique Customer IDs)` |
| Average Order Value | `Total Revenue / Total Orders` |
| Profit Margin % | `Total Profit / Total Revenue × 100` |
| Average Discount % | `mean(Discount) × 100` |
| YoY Growth % | `(Current Year Sales − Prior Year Sales) / Prior Year Sales × 100` |

> All KPI values are computed dynamically at runtime. No values are hard-coded.

---

## EDA Dimensions

| Dimension | Analysis |
|:----------|:---------|
| Time Trends | Monthly revenue, profit, margin; annual revenue |
| Product / Category | Revenue and profit by category and sub-category; top/bottom products |
| Regional | Revenue, profit, and margin by region |
| Customer Segment | Revenue and margin by Consumer, Corporate, Home Office |
| Discount | Discount band vs profit margin (association analysis, not causal) |
| Shipping | Order volume and profit by Ship Mode |

---

## Forecasting Approach

### Monthly Time Series
Transaction data is aggregated monthly. Target: `total_sales` (sum of Sales per month).

### Forecast Features
- Lag features: `lag_1` through `lag_12`
- Rolling averages: `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12` (shifted to avoid leakage)
- Seasonality: `month_number`
- Trend: `year`, `time_index`

### Data Leakage Prevention
All rolling and lag features use `.shift(1)` before `.rolling()` so no current-period information is used in any feature.

---

## Chronological Train/Test Validation

**Chronological splitting** is used — NOT random shuffling.

| Set | Period | Size |
|:----|:-------|:-----|
| Training | Earliest 80% of monthly observations | 80% |
| Test | Latest 20% of monthly observations | 20% |

Random shuffling is intentionally avoided: it would allow the model to see future data during training, producing inflated metrics that do not reflect real-world forecast performance.

---

## ML Models & Evaluation

### Models Trained

| Model | Description |
|:------|:------------|
| Linear Regression | OLS baseline; interpretable; reveals linear trend |
| Ridge Regression | L2-regularised linear; handles correlated lag features |
| Random Forest | 200-tree ensemble; captures non-linear patterns |

### Evaluation Metrics (on held-out test set)

| Metric | Primary Role |
|:-------|:-------------|
| **MAE** | **Primary selection criterion** — average dollar error per month |
| RMSE | Secondary — penalises large errors more |
| R² | Informational — proportion of variance explained |
| MAPE | Informational — percentage error (zero actuals excluded) |

### Refit on Full Data
After selecting the best model (by MAE on the test set), the selected model architecture is **refit on all available historical data** before future forecasting. Evaluation metrics remain those from the held-out test set.

---

## Dashboard Pages

### Page 1 — Executive Overview
- 7 KPI cards (Revenue, Profit, Orders, Customers, AOV, Profit Margin, Avg Discount)
- Monthly Revenue Trend
- Monthly Profit Trend
- Revenue & Profit by Category
- Year-over-Year performance
- 5 dynamic executive insights

### Page 2 — Sales & Product Analysis
- Revenue and Profit by Sub-Category
- Top 10 Products by Revenue / Bottom 10 by Profit
- Regional Revenue, Profit, and Margin
- Segment Revenue and Margin
- Discount Band vs Profit Margin (observed association — labelled)
- Discount vs Profit scatter
- Ship Mode analysis
- Key Drivers summary

### Page 3 — Forecast, Risk & Action
- **Actual Sales** — full historical monthly series
- **Backtest Prediction** — model predictions on test period only (not training data)
- **Future Forecast** — future monthly predictions from refit model
- Forecast horizon selector (3 / 6 / 12 months)
- Model evaluation metrics table
- Interactive Forecast Explorer (by Category or Region)
- Risk identification cards (High / Medium / Low)
- Opportunity cards
- Illustrative profit scenario analysis
- 5 recommended actions
- Data & Model Information panel

---

## Risk/Opportunity/Action Framework

### Risk Detection Rules

| Rule | Level | Definition |
|:-----|:------|:-----------|
| Category total profit < 0 | High | Computed dynamically |
| Sub-category total profit < 0 | High | Computed dynamically |
| High-discount product with negative margin | Medium | avg_discount > 25% AND margin < 0 |
| Latest month > 15% below 6-month average | Medium | Computed dynamically |
| Region profit < 50% of cross-region average | Low | Computed dynamically |

### Opportunity Detection Rules
- Top-2 categories by positive profit → High-Profit Category
- Top-3 sub-categories by margin (positive only) → High-Margin Sub-Category
- Regions above cross-region average profit → Above-Average Profit Region
- Recent 3-month average > prior 3-month by > 5% → Improving Revenue Trend

### Recommended Actions
Framed as hypotheses for investigation and testing — not guaranteed outcomes.

---

## Installation Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yashaswi-catalyst/E-commerce-sales-forecasting.git
cd E-commerce-sales-forecasting
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Download and place the dataset
Download `Superstore.csv` from:
- https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Place the file at:
```
data/Superstore.csv
```

---

## How to Run the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501`.

**First run:** Trains all three models and saves the best to `models/sales_forecast_model.joblib`.

**Subsequent runs:** Loads the saved model. If the dataset has changed, a staleness warning is shown. Click **Retrain Model** in the sidebar to retrain.

---

## How to Generate and Run the Jupyter Notebook

### Step 1: Generate the notebook
```bash
python create_notebook.py
```

### Step 2: Open the notebook
```bash
jupyter notebook notebooks/Yashaswi_EcommerceSalesForecasting.ipynb
```

Or open in VS Code with the Jupyter extension.

---

## How to Generate the Word Report

```bash
python create_report.py
```

Output: `Yashaswi_EcommerceSalesForecastingReport.docx`

---

## How to Run the Core Pipeline (CLI)

```bash
python Yashaswi_EcommerceSalesForecasting.py
```

---

## Project Structure

```
E-commerce-sales-forecasting/
│
├── data/
│   └── Superstore.csv                              ← Dataset (user must provide)
│
├── models/
│   └── sales_forecast_model.joblib                 ← Saved model (generated on first run)
│
├── notebooks/
│   └── Yashaswi_EcommerceSalesForecasting.ipynb    ← Jupyter Notebook
│
├── Yashaswi_EcommerceSalesForecasting.py           ← Core analytics + ML pipeline
├── dashboard.py                                    ← Streamlit executive dashboard
├── create_notebook.py                              ← Notebook generator script
├── create_report.py                                ← Word report generator script
├── Yashaswi_EcommerceSalesForecastingReport.docx   ← Word report (generated)
│
├── requirements.txt                                ← Python dependencies
└── README.md                                       ← This file
```

---

## Analytical Integrity

> **No values are hard-coded.** All KPIs, rankings, insights, and forecast outputs are computed dynamically from the loaded dataset.

> **Associations ≠ Causation.** Observed patterns (e.g., high discount is associated with lower margin) are stated as associations, not causal claims.

> **Forecasts are estimates.** All forecast outputs carry explicit disclaimers. They are planning aids, not guaranteed outcomes.

> **Recommendations are hypotheses.** All recommended actions are framed for investigation and testing, not as certain interventions.

---

## Limitations

- Forecasting model assumes historical patterns continue; structural breaks are not modelled
- Profit scenario analysis uses a simple OLS association model only
- External factors (macroeconomic conditions, marketing spend) are not incorporated
- Segmented forecasts require at least 15 monthly observations per segment
- The dataset is a static CSV; real-time pipeline integration is out of scope

---

## Future Scope

- Integrate external signals (marketing spend, economic indices)
- Explore Prophet or SARIMA for explicit seasonality modelling
- Product-level demand forecasting for inventory planning
- Formal A/B test design for discount policy evaluation
- Deploy dashboard on Streamlit Cloud or IBM Cloud
- Real-time data pipeline integration

---

## Program Information

- **Program:** IBM SkillsBuild Data Analytics with AI — Academic Internship
- **Project Type:** End-to-End Business Intelligence and Machine Learning Project
- **Dataset:** Sample Superstore Retail Transactions
- **Author:** Yashaswi
