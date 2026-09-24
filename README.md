# 🛒 AI-Powered E-Commerce Sales Forecasting & Profit Optimization Dashboard

**IBM SkillsBuild Data Analytics with AI — Academic Internship Project**

---

## Overview

This project is a complete, production-quality Business Intelligence (BI) and Machine Learning system that transforms raw e-commerce transaction data into actionable executive insights and a quantitative sales forecast.

It follows the IBM Business Intelligence framework:

```
DATA → INFORMATION → INSIGHT → DECISION → ACTION
```

| BI Level | Question |
|:---------|:---------|
| L1 — KPIs | What is happening? |
| L2 — Trends | How is it changing? |
| L3 — Drivers | Why? |
| L4 — Risk | What could go wrong? |
| L5 — Action | What should management investigate or test? |

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

## Dataset

| Field | Details |
|:------|:--------|
| **Name** | Sample Superstore |
| **File** | `data/Superstore.csv` |
| **Source** | Tableau Sample Data / Kaggle: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final |
| **Type** | Retail order transactions |
| **Key Columns** | Order Date, Ship Date, Customer ID, Segment, Region, State, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit |

> **Note:** Download the dataset and place it at `data/Superstore.csv` before running the application.

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
     ↓ calculate_kpis()             → KPI Dictionary
     ↓ create_monthly_series()      → Monthly Aggregated Series
     ↓ create_forecast_features()   → Supervised ML Feature Set
     ↓ chronological_split()        → Train / Test Sets
     ↓ train_forecast_models()      → 3 Trained Models
     ↓ evaluate_models()            → MAE / RMSE / R² / MAPE
     ↓ select_best_model()          → Selected Model (lowest MAE)
     ↓ save_model()                 → models/sales_forecast_model.joblib
     ↓ generate_forecast()          → Historical + Backtest + Future Forecast
     ↓ generate_business_insights() → Findings / Risks / Opportunities
     ↓ identify_risks()             → Evidence-Based Risk List
     ↓ identify_opportunities()     → Evidence-Based Opportunity List
     ↓ analyze_profitability()      → Discount Scenario Analysis
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
2. Drop rows with null `Order Date` (cannot be placed on time axis)
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

## Forecasting Approach

### Monthly Time Series
Transaction data is aggregated monthly. Target: `total_sales` (sum of Sales per month).

### Forecast Features
- Lag features: `lag_1` through `lag_12` (past sales)
- Rolling averages: `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12` (shifted to avoid leakage)
- Seasonality: `month_number`
- Trend: `year`, `time_index`

### Data Leakage Prevention
All rolling and lag features use only past observations. `.shift(1)` is applied before `.rolling()` so no current-period information is used in any feature.

### Train/Test Split
**Chronological** — earliest 80% for training, latest 20% for testing.

> Random shuffling is intentionally avoided: it would allow the model to see future data during training, producing inflated metrics that do not reflect real-world forecast performance.

---

## ML Models

| Model | Description |
|:------|:------------|
| Linear Regression | OLS baseline; interpretable; reveals linear trend |
| Ridge Regression | L2-regularised linear; handles correlated lag features |
| Random Forest | 200-tree ensemble; captures non-linear patterns |

---

## Evaluation Metrics

| Metric | Primary Role |
|:-------|:-------------|
| **MAE** | **Primary selection criterion** — average dollar error per month |
| RMSE | Secondary — penalises large errors more |
| R² | Informational — proportion of variance explained |
| MAPE | Informational — percentage error (zero actuals excluded) |

The model with the **lowest MAE** is selected as the final forecaster.

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
- Top 10 Products by Revenue
- Bottom 10 Products by Profit
- Regional Revenue, Profit, and Margin
- Segment Revenue and Margin
- Discount Band vs Profit Margin (association analysis)
- Discount vs Profit scatter
- Ship Mode analysis
- Key Drivers summary

### Page 3 — Forecast, Risk & Action
- Sales forecast chart (Actual / Backtest / Forecast)
- Model evaluation metrics
- Interactive Forecast Explorer (by Category or Region)
- Risk identification cards (High / Medium / Low)
- Opportunity cards
- Profit scenario analysis table
- Recommended Actions
- Data & Model Information panel

---

## Risk/Opportunity/Action Framework

### Risk Detection Rules
| Rule | Level | Threshold |
|:-----|:------|:----------|
| Category negative total profit | High | Profit < 0 |
| Sub-category negative profit | High | Profit < 0 |
| High-discount + negative margin product | Medium | Avg discount > 25% AND margin < 0 |
| Declining recent revenue | Medium | Latest month > 15% below 6-month average |
| Below-average region profit | Low | Region profit < 50% of cross-region average |

### Opportunity Detection Rules
- Top-2 categories by profit → High-Profit Category
- Top-3 sub-categories by margin (positive) → High-Margin Sub-Category
- Regions above average profit → Above-Average Profit Region
- Recent 3-month average > prior 3-month by >5% → Improving Revenue Trend

### Recommended Actions
Framed as hypotheses for management investigation and testing — never as guaranteed outcomes.

---

## Installation Instructions

### 1. Prerequisites
- Python 3.9 or later
- pip

### 2. Clone or Download the Project
```bash
git clone <repository-url>
cd Ecommerce-Sales-Forecasting-Dashboard
```

### 3. Create a Virtual Environment

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

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Download and Place the Dataset
Download the Superstore dataset from:
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

The dashboard will open at `http://localhost:8501`.

**First run:** The pipeline will train all three models and save the best to `models/sales_forecast_model.joblib`.

**Subsequent runs:** The saved model is loaded automatically. Click **Retrain Model** in the sidebar to retrain from scratch.

---

## How to Run the Jupyter Notebook

### Step 1: Generate the notebook
```bash
python create_notebook.py
```

### Step 2: Open the notebook
```bash
jupyter notebook notebooks/YourName_EcommerceSalesForecasting.ipynb
```

Or open in VS Code using the Jupyter extension.

### Step 3: Run all cells
The notebook imports from `YourName_EcommerceSalesForecasting.py` using a relative path adjustment. Ensure the kernel's working directory is `notebooks/`.

---

## How to Generate the Word Report

```bash
python create_report.py
```

Output: `YourName_EcommerceSalesForecastingReport.docx`

---

## How to Run the Core Pipeline (CLI)

```bash
python YourName_EcommerceSalesForecasting.py
```

This runs the full pipeline and prints KPI results to the terminal.

---

## Project Structure

```
Ecommerce-Sales-Forecasting-Dashboard/
│
├── data/
│   └── Superstore.csv                          ← Dataset (user must provide)
│
├── models/
│   └── sales_forecast_model.joblib             ← Saved model (generated on first run)
│
├── notebooks/
│   └── YourName_EcommerceSalesForecasting.ipynb← Jupyter Notebook (generated by create_notebook.py)
│
├── YourName_EcommerceSalesForecasting.py        ← Core analytics + ML pipeline
├── dashboard.py                                 ← Streamlit executive dashboard
├── create_notebook.py                           ← Notebook generator script
├── create_report.py                             ← Word report generator script
├── YourName_EcommerceSalesForecastingReport.docx← Word report (generated by create_report.py)
│
├── requirements.txt                             ← Python dependencies
├── README.md                                    ← This file
└── gitignore.txt                                ← Rename to .gitignore
```

---

## Analytical Integrity

> **No values are hard-coded.** All KPIs, rankings, insights, and forecast outputs are computed dynamically from the loaded dataset.

> **Associations ≠ Causation.** Observed patterns (e.g., high discount → lower margin) are stated as associations in the dataset, not as causal claims.

> **Forecasts are estimates.** All forecast outputs carry explicit disclaimers. They are planning aids, not guaranteed outcomes.

> **Recommendations are hypotheses.** All recommended actions are framed for investigation and testing, not as certain interventions.

---

## Limitations

- Forecasting model assumes historical patterns continue; structural breaks are not modelled
- Profit scenario analysis uses a simple OLS association model only
- External factors (macroeconomic conditions, marketing spend) are not incorporated
- Segmented forecasts require at least 15 monthly observations per segment
- The dataset is static; real-time pipeline integration is out of scope

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
- **Project Type:** End-to-End BI + ML + Executive Dashboard
- **Dataset:** Sample Superstore Retail Transactions
- **Author:** [Your Name]
