"""
Script to generate the Jupyter Notebook programmatically.
Run: python create_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ─────────────────────────────────────────────────────────────────────────────
# COVER
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# 🛒 AI-Powered E-Commerce Sales Forecasting & Profit Optimization
## IBM SkillsBuild Data Analytics with AI Academic Internship

---

| Field         | Details |
|:---|:---|
| **Project**   | AI-Powered E-Commerce Sales Forecasting & Profit Optimization Dashboard |
| **Type**      | Business Intelligence + Machine Learning |
| **Dataset**   | Sample Superstore Retail Transactions |
| **Author**    | [Your Name] |
| **Program**   | IBM SkillsBuild Data Analytics with AI |

---

### Business Intelligence Framework

```
DATA → INFORMATION → INSIGHT → DECISION → ACTION
```

| Level | Question |
|:------|:---------|
| L1 — KPIs  | What is happening? |
| L2 — Trends | How is it changing? |
| L3 — Drivers | Why? |
| L4 — Risk | What could go wrong? |
| L5 — Action | What should management investigate or test? |
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJECT INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 1. Project Introduction & Business Problem

### Business Objective
This project builds an end-to-end e-commerce analytics system that transforms raw transaction data into actionable business intelligence and a sales forecast. It helps management answer:

1. **How is the business performing?** — Revenue, profit, orders, customers
2. **What is changing?** — Monthly and YoY trends
3. **What drives performance?** — Products, categories, regions, segments
4. **Where are the risks?** — Negative-margin products, declining categories
5. **What might happen next?** — 3–12 month sales forecast
6. **What should management test?** — Evidence-based recommendations

### Why E-Commerce Sales Forecasting?
Accurate sales forecasting enables:
- **Inventory planning** — avoid over/under-stocking
- **Cash-flow forecasting** — anticipate revenue cycles
- **Marketing budget allocation** — target high-potential periods
- **Supplier negotiations** — backed by demand evidence

### Dataset
The **Sample Superstore** dataset contains retail order transactions including:
order dates, product categories, customer segments, regional data, sales, discounts, profit, and quantity.

> **Note:** All metrics, rankings, and insights in this notebook are computed dynamically
> from the loaded dataset. No values are hard-coded.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 2. Environment Setup & Imports"))

cells.append(code("""# Standard library
import os
import sys
import warnings
from datetime import datetime

# Data manipulation
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Machine learning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings('ignore')

# Plotting style
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})
sns.set_palette('Set2')

# Ensure project root is on path
ROOT = os.path.abspath('..')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print('Environment ready.')
print(f'pandas {pd.__version__} | numpy {np.__version__}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 3. Data Loading & Validation

We load the Superstore CSV, normalise column names, parse dates, and validate
that all required columns are present.
"""))

cells.append(code("""# Import the shared pipeline module
from YourName_EcommerceSalesForecasting import (
    DATA_PATH, load_data, validate_columns,
    clean_data, create_features, calculate_kpis,
    create_monthly_series, create_forecast_features,
    train_forecast_models, evaluate_models, select_best_model,
    save_model, load_model, generate_forecast,
    generate_business_insights, identify_risks, identify_opportunities,
    analyze_profitability, chronological_split, FEATURE_COLS,
    MODEL_PATH,
)

# Load from data/ directory (one level up from notebooks/)
data_path = os.path.join('..', 'data', 'Superstore.csv')

try:
    raw_df = load_data(data_path)
    print(f'Dataset loaded successfully.')
except FileNotFoundError as e:
    print(e)
    raise
"""))

cells.append(code("""# Dataset overview
print(f'Shape: {raw_df.shape}')
print(f'Columns ({len(raw_df.columns)}):')
for col in raw_df.columns:
    print(f'  {col}: {raw_df[col].dtype}')
"""))

cells.append(code("""# First 5 rows
raw_df.head()
"""))

cells.append(code("""# Basic statistics
raw_df.describe(include='all').T
"""))

cells.append(code("""# Missing value summary
missing = raw_df.isnull().sum()
missing_pct = (missing / len(raw_df) * 100).round(2)
pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct}).query('`Missing Count` > 0')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 4. Data Cleaning

### Cleaning Steps
1. Detect and remove exact duplicate rows
2. Drop rows with null `Order Date` (cannot be placed on time axis)
3. Drop rows where `Sales ≤ 0` (invalid revenue records)
4. Impute remaining numeric nulls with column median
5. Clip `Discount` to valid range [0, 1]
6. Strip whitespace from categorical columns

A **cleaning log** is produced to provide full transparency.
"""))

cells.append(code("""clean_df, cleaning_log = clean_data(raw_df)

print('=== CLEANING LOG ===')
for k, v in cleaning_log.items():
    print(f'  {k}: {v}')
"""))

cells.append(code("""print(f'Raw rows: {cleaning_log["raw_row_count"]:,}')
print(f'Duplicates removed: {cleaning_log["duplicate_rows_removed"]:,}')
print(f'Null order-date removed: {cleaning_log["null_order_date_removed"]:,}')
print(f'Invalid sales removed: {cleaning_log["invalid_sales_removed"]:,}')
print(f'Final analytical rows: {cleaning_log["final_row_count"]:,}')
print(f'Date range: {cleaning_log["date_range_start"]} to {cleaning_log["date_range_end"]}')
print(f'Total months: {cleaning_log["total_months"]}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 5. Feature Engineering

Derived columns added for analytics and ML:
- **Time**: Year, Quarter, Month, Month Number, Year-Month, Week, Day of Week
- **Shipping**: Shipping Days (Ship Date - Order Date)
- **Financial**: Profit Margin, Revenue per Unit
- **Discount Band**: categorical grouping of discount rates
- **Customer**: Customer Order Count, Repeat Customer flag
"""))

cells.append(code("""feat_df = create_features(clean_df)

new_cols = [c for c in feat_df.columns if c not in clean_df.columns]
print(f'New engineered columns ({len(new_cols)}):')
for c in new_cols:
    print(f'  {c}')

feat_df[new_cols].head(3)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 6. KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 6. Business KPI Calculations

### KPI Definitions
| KPI | Definition |
|:----|:-----------|
| **Total Revenue** | `sum(Sales)` |
| **Total Profit** | `sum(Profit)` |
| **Total Orders** | `count(unique Order IDs)` |
| **Total Customers** | `count(unique Customer IDs)` |
| **Average Order Value** | `Total Revenue / Total Orders` |
| **Profit Margin %** | `Total Profit / Total Revenue × 100` |
| **Average Discount %** | `mean(Discount) × 100` |
| **YoY Growth %** | `(Current Year Sales − Prior Year Sales) / Prior Year Sales × 100` |

> **Note:** Total Orders uses unique Order IDs — not row count — to avoid double-counting
> multi-item orders. All values computed dynamically.
"""))

cells.append(code("""kpis = calculate_kpis(feat_df)

print('=== BUSINESS KPIs ===')
for k, v in kpis.items():
    if v is None:
        print(f'  {k}: N/A')
    elif isinstance(v, float):
        print(f'  {k}: {v:,.2f}')
    elif isinstance(v, int):
        print(f'  {k}: {v:,}')
    else:
        print(f'  {k}: {v}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7. EDA — OVERALL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 7. Exploratory Data Analysis

### 7A. Overall Performance
"""))

cells.append(code("""fig, axes = plt.subplots(1, 4, figsize=(16, 4))

metrics = ['Sales', 'Profit', 'Quantity']
titles  = ['Sales Distribution', 'Profit Distribution', 'Quantity Distribution']
colors  = ['#1f4e79', '#2d6a4f', '#f18f01']

for ax, col, title, color in zip(axes[:3], metrics, titles, colors):
    feat_df[col].hist(bins=40, ax=ax, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

# Profit margin
axes[3].hist(feat_df['Profit Margin'].clip(-100, 100), bins=40,
             color='#7c5cd8', edgecolor='white', alpha=0.85)
axes[3].set_title('Profit Margin Distribution', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Profit Margin %')
axes[3].set_ylabel('Frequency')

plt.suptitle('Overall Business Metric Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7B. TIME TRENDS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("### 7B. Time Trends"))

cells.append(code("""monthly = create_monthly_series(feat_df)
print(f'Monthly series: {len(monthly)} periods')
monthly.head(3)
"""))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Monthly sales
ax = axes[0, 0]
ax.plot(monthly['month_dt'], monthly['total_sales'], color='#1f4e79', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(monthly['month_dt'], monthly['total_sales'], alpha=0.1, color='#1f4e79')
ax.set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales ($)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Monthly profit
ax = axes[0, 1]
colors_p = ['#2d6a4f' if v >= 0 else '#c1121f' for v in monthly['total_profit']]
ax.bar(monthly['month_dt'], monthly['total_profit'], color=colors_p, width=20)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Monthly Profit Trend', fontsize=13, fontweight='bold')
ax.set_ylabel('Profit ($)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Profit margin monthly
ax = axes[1, 0]
ax.plot(monthly['month_dt'], monthly['profit_margin'], color='#7c5cd8', linewidth=2, marker='s', markersize=4)
ax.axhline(monthly['profit_margin'].mean(), color='#f18f01', linewidth=1.5, linestyle='--',
           label=f"Avg: {monthly['profit_margin'].mean():.1f}%")
ax.set_title('Monthly Profit Margin %', fontsize=13, fontweight='bold')
ax.set_ylabel('Profit Margin (%)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Yearly revenue
ax = axes[1, 1]
yearly = feat_df.groupby('Year')['Sales'].sum().reset_index()
bars = ax.bar(yearly['Year'].astype(str), yearly['Sales'], color='#2e86ab', edgecolor='white')
for bar, val in zip(bars, yearly['Sales']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + yearly['Sales'].max()*0.01,
            f'${val/1e6:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Annual Revenue', fontsize=13, fontweight='bold')
ax.set_ylabel('Sales ($)')

plt.tight_layout()
plt.savefig('time_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print('Time trends chart saved.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7C. PRODUCT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("### 7C. Product & Category Analysis"))

cells.append(code("""# Category-level
cat_agg = feat_df.groupby('Category').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
).reset_index()
cat_agg['profit_margin'] = cat_agg['total_profit'] / cat_agg['total_sales'] * 100

print('Category Performance:')
print(cat_agg.to_string(index=False))
"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Sales by category
axes[0].bar(cat_agg['Category'], cat_agg['total_sales'], color=sns.color_palette('Set2')[:3], edgecolor='white')
axes[0].set_title('Revenue by Category', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Sales ($)')

# Profit by category
colors_cat = ['#2d6a4f' if v >= 0 else '#c1121f' for v in cat_agg['total_profit']]
axes[1].bar(cat_agg['Category'], cat_agg['total_profit'], color=colors_cat, edgecolor='white')
axes[1].set_title('Profit by Category', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Profit ($)')

# Profit margin
axes[2].bar(cat_agg['Category'], cat_agg['profit_margin'], color='#7c5cd8', edgecolor='white')
axes[2].set_title('Profit Margin % by Category', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Profit Margin (%)')

plt.suptitle('Category Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""if 'Sub-Category' in feat_df.columns:
    sc_agg = feat_df.groupby('Sub-Category').agg(
        total_sales=('Sales', 'sum'),
        total_profit=('Profit', 'sum'),
    ).reset_index()
    sc_agg['profit_margin'] = np.where(
        sc_agg['total_sales'] != 0,
        sc_agg['total_profit'] / sc_agg['total_sales'] * 100, 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc_sorted_sales  = sc_agg.sort_values('total_sales', ascending=True)
    sc_sorted_profit = sc_agg.sort_values('total_profit', ascending=True)

    axes[0].barh(sc_sorted_sales['Sub-Category'], sc_sorted_sales['total_sales'],
                 color='#2e86ab', edgecolor='white')
    axes[0].set_title('Revenue by Sub-Category', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Sales ($)')

    colors_sc = ['#c1121f' if v < 0 else '#2d6a4f' for v in sc_sorted_profit['total_profit']]
    axes[1].barh(sc_sorted_profit['Sub-Category'], sc_sorted_profit['total_profit'],
                 color=colors_sc, edgecolor='white')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title('Profit by Sub-Category', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Profit ($)')

    plt.tight_layout()
    plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7D. REGIONAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("### 7D. Regional Analysis"))

cells.append(code("""reg_agg = feat_df.groupby('Region').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
).reset_index()
reg_agg['profit_margin'] = reg_agg['total_profit'] / reg_agg['total_sales'] * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(reg_agg['Region'], reg_agg['total_sales'],
            color=sns.color_palette('Set2')[:len(reg_agg)], edgecolor='white')
axes[0].set_title('Revenue by Region', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Sales ($)')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20)

colors_reg = ['#2d6a4f' if v >= 0 else '#c1121f' for v in reg_agg['total_profit']]
axes[1].bar(reg_agg['Region'], reg_agg['total_profit'], color=colors_reg, edgecolor='white')
axes[1].set_title('Profit by Region', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Profit ($)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20)

axes[2].bar(reg_agg['Region'], reg_agg['profit_margin'], color='#7c5cd8', edgecolor='white')
axes[2].set_title('Profit Margin % by Region', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Profit Margin (%)')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=20)

plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7E. CUSTOMER & SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("### 7E. Customer Segment Analysis"))

cells.append(code("""if 'Segment' in feat_df.columns:
    seg_agg = feat_df.groupby('Segment').agg(
        total_sales=('Sales', 'sum'),
        total_profit=('Profit', 'sum'),
        unique_customers=('Customer ID', 'nunique') if 'Customer ID' in feat_df.columns
            else ('Sales', 'count'),
    ).reset_index()
    seg_agg['profit_margin'] = seg_agg['total_profit'] / seg_agg['total_sales'] * 100

    print('Segment Performance:')
    print(seg_agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_seg = sns.color_palette('Set2')[:len(seg_agg)]
    axes[0].bar(seg_agg['Segment'], seg_agg['total_sales'], color=colors_seg, edgecolor='white')
    axes[0].set_title('Revenue by Segment', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Sales ($)')

    axes[1].bar(seg_agg['Segment'], seg_agg['profit_margin'], color=colors_seg, edgecolor='white')
    axes[1].set_title('Profit Margin % by Segment', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Profit Margin (%)')

    plt.tight_layout()
    plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 7F. DISCOUNT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("### 7F. Discount vs Profit Analysis\n> **Important:** The patterns below are observed associations in the dataset. They do NOT establish causal relationships."))

cells.append(code("""if 'Discount' in feat_df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: discount vs profit (sample)
    sample = feat_df.sample(min(3000, len(feat_df)), random_state=42)
    axes[0].scatter(sample['Discount'], sample['Profit'], alpha=0.3, s=15, color='#2e86ab')
    axes[0].axhline(0, color='red', linewidth=1, linestyle='--')
    axes[0].set_title('Discount vs Profit (Sample)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Discount Rate')
    axes[0].set_ylabel('Transaction Profit ($)')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Discount band vs avg profit margin
    if 'Discount Band' in feat_df.columns:
        band_agg = feat_df.groupby('Discount Band')['Profit Margin'].mean().reset_index()
        order = ['No Discount','Low (0–10%)','Moderate (11–20%)','High (21–30%)','Very High (>30%)']
        band_agg = band_agg.set_index('Discount Band').reindex(
            [o for o in order if o in band_agg['Discount Band'].values]).reset_index()
        colors_b = ['#2d6a4f' if v >= 0 else '#c1121f' for v in band_agg['Profit Margin']]
        axes[1].bar(band_agg['Discount Band'], band_agg['Profit Margin'], color=colors_b, edgecolor='white')
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_title('Avg Profit Margin by Discount Band\n(Observed Association)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Avg Profit Margin (%)')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Discount vs sales
    axes[2].scatter(sample['Discount'], sample['Sales'], alpha=0.3, s=15, color='#f18f01')
    axes[2].set_title('Discount vs Sales (Sample)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Discount Rate')
    axes[2].set_ylabel('Transaction Sales ($)')
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 8. BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 8. Business Insights

All insights below are generated dynamically from the dataset.
Associations are stated as observed patterns — **not causal claims**.
"""))

cells.append(code("""insights = generate_business_insights(feat_df, kpis, monthly)

print('=== KEY FINDINGS ===')
for i, f in enumerate(insights['findings'], 1):
    print(f'\\n{i}. {f}')

print('\\n=== RISKS ===')
for i, r in enumerate(insights['risks'], 1):
    print(f'\\n{i}. {r}')

print('\\n=== OPPORTUNITIES ===')
for i, o in enumerate(insights['opportunities'], 1):
    print(f'\\n{i}. {o}')

print('\\n=== RECOMMENDATIONS ===')
for i, r in enumerate(insights['recommendations'], 1):
    print(f'\\n{i}. {r}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 9. MONTHLY TIME SERIES & FORECAST FEATURES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 9. Monthly Time Series & Forecast Feature Engineering
"""))

cells.append(code("""# Already created above — display structure
print(f'Monthly series shape: {monthly.shape}')
print('\\nColumns:', list(monthly.columns))
monthly.head(6)
"""))

cells.append(code("""# Create forecast features
forecast_feat_df = create_forecast_features(monthly)
feature_cols = [c for c in FEATURE_COLS if c in forecast_feat_df.columns]

print(f'Forecast feature rows: {len(forecast_feat_df)}')
print(f'Feature columns ({len(feature_cols)}): {feature_cols}')
forecast_feat_df[['month_dt','total_sales'] + feature_cols[:5]].head(4)
"""))

cells.append(md("""### Why These Features?

| Feature | Purpose |
|:--------|:--------|
| `lag_1` … `lag_12` | Capture autocorrelation — past months predict future |
| `rolling_mean_3/6/12` | Capture short/medium/long-term trend smoothing |
| `month_number` | Capture seasonality (e.g., Q4 holiday peaks) |
| `year` | Capture long-term linear growth trend |
| `time_index` | Integer time step — additional trend signal |

> **Data leakage prevention:** All lag and rolling features use only past periods (`.shift(1)` applied before `.rolling()`). No future information is used to predict the current period.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 10. Chronological Train/Test Split

### Why Chronological — Not Random?
Random shuffling would allow the model to see future data during training,
producing artificially inflated metrics that do not reflect real-world
forecasting performance. **Chronological splitting correctly simulates deployment**:
the model is trained on earlier data and evaluated on unseen future data.
"""))

cells.append(code("""train_df, test_df = chronological_split(forecast_feat_df, test_ratio=0.20)

print(f'Training set: {len(train_df)} months')
print(f'  Range: {train_df["month_dt"].min().date()} → {train_df["month_dt"].max().date()}')
print(f'Test set: {len(test_df)} months')
print(f'  Range: {test_df["month_dt"].min().date()} → {test_df["month_dt"].max().date()}')

# Visualise split
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df['month_dt'], train_df['total_sales'],
        color='#1f4e79', linewidth=2.5, label='Training Set', marker='o', markersize=4)
ax.plot(test_df['month_dt'],  test_df['total_sales'],
        color='#c1121f', linewidth=2.5, label='Test Set', marker='s', markersize=4)
ax.axvline(test_df['month_dt'].min(), color='#f18f01', linestyle='--', linewidth=2, label='Split Point')
ax.set_title('Chronological Train / Test Split', fontsize=14, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 11. Model Training

Three regression models are trained on the chronological training set:

| Model | Rationale |
|:------|:----------|
| **Linear Regression** | Simple baseline; interpretable coefficients; reveals linear trend |
| **Ridge Regression** | Linear with L2 regularisation; prevents overfitting when features are correlated |
| **Random Forest** | Ensemble of decision trees; captures non-linear patterns and feature interactions |
"""))

cells.append(code("""trained_models = train_forecast_models(train_df, feature_cols)
print(f'Trained models: {list(trained_models.keys())}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 12. MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 12. Model Evaluation

### Metric Definitions
| Metric | Meaning | Better |
|:-------|:--------|:-------|
| **MAE** | Mean Absolute Error — average dollar error per month | Lower |
| **RMSE** | Root Mean Squared Error — penalises large errors more | Lower |
| **R²** | Proportion of variance explained (0–1) | Higher |
| **MAPE** | Mean Absolute Percentage Error | Lower |

### Why MAE as Primary Selection Criterion?
For business sales forecasting, the **practical question is**:
*"By how many dollars is my forecast off on average?"*
MAE directly answers this. R² measures statistical fit and can be misleading
when variance is dominated by trend rather than genuine model skill.
RMSE is reported but penalises outlier months more heavily than is
appropriate for operational monthly planning.
"""))

cells.append(code("""eval_df = evaluate_models(trained_models, test_df, feature_cols)
print('Model Evaluation on Held-Out Test Set (Chronological):')
eval_df.style.background_gradient(subset=['MAE','RMSE'], cmap='RdYlGn_r') \\
             .background_gradient(subset=['R2'], cmap='RdYlGn') \\
             .format({'MAE': '${:,.0f}', 'RMSE': '${:,.0f}',
                      'R2': '{:.4f}', 'MAPE': '{:.2f}%'})
"""))

cells.append(code("""# Select best model
best_name, best_model = select_best_model(eval_df, trained_models, primary_metric='MAE')
print(f'\\nSelected model: {best_name}')
print(f'MAE  = ${eval_df.loc[best_name, \"MAE\"]:,.0f}')
print(f'RMSE = ${eval_df.loc[best_name, \"RMSE\"]:,.0f}')
print(f'R²   = {eval_df.loc[best_name, \"R2\"]:.4f}')
"""))

cells.append(code("""# Visualise predictions vs actual on test set
X_test = test_df[feature_cols]
y_pred = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(test_df['month_dt'], test_df['total_sales'],
             color='#1f4e79', linewidth=2.5, label='Actual', marker='o')
axes[0].plot(test_df['month_dt'], y_pred,
             color='#f18f01', linewidth=2.5, label=f'{best_name} Prediction',
             linestyle='--', marker='s')
axes[0].set_title(f'Test Period: Actual vs {best_name}', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Monthly Sales ($)')
axes[0].legend()

residuals = test_df['total_sales'].values - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color='#7c5cd8', edgecolors='white', s=60)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Sales ($)')
axes[1].set_ylabel('Residual ($)')

plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 13. Save Selected Model"))

cells.append(code("""model_path = os.path.join('..', 'models', 'sales_forecast_model.joblib')
save_model(best_model, best_name, eval_df, feature_cols, monthly, filepath=model_path)
print(f'Model saved: {model_path}')

# Verify it can be reloaded
payload = load_model(model_path)
print(f'Reload check — model name: {payload[\"model_name\"]}, trained at: {payload[\"trained_at\"]}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 14. FORECAST GENERATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 14. Forecast Generation

The selected model forecasts future monthly sales by iterating one step ahead,
feeding each prediction back as a lag feature for the next period.

> **Disclaimer:** Forecast values are estimates produced by the selected model and
> are subject to uncertainty. They should not be treated as guaranteed business outcomes.
"""))

cells.append(code("""FORECAST_HORIZON = 6  # months

forecast_df = generate_forecast(
    best_model, forecast_feat_df, monthly,
    horizon=FORECAST_HORIZON, feature_cols=feature_cols,
)

future = forecast_df[forecast_df['type'] == 'Forecast'][['month_dt','predicted']].copy()
future.columns = ['Month', 'Forecast Sales ($)']
future['Month'] = future['Month'].dt.strftime('%B %Y')
future['Forecast Sales ($)'] = future['Forecast Sales ($)'].apply(lambda v: f'${v:,.0f}')
print(f'\\nForecasted months ({FORECAST_HORIZON}-month horizon):')
print(future.to_string(index=False))
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(16, 5))

actual_rows   = forecast_df[forecast_df['type'] == 'Actual']
backtest_rows = forecast_df[forecast_df['type'] == 'Backtest']
fc_rows       = forecast_df[forecast_df['type'] == 'Forecast']

ax.plot(actual_rows['month_dt'], actual_rows['actual'],
        color='#1f4e79', linewidth=2.5, label='Actual (Historical)', marker='o', markersize=4)
ax.plot(backtest_rows['month_dt'], backtest_rows['actual'],
        color='#1f4e79', linewidth=1.5, linestyle=':', label='Actual (Test Period)', marker='o', markersize=3)
ax.plot(backtest_rows['month_dt'], backtest_rows['predicted'],
        color='#2e86ab', linewidth=2, linestyle='--', label='Model Backtest', marker='s', markersize=5)
if len(fc_rows) > 0:
    ax.plot(fc_rows['month_dt'], fc_rows['predicted'],
            color='#7c5cd8', linewidth=2.5, linestyle='--', label=f'Forecast ({FORECAST_HORIZON}M)',
            marker='D', markersize=7)
    ax.axvspan(fc_rows['month_dt'].min(), fc_rows['month_dt'].max(),
               alpha=0.08, color='#7c5cd8', label='Forecast Window')

ax.set_title(f'Sales Forecast — Actual · Backtest · Future ({FORECAST_HORIZON}-month horizon)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)')
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print('Forecast chart saved.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 15. PROFIT SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 15. Profit Scenario Analysis (Discount vs Profit)

> **DISCLAIMER:** These scenarios use a simple OLS regression between average discount
> rate and profit margin. This captures historical association **only** — it does NOT
> establish that changing the discount will cause the estimated profit outcome.
> Real outcomes depend on customer price sensitivity, competition, cost structure, and
> many other factors not modelled here.
"""))

cells.append(code("""profit_analysis = analyze_profitability(feat_df)

print('=== DISCOUNT BAND SUMMARY ===')
print(profit_analysis['discount_band_summary'].to_string(index=False))

print('\\n=== HYPOTHETICAL DISCOUNT SCENARIOS ===')
print('(Based on historical discount-margin association only)')
print(profit_analysis['scenario_estimates'].to_string(index=False))

print(f'\\n=== DISCLAIMER ===')
print(profit_analysis['caveats'])
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 16. RISK & OPPORTUNITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 16. Risk Analysis"))

cells.append(code("""risks = identify_risks(feat_df, monthly)

print(f'Total risks identified: {len(risks)}')
for risk in risks:
    print(f'\\n  [{risk["risk_level"]}] {risk["type"]}')
    print(f'  Entity : {risk["entity"]}')
    print(f'  Detail : {risk["detail"]}')
"""))

cells.append(md("### Opportunity Analysis"))

cells.append(code("""opportunities = identify_opportunities(feat_df, monthly)

print(f'Total opportunities identified: {len(opportunities)}')
for opp in opportunities:
    print(f'\\n  {opp["type"]}')
    print(f'  Entity : {opp["entity"]}')
    print(f'  Detail : {opp["detail"]}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 17. RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 17. Recommended Actions"))

cells.append(code("""print('=== RECOMMENDED ACTIONS ===')
print('(Evidence-based hypotheses for investigation/testing — not guaranteed outcomes)')
for i, rec in enumerate(insights['recommendations'], 1):
    print(f'\\nAction {i}: {rec}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# 18. CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 18. Conclusion

### Summary of Analytical Findings

This notebook demonstrated a complete end-to-end analytics pipeline:

1. **Data Loading & Cleaning** — The Superstore dataset was loaded, validated, and cleaned. A full cleaning log provides transparency on all transformations.

2. **Feature Engineering** — Time, financial, and discount features were derived to support both EDA and ML.

3. **KPI Calculation** — Core business KPIs (Revenue, Profit, Margin, AOV, YoY Growth) were computed dynamically.

4. **EDA** — Focused analyses addressed: time trends, category performance, regional patterns, segment contributions, and discount-profit associations.

5. **ML Forecasting** — Three regression models (Linear Regression, Ridge, Random Forest) were trained on a **chronological** split. The model with the lowest MAE was selected as the final forecaster.

6. **Forecast Generation** — Future monthly sales were forecast using iterative one-step-ahead prediction. All values are **estimates** subject to uncertainty.

7. **Profit Scenario Analysis** — Historical discount-margin associations were explored. Scenarios are hypothetical only.

8. **Risk & Opportunity Identification** — Evidence-based risks and opportunities were identified using defined detection rules.

9. **Recommended Actions** — Five practical actions were generated for management investigation and testing.

### Analytical Integrity Statement
- All metrics are computed from the loaded dataset. No values are hard-coded.
- Observed associations are not claimed as causal relationships.
- Forecast values are model estimates, not guaranteed outcomes.
- Recommended actions are framed as hypotheses to test, not certainties.
- The IBM BI framework (KPIs → Trends → Drivers → Risk → Action) guided the project structure.

### Limitations
- The forecasting model assumes past patterns continue. Structural breaks (new competitors, economic shifts) are not modelled.
- The profit scenario analysis uses a simple OLS association model only.
- Geographic analysis (state-level maps) requires additional tooling beyond this notebook.
- The model does not capture external factors (marketing spend, seasonality events, macroeconomic conditions).

### Future Scope
- Integrate external signals (web traffic, marketing spend, macroeconomic indices)
- Explore SARIMA or Prophet for explicit seasonality modelling
- Build a product-level demand forecasting model
- Implement a formal A/B test design for discount policy evaluation
- Add real-time data pipeline integration
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language":     "python",
        "name":         "python3",
    },
    "language_info": {
        "name":    "python",
        "version": "3.10.0",
    },
}

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'notebooks',
    'YourName_EcommerceSalesForecasting.ipynb',
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written: {out_path}')
print(f'Cells: {len(nb.cells)}')
