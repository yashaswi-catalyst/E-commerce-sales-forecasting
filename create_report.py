"""
Generate the Word Report for:
AI-Powered E-Commerce Sales Forecasting & Profit Optimization
IBM SkillsBuild Data Analytics with AI Academic Internship

Run: python create_report.py
Output: Yashaswi_EcommerceSalesForecastingReport.docx
"""

import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return h

def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    return p

def add_table_row(table, cells, bold=False):
    row = table.add_row()
    for i, val in enumerate(cells):
        cell = row.cells[i]
        cell.text = str(val)
        if bold:
            for run in cell.paragraphs[0].runs:
                run.bold = True
    return row

def section_divider(doc):
    doc.add_paragraph("")

# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------

def create_report(output_path: str):
    doc = Document()

    # Narrow margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ─────────────────────────────────────────────────────────────────────────
    # TITLE PAGE
    # ─────────────────────────────────────────────────────────────────────────
    doc.add_paragraph("")
    doc.add_paragraph("")
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run("AI-Powered E-Commerce Sales Forecasting\n& Profit Optimization Dashboard")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph("")
    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run(
        "IBM SkillsBuild Data Analytics with AI — Academic Internship Project\n"
        "End-to-End Business Intelligence + Machine Learning System"
    )
    sub_run.font.size = Pt(13)
    sub_run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    doc.add_paragraph("")
    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.add_run(
        f"Author: Yashaswi\n"
        f"Date: {datetime.date.today().strftime('%B %Y')}\n"
        f"Dataset: Sample Superstore Retail Transactions\n"
        f"Technologies: Python · Pandas · Scikit-Learn · Streamlit · Plotly"
    ).font.size = Pt(12)

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────────────────
    # ABSTRACT
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "Abstract", level=1)
    add_para(doc, (
        "This report presents an end-to-end Business Intelligence and Machine Learning system "
        "designed to transform raw e-commerce transaction data into actionable executive insights "
        "and a quantitative sales forecast. The system uses the Sample Superstore retail dataset "
        "containing order transactions with product, customer, regional, and financial attributes. "
        "The project covers data loading, validation, cleaning, feature engineering, "
        "Key Performance Indicator (KPI) calculation, Exploratory Data Analysis (EDA), "
        "time-series sales forecasting using three regression models (Linear Regression, "
        "Ridge Regression, and Random Forest Regressor), profit scenario analysis, "
        "risk identification, opportunity detection, and an interactive three-page Streamlit "
        "executive dashboard. All metrics, insights, and forecast outputs are computed "
        "dynamically from the loaded dataset. No values are hard-coded. The project follows "
        "the IBM BI framework: KPIs → Trends → Drivers → Risk → Action."
    ))

    doc.add_page_break()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. INTRODUCTION
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "1. Introduction", level=1)
    add_para(doc, (
        "The rapid growth of e-commerce has generated large volumes of transactional data "
        "that, when analysed systematically, can yield significant competitive intelligence. "
        "This project demonstrates how a B.Tech Computer Science student can apply "
        "standard data analytics and machine learning techniques to derive board-ready "
        "business insights from a publicly available retail dataset."
    ))
    add_para(doc, (
        "The project is structured around the IBM Business Intelligence hierarchy: "
        "Level 1 (What is happening?), Level 2 (How is it changing?), "
        "Level 3 (Why?), Level 4 (What could go wrong?), and "
        "Level 5 (What should management investigate or test?). "
        "Every dashboard section, KPI, and analytical output is mapped to one of these levels."
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. BUSINESS PROBLEM
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "2. Business Problem", level=1)
    add_para(doc, "The project addresses the following management questions:")
    questions = [
        "How is the overall business performing in terms of revenue, profit, and orders?",
        "How are these metrics changing over time (monthly, quarterly, annually)?",
        "Which products, categories, regions, and customer segments drive performance?",
        "Where are the business risks — negative margins, declining categories, weak regions?",
        "Where are the strongest opportunities for improvement?",
        "What monthly sales volumes can management expect over the next 3–12 months?",
        "What operational or strategic actions should management investigate or test?",
    ]
    for q in questions:
        add_bullet(doc, q)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PROJECT OBJECTIVES
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "3. Project Objectives", level=1)
    objectives = [
        "Build a modular, reusable Python analytics pipeline for e-commerce transaction data.",
        "Compute and present all core business KPIs dynamically from the loaded dataset.",
        "Conduct focused EDA across five analytical dimensions: time, product, region, segment, discount.",
        "Train, evaluate, and compare three supervised regression models for monthly sales forecasting.",
        "Implement chronological train/test splitting to prevent future data leakage.",
        "Generate a configurable 3–12 month sales forecast with clear uncertainty disclaimers.",
        "Build a three-page interactive Streamlit executive BI dashboard.",
        "Identify evidence-based business risks and opportunities using defined detection rules.",
        "Generate dynamic, data-driven recommended actions for management investigation.",
        "Produce a clean, reproducible Jupyter Notebook documenting all steps.",
    ]
    for obj in objectives:
        add_bullet(doc, obj)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DATASET DESCRIPTION
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "4. Dataset Description", level=1)
    add_para(doc, (
        "The project uses the Sample Superstore dataset — a widely used retail analytics "
        "benchmark dataset published by Tableau. It is also available on Kaggle "
        "(https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)."
    ))
    add_para(doc, "Key dataset characteristics:", bold=True)
    dataset_attrs = [
        ("File name",    "Superstore.csv"),
        ("File path",    "data/Superstore.csv"),
        ("Data type",    "Retail transaction records"),
        ("Granularity",  "One row per order line (product within an order)"),
        ("Time coverage","Multiple calendar years of transactions"),
    ]
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text = "Attribute", "Description"
    for cell in hdr:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for attr, desc in dataset_attrs:
        row = tbl.add_row()
        row.cells[0].text = attr
        row.cells[1].text = desc

    section_divider(doc)
    add_para(doc, "Key columns used in this project:", bold=True)
    columns = [
        "Order Date — transaction date (parsed as datetime)",
        "Ship Date — fulfilment date (used to compute Shipping Days)",
        "Customer ID — unique customer identifier",
        "Segment — Consumer, Corporate, Home Office",
        "Region, State, City — geographic attributes",
        "Category, Sub-Category — product taxonomy",
        "Product Name, Product ID — product identifiers",
        "Sales — transaction revenue ($)",
        "Quantity — units sold",
        "Discount — fractional discount applied (0–1)",
        "Profit — transaction profit ($)",
        "Ship Mode — shipping method",
    ]
    for col in columns:
        add_bullet(doc, col)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. DATA CLEANING
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "5. Data Cleaning", level=1)
    add_para(doc, (
        "Data is loaded by load_data(), which reads the CSV with encoding fallback, "
        "validates required columns, and parses Order Date and Ship Date as datetime. "
        "Data cleaning is then implemented in the clean_data() function. "
        "All transformation steps are logged in a cleaning log for full transparency. "
        "No rows are removed arbitrarily — each removal has an explicit justification."
    ))
    cleaning_steps = [
        "Exact duplicate rows: detected and removed.",
        "Null Order Date rows: removed — records without an order date cannot be placed on the time axis.",
        "Invalid Sales rows (Sales ≤ 0): removed — negative or zero revenue records are invalid for analytics.",
        "Remaining numeric nulls (Profit, Quantity, Discount): imputed with column median to preserve analytical rows.",
        "Discount values outside [0, 1]: clipped to valid range.",
        "String whitespace: stripped from all categorical columns.",
    ]
    for step in cleaning_steps:
        add_bullet(doc, step)
    add_para(doc, (
        "The cleaning log records: raw row count, duplicates removed, null-date rows removed, "
        "invalid-sales rows removed, final analytical row count, and date range. "
        "All counts are computed at runtime — none are hard-coded."
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 6. EDA
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "6. Exploratory Data Analysis", level=1)
    add_para(doc, (
        "EDA is organised around five business-focused analytical dimensions. "
        "Each chart answers a specific business question rather than serving as a 'data dump'."
    ))

    dims = [
        ("A. Overall Performance", "Sales, profit, and quantity distribution analysis."),
        ("B. Time Trends", "Monthly revenue, monthly profit, monthly profit margin, annual revenue."),
        ("C. Product Analysis", "Revenue/profit by category and sub-category; top-10 and bottom-10 products."),
        ("D. Regional Analysis", "Revenue, profit, and profit margin by region."),
        ("E. Customer Segment", "Revenue and profit margin by Consumer, Corporate, and Home Office segments."),
        ("F. Discount Analysis", "Discount rate vs profit scatter; average profit margin by discount band. Stated as association — not causation."),
        ("G. Shipping Analysis", "Order volume and profit by Ship Mode."),
    ]
    for dim, desc in dims:
        add_para(doc, f"{dim}: {desc}", bold=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. BUSINESS KPIs
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "7. Business KPI Definitions", level=1)
    add_para(doc, (
        "All KPIs are computed dynamically at runtime by calculate_kpis(), "
        "which takes the cleaned and feature-engineered dataset as input. "
        "No KPI value is hard-coded."
    ))
    kpi_defs = [
        ("Total Revenue",        "sum(Sales)"),
        ("Total Profit",         "sum(Profit)"),
        ("Total Orders",         "count(unique Order IDs) — not row count"),
        ("Total Customers",      "count(unique Customer IDs)"),
        ("Average Order Value",  "Total Revenue / Total Orders"),
        ("Profit Margin %",      "Total Profit / Total Revenue × 100"),
        ("Average Discount %",   "mean(Discount) × 100"),
        ("YoY Growth %",         "(Current Year Sales − Prior Year Sales) / Prior Year Sales × 100"),
    ]
    tbl2 = doc.add_table(rows=1, cols=2)
    tbl2.style = "Table Grid"
    hdr2 = tbl2.rows[0].cells
    hdr2[0].text, hdr2[1].text = "KPI", "Formula"
    for cell in hdr2:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for name, formula in kpi_defs:
        row = tbl2.add_row()
        row.cells[0].text = name
        row.cells[1].text = formula

    # ─────────────────────────────────────────────────────────────────────────
    # 8. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "8. Feature Engineering", level=1)
    add_para(doc, "The create_features() function derives the following analytical columns:")
    feat_table_data = [
        ("Year / Quarter / Month / Month Number", "Time decomposition for trend and seasonality analysis"),
        ("Year-Month",       "String period label for display"),
        ("Week / Day of Week","Sub-monthly time analysis"),
        ("Shipping Days",    "Ship Date − Order Date (clipped at 0)"),
        ("Profit Margin",    "Profit / Sales × 100"),
        ("Revenue per Unit", "Sales / Quantity"),
        ("Discount Band",    "Categorical grouping: No Discount, Low, Moderate, High, Very High"),
        ("Customer Order Count","Number of unique orders per Customer ID"),
        ("Repeat Customer",  "Flag: >1 order = Repeat, else One-time"),
    ]
    tbl3 = doc.add_table(rows=1, cols=2)
    tbl3.style = "Table Grid"
    hdr3 = tbl3.rows[0].cells
    hdr3[0].text, hdr3[1].text = "Feature", "Description"
    for cell in hdr3:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for feat, desc in feat_table_data:
        row = tbl3.add_row()
        row.cells[0].text = feat
        row.cells[1].text = desc

    # ─────────────────────────────────────────────────────────────────────────
    # 9. MACHINE LEARNING METHODOLOGY
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "9. Machine Learning Methodology", level=1)
    add_para(doc, (
        "The ML task is supervised regression — predicting monthly total sales "
        "as a continuous numeric target."
    ))

    add_heading(doc, "9.1 Monthly Time Series Creation", level=2)
    add_para(doc, (
        "Transaction-level data is aggregated to a monthly time series using "
        "create_monthly_series(). The target variable is total_sales (sum of Sales per month). "
        "Additional aggregated columns include total_profit, total_quantity, "
        "total_orders, unique_customers, average_order_value, and profit_margin."
    ))

    add_heading(doc, "9.2 Forecast Feature Engineering", level=2)
    add_para(doc, "create_forecast_features() creates supervised learning features:")
    feat_ml_data = [
        ("lag_1 … lag_12",    "Sales lagged 1 to 12 months"),
        ("rolling_mean_3/6/12","3-, 6-, 12-month rolling average (shifted by 1 to prevent leakage)"),
        ("month_number",       "Calendar month (1–12) — seasonality proxy"),
        ("year",               "Calendar year — trend proxy"),
        ("time_index",         "Integer index — linear trend signal"),
    ]
    for feat, desc in feat_ml_data:
        add_bullet(doc, f"{feat}: {desc}")
    add_para(doc, (
        "All lag and rolling features use only past observations. "
        "Rolling means are computed on a shifted series (.shift(1)) "
        "to ensure no current-period information is used. "
        "Rows with NaN values from the lag initialisation period are dropped."
    ), italic=True)

    add_heading(doc, "9.3 Models", level=2)
    model_descriptions = [
        ("Linear Regression", "Simple OLS baseline. Interpretable coefficients. Reveals linear trend component."),
        ("Ridge Regression",  "Linear regression with L2 regularisation (alpha=1.0). Handles correlated lag features."),
        ("Random Forest",     "Ensemble of 200 decision trees (max_depth=8, min_samples_leaf=2). Captures non-linear patterns."),
    ]
    tbl4 = doc.add_table(rows=1, cols=2)
    tbl4.style = "Table Grid"
    hdr4 = tbl4.rows[0].cells
    hdr4[0].text, hdr4[1].text = "Model", "Description"
    for cell in hdr4:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for name, desc in model_descriptions:
        row = tbl4.add_row()
        row.cells[0].text = name
        row.cells[1].text = desc

    # ─────────────────────────────────────────────────────────────────────────
    # 10. FORECASTING METHODOLOGY
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "10. Forecasting Methodology", level=1)

    add_heading(doc, "10.1 Chronological Train/Test Split", level=2)
    add_para(doc, (
        "The dataset is split chronologically: the earliest 80% of monthly observations "
        "form the training set; the latest 20% form the test set. "
        "This is implemented in chronological_split()."
    ))
    add_para(doc, (
        "WHY CHRONOLOGICAL: Random shuffling would allow the model to see future data "
        "during training, producing artificially inflated metrics that do not reflect "
        "real-world forecasting performance. Chronological splitting simulates deployment: "
        "the model is trained on the past and evaluated on genuinely unseen future observations."
    ), italic=True)

    add_heading(doc, "10.2 Model Refit on All Historical Data", level=2)
    add_para(doc, (
        "After evaluation and model selection (step 10.1), the selected model architecture "
        "is refit on ALL available historical observations using refit_selected_model(). "
        "This ensures the final forecasting model benefits from the complete dataset, not "
        "just the training split. The evaluation metrics reported in Section 11 are those "
        "computed on the held-out test set — NOT on this refit model."
    ))

    add_heading(doc, "10.3 Future Forecast Generation", level=2)
    add_para(doc, (
        "generate_future_forecast() uses the refit model to produce future monthly sales "
        "estimates. It iterates one month ahead at a time, feeding each prediction back "
        "into the lag feature window for the next period. "
        "The forecast horizon is configurable (3, 6, or 12 months). "
        "Predicted sales values are clipped at 0 (sales cannot be negative)."
    ))

    add_heading(doc, "10.4 Backtest", level=2)
    add_para(doc, (
        "generate_backtest() produces predictions on the chronological TEST PERIOD ONLY — "
        "observations the model did not see during training. It does NOT produce predictions "
        "over the training period. The backtest is displayed in the dashboard forecast chart "
        "to allow visual inspection of model accuracy on genuinely unseen data."
    ))

    add_heading(doc, "10.5 Forecast Uncertainty", level=2)
    add_para(doc, (
        "DISCLAIMER: Forecast values are estimates produced by the selected model and are "
        "subject to uncertainty. They are intended as planning aids, not guaranteed outcomes. "
        "Actual results will be influenced by market conditions, competitive dynamics, and "
        "other factors not captured in the historical sales data. "
        "Formal prediction intervals are not implemented in the current version."
    ), italic=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 11. MODEL EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "11. Model Evaluation", level=1)
    add_para(doc, "All models are evaluated on the held-out chronological test set.")
    metric_defs = [
        ("MAE (Mean Absolute Error)",      "Average dollar error per month. Lower is better.",
         "Primary selection criterion — directly answers 'how wrong is the forecast on average?'"),
        ("RMSE (Root Mean Squared Error)", "Penalises large errors more. Lower is better.",
         "Reported but not used as primary criterion to avoid over-penalising outlier months."),
        ("R² (Coefficient of Determination)", "Proportion of variance explained (0–1). Higher is better.",
         "Not used as primary criterion — can be misleading when variance is dominated by trend."),
        ("MAPE (Mean Absolute Percentage Error)", "Average % error. Lower is better.",
         "Informational — zero actual values are excluded to prevent division by zero."),
    ]
    tbl5 = doc.add_table(rows=1, cols=3)
    tbl5.style = "Table Grid"
    hdr5 = tbl5.rows[0].cells
    hdr5[0].text, hdr5[1].text, hdr5[2].text = "Metric", "Definition", "Selection Role"
    for cell in hdr5:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for name, defn, role in metric_defs:
        row = tbl5.add_row()
        row.cells[0].text = name
        row.cells[1].text = defn
        row.cells[2].text = role

    add_para(doc, (
        "The model with the lowest MAE is selected by select_best_model(). "
        "The selected architecture is then refit on all historical data via refit_selected_model() "
        "before future forecasting. The refit model is saved to models/sales_forecast_model.joblib "
        "using joblib. The payload includes: model, model name, evaluation DataFrame (test-set metrics), "
        "feature column list, dataset hash, and training timestamp. "
        "The dataset hash enables staleness detection — the dashboard warns the user if the dataset "
        "has changed since the model was trained."
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 12. FORECAST RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "12. Forecast Results", level=1)
    add_para(doc, (
        "Forecast results are presented in the dashboard Page 3 (Forecast, Risk & Action). "
        "The forecast chart displays three distinct series:"
    ))
    series = [
        "Actual Sales — full historical monthly sales values",
        "Backtest Prediction (Test Period) — model predictions on the held-out test period only "
        "(observations not used during training; displayed for visual accuracy assessment)",
        "Future Forecast — iterative one-step-ahead predictions from the refit model "
        "for the selected horizon (3, 6, or 12 months)",
    ]
    for s in series:
        add_bullet(doc, s)
    add_para(doc, (
        "The evaluation metrics (MAE, RMSE, R²) computed on the test set are displayed "
        "alongside the forecast chart. Forecast values are clearly labelled as estimates."
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 13. PROFIT INTELLIGENCE / SCENARIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "13. Profit Intelligence / Scenario Analysis", level=1)
    add_para(doc, (
        "The analyze_profitability() function examines the historical relationship "
        "between discount rate and profit margin. It provides:"
    ))
    prof_items = [
        "Discount Band Summary: average profit margin, total sales, and order count for each discount band.",
        "Hypothetical Scenarios: estimated profit margin under modified discount assumptions "
        "(current, reduced by 5pp, reduced by 10pp, increased by 5pp).",
    ]
    for item in prof_items:
        add_bullet(doc, item)
    add_para(doc, (
        "IMPORTANT DISCLAIMER: Scenario estimates are derived from a simple linear regression "
        "between average discount rate and profit margin. This is an association model only. "
        "It does NOT establish that changing the discount rate will cause the estimated profit outcome. "
        "Real outcomes depend on customer price sensitivity, competitive dynamics, cost structure, "
        "and many other factors not modelled here."
    ), italic=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 14. DASHBOARD DESIGN
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "14. Dashboard Design", level=1)
    add_para(doc, (
        "The Streamlit dashboard (dashboard.py) uses a clean executive BI design. "
        "It is organised into three pages, each answering a distinct business question."
    ))
    pages = [
        ("Page 1 — Executive Overview",
         "How healthy is the business? KPIs, monthly revenue/profit trends, "
         "category performance, YoY analysis, and 5 dynamic executive insights."),
        ("Page 2 — Sales & Product Analysis",
         "What drives sales and profit? Sub-category analysis, top/bottom products, "
         "regional performance, segment performance, discount analysis, shipping analysis, "
         "and a key-drivers narrative."),
        ("Page 3 — Forecast, Risk & Action",
         "What might happen next? Sales forecast with model metrics, "
         "interactive forecast explorer, risk identification, opportunity detection, "
         "profit scenario analysis, recommended actions, and model information."),
    ]
    for name, desc in pages:
        add_para(doc, f"{name}:", bold=True)
        add_para(doc, desc)

    add_para(doc, "Design principles applied:", bold=True)
    design_principles = [
        "KPI cards at the top of each page with dynamic values",
        "Plotly interactive charts (hover, zoom, filter)",
        "Sidebar filters: Date Range, Region, Segment, Category, Sub-Category, Ship Mode",
        "All displayed numbers are computed from the filtered dataset",
        "Clear section headings mapping to the IBM BI framework levels",
        "Disclaimer banners on all association analyses and forecasts",
        "Colour coding: green/positive = profit, red/negative = loss, purple = forecast",
    ]
    for p in design_principles:
        add_bullet(doc, p)

    # ─────────────────────────────────────────────────────────────────────────
    # 15. KEY FINDINGS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "15. Key Findings", level=1)
    add_para(doc, (
        "All findings below are generated dynamically by generate_business_insights() "
        "from the loaded dataset. The specific values will reflect the actual dataset "
        "loaded at runtime. The structure of each finding is shown below."
    ), italic=True)
    finding_templates = [
        "Total revenue across the analysis period with profit margin comparison.",
        "Year-over-year revenue growth or decline with comparison year stated.",
        "Highest-profit and lowest-profit categories with dollar values.",
        "High-discount vs low-discount profit margin comparison (association only).",
        "Top sub-category by revenue and top sub-category by profit.",
    ]
    for i, tmpl in enumerate(finding_templates, 1):
        add_bullet(doc, f"Finding {i}: {tmpl}")

    # ─────────────────────────────────────────────────────────────────────────
    # 16. BUSINESS RISKS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "16. Business Risks", level=1)
    add_para(doc, "Risks are identified by identify_risks() using defined detection rules:")
    risk_rules = [
        ("Negative Profit — Category", "High", "Any category with total profit < 0"),
        ("Negative Profit — Sub-Category", "High", "Any sub-category with total profit < 0"),
        ("High Discount + Negative Margin — Product", "Medium",
         "Products with average discount > 25% AND negative profit margin"),
        ("Declining Recent Revenue", "Medium",
         "Latest month sales > 15% below the 6-month rolling average"),
        ("Below-Average Profit — Region", "Low",
         "Region with profit < 50% of the cross-region profit average"),
    ]
    tbl6 = doc.add_table(rows=1, cols=3)
    tbl6.style = "Table Grid"
    hdr6 = tbl6.rows[0].cells
    hdr6[0].text, hdr6[1].text, hdr6[2].text = "Risk Type", "Level", "Detection Rule"
    for cell in hdr6:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for rtype, rlevel, rdef in risk_rules:
        row = tbl6.add_row()
        row.cells[0].text = rtype
        row.cells[1].text = rlevel
        row.cells[2].text = rdef

    # ─────────────────────────────────────────────────────────────────────────
    # 17. BUSINESS OPPORTUNITIES
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "17. Business Opportunities", level=1)
    add_para(doc, "Opportunities are identified by identify_opportunities() using these rules:")
    opp_rules = [
        "Top-2 categories with positive profit → High-Profit Category opportunity",
        "Top-3 sub-categories by profit margin (positive only) → High-Margin Sub-Category",
        "Regions with profit above the cross-region average → Above-Average Profit Region",
        "3-month average sales > 3-month prior average by >5% → Improving Revenue Trend",
    ]
    for r in opp_rules:
        add_bullet(doc, r)

    # ─────────────────────────────────────────────────────────────────────────
    # 18. RECOMMENDED ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "18. Recommended Actions", level=1)
    add_para(doc, (
        "Five evidence-based recommended actions are generated dynamically. "
        "They are framed as hypotheses for investigation and testing — not guaranteed outcomes."
    ))
    actions = [
        "Investigate the discount policy for sub-categories with negative or below-average profit margins.",
        "Evaluate inventory and marketing investment allocation toward the highest-profit category and sub-category.",
        "Investigate the root cause of weak regional performance; pilot targeted interventions.",
        "Monitor forecasted soft sales periods; prepare inventory and cash-flow plans accordingly.",
        "Experiment with segment-differentiated discount strategies; measure profit impact.",
    ]
    for i, action in enumerate(actions, 1):
        add_bullet(doc, f"Action {i}: {action}")

    # ─────────────────────────────────────────────────────────────────────────
    # 19. LIMITATIONS
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "19. Limitations", level=1)
    limitations = [
        "The forecasting model assumes historical patterns continue. Structural breaks (new competitors, economic shocks) are not captured.",
        "The profit scenario analysis uses a simple OLS association model. It does not establish causal relationships.",
        "External factors (marketing spend, seasonality events, macroeconomic conditions) are not incorporated.",
        "Geographic map visualisation (state-level choropleth) is not included in the current version.",
        "The customer-level repeat-purchase analysis relies on Customer ID being available in the dataset.",
        "Forecasts for low-data segments (fewer than 15 monthly observations) are intentionally withheld.",
        "The dataset is a static CSV; real-time data pipeline integration is outside the current scope.",
    ]
    for lim in limitations:
        add_bullet(doc, lim)

    # ─────────────────────────────────────────────────────────────────────────
    # 20. CONCLUSION
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "20. Conclusion", level=1)
    add_para(doc, (
        "This project demonstrates a complete, end-to-end Business Intelligence "
        "and Machine Learning pipeline built on publicly available e-commerce transaction data. "
        "Starting from raw CSV records, it delivers dynamic KPIs, focused EDA, a validated "
        "sales forecast, evidence-based risk and opportunity identification, and actionable "
        "management recommendations — all presented through a professional three-page "
        "Streamlit executive dashboard."
    ))
    add_para(doc, (
        "The project rigorously maintains analytical integrity: all metrics are computed "
        "at runtime, all associations are qualified as observed patterns rather than causal claims, "
        "and all forecast values carry explicit uncertainty disclaimers. "
        "It follows the IBM BI framework throughout, ensuring that every KPI, chart, and "
        "recommendation serves a clear business purpose."
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 21. FUTURE SCOPE
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "21. Future Scope", level=1)
    future = [
        "Integrate external signals: web traffic, marketing spend, macroeconomic indices.",
        "Explore explicit seasonality models: Facebook Prophet, SARIMA, or Theta.",
        "Build a product-level demand forecasting model for inventory planning.",
        "Implement a formal A/B test design framework for discount policy evaluation.",
        "Add a real-time data pipeline using Apache Kafka or cloud ETL tools.",
        "Develop a geographic heatmap using Plotly Choropleth for state-level sales visualisation.",
        "Introduce customer lifetime value (CLV) modelling for retention prioritisation.",
        "Deploy the Streamlit dashboard on Streamlit Cloud or IBM Cloud.",
    ]
    for f in future:
        add_bullet(doc, f)

    # ─────────────────────────────────────────────────────────────────────────
    # 22. REFERENCES
    # ─────────────────────────────────────────────────────────────────────────
    add_heading(doc, "22. References", level=1)
    refs = [
        "Tableau Sample Superstore Dataset. Available at: https://www.tableau.com/",
        "Kaggle Superstore Dataset: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final",
        "McKinney, W. (2022). Python for Data Analysis, 3rd Ed. O'Reilly.",
        "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.",
        "Streamlit Documentation. https://docs.streamlit.io/",
        "Plotly Python Documentation. https://plotly.com/python/",
        "Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd Ed.",
        "IBM SkillsBuild Data Analytics with AI Program. https://skillsbuild.org/",
    ]
    for i, ref in enumerate(refs, 1):
        add_bullet(doc, f"[{i}] {ref}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    doc.save(output_path)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Yashaswi_EcommerceSalesForecastingReport.docx",
    )
    create_report(out)
