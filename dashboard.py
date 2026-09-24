"""
=============================================================================
AI-Powered E-Commerce Sales Forecasting & Profit Optimization Dashboard
IBM SkillsBuild Data Analytics with AI Academic Internship
=============================================================================
Author  : Yashaswi
Streamlit Executive BI Dashboard — 3 Pages:
  Page 1: Executive Overview       — "How healthy is the business?"
  Page 2: Sales & Product Analysis — "What is driving sales and profit?"
  Page 3: Forecast, Risk & Action  — "What might happen next?"

ML Workflow (correctly implemented)
--------------------------------------
1. Chronological train/test split
2. Train candidates on training set
3. Evaluate candidates on HELD-OUT test set → select best (lowest MAE)
4. Refit selected model on ALL historical data
5. Backtest = predictions on test period only (not training data)
6. Future Forecast = refit-model predictions on future months
=============================================================================
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure the project root is on the path so we can import the pipeline module
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Yashaswi_EcommerceSalesForecasting import (
    DATA_PATH, MODEL_PATH,
    load_data, clean_data, create_features, calculate_kpis,
    create_monthly_series, create_forecast_features,
    train_forecast_models, evaluate_models, select_best_model,
    refit_selected_model, save_model, load_model,
    check_model_staleness, compute_dataset_hash,
    generate_backtest, generate_future_forecast,
    generate_business_insights, identify_risks, identify_opportunities,
    analyze_profitability, chronological_split, FEATURE_COLS,
)

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="E-Commerce Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Global colour palette (restrained, professional)
# =============================================================================
COLORS = {
    "primary":   "#1f4e79",
    "secondary": "#2e86ab",
    "accent":    "#f18f01",
    "positive":  "#2d6a4f",
    "negative":  "#c1121f",
    "neutral":   "#6b7280",
    "forecast":  "#7c5cd8",
    "actual":    "#1f4e79",
    "backtest":  "#e07b39",
}

CAT_COLORS = px.colors.qualitative.Set2

# =============================================================================
# Custom CSS — clean executive style
# =============================================================================
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }

    [data-testid="stSidebar"] { background-color: #1f4e79; color: white; }
    [data-testid="stSidebar"] .stMarkdown p { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #e2e8f0 !important; }

    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px 16px 16px 16px;
        border-left: 4px solid #1f4e79;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 12px; color: #6b7280; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
    }
    .kpi-value { font-size: 26px; font-weight: 700; color: #1f2328; line-height: 1.2; }
    .kpi-delta { font-size: 13px; color: #6b7280; margin-top: 4px; }
    .kpi-positive { color: #2d6a4f; }
    .kpi-negative { color: #c1121f; }

    .section-header {
        font-size: 18px; font-weight: 700; color: #1f4e79;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px; margin-top: 28px; margin-bottom: 16px;
    }
    .insight-card {
        background: #f0f7ff; border-left: 4px solid #2e86ab;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .risk-card-high {
        background: #fff0f0; border-left: 4px solid #c1121f;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .risk-card-medium {
        background: #fff8e6; border-left: 4px solid #f18f01;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .risk-card-low {
        background: #f0f4f8; border-left: 4px solid #6b7280;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .opp-card {
        background: #f0faf4; border-left: 4px solid #2d6a4f;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .action-card {
        background: #f5f0ff; border-left: 4px solid #7c5cd8;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px;
    }
    .disclaimer {
        background: #fefce8; border: 1px solid #fde68a; border-radius: 6px;
        padding: 12px 16px; font-size: 12px; color: #6b7280; margin: 12px 0;
    }
    .stale-warning {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
        padding: 12px 16px; font-size: 13px; color: #856404; margin: 10px 0;
    }
    .info-box {
        background: #f0f4f8; border-radius: 8px;
        padding: 14px 18px; font-size: 13px; color: #374151;
    }
    .page-title { font-size: 28px; font-weight: 800; color: #1f4e79; margin-bottom: 4px; }
    .page-subtitle { font-size: 14px; color: #6b7280; margin-bottom: 24px; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper: KPI card HTML
# =============================================================================
def kpi_card(label: str, value: str, delta: str = "", delta_positive: bool = None) -> str:
    delta_class = ""
    if delta_positive is True:
        delta_class = "kpi-positive"
    elif delta_positive is False:
        delta_class = "kpi-negative"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# =============================================================================
# Pipeline caching
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_pipeline(data_path: str, retrain: bool = False):
    """
    Load data, run the full analytics and ML pipeline, and cache results.

    ML Workflow
    -----------
    1. Chronological train/test split (80/20)
    2. Train Linear Regression, Ridge, Random Forest on training set
    3. Evaluate all three on HELD-OUT test set
    4. Select best model by lowest MAE
    5. Refit selected model on ALL available historical data
    6. Use refit model for future forecasting
    7. Evaluation metrics remain those from step 3 (unseen test data)
    """
    raw_df = load_data(data_path)
    clean_df, cleaning_log = clean_data(raw_df)
    feat_df  = create_features(clean_df)
    kpis     = calculate_kpis(feat_df)
    monthly  = create_monthly_series(feat_df)
    ffd      = create_forecast_features(monthly)
    fcols    = [c for c in FEATURE_COLS if c in ffd.columns]

    data_hash = compute_dataset_hash(data_path) if os.path.exists(data_path) else ""

    # Attempt to load saved model
    existing = load_model(MODEL_PATH) if not retrain else None

    if existing and not retrain:
        is_stale = check_model_staleness(existing, data_path)
        if is_stale:
            # Dataset changed — must retrain
            existing = None
    else:
        is_stale = False

    if existing and not retrain:
        best_name      = existing["model_name"]
        eval_df        = existing.get("eval_df")
        trained_at     = existing.get("trained_at", "unknown")
        # Refit on current full dataset (even when loading a saved selection)
        final_model    = refit_selected_model(best_name, ffd, fcols)
        # Rebuild eval model for backtest display
        train_df, test_df = chronological_split(ffd)
        tmp_models     = train_forecast_models(train_df, fcols)
        eval_model     = tmp_models[best_name]
    else:
        if len(ffd) < 10:
            raise ValueError(
                "Insufficient monthly data for forecasting. "
                f"Only {len(ffd)} usable observations after lag feature creation. "
                "At least 24 months of transaction data are recommended."
            )
        train_df, test_df = chronological_split(ffd)
        models     = train_forecast_models(train_df, fcols)
        eval_df    = evaluate_models(models, test_df, fcols)
        best_name, eval_model = select_best_model(eval_df, models)
        final_model = refit_selected_model(best_name, ffd, fcols)
        trained_at  = datetime.now().isoformat()
        save_model(final_model, best_name, eval_df, fcols, monthly, MODEL_PATH, data_hash)

    insights        = generate_business_insights(feat_df, kpis, monthly)
    risks           = identify_risks(feat_df, monthly)
    opportunities   = identify_opportunities(feat_df, monthly)
    profit_analysis = analyze_profitability(feat_df)

    return {
        "raw_df":        raw_df,
        "clean_df":      clean_df,
        "feat_df":       feat_df,
        "cleaning_log":  cleaning_log,
        "kpis":          kpis,
        "monthly":       monthly,
        "ffd":           ffd,
        "feature_cols":  fcols,
        "train_df":      train_df,
        "test_df":       test_df,
        "eval_df":       eval_df,
        "eval_model":    eval_model,
        "final_model":   final_model,
        "best_model_name": best_name,
        "trained_at":    trained_at,
        "is_stale":      is_stale,
        "insights":      insights,
        "risks":         risks,
        "opportunities": opportunities,
        "profit_analysis": profit_analysis,
    }


# =============================================================================
# Sidebar filter application
# =============================================================================
def apply_filters(feat_df: pd.DataFrame, f: dict) -> pd.DataFrame:
    df = feat_df.copy()
    if "date_range" in f and len(f["date_range"]) == 2:
        start, end = f["date_range"]
        df = df[(df["Order Date"].dt.date >= start) & (df["Order Date"].dt.date <= end)]
    if f.get("regions"):
        df = df[df["Region"].isin(f["regions"])]
    if f.get("segments") and "Segment" in df.columns:
        df = df[df["Segment"].isin(f["segments"])]
    if f.get("categories"):
        df = df[df["Category"].isin(f["categories"])]
    if f.get("subcategories") and "Sub-Category" in df.columns:
        df = df[df["Sub-Category"].isin(f["subcategories"])]
    if f.get("ship_modes") and "Ship Mode" in df.columns:
        df = df[df["Ship Mode"].isin(f["ship_modes"])]
    return df


# =============================================================================
# Plotly layout helper
# =============================================================================
def fig_layout(fig, title: str = "", height: int = 360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1f4e79"), x=0),
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(feat_df: pd.DataFrame) -> tuple:
    with st.sidebar:
        st.markdown("### 📊 E-Commerce BI Dashboard")
        st.markdown("*AI-Powered Sales Forecasting*")
        st.markdown("---")
        st.markdown("**Filters**")

        date_min = feat_df["Order Date"].min().date()
        date_max = feat_df["Order Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

        regions   = sorted(feat_df["Region"].unique()) if "Region" in feat_df.columns else []
        segments  = sorted(feat_df["Segment"].unique()) if "Segment" in feat_df.columns else []
        cats      = sorted(feat_df["Category"].unique()) if "Category" in feat_df.columns else []
        subcats   = sorted(feat_df["Sub-Category"].unique()) if "Sub-Category" in feat_df.columns else []
        ship_modes= sorted(feat_df["Ship Mode"].unique()) if "Ship Mode" in feat_df.columns else []

        sel_regions   = st.multiselect("Region",      regions,    default=regions)
        sel_segments  = st.multiselect("Segment",     segments,   default=segments)
        sel_cats      = st.multiselect("Category",    cats,       default=cats)
        sel_subcats   = st.multiselect("Sub-Category",subcats,    default=subcats)
        sel_ship      = st.multiselect("Ship Mode",   ship_modes, default=ship_modes)

        st.markdown("---")
        st.markdown("**Model Controls**")
        retrain_btn = st.button(
            "🔄 Retrain Model",
            help="Retrain all forecasting models from scratch on current dataset",
        )

        st.markdown("---")
        st.markdown("**Navigation**")
        page = st.radio(
            "",
            ["📈 Executive Overview",
             "🔍 Sales & Product Analysis",
             "🔮 Forecast, Risk & Action"],
            label_visibility="collapsed",
        )

    filters = {
        "date_range":    date_range,
        "regions":       sel_regions,
        "segments":      sel_segments,
        "categories":    sel_cats,
        "subcategories": sel_subcats,
        "ship_modes":    sel_ship,
    }
    return page, filters, retrain_btn


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
def page_executive_overview(filtered_df, monthly_all, insights):
    st.markdown('<div class="page-title">📈 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">How is the business performing? — KPIs, trends, and key findings</div>',
        unsafe_allow_html=True,
    )

    kpis = calculate_kpis(filtered_df)

    # --- KPI ROW ---
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    cols = st.columns(7)

    def _fmt_currency(v):
        if abs(v) >= 1_000_000:  return f"${v/1_000_000:.2f}M"
        elif abs(v) >= 1_000:    return f"${v/1_000:.1f}K"
        return f"${v:,.0f}"

    def _yoy_delta():
        g = kpis.get("YoY Growth %")
        if g is None: return "", None
        sign = "+" if g >= 0 else ""
        return f"YoY {sign}{g:.1f}% ({kpis.get('YoY Comparison','')})", g >= 0

    yoy_text, yoy_pos = _yoy_delta()

    kpi_data = [
        ("Total Revenue",   _fmt_currency(kpis["Total Revenue"]),  yoy_text, yoy_pos),
        ("Total Profit",    _fmt_currency(kpis["Total Profit"]),   f"Margin: {kpis['Profit Margin %']:.1f}%", kpis["Profit Margin %"] > 0),
        ("Total Orders",    f"{kpis['Total Orders']:,}",           "", None),
        ("Total Customers", f"{kpis['Total Customers']:,}" if kpis["Total Customers"] else "N/A", "", None),
        ("Avg Order Value", _fmt_currency(kpis["Average Order Value"]), "", None),
        ("Profit Margin",   f"{kpis['Profit Margin %']:.1f}%",    "", kpis["Profit Margin %"] > 0),
        ("Avg Discount",    f"{kpis['Average Discount %']:.1f}%", "", None),
    ]
    for col, (label, value, delta, pos) in zip(cols, kpi_data):
        col.markdown(kpi_card(label, value, delta, pos), unsafe_allow_html=True)

    # --- MONTHLY TRENDS ---
    st.markdown('<div class="section-header">Revenue & Profit Trends</div>', unsafe_allow_html=True)
    from Yashaswi_EcommerceSalesForecasting import create_monthly_series as _cms
    monthly_f = _cms(filtered_df)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_f["month_dt"], y=monthly_f["total_sales"],
            mode="lines+markers", name="Revenue",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(31,78,121,0.07)",
        ))
        fig_layout(fig, "Monthly Revenue Trend")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        colors_p = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                    for v in monthly_f["total_profit"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_f["month_dt"], y=monthly_f["total_profit"],
            marker_color=colors_p, name="Profit",
        ))
        fig_layout(fig, "Monthly Profit Trend")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # --- CATEGORY PERFORMANCE ---
    st.markdown('<div class="section-header">Category Performance</div>', unsafe_allow_html=True)
    if "Category" in filtered_df.columns:
        cat_agg = filtered_df.groupby("Category").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cat_agg, x="Category", y="total_sales",
                         title="Revenue by Category",
                         color="Category", color_discrete_sequence=CAT_COLORS,
                         text=cat_agg["total_sales"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=340)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors_cat = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                          for v in cat_agg["total_profit"]]
            fig = go.Figure(go.Bar(
                x=cat_agg["Category"], y=cat_agg["total_profit"],
                marker_color=colors_cat,
                text=cat_agg["total_profit"].apply(lambda v: f"${v/1000:.0f}K"),
                textposition="outside",
            ))
            fig_layout(fig, "Profit by Category", height=340)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- YEARLY PERFORMANCE ---
    st.markdown('<div class="section-header">Year-over-Year Performance</div>', unsafe_allow_html=True)
    if "Year" in filtered_df.columns:
        yr_agg = filtered_df.groupby("Year").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()
        yr_agg["profit_margin"] = np.where(
            yr_agg["total_sales"] != 0,
            yr_agg["total_profit"] / yr_agg["total_sales"] * 100, 0
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=yr_agg["Year"], y=yr_agg["total_sales"],
            name="Revenue", marker_color=COLORS["primary"],
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=yr_agg["Year"], y=yr_agg["profit_margin"],
            name="Profit Margin %", mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2.5),
        ), secondary_y=True)
        fig.update_layout(
            title="Annual Revenue vs Profit Margin",
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="system-ui, sans-serif", size=12),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        fig.update_yaxes(title_text="Revenue ($)", tickprefix="$", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- EXECUTIVE INSIGHTS ---
    st.markdown('<div class="section-header">Executive Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6b7280'>Dynamically generated from the loaded dataset. "
        "All associations are stated as observed patterns — not causal claims.</small>",
        unsafe_allow_html=True,
    )
    for i, finding in enumerate(insights["findings"], 1):
        st.markdown(
            f'<div class="insight-card"><strong>Finding {i}:</strong> {finding}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: SALES & PRODUCT ANALYSIS
# =============================================================================
def page_sales_product(filtered_df):
    st.markdown('<div class="page-title">🔍 Sales & Product Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">What is driving revenue and profit? — Products, regions, segments, discounts</div>',
        unsafe_allow_html=True,
    )

    # --- SUB-CATEGORY ---
    st.markdown('<div class="section-header">Sub-Category Performance</div>', unsafe_allow_html=True)
    if "Sub-Category" in filtered_df.columns:
        sc_agg = filtered_df.groupby("Sub-Category").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            sc_s = sc_agg.sort_values("total_sales", ascending=True)
            fig = px.bar(sc_s, x="total_sales", y="Sub-Category", orientation="h",
                         title="Revenue by Sub-Category",
                         color="total_sales", color_continuous_scale=["#dbeafe","#1f4e79"])
            fig.update_coloraxes(showscale=False)
            fig_layout(fig, height=420)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sc_p = sc_agg.sort_values("total_profit", ascending=True)
            colors_sc = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                         for v in sc_p["total_profit"]]
            fig = go.Figure(go.Bar(
                x=sc_p["total_profit"], y=sc_p["Sub-Category"],
                orientation="h", marker_color=colors_sc,
            ))
            fig_layout(fig, "Profit by Sub-Category", height=420)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- TOP & BOTTOM PRODUCTS ---
    st.markdown('<div class="section-header">Product-Level Analysis</div>', unsafe_allow_html=True)
    if "Product Name" in filtered_df.columns:
        prod = filtered_df.groupby("Product Name").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()
        prod["short"] = prod["Product Name"].str[:40] + "…"

        c1, c2 = st.columns(2)
        with c1:
            top10 = prod.sort_values("total_sales", ascending=True).tail(10)
            fig = px.bar(top10, x="total_sales", y="short", orientation="h",
                         title="Top 10 Products by Revenue",
                         color="total_sales", color_continuous_scale=["#bfdbfe","#1f4e79"])
            fig.update_coloraxes(showscale=False)
            fig_layout(fig, height=380)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bot10 = prod.sort_values("total_profit", ascending=True).head(10)
            colors_b = [COLORS["negative"] if v < 0 else COLORS["neutral"]
                        for v in bot10["total_profit"]]
            fig = go.Figure(go.Bar(
                x=bot10["total_profit"], y=bot10["short"],
                orientation="h", marker_color=colors_b,
            ))
            fig_layout(fig, "Bottom 10 Products by Profit", height=380)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- REGIONAL ---
    st.markdown('<div class="section-header">Regional Performance</div>', unsafe_allow_html=True)
    if "Region" in filtered_df.columns:
        reg = filtered_df.groupby("Region").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()
        reg["profit_margin"] = np.where(
            reg["total_sales"] != 0, reg["total_profit"] / reg["total_sales"] * 100, 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.pie(reg, values="total_sales", names="Region",
                         title="Revenue Share by Region",
                         color_discrete_sequence=CAT_COLORS, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors_r = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                        for v in reg["total_profit"]]
            fig = go.Figure(go.Bar(
                x=reg["Region"], y=reg["total_profit"], marker_color=colors_r,
                text=reg["total_profit"].apply(lambda v: f"${v/1000:.0f}K"),
                textposition="outside",
            ))
            fig_layout(fig, "Profit by Region", height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.bar(reg, x="Region", y="profit_margin",
                         title="Profit Margin % by Region",
                         color="profit_margin",
                         color_continuous_scale=["#fee2e2","#dcfce7"],
                         text=reg["profit_margin"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig_layout(fig, height=320)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    # --- SEGMENT ---
    st.markdown('<div class="section-header">Customer Segment Performance</div>', unsafe_allow_html=True)
    if "Segment" in filtered_df.columns:
        seg = filtered_df.groupby("Segment").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
        ).reset_index()
        seg["profit_margin"] = np.where(
            seg["total_sales"] != 0, seg["total_profit"] / seg["total_sales"] * 100, 0
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(seg, x="Segment", y="total_sales",
                         title="Revenue by Segment",
                         color="Segment", color_discrete_sequence=CAT_COLORS,
                         text=seg["total_sales"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(seg, x="Segment", y="profit_margin",
                         title="Profit Margin % by Segment",
                         color="Segment", color_discrete_sequence=CAT_COLORS,
                         text=seg["profit_margin"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    # --- DISCOUNT ANALYSIS ---
    st.markdown('<div class="section-header">Discount vs Profit Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠ The patterns below are observed associations in the dataset. '
        'They do not establish that discounting causes the observed profit outcome.</div>',
        unsafe_allow_html=True,
    )
    if "Discount Band" in filtered_df.columns:
        c1, c2 = st.columns(2)
        with c1:
            order_map = {
                "No Discount":0,"Low (0–10%)":1,
                "Moderate (11–20%)":2,"High (21–30%)":3,"Very High (>30%)":4
            }
            band_agg = filtered_df.groupby("Discount Band").agg(
                avg_margin=("Profit Margin","mean")
            ).reset_index()
            band_agg["order"] = band_agg["Discount Band"].map(order_map).fillna(99)
            band_agg = band_agg.sort_values("order")
            colors_bd = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                         for v in band_agg["avg_margin"]]
            fig = go.Figure(go.Bar(
                x=band_agg["Discount Band"], y=band_agg["avg_margin"],
                marker_color=colors_bd,
                text=band_agg["avg_margin"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_layout(fig, "Avg Profit Margin by Discount Band (Observed)", height=340)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sample = filtered_df.sample(min(2000, len(filtered_df)), random_state=42)
            fig = px.scatter(
                sample, x="Discount", y="Profit",
                opacity=0.4,
                color="Category" if "Category" in sample.columns else None,
                color_discrete_sequence=CAT_COLORS,
                title="Discount vs Profit (Sample — Observed Association)",
            )
            fig.update_traces(marker=dict(size=5))
            fig_layout(fig, height=340)
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- SHIPPING ---
    if "Ship Mode" in filtered_df.columns:
        st.markdown('<div class="section-header">Shipping Mode Analysis</div>', unsafe_allow_html=True)
        ship = filtered_df.groupby("Ship Mode").agg(
            total_sales=("Sales",  "sum"),
            total_profit=("Profit","sum"),
            order_count=("Sales",  "count"),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(ship, values="order_count", names="Ship Mode",
                         title="Order Volume by Ship Mode",
                         color_discrete_sequence=CAT_COLORS, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(ship, x="Ship Mode", y="total_profit",
                         title="Profit by Ship Mode",
                         color="Ship Mode", color_discrete_sequence=CAT_COLORS,
                         text=ship["total_profit"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- KEY DRIVERS ---
    st.markdown('<div class="section-header">Key Drivers Summary</div>', unsafe_allow_html=True)
    _key_drivers_text(filtered_df)


def _key_drivers_text(df):
    lines = []
    if "Category" in df.columns:
        cp = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top profit driver:</strong> '{cp.index[0]}' category — ${cp.iloc[0]:,.0f} total profit."
        )
    if "Region" in df.columns:
        rs = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top revenue region:</strong> '{rs.index[0]}' — ${rs.iloc[0]:,.0f} in sales."
        )
    if "Sub-Category" in df.columns:
        ss = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top sub-category by revenue:</strong> '{ss.index[0]}' — ${ss.iloc[0]:,.0f}."
        )
    if "Segment" in df.columns:
        segs = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top customer segment:</strong> '{segs.index[0]}' — ${segs.iloc[0]:,.0f} in revenue."
        )
    for line in lines:
        st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: FORECAST, RISK & ACTION
# =============================================================================
def page_forecast_risk_action(pipeline):
    st.markdown('<div class="page-title">🔮 Forecast, Risk & Action</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">What might happen next? — Forecast, risk detection, opportunities, recommended actions</div>',
        unsafe_allow_html=True,
    )

    monthly      = pipeline["monthly"]
    eval_model   = pipeline["eval_model"]
    final_model  = pipeline["final_model"]
    test_df      = pipeline["test_df"]
    eval_df      = pipeline["eval_df"]
    feature_cols = pipeline["feature_cols"]
    best_name    = pipeline["best_model_name"]
    trained_at   = pipeline["trained_at"]
    insights     = pipeline["insights"]
    risks        = pipeline["risks"]
    opportunities= pipeline["opportunities"]
    profit_analysis = pipeline["profit_analysis"]

    # Staleness warning
    if pipeline.get("is_stale"):
        st.markdown(
            '<div class="stale-warning">⚠ <strong>Model may be stale:</strong> The saved model '
            'was trained on a different version of the dataset. Click <strong>Retrain Model</strong> '
            'in the sidebar to update.</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Sales Forecast</div>', unsafe_allow_html=True)

    col_h, col_info = st.columns([1, 2])
    with col_h:
        horizon = st.selectbox(
            "Forecast Horizon",
            [3, 6, 12],
            index=1,
            format_func=lambda h: f"Next {h} months",
        )
    with col_info:
        if eval_df is not None and best_name in eval_df.index:
            m = eval_df.loc[best_name]
            st.markdown(
                f'<div class="info-box">'
                f'<strong>Model:</strong> {best_name} &nbsp;|&nbsp; '
                f'<strong>MAE:</strong> ${m["MAE"]:,.0f} &nbsp;|&nbsp; '
                f'<strong>RMSE:</strong> ${m["RMSE"]:,.0f} &nbsp;|&nbsp; '
                f'<strong>R²:</strong> {m["R2"]:.4f}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Generate backtest (test period only) + future forecast
    bt_df = generate_backtest(eval_model, test_df, feature_cols)
    fc_df = generate_future_forecast(final_model, monthly, horizon, feature_cols)

    # Build chart: all actuals + backtest (test period) + forecast
    actual_df = monthly[["month_dt","total_sales"]].copy()

    fig = go.Figure()

    # All historical actuals
    fig.add_trace(go.Scatter(
        x=actual_df["month_dt"], y=actual_df["total_sales"],
        mode="lines+markers", name="Actual Sales",
        line=dict(color=COLORS["actual"], width=2.5),
    ))

    # Backtest — test period only
    fig.add_trace(go.Scatter(
        x=bt_df["month_dt"], y=bt_df["predicted"],
        mode="lines+markers", name="Backtest Prediction (Test Period)",
        line=dict(color=COLORS["backtest"], width=2, dash="dash"),
        marker=dict(size=7, symbol="circle-open"),
    ))

    # Future forecast
    if len(fc_df) > 0:
        fig.add_trace(go.Scatter(
            x=fc_df["month_dt"], y=fc_df["predicted"],
            mode="lines+markers", name=f"Future Forecast ({horizon}M)",
            line=dict(color=COLORS["forecast"], width=2.5, dash="longdash"),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig.add_vrect(
            x0=fc_df["month_dt"].min(), x1=fc_df["month_dt"].max(),
            fillcolor="rgba(124,92,216,0.07)", layer="below", line_width=0,
        )

    # Highlight test period window
    if len(test_df) > 0:
        fig.add_vrect(
            x0=test_df["month_dt"].min(), x1=test_df["month_dt"].max(),
            fillcolor="rgba(224,123,57,0.08)", layer="below", line_width=0,
            annotation_text="Test Period", annotation_position="top left",
        )

    fig_layout(fig, "Sales Forecast — Actual Sales · Backtest Prediction · Future Forecast", height=440)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="disclaimer">'
        '⚠ <strong>Forecast Disclaimer:</strong> '
        f'Future Forecast values are estimates produced by the {best_name} model '
        '(refit on all available historical data). They are subject to uncertainty '
        'and should not be treated as guaranteed business outcomes. '
        'The Backtest Prediction shows model performance on the held-out chronological '
        'test period — observations not used during training.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Future forecast table
    if len(fc_df) > 0:
        with st.expander("View Future Forecast Values"):
            disp = fc_df[["month_dt","predicted"]].copy()
            disp.columns = ["Month", "Forecast Sales ($)"]
            disp["Month"] = disp["Month"].dt.strftime("%B %Y")
            disp["Forecast Sales ($)"] = disp["Forecast Sales ($)"].apply(lambda v: f"${v:,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # Model comparison
    if eval_df is not None:
        with st.expander("Model Comparison — Evaluation Metrics (Held-Out Test Set)"):
            st.markdown("""
**Metric Definitions:**
- **MAE** (Mean Absolute Error): Average absolute dollar error per month on the test set — lower is better. *Primary selection criterion.*
- **RMSE** (Root Mean Squared Error): Penalises large forecast errors more — lower is better.
- **R²** (Coefficient of Determination): Proportion of variance explained — higher is better (max 1.0). Not used as primary criterion.
- **MAPE** (Mean Absolute Percentage Error): Average % error — lower is better.

*All metrics computed on the chronological held-out test set — observations the model did NOT see during training.*
*The final model is then refit on all available historical data before future forecasting.*
            """)
            disp_ev = eval_df.copy().reset_index()
            for col in ["MAE","RMSE"]:
                disp_ev[col] = disp_ev[col].apply(lambda v: f"${v:,.0f}")
            disp_ev["R2"]   = disp_ev["R2"].apply(lambda v: f"{v:.4f}")
            disp_ev["MAPE"] = disp_ev["MAPE"].apply(lambda v: f"{v:.2f}%" if not pd.isna(v) else "N/A")
            st.dataframe(disp_ev, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE FORECAST EXPLORER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔭 Interactive Forecast Explorer</div>', unsafe_allow_html=True)
    with st.expander("Explore Forecasts by Category or Region"):
        st.markdown(
            "**Note:** Segmented forecasts require at least 15 monthly observations per segment. "
            "If fewer are available, the forecast will not be generated."
        )
        seg_type = st.radio("Segment by", ["Category", "Region"], horizontal=True)
        feat_df  = pipeline["feat_df"]

        if seg_type in feat_df.columns:
            seg_options = sorted(feat_df[seg_type].unique().tolist())
            chosen      = st.selectbox(f"Select {seg_type}", seg_options)
            seg_df      = feat_df[feat_df[seg_type] == chosen].copy()

            from Yashaswi_EcommerceSalesForecasting import create_monthly_series as _cms
            seg_monthly = _cms(seg_df)
            from Yashaswi_EcommerceSalesForecasting import create_forecast_features as _cff
            seg_ffd     = _cff(seg_monthly)
            seg_fcols   = [c for c in feature_cols if c in seg_ffd.columns]

            MIN_OBS = 15
            if len(seg_ffd) < MIN_OBS:
                st.warning(
                    f"Only {len(seg_ffd)} monthly observations available for '{chosen}'. "
                    f"At least {MIN_OBS} are required for a reliable forecast. "
                    "Please select a different segment or expand the date range."
                )
            else:
                seg_train, seg_test = chronological_split(seg_ffd)
                seg_models  = train_forecast_models(seg_train, seg_fcols)
                seg_eval    = evaluate_models(seg_models, seg_test, seg_fcols)
                seg_bname, seg_eval_model = select_best_model(seg_eval, seg_models)
                seg_final   = refit_selected_model(seg_bname, seg_ffd, seg_fcols)

                seg_bt = generate_backtest(seg_eval_model, seg_test, seg_fcols)
                seg_fc = generate_future_forecast(seg_final, seg_monthly, horizon, seg_fcols)

                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=seg_monthly["month_dt"], y=seg_monthly["total_sales"],
                    mode="lines+markers", name="Actual Sales",
                    line=dict(color=COLORS["actual"], width=2),
                ))
                fig_s.add_trace(go.Scatter(
                    x=seg_bt["month_dt"], y=seg_bt["predicted"],
                    mode="lines+markers", name="Backtest Prediction",
                    line=dict(color=COLORS["backtest"], width=2, dash="dash"),
                ))
                if len(seg_fc) > 0:
                    fig_s.add_trace(go.Scatter(
                        x=seg_fc["month_dt"], y=seg_fc["predicted"],
                        mode="lines+markers", name="Future Forecast",
                        line=dict(color=COLORS["forecast"], width=2.5, dash="longdash"),
                    ))
                fig_layout(fig_s, f"Sales Forecast — {chosen}", height=360)
                fig_s.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig_s, use_container_width=True)
                st.caption(
                    f"Model: {seg_bname} | "
                    f"MAE: ${seg_eval.loc[seg_bname,'MAE']:,.0f} | "
                    f"R²: {seg_eval.loc[seg_bname,'R2']:.4f} (test set)"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # RISK
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠ Business Risks</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6b7280'>Each risk is identified using an explicitly defined detection rule "
        "applied to the loaded dataset.</small>",
        unsafe_allow_html=True,
    )
    if risks:
        for risk in risks:
            level = risk.get("risk_level", "Low")
            css_class = f"risk-card-{level.lower()}"
            badge = {"High":"🔴","Medium":"🟡","Low":"🔵"}.get(level,"⚪")
            st.markdown(
                f'<div class="{css_class}">'
                f'{badge} <strong>[{level}]</strong> {risk["type"]} — '
                f'<em>{risk["entity"]}</em><br>'
                f'<small>{risk["detail"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No significant risks detected in the current dataset.")

    # ─────────────────────────────────────────────────────────────────────────
    # OPPORTUNITY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Business Opportunities</div>', unsafe_allow_html=True)
    if opportunities:
        for opp in opportunities:
            st.markdown(
                f'<div class="opp-card">'
                f'✅ <strong>{opp["type"]}</strong> — <em>{opp["entity"]}</em><br>'
                f'<small>{opp["detail"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No distinct opportunities identified in the current dataset.")

    # ─────────────────────────────────────────────────────────────────────────
    # PROFIT SCENARIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Profit Scenario Analysis (Illustrative)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠ <strong>Disclaimer:</strong> '
        + profit_analysis.get("caveats", "")
        + '</div>',
        unsafe_allow_html=True,
    )
    if "discount_band_summary" in profit_analysis:
        band_sum = profit_analysis["discount_band_summary"].copy()
        order_map = {
            "No Discount":0,"Low (0–10%)":1,
            "Moderate (11–20%)":2,"High (21–30%)":3,"Very High (>30%)":4
        }
        band_sum["order"] = band_sum["Discount Band"].map(order_map).fillna(99)
        band_sum = band_sum.sort_values("order")

        c1, c2 = st.columns(2)
        with c1:
            colors_b = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                        for v in band_sum["profit_margin"]]
            fig = go.Figure(go.Bar(
                x=band_sum["Discount Band"], y=band_sum["profit_margin"],
                marker_color=colors_b,
                text=band_sum["profit_margin"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_layout(fig, "Historical: Profit Margin by Discount Band", height=340)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "scenario_estimates" in profit_analysis:
                st.markdown("**Illustrative Discount Scenarios**")
                st.dataframe(
                    profit_analysis["scenario_estimates"],
                    use_container_width=True, hide_index=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMENDED ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Recommended Actions</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6b7280'>These recommendations are evidence-based hypotheses for "
        "management investigation and testing. They are not guaranteed outcomes.</small>",
        unsafe_allow_html=True,
    )
    for i, rec in enumerate(insights["recommendations"], 1):
        st.markdown(
            f'<div class="action-card">🔷 <strong>Action {i}:</strong> {rec}</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DATA & MODEL INFORMATION
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ℹ Data & Model Information</div>', unsafe_allow_html=True)
    log = pipeline["cleaning_log"]
    info_items = [
        f"<strong>Dataset:</strong> {DATA_PATH}",
        f"<strong>Raw Rows:</strong> {log['raw_row_count']:,}",
        f"<strong>Analytical Rows:</strong> {log['final_row_count']:,}",
        f"<strong>Date Range:</strong> {log['date_range_start']} – {log['date_range_end']}",
        f"<strong>Selected Model:</strong> {best_name}",
        f"<strong>Model Trained At:</strong> {trained_at}",
    ]
    if eval_df is not None and best_name in eval_df.index:
        m = eval_df.loc[best_name]
        info_items += [
            f"<strong>Test MAE:</strong> ${m['MAE']:,.0f}",
            f"<strong>Test RMSE:</strong> ${m['RMSE']:,.0f}",
            f"<strong>Test R²:</strong> {m['R2']:.4f}",
        ]
    info_items.append(f"<strong>Forecast Horizon:</strong> {horizon} months")
    st.markdown(
        '<div class="info-box">' + "<br>".join(info_items) + "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    try:
        pipeline = st.session_state.get("pipeline")
        retrain_requested = st.session_state.get("retrain_requested", False)

        if pipeline is None or retrain_requested:
            if retrain_requested:
                st.session_state["retrain_requested"] = False
                load_pipeline.clear()

            with st.spinner("Loading data and running analytics pipeline …"):
                pipeline = load_pipeline(DATA_PATH, retrain=retrain_requested)
                st.session_state["pipeline"] = pipeline

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(f"Data validation error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        with st.expander("Technical details (for debugging)"):
            st.code(traceback.format_exc())
        st.stop()

    feat_df = pipeline["feat_df"]
    page, filters, retrain_btn = render_sidebar(feat_df)

    if retrain_btn:
        st.session_state["retrain_requested"] = True
        st.rerun()

    filtered_df = apply_filters(feat_df, filters)
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust the filter selections.")
        st.stop()

    # Dataset summary expander
    log = pipeline["cleaning_log"]
    with st.expander("📋 Dataset Summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows",       f"{log['raw_row_count']:,}")
        c2.metric("Analytical Rows",f"{log['final_row_count']:,}")
        c3.metric("Date Start",     log["date_range_start"])
        c4.metric("Date End",       log["date_range_end"])
        st.caption(
            f"Duplicates removed: {log.get('duplicate_rows_removed',0)} | "
            f"Invalid sales removed: {log.get('invalid_sales_removed',0)} | "
            f"Null order-date removed: {log.get('null_order_date_removed',0)}"
        )

    if page == "📈 Executive Overview":
        page_executive_overview(filtered_df, pipeline["monthly"], pipeline["insights"])
    elif page == "🔍 Sales & Product Analysis":
        page_sales_product(filtered_df)
    elif page == "🔮 Forecast, Risk & Action":
        page_forecast_risk_action(pipeline)


if __name__ == "__main__":
    main()
