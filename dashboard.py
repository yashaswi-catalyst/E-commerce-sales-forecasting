"""
=============================================================================
AI-Powered E-Commerce Sales Forecasting & Profit Optimization Dashboard
IBM SkillsBuild Data Analytics with AI Academic Internship
=============================================================================
Streamlit Executive BI Dashboard — 3 Pages:
  Page 1: Executive Overview      — "How healthy is the business?"
  Page 2: Sales & Product Analysis— "What is driving sales and profit?"
  Page 3: Forecast, Risk & Action — "What might happen next?"
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

from YourName_EcommerceSalesForecasting import (
    DATA_PATH, MODEL_PATH,
    load_data, clean_data, create_features, calculate_kpis,
    create_monthly_series, create_forecast_features,
    train_forecast_models, evaluate_models, select_best_model,
    save_model, load_model, generate_forecast,
    generate_business_insights, identify_risks, identify_opportunities,
    analyze_profitability, chronological_split, FEATURE_COLS,
)

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="E-Commerce BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Global colour palette (restrained, professional)
# =============================================================================
COLORS = {
    "primary":    "#1f4e79",
    "secondary":  "#2e86ab",
    "accent":     "#f18f01",
    "positive":   "#2d6a4f",
    "negative":   "#c1121f",
    "neutral":    "#6b7280",
    "bg_card":    "#f0f4f8",
    "forecast":   "#7c5cd8",
    "actual":     "#1f4e79",
    "backtest":   "#2e86ab",
}

CAT_COLORS = px.colors.qualitative.Set2

# =============================================================================
# Custom CSS — clean executive style
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1f4e79; color: white; }
    [data-testid="stSidebar"] .stMarkdown p { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #e2e8f0 !important; }

    /* KPI card style */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 20px 16px 16px 16px;
        border-left: 4px solid #1f4e79;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 12px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #1f2328;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 13px;
        color: #6b7280;
        margin-top: 4px;
    }
    .kpi-positive { color: #2d6a4f; }
    .kpi-negative { color: #c1121f; }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1f4e79;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px;
        margin-top: 28px;
        margin-bottom: 16px;
    }

    /* Insight card */
    .insight-card {
        background: #f0f7ff;
        border-left: 4px solid #2e86ab;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #1f2328;
    }

    /* Risk card */
    .risk-card-high {
        background: #fff0f0;
        border-left: 4px solid #c1121f;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .risk-card-medium {
        background: #fff8e6;
        border-left: 4px solid #f18f01;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .risk-card-low {
        background: #f0f4f8;
        border-left: 4px solid #6b7280;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Opportunity card */
    .opp-card {
        background: #f0faf4;
        border-left: 4px solid #2d6a4f;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Action card */
    .action-card {
        background: #f5f0ff;
        border-left: 4px solid #7c5cd8;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 12px;
        color: #6b7280;
        margin: 12px 0;
    }

    /* Data info box */
    .info-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 13px;
        color: #374151;
    }

    /* Page title */
    .page-title {
        font-size: 28px;
        font-weight: 800;
        color: #1f4e79;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 24px;
    }

    /* Hide Streamlit default chrome */
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
# Cache the pipeline — only re-runs when data changes
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_pipeline(data_path: str, retrain: bool = False):
    """
    Load data, run the full analytics pipeline, and cache the results.
    Re-trains the model only if retrain=True or no saved model exists.
    """
    raw_df = load_data(data_path)
    clean_df, cleaning_log = clean_data(raw_df)
    feat_df = create_features(clean_df)
    kpis    = calculate_kpis(feat_df)
    monthly = create_monthly_series(feat_df)
    forecast_feat_df = create_forecast_features(monthly)
    feature_cols = [c for c in FEATURE_COLS if c in forecast_feat_df.columns]

    # Model loading or training
    existing = load_model(MODEL_PATH) if not retrain else None
    if existing and not retrain:
        best_model    = existing["model"]
        best_name     = existing["model_name"]
        eval_df       = existing.get("eval_df")
        trained_at    = existing.get("trained_at", "unknown")
    else:
        if len(forecast_feat_df) < 10:
            raise ValueError(
                "Insufficient monthly data for forecasting. "
                f"Only {len(forecast_feat_df)} usable rows. "
                "At least 24 months of data are recommended."
            )
        train_df, test_df = chronological_split(forecast_feat_df)
        models      = train_forecast_models(train_df, feature_cols)
        eval_df     = evaluate_models(models, test_df, feature_cols)
        best_name, best_model = select_best_model(eval_df, models)
        save_model(best_model, best_name, eval_df, feature_cols, monthly, MODEL_PATH)
        trained_at  = datetime.now().isoformat()

    insights        = generate_business_insights(feat_df, kpis, monthly)
    risks           = identify_risks(feat_df, monthly)
    opportunities   = identify_opportunities(feat_df, monthly)
    profit_analysis = analyze_profitability(feat_df)

    return {
        "raw_df":          raw_df,
        "clean_df":        clean_df,
        "feat_df":         feat_df,
        "cleaning_log":    cleaning_log,
        "kpis":            kpis,
        "monthly":         monthly,
        "forecast_feat_df":forecast_feat_df,
        "feature_cols":    feature_cols,
        "eval_df":         eval_df,
        "best_model_name": best_name,
        "best_model":      best_model,
        "trained_at":      trained_at,
        "insights":        insights,
        "risks":           risks,
        "opportunities":   opportunities,
        "profit_analysis": profit_analysis,
    }


# =============================================================================
# Apply sidebar filters to the feature DataFrame
# =============================================================================
def apply_filters(feat_df: pd.DataFrame, sidebar_filters: dict) -> pd.DataFrame:
    df = feat_df.copy()
    if "date_range" in sidebar_filters:
        start, end = sidebar_filters["date_range"]
        df = df[(df["Order Date"].dt.date >= start) & (df["Order Date"].dt.date <= end)]
    if sidebar_filters.get("regions"):
        df = df[df["Region"].isin(sidebar_filters["regions"])]
    if sidebar_filters.get("segments") and "Segment" in df.columns:
        df = df[df["Segment"].isin(sidebar_filters["segments"])]
    if sidebar_filters.get("categories"):
        df = df[df["Category"].isin(sidebar_filters["categories"])]
    if sidebar_filters.get("subcategories") and "Sub-Category" in df.columns:
        df = df[df["Sub-Category"].isin(sidebar_filters["subcategories"])]
    if sidebar_filters.get("ship_modes") and "Ship Mode" in df.columns:
        df = df[df["Ship Mode"].isin(sidebar_filters["ship_modes"])]
    return df


# =============================================================================
# Plotly chart helpers
# =============================================================================
def fig_layout(fig, title: str = "", height: int = 360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1f4e79"), x=0),
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
    )
    return fig


def bar_chart(df, x, y, title, color=None, orientation="v", height=360):
    if orientation == "h":
        fig = px.bar(df, x=y, y=x, orientation="h", title=title,
                     color=color, color_discrete_sequence=CAT_COLORS)
    else:
        fig = px.bar(df, x=x, y=y, title=title,
                     color=color, color_discrete_sequence=CAT_COLORS)
    return fig_layout(fig, height=height)


def line_chart(df, x, y, title, color=None, height=360):
    fig = px.line(df, x=x, y=y, title=title,
                  color=color, color_discrete_sequence=CAT_COLORS,
                  markers=True)
    return fig_layout(fig, height=height)


def pie_chart(df, values, names, title, height=360):
    fig = px.pie(df, values=values, names=names, title=title,
                 color_discrete_sequence=CAT_COLORS, hole=0.4)
    fig.update_traces(textinfo="percent+label")
    return fig_layout(fig, height=height)


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(feat_df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("### 📊 E-Commerce BI")
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

        regions = sorted(feat_df["Region"].unique().tolist()) if "Region" in feat_df.columns else []
        sel_regions = st.multiselect("Region", regions, default=regions)

        segments = sorted(feat_df["Segment"].unique().tolist()) if "Segment" in feat_df.columns else []
        sel_segments = st.multiselect("Segment", segments, default=segments)

        categories = sorted(feat_df["Category"].unique().tolist()) if "Category" in feat_df.columns else []
        sel_categories = st.multiselect("Category", categories, default=categories)

        subcats = sorted(feat_df["Sub-Category"].unique().tolist()) if "Sub-Category" in feat_df.columns else []
        sel_subcats = st.multiselect("Sub-Category", subcats, default=subcats)

        ship_modes = sorted(feat_df["Ship Mode"].unique().tolist()) if "Ship Mode" in feat_df.columns else []
        sel_ship = st.multiselect("Ship Mode", ship_modes, default=ship_modes)

        st.markdown("---")
        st.markdown("**Model Controls**")
        retrain_btn = st.button("🔄 Retrain Model", help="Retrain all forecasting models from scratch")

        st.markdown("---")
        st.markdown("**Pages**")
        page = st.radio(
            "",
            ["📈 Executive Overview", "🔍 Sales & Product Analysis", "🔮 Forecast, Risk & Action"],
            label_visibility="collapsed",
        )

    filters = {
        "date_range":  date_range if len(date_range) == 2 else (date_min, date_max),
        "regions":     sel_regions,
        "segments":    sel_segments,
        "categories":  sel_categories,
        "subcategories": sel_subcats,
        "ship_modes":  sel_ship,
    }
    return page, filters, retrain_btn


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
def page_executive_overview(filtered_df, kpis_all, monthly, insights):
    st.markdown('<div class="page-title">📈 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">How is the business performing? — KPIs, trends, and key findings</div>',
        unsafe_allow_html=True,
    )

    # Recalculate KPIs on filtered data
    from YourName_EcommerceSalesForecasting import calculate_kpis as _ck
    kpis = _ck(filtered_df)

    # --- KPI ROW ---
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    cols = st.columns(7)

    def _fmt_currency(v):
        if v >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        elif v >= 1_000:
            return f"${v/1_000:.1f}K"
        return f"${v:,.0f}"

    def _yoy_delta(kpis):
        g = kpis.get("YoY Growth %")
        if g is None:
            return "", None
        sign = "+" if g >= 0 else ""
        return f"YoY {sign}{g:.1f}% ({kpis.get('YoY Comparison','')})", g >= 0

    yoy_text, yoy_pos = _yoy_delta(kpis)

    kpi_data = [
        ("Total Revenue",    _fmt_currency(kpis["Total Revenue"]),   yoy_text, yoy_pos),
        ("Total Profit",     _fmt_currency(kpis["Total Profit"]),    f"Margin: {kpis['Profit Margin %']:.1f}%", kpis["Profit Margin %"] > 0),
        ("Total Orders",     f"{kpis['Total Orders']:,}",            "", None),
        ("Total Customers",  f"{kpis['Total Customers']:,}" if kpis["Total Customers"] else "N/A", "", None),
        ("Avg Order Value",  _fmt_currency(kpis["Average Order Value"]), "", None),
        ("Profit Margin",    f"{kpis['Profit Margin %']:.1f}%",      "", kpis["Profit Margin %"] > 0),
        ("Avg Discount",     f"{kpis['Average Discount %']:.1f}%",   "", None),
    ]
    for col, (label, value, delta, pos) in zip(cols, kpi_data):
        col.markdown(kpi_card(label, value, delta, pos), unsafe_allow_html=True)

    # --- MONTHLY TRENDS ---
    st.markdown('<div class="section-header">Revenue & Profit Trends</div>', unsafe_allow_html=True)

    # Recompute monthly from filtered data
    from YourName_EcommerceSalesForecasting import create_monthly_series as _cms
    monthly_filtered = _cms(filtered_df)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_filtered["month_dt"],
            y=monthly_filtered["total_sales"],
            mode="lines+markers",
            name="Revenue",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31,78,121,0.08)",
        ))
        fig_layout(fig, "Monthly Revenue Trend")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        colors_profit = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                         for v in monthly_filtered["total_profit"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_filtered["month_dt"],
            y=monthly_filtered["total_profit"],
            marker_color=colors_profit,
            name="Profit",
        ))
        fig_layout(fig, "Monthly Profit Trend")
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # --- CATEGORY PERFORMANCE ---
    st.markdown('<div class="section-header">Category Performance</div>', unsafe_allow_html=True)
    if "Category" in filtered_df.columns:
        cat_agg = filtered_df.groupby("Category").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        cat_agg["profit_margin"] = np.where(
            cat_agg["total_sales"] != 0,
            cat_agg["total_profit"] / cat_agg["total_sales"] * 100, 0
        )

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
                marker_color=colors_cat, text=cat_agg["total_profit"].apply(
                    lambda v: f"${v/1000:.0f}K"),
                textposition="outside",
            ))
            fig_layout(fig, "Profit by Category", height=340)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- YEARLY PERFORMANCE ---
    st.markdown('<div class="section-header">Year-over-Year Performance</div>', unsafe_allow_html=True)
    if "Year" in filtered_df.columns:
        yr_agg = filtered_df.groupby("Year").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
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
        "Associations are stated as observed patterns — not causal claims.</small>",
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
        '<div class="page-subtitle">What is driving revenue and profit? — Products, regions, segments, and discounts</div>',
        unsafe_allow_html=True,
    )

    # --- SUB-CATEGORY ANALYSIS ---
    st.markdown('<div class="section-header">Sub-Category Performance</div>', unsafe_allow_html=True)
    if "Sub-Category" in filtered_df.columns:
        sc_agg = filtered_df.groupby("Sub-Category").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        sc_agg["profit_margin"] = np.where(
            sc_agg["total_sales"] != 0,
            sc_agg["total_profit"] / sc_agg["total_sales"] * 100, 0
        )
        sc_sales_sorted  = sc_agg.sort_values("total_sales", ascending=True)
        sc_profit_sorted = sc_agg.sort_values("total_profit", ascending=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(sc_sales_sorted, x="total_sales", y="Sub-Category",
                         orientation="h", title="Revenue by Sub-Category",
                         color="total_sales",
                         color_continuous_scale=["#dbeafe","#1f4e79"])
            fig_layout(fig, height=420)
            fig.update_coloraxes(showscale=False)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors_sc = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                         for v in sc_profit_sorted["total_profit"]]
            fig = go.Figure(go.Bar(
                x=sc_profit_sorted["total_profit"],
                y=sc_profit_sorted["Sub-Category"],
                orientation="h",
                marker_color=colors_sc,
            ))
            fig_layout(fig, "Profit by Sub-Category", height=420)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- TOP & BOTTOM PRODUCTS ---
    st.markdown('<div class="section-header">Product-Level Analysis</div>', unsafe_allow_html=True)
    if "Product Name" in filtered_df.columns:
        prod_agg = filtered_df.groupby("Product Name").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        prod_agg["short_name"] = prod_agg["Product Name"].str[:40] + "…"

        top10_sales   = prod_agg.sort_values("total_sales", ascending=False).head(10)
        top10_profit  = prod_agg.sort_values("total_profit", ascending=False).head(10)
        worst10_profit= prod_agg.sort_values("total_profit", ascending=True).head(10)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(top10_sales.sort_values("total_sales"),
                         x="total_sales", y="short_name", orientation="h",
                         title="Top 10 Products by Revenue",
                         color="total_sales",
                         color_continuous_scale=["#bfdbfe","#1f4e79"])
            fig.update_coloraxes(showscale=False)
            fig_layout(fig, height=380)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors_wp = [COLORS["negative"] if v < 0 else COLORS["neutral"]
                         for v in worst10_profit["total_profit"]]
            fig = go.Figure(go.Bar(
                x=worst10_profit["total_profit"],
                y=worst10_profit["short_name"],
                orientation="h",
                marker_color=colors_wp,
            ))
            fig_layout(fig, "Bottom 10 Products by Profit", height=380)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- REGIONAL ANALYSIS ---
    st.markdown('<div class="section-header">Regional Performance</div>', unsafe_allow_html=True)
    if "Region" in filtered_df.columns:
        reg_agg = filtered_df.groupby("Region").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        reg_agg["profit_margin"] = np.where(
            reg_agg["total_sales"] != 0,
            reg_agg["total_profit"] / reg_agg["total_sales"] * 100, 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.pie(reg_agg, values="total_sales", names="Region",
                         title="Revenue Share by Region",
                         color_discrete_sequence=CAT_COLORS, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            colors_reg = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                          for v in reg_agg["total_profit"]]
            fig = go.Figure(go.Bar(
                x=reg_agg["Region"], y=reg_agg["total_profit"],
                marker_color=colors_reg,
                text=reg_agg["total_profit"].apply(lambda v: f"${v/1000:.0f}K"),
                textposition="outside",
            ))
            fig_layout(fig, "Profit by Region", height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.bar(reg_agg, x="Region", y="profit_margin",
                         title="Profit Margin % by Region",
                         color="profit_margin",
                         color_continuous_scale=["#fee2e2","#dcfce7"],
                         text=reg_agg["profit_margin"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig_layout(fig, height=320)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    # --- SEGMENT ANALYSIS ---
    st.markdown('<div class="section-header">Customer Segment Performance</div>', unsafe_allow_html=True)
    if "Segment" in filtered_df.columns:
        seg_agg = filtered_df.groupby("Segment").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        seg_agg["profit_margin"] = np.where(
            seg_agg["total_sales"] != 0,
            seg_agg["total_profit"] / seg_agg["total_sales"] * 100, 0
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(seg_agg, x="Segment", y="total_sales",
                         title="Revenue by Segment",
                         color="Segment", color_discrete_sequence=CAT_COLORS,
                         text=seg_agg["total_sales"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(seg_agg, x="Segment", y="profit_margin",
                         title="Profit Margin % by Segment",
                         color="Segment", color_discrete_sequence=CAT_COLORS,
                         text=seg_agg["profit_margin"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    # --- DISCOUNT ANALYSIS ---
    st.markdown('<div class="section-header">Discount vs Profit Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">⚠ The pattern shown below is an observed association in the dataset. '
        'It does not establish that discounting causes the profit outcome shown.</div>',
        unsafe_allow_html=True,
    )

    if "Discount" in filtered_df.columns and "Discount Band" in filtered_df.columns:
        c1, c2 = st.columns(2)
        with c1:
            band_agg = filtered_df.groupby("Discount Band").agg(
                avg_margin=("Profit Margin", "mean"),
                count=("Sales", "count"),
            ).reset_index()
            order_map = {
                "No Discount":0,"Low (0–10%)":1,
                "Moderate (11–20%)":2,"High (21–30%)":3,"Very High (>30%)":4
            }
            band_agg["order"] = band_agg["Discount Band"].map(order_map).fillna(99)
            band_agg = band_agg.sort_values("order")
            colors_band = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                           for v in band_agg["avg_margin"]]
            fig = go.Figure(go.Bar(
                x=band_agg["Discount Band"], y=band_agg["avg_margin"],
                marker_color=colors_band,
                text=band_agg["avg_margin"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_layout(fig, "Avg Profit Margin by Discount Band", height=340)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sample = filtered_df.sample(min(2000, len(filtered_df)), random_state=42)
            fig = px.scatter(
                sample, x="Discount", y="Profit",
                opacity=0.4,
                color="Category" if "Category" in sample.columns else None,
                color_discrete_sequence=CAT_COLORS,
                title="Discount vs Profit (Sample)",
                labels={"Discount": "Discount Rate", "Profit": "Transaction Profit"},
            )
            fig.update_traces(marker=dict(size=5))
            fig_layout(fig, height=340)
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- SHIPPING ANALYSIS ---
    if "Ship Mode" in filtered_df.columns:
        st.markdown('<div class="section-header">Shipping Mode Analysis</div>', unsafe_allow_html=True)
        ship_agg = filtered_df.groupby("Ship Mode").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Sales", "count"),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(ship_agg, values="order_count", names="Ship Mode",
                         title="Order Volume by Ship Mode",
                         color_discrete_sequence=CAT_COLORS, hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(ship_agg, x="Ship Mode", y="total_profit",
                         title="Profit by Ship Mode",
                         color="Ship Mode", color_discrete_sequence=CAT_COLORS,
                         text=ship_agg["total_profit"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig_layout(fig, height=320)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # --- KEY DRIVERS ---
    st.markdown('<div class="section-header">Key Drivers Summary</div>', unsafe_allow_html=True)
    _key_drivers_text(filtered_df)


def _key_drivers_text(df):
    """Generate a brief dynamic key-drivers narrative."""
    lines = []

    if "Category" in df.columns:
        cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        best = cat_profit.index[0]
        lines.append(
            f"<strong>Top profit driver:</strong> '{best}' category — "
            f"${cat_profit.iloc[0]:,.0f} total profit."
        )

    if "Region" in df.columns:
        reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top revenue region:</strong> '{reg_sales.index[0]}' — "
            f"${reg_sales.iloc[0]:,.0f} in sales."
        )

    if "Sub-Category" in df.columns:
        sc_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top sub-category by revenue:</strong> '{sc_sales.index[0]}' — "
            f"${sc_sales.iloc[0]:,.0f}."
        )

    if "Segment" in df.columns:
        seg_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
        lines.append(
            f"<strong>Top customer segment:</strong> '{seg_sales.index[0]}' — "
            f"${seg_sales.iloc[0]:,.0f} in revenue."
        )

    for line in lines:
        st.markdown(f'<div class="insight-card">{line}</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: FORECAST, RISK & ACTION
# =============================================================================
def page_forecast_risk_action(filtered_df, monthly_all, best_model, best_name,
                               eval_df, feature_cols, forecast_feat_df,
                               insights, risks, opportunities,
                               profit_analysis, trained_at):

    st.markdown('<div class="page-title">🔮 Forecast, Risk & Action</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'What might happen next? — Forecast, risk detection, opportunities, and recommended actions'
        '</div>',
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST SECTION
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

    # Generate forecast with selected horizon
    forecast_df = generate_forecast(
        best_model, forecast_feat_df, monthly_all, horizon=horizon,
        feature_cols=feature_cols,
    )

    # Forecast chart
    fig = go.Figure()
    actual_rows   = forecast_df[forecast_df["type"] == "Actual"]
    backtest_rows = forecast_df[forecast_df["type"] == "Backtest"]
    fc_rows       = forecast_df[forecast_df["type"] == "Forecast"]

    fig.add_trace(go.Scatter(
        x=actual_rows["month_dt"], y=actual_rows["actual"],
        mode="lines+markers", name="Actual (Historical)",
        line=dict(color=COLORS["actual"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=backtest_rows["month_dt"],
        y=backtest_rows["actual"],
        mode="lines", name="Actual (Test Period)",
        line=dict(color=COLORS["actual"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=backtest_rows["month_dt"], y=backtest_rows["predicted"],
        mode="lines+markers", name="Model Backtest",
        line=dict(color=COLORS["backtest"], width=2, dash="dash"),
        marker=dict(size=6, symbol="circle-open"),
    ))
    if len(fc_rows) > 0:
        fig.add_trace(go.Scatter(
            x=fc_rows["month_dt"], y=fc_rows["predicted"],
            mode="lines+markers", name=f"Forecast ({horizon}M)",
            line=dict(color=COLORS["forecast"], width=2.5, dash="longdash"),
            marker=dict(size=8, symbol="diamond"),
        ))
        # Shaded forecast region
        fig.add_vrect(
            x0=fc_rows["month_dt"].min(), x1=fc_rows["month_dt"].max(),
            fillcolor="rgba(124,92,216,0.07)", layer="below", line_width=0,
        )

    fig_layout(fig, "Sales Forecast — Actual · Backtest · Future Forecast", height=420)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="disclaimer">'
        '⚠ <strong>Forecast Disclaimer:</strong> Forecast values are estimates produced by the '
        f'{best_name} model and are subject to uncertainty. They are intended as a planning aid, '
        'not as guaranteed business outcomes. Actual results will be influenced by market conditions, '
        'competitive dynamics, and factors not captured in the historical sales data.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Forecast table
    if len(fc_rows) > 0:
        with st.expander("View Forecast Values"):
            display_fc = fc_rows[["month_dt", "predicted"]].copy()
            display_fc.columns = ["Month", "Forecast Sales ($)"]
            display_fc["Month"] = display_fc["Month"].dt.strftime("%B %Y")
            display_fc["Forecast Sales ($)"] = display_fc["Forecast Sales ($)"].apply(
                lambda v: f"${v:,.0f}"
            )
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

    # Model Comparison Table
    if eval_df is not None:
        with st.expander("Model Comparison — Evaluation Metrics"):
            st.markdown("""
            **Metric Definitions:**
            - **MAE** (Mean Absolute Error): Average absolute dollar error per month — lower is better.
            - **RMSE** (Root Mean Squared Error): Penalises large forecast errors more heavily — lower is better.
            - **R²** (Coefficient of Determination): Proportion of variance explained — higher is better (max 1.0).
            - **MAPE** (Mean Absolute Percentage Error): Average % error — lower is better.

            *Model selected on lowest MAE (practical forecasting error metric).*
            """)
            display_eval = eval_df.copy().reset_index()
            display_eval["MAE"]  = display_eval["MAE"].apply(lambda v: f"${v:,.0f}")
            display_eval["RMSE"] = display_eval["RMSE"].apply(lambda v: f"${v:,.0f}")
            display_eval["R2"]   = display_eval["R2"].apply(lambda v: f"{v:.4f}")
            display_eval["MAPE"] = display_eval["MAPE"].apply(lambda v: f"{v:.2f}%" if not pd.isna(v) else "N/A")
            st.dataframe(display_eval, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE FORECAST EXPLORER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔭 Interactive Forecast Explorer</div>', unsafe_allow_html=True)
    with st.expander("Explore Forecasts by Category or Region"):
        st.markdown(
            "**Note:** Segmented forecasts require sufficient historical data per segment. "
            "If fewer than 15 monthly observations are available for a segment, the forecast "
            "will not be generated to avoid unreliable results."
        )
        seg_type = st.radio("Segment by", ["Category", "Region"], horizontal=True)
        seg_col  = seg_type

        if seg_col in filtered_df.columns:
            seg_options = sorted(filtered_df[seg_col].unique().tolist())
            chosen_seg  = st.selectbox(f"Select {seg_col}", seg_options)

            seg_df = filtered_df[filtered_df[seg_col] == chosen_seg].copy()
            from YourName_EcommerceSalesForecasting import create_monthly_series as _cms
            seg_monthly = _cms(seg_df)

            MIN_MONTHS = 15
            seg_ffd = create_forecast_features(seg_monthly)
            seg_fcols = [c for c in feature_cols if c in seg_ffd.columns]

            if len(seg_ffd) < MIN_MONTHS:
                st.warning(
                    f"Only {len(seg_ffd)} monthly observations available for '{chosen_seg}'. "
                    f"At least {MIN_MONTHS} are required for a reliable forecast. "
                    "Please select a different segment or increase the date range."
                )
            else:
                seg_train, seg_test = chronological_split(seg_ffd)
                seg_models  = train_forecast_models(seg_train, seg_fcols)
                seg_eval    = evaluate_models(seg_models, seg_test, seg_fcols)
                seg_bname, seg_bmodel = select_best_model(seg_eval, seg_models)
                seg_forecast = generate_forecast(
                    seg_bmodel, seg_ffd, seg_monthly,
                    horizon=horizon, feature_cols=seg_fcols,
                )
                fig_seg = go.Figure()
                fig_seg.add_trace(go.Scatter(
                    x=seg_forecast["month_dt"], y=seg_forecast["actual"],
                    mode="lines+markers", name="Actual",
                    line=dict(color=COLORS["actual"], width=2),
                ))
                seg_fc = seg_forecast[seg_forecast["type"] == "Forecast"]
                if len(seg_fc) > 0:
                    fig_seg.add_trace(go.Scatter(
                        x=seg_fc["month_dt"], y=seg_fc["predicted"],
                        mode="lines+markers", name="Forecast",
                        line=dict(color=COLORS["forecast"], width=2.5, dash="longdash"),
                    ))
                fig_layout(fig_seg, f"Sales Forecast — {chosen_seg}", height=360)
                fig_seg.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig_seg, use_container_width=True)
                st.caption(
                    f"Model: {seg_bname} | "
                    f"MAE: ${seg_eval.loc[seg_bname,'MAE']:,.0f} | "
                    f"R²: {seg_eval.loc[seg_bname,'R2']:.4f}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # RISK SECTION
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠ Business Risks</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6b7280'>Risks are identified using defined detection rules applied "
        "to the dataset. Thresholds are stated explicitly for each risk type.</small>",
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
        st.success("No significant risks detected in the current filtered dataset.")

    # ─────────────────────────────────────────────────────────────────────────
    # OPPORTUNITY SECTION
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
        st.info("No distinct opportunities identified in the current filtered dataset.")

    # ─────────────────────────────────────────────────────────────────────────
    # PROFIT SCENARIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Profit Scenario Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">'
        '⚠ <strong>Important Disclaimer:</strong> '
        + profit_analysis.get("caveats", "")
        + '</div>',
        unsafe_allow_html=True,
    )

    if "discount_band_summary" in profit_analysis:
        band_sum = profit_analysis["discount_band_summary"]
        order_map = {
            "No Discount":0,"Low (0–10%)":1,
            "Moderate (11–20%)":2,"High (21–30%)":3,"Very High (>30%)":4
        }
        band_sum = band_sum.copy()
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
                st.markdown("**Hypothetical Discount Scenarios**")
                st.dataframe(
                    profit_analysis["scenario_estimates"],
                    use_container_width=True,
                    hide_index=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMENDED ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Recommended Actions</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#6b7280'>These recommendations are evidence-based hypotheses for management "
        "investigation and testing — not guaranteed outcomes.</small>",
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
    from YourName_EcommerceSalesForecasting import DATA_PATH as _dp
    info_lines = [
        f"<strong>Dataset File:</strong> {_dp}",
        f"<strong>Selected Model:</strong> {best_name}",
        f"<strong>Model Trained At:</strong> {trained_at if trained_at else 'N/A'}",
    ]
    if eval_df is not None and best_name in eval_df.index:
        m = eval_df.loc[best_name]
        info_lines += [
            f"<strong>MAE:</strong> ${m['MAE']:,.0f}",
            f"<strong>RMSE:</strong> ${m['RMSE']:,.0f}",
            f"<strong>R²:</strong> {m['R2']:.4f}",
        ]
    info_lines.append(f"<strong>Forecast Horizon:</strong> {horizon} months")
    st.markdown(
        '<div class="info-box">' + " &nbsp;|&nbsp; ".join(info_lines) + "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Sidebar & filters
    # We need feat_df for the sidebar — attempt a lightweight load
    try:
        pipeline = st.session_state.get("pipeline")
        retrain_requested = st.session_state.get("retrain_requested", False)

        if pipeline is None or retrain_requested:
            if retrain_requested:
                st.session_state["retrain_requested"] = False
                # Clear cache to force re-computation
                load_pipeline.clear()

            with st.spinner("Loading data and pipeline …"):
                pipeline = load_pipeline(DATA_PATH, retrain=retrain_requested)
                st.session_state["pipeline"] = pipeline

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(f"Data validation error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading pipeline: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()

    feat_df = pipeline["feat_df"]
    page, filters, retrain_btn = render_sidebar(feat_df)

    if retrain_btn:
        st.session_state["retrain_requested"] = True
        st.rerun()

    # Apply filters
    filtered_df = apply_filters(feat_df, filters)

    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust the filter selections.")
        st.stop()

    # --- Cleaning Log Banner ---
    log = pipeline["cleaning_log"]
    with st.expander("📋 Dataset Summary", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows",         f"{log['raw_row_count']:,}")
        c2.metric("Analytical Rows",  f"{log['final_row_count']:,}")
        c3.metric("Date Range Start", log["date_range_start"])
        c4.metric("Date Range End",   log["date_range_end"])
        st.markdown(
            f"Duplicates removed: **{log.get('duplicate_rows_removed',0)}** | "
            f"Invalid sales removed: **{log.get('invalid_sales_removed',0)}** | "
            f"Null order-date removed: **{log.get('null_order_date_removed',0)}**"
        )

    # --- Route to selected page ---
    if page == "📈 Executive Overview":
        page_executive_overview(
            filtered_df,
            pipeline["kpis"],
            pipeline["monthly"],
            pipeline["insights"],
        )
    elif page == "🔍 Sales & Product Analysis":
        page_sales_product(filtered_df)
    elif page == "🔮 Forecast, Risk & Action":
        page_forecast_risk_action(
            filtered_df=filtered_df,
            monthly_all=pipeline["monthly"],
            best_model=pipeline["best_model"],
            best_name=pipeline["best_model_name"],
            eval_df=pipeline["eval_df"],
            feature_cols=pipeline["feature_cols"],
            forecast_feat_df=pipeline["forecast_feat_df"],
            insights=pipeline["insights"],
            risks=pipeline["risks"],
            opportunities=pipeline["opportunities"],
            profit_analysis=pipeline["profit_analysis"],
            trained_at=pipeline.get("trained_at"),
        )


if __name__ == "__main__":
    main()
