"""
=============================================================================
AI-Powered E-Commerce Sales Forecasting & Profit Optimization
IBM SkillsBuild Data Analytics with AI Academic Internship
=============================================================================
Author  : [Your Name]
Dataset : Sample Superstore (retail transaction dataset)
Purpose : End-to-end analytics pipeline — data loading, cleaning, EDA,
          KPI calculation, feature engineering, ML forecasting, insight
          generation, risk and opportunity analysis, recommendations.

IMPORTANT: All metrics, insights and forecast outputs are computed
           dynamically from the loaded dataset. No values are hard-coded.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import os
import warnings
import logging
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server contexts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "Superstore.csv")
MODEL_PATH = os.path.join("models", "sales_forecast_model.joblib")

# Column name aliases — map common spelling variations to canonical names
COLUMN_ALIASES = {
    "order date":       "Order Date",
    "orderdate":        "Order Date",
    "ship date":        "Ship Date",
    "shipdate":         "Ship Date",
    "customer id":      "Customer ID",
    "customerid":       "Customer ID",
    "customer name":    "Customer Name",
    "customername":     "Customer Name",
    "order id":         "Order ID",
    "orderid":          "Order ID",
    "product id":       "Product ID",
    "productid":        "Product ID",
    "product name":     "Product Name",
    "productname":      "Product Name",
    "sub-category":     "Sub-Category",
    "subcategory":      "Sub-Category",
    "ship mode":        "Ship Mode",
    "shipmode":         "Ship Mode",
    "postal code":      "Postal Code",
    "postalcode":       "Postal Code",
}

# Required columns (after alias normalisation)
REQUIRED_COLUMNS = [
    "Order Date", "Sales", "Profit", "Quantity", "Discount",
    "Category", "Region",
]

# Optional but highly useful
OPTIONAL_COLUMNS = [
    "Order ID", "Customer ID", "Customer Name", "Segment",
    "Sub-Category", "Product Name", "Product ID",
    "Ship Date", "Ship Mode", "State", "City",
]


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Superstore CSV dataset.

    Steps
    -----
    1. Verify the file exists.
    2. Read with pandas, trying common encodings.
    3. Normalise column names (strip whitespace, apply aliases).
    4. Validate that required columns are present.
    5. Parse Order Date and Ship Date as datetime.
    6. Log dataset shape, dtypes, and missing-value summary.

    Returns
    -------
    pd.DataFrame  — raw (uncleaned) dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"Dataset not found: {filepath}\n\n"
            f"Please download the Sample Superstore dataset and place it at:\n"
            f"  {os.path.abspath(filepath)}\n\n"
            f"Dataset source:\n"
            f"  Tableau Sample – Superstore Sales (superstore.csv)\n"
            f"  Available on Kaggle: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final\n"
            f"  Or Tableau's public data resources.\n"
            f"{'='*60}\n"
        )

    # Try different encodings
    for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            logger.info(f"Loaded '{filepath}' with encoding={enc}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError(f"Could not read '{filepath}' with any standard encoding.")

    # Normalise column names
    df = _normalise_columns(df)

    # Validate required columns
    validate_columns(df)

    # Parse dates
    for date_col in ("Order Date", "Ship Date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")

    # Ensure numeric types
    for num_col in ("Sales", "Profit", "Quantity", "Discount"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Log summary
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"Missing values:\n{missing}")
    else:
        logger.info("No missing values detected.")

    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply COLUMN_ALIASES to column names."""
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        key = col.lower().replace("_", " ")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.info(f"Renamed columns: {rename_map}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check that all REQUIRED_COLUMNS are present.
    Raises ValueError with a clear diagnostic message if any are missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n{'='*60}\n"
            f"Missing required columns: {missing}\n\n"
            f"Dataset columns found: {list(df.columns)}\n\n"
            f"Please verify your dataset is the Superstore sales dataset\n"
            f"and that column names match or have common aliases.\n"
            f"{'='*60}\n"
        )
    logger.info("All required columns present.")


# =============================================================================
# 2. DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataset and return (cleaned_df, cleaning_log).

    Steps
    -----
    - Detect and remove exact duplicate rows.
    - Drop rows with null Order Date (cannot be placed on a time axis).
    - Drop rows where Sales <= 0 (invalid revenue records).
    - Handle remaining nulls in numeric columns (fill with median).
    - Clip Discount to [0, 1] range.
    - Strip whitespace from categorical columns.
    - Calculate Shipping Days if both dates are present.

    Returns
    -------
    cleaned_df : pd.DataFrame
    cleaning_log : dict — audit trail for transparency.
    """
    log = {}
    log["raw_row_count"] = len(df)
    log["raw_column_count"] = len(df.columns)

    # 1. Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    log["duplicate_rows_removed"] = before - len(df)
    logger.info(f"Duplicates removed: {log['duplicate_rows_removed']}")

    # 2. Drop rows with null Order Date
    before = len(df)
    df = df.dropna(subset=["Order Date"])
    log["null_order_date_removed"] = before - len(df)

    # 3. Drop rows where Sales is null or <= 0
    before = len(df)
    df = df[df["Sales"].notna() & (df["Sales"] > 0)]
    log["invalid_sales_removed"] = before - len(df)
    logger.info(f"Invalid sales rows removed: {log['invalid_sales_removed']}")

    # 4. Handle remaining numeric nulls
    for col in ("Profit", "Quantity", "Discount"):
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {null_count} nulls in '{col}' with median={median_val:.4f}")

    # 5. Clip Discount to valid range [0, 1]
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].clip(0, 1)

    # 6. Strip whitespace from object/string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 7. Final counts and date range
    log["final_row_count"] = len(df)
    log["date_range_start"] = str(df["Order Date"].min().date())
    log["date_range_end"] = str(df["Order Date"].max().date())
    log["total_months"] = (
        (df["Order Date"].max().year - df["Order Date"].min().year) * 12
        + df["Order Date"].max().month - df["Order Date"].min().month + 1
    )

    logger.info(
        f"Cleaning complete. Rows: {log['raw_row_count']} → {log['final_row_count']}. "
        f"Date range: {log['date_range_start']} to {log['date_range_end']}"
    )

    df = df.reset_index(drop=True)
    return df, log


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytical columns from the cleaned dataset.

    New columns
    -----------
    Year, Quarter, Month, Month Number, Year-Month,
    Week, Day of Week, Shipping Days, Profit Margin,
    Revenue per Unit, Discount Band,
    (if Order ID present) Customer Order Count, Repeat Customer Flag.
    """
    df = df.copy()

    # Time features
    df["Year"]         = df["Order Date"].dt.year
    df["Quarter"]      = df["Order Date"].dt.quarter.apply(lambda q: f"Q{q}")
    df["Month"]        = df["Order Date"].dt.month_name()
    df["Month Number"] = df["Order Date"].dt.month
    df["Year-Month"]   = df["Order Date"].dt.to_period("M").astype(str)
    df["Week"]         = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Day of Week"]  = df["Order Date"].dt.day_name()

    # Shipping days
    if "Ship Date" in df.columns:
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
        df["Shipping Days"] = df["Shipping Days"].clip(lower=0)

    # Financial derived metrics
    df["Profit Margin"]    = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, 0)
    df["Revenue per Unit"] = np.where(df["Quantity"] != 0, df["Sales"] / df["Quantity"], 0)

    # Discount bands
    def _discount_band(d):
        if d == 0:            return "No Discount"
        elif d <= 0.10:       return "Low (0–10%)"
        elif d <= 0.20:       return "Moderate (11–20%)"
        elif d <= 0.30:       return "High (21–30%)"
        else:                 return "Very High (>30%)"

    if "Discount" in df.columns:
        df["Discount Band"] = df["Discount"].apply(_discount_band)

    # Customer-level enrichment
    if "Customer ID" in df.columns and "Order ID" in df.columns:
        order_counts = (
            df.groupby("Customer ID")["Order ID"]
            .nunique()
            .reset_index(name="Customer Order Count")
        )
        df = df.merge(order_counts, on="Customer ID", how="left")
        df["Repeat Customer"] = df["Customer Order Count"].apply(
            lambda x: "Repeat" if x > 1 else "One-time"
        )

    logger.info("Feature engineering complete.")
    return df


# =============================================================================
# 4. KPI CALCULATIONS
# =============================================================================

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Compute core business KPIs dynamically from the dataset.

    Definitions
    -----------
    Total Revenue      = sum(Sales)
    Total Profit       = sum(Profit)
    Total Orders       = count of unique Order IDs (or row count if absent)
    Total Customers    = count of unique Customer IDs (or N/A)
    Average Order Value= Total Revenue / Total Orders
    Profit Margin %    = Total Profit / Total Revenue × 100
    Total Qty Sold     = sum(Quantity)
    Average Discount % = mean(Discount) × 100
    Revenue per Customer = Total Revenue / Total Customers
    YoY Growth %       = growth from penultimate to latest calendar year
    """
    kpis = {}

    kpis["Total Revenue"]  = df["Sales"].sum()
    kpis["Total Profit"]   = df["Profit"].sum()
    kpis["Total Quantity"] = int(df["Quantity"].sum())

    if "Order ID" in df.columns:
        kpis["Total Orders"] = df["Order ID"].nunique()
    else:
        kpis["Total Orders"] = len(df)

    if "Customer ID" in df.columns:
        kpis["Total Customers"] = df["Customer ID"].nunique()
    else:
        kpis["Total Customers"] = None

    kpis["Average Order Value"] = (
        kpis["Total Revenue"] / kpis["Total Orders"]
        if kpis["Total Orders"] > 0 else 0
    )
    kpis["Profit Margin %"] = (
        kpis["Total Profit"] / kpis["Total Revenue"] * 100
        if kpis["Total Revenue"] != 0 else 0
    )
    kpis["Average Discount %"] = (
        df["Discount"].mean() * 100 if "Discount" in df.columns else 0
    )
    kpis["Revenue per Customer"] = (
        kpis["Total Revenue"] / kpis["Total Customers"]
        if kpis["Total Customers"] and kpis["Total Customers"] > 0 else None
    )

    # Year-over-Year growth (latest vs previous year)
    if "Year" in df.columns:
        yearly = df.groupby("Year")["Sales"].sum().sort_index()
        if len(yearly) >= 2:
            prev_yr = yearly.iloc[-2]
            curr_yr = yearly.iloc[-1]
            kpis["YoY Growth %"] = (
                (curr_yr - prev_yr) / prev_yr * 100 if prev_yr != 0 else None
            )
            kpis["YoY Comparison"] = f"{yearly.index[-2]} vs {yearly.index[-1]}"
        else:
            kpis["YoY Growth %"] = None
            kpis["YoY Comparison"] = "Insufficient years"

    logger.info("KPIs calculated.")
    return kpis


# =============================================================================
# 5. MONTHLY TIME SERIES
# =============================================================================

def create_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction data to a monthly time series.

    Columns
    -------
    month              : Period (YYYY-MM)
    total_sales        : sum of Sales
    total_profit       : sum of Profit
    total_orders       : count of unique Order IDs
    unique_customers   : count of unique Customer IDs
    average_order_value: total_sales / total_orders
    profit_margin      : total_profit / total_sales × 100
    total_quantity     : sum of Quantity
    """
    df = df.copy()
    df["_period"] = df["Order Date"].dt.to_period("M")

    agg_dict = {
        "Sales":    "sum",
        "Profit":   "sum",
        "Quantity": "sum",
    }
    if "Order ID" in df.columns:
        agg_dict["Order ID"] = pd.NamedAgg(column="Order ID", aggfunc="nunique")
    if "Customer ID" in df.columns:
        agg_dict["Customer ID"] = pd.NamedAgg(column="Customer ID", aggfunc="nunique")

    monthly = df.groupby("_period").agg(**{
        "total_sales":   pd.NamedAgg(column="Sales", aggfunc="sum"),
        "total_profit":  pd.NamedAgg(column="Profit", aggfunc="sum"),
        "total_quantity":pd.NamedAgg(column="Quantity", aggfunc="sum"),
        **({"total_orders": pd.NamedAgg(column="Order ID", aggfunc="nunique")}
           if "Order ID" in df.columns else {}),
        **({"unique_customers": pd.NamedAgg(column="Customer ID", aggfunc="nunique")}
           if "Customer ID" in df.columns else {}),
    }).reset_index()

    monthly.rename(columns={"_period": "month"}, inplace=True)
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly.sort_values("month_dt", inplace=True)

    if "total_orders" in monthly.columns:
        monthly["average_order_value"] = np.where(
            monthly["total_orders"] > 0,
            monthly["total_sales"] / monthly["total_orders"], 0
        )
    monthly["profit_margin"] = np.where(
        monthly["total_sales"] != 0,
        monthly["total_profit"] / monthly["total_sales"] * 100, 0
    )
    monthly = monthly.reset_index(drop=True)
    logger.info(f"Monthly series: {len(monthly)} periods.")
    return monthly


# =============================================================================
# 6. FORECAST FEATURE ENGINEERING
# =============================================================================

def create_forecast_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Build supervised-learning features from the monthly time series
    using ONLY historical (past) information — no future leakage.

    Features
    --------
    lag_1  … lag_12 : sales lagged 1–12 months
    rolling_mean_3  : 3-month rolling mean of sales (shift by 1 to avoid leakage)
    rolling_mean_6  : 6-month rolling mean
    rolling_mean_12 : 12-month rolling mean
    month_number    : calendar month (1–12) — seasonality proxy
    year            : calendar year — trend proxy
    time_index      : integer index from 0 — linear trend

    All rolling / lag features are computed from past observations only
    (using .shift(1) before .rolling()) to ensure no data leakage.

    Returns a DataFrame with NaN rows (from lag initialisation) dropped.
    """
    df = monthly.copy().sort_values("month_dt").reset_index(drop=True)
    df["time_index"]    = np.arange(len(df))
    df["month_number"]  = df["month_dt"].dt.month
    df["year"]          = df["month_dt"].dt.year

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["total_sales"].shift(lag)

    # Rolling means use .shift(1) so the window never includes the current month
    shifted = df["total_sales"].shift(1)
    df["rolling_mean_3"]  = shifted.rolling(3).mean()
    df["rolling_mean_6"]  = shifted.rolling(6).mean()
    df["rolling_mean_12"] = shifted.rolling(12).mean()

    # Drop rows where ANY feature is NaN (lag initialisation period)
    feature_cols = (
        [f"lag_{l}" for l in [1, 2, 3, 6, 12]]
        + ["rolling_mean_3", "rolling_mean_6", "rolling_mean_12"]
    )
    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Forecast features created: {len(df)} usable rows.")
    return df


# =============================================================================
# 7. TRAIN / TEST SPLIT (CHRONOLOGICAL)
# =============================================================================

def chronological_split(df: pd.DataFrame, test_ratio: float = 0.20):
    """
    Split the feature DataFrame chronologically: earlier rows for training,
    later rows for testing.

    WHY CHRONOLOGICAL (not random)?
    --------------------------------
    Forecasting models must learn from the past to predict the future.
    Random shuffling would allow the model to see future data during training,
    producing artificially inflated metrics that do not reflect real-world
    forecasting performance. Chronological splitting correctly simulates
    the deployment scenario.

    Returns
    -------
    (train_df, test_df)  both sorted by month_dt.
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    logger.info(
        f"Chronological split: train={len(train)} rows "
        f"({train['month_dt'].min().date()} – {train['month_dt'].max().date()}), "
        f"test={len(test)} rows "
        f"({test['month_dt'].min().date()} – {test['month_dt'].max().date()})"
    )
    return train, test


# =============================================================================
# 8. MODEL TRAINING & EVALUATION
# =============================================================================

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
    "month_number", "year", "time_index",
]
TARGET_COL = "total_sales"


def train_forecast_models(
    train_df: pd.DataFrame,
    feature_cols: list = None,
) -> dict:
    """
    Train Linear Regression, Ridge Regression, and Random Forest Regressor.

    Returns
    -------
    dict: {model_name: fitted_pipeline}
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            # Random Forest is scale-invariant; scaler included for uniformity
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
            )),
        ]),
    }

    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        logger.info(f"Trained: {name}")

    return trained


def evaluate_models(
    models: dict,
    test_df: pd.DataFrame,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the held-out test set.

    Metrics
    -------
    MAE    : Mean Absolute Error — average dollar error per month.
    RMSE   : Root Mean Squared Error — penalises large errors more.
    R²     : Coefficient of determination — proportion of variance explained.
    MAPE   : Mean Absolute Percentage Error — percentage error (skips zero actuals).

    Returns
    -------
    pd.DataFrame  with one row per model and columns [MAE, RMSE, R2, MAPE].
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in test_df.columns]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    results = []
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)

        nonzero = y_test != 0
        mape = (
            np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100
            if nonzero.sum() > 0 else np.nan
        )
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape})
        logger.info(f"{name}: MAE={mae:,.2f}  RMSE={rmse:,.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    return pd.DataFrame(results).set_index("Model")


def select_best_model(
    eval_df: pd.DataFrame,
    models: dict,
    primary_metric: str = "MAE",
) -> tuple[str, object]:
    """
    Select the model with the lowest MAE (primary forecasting error metric).

    WHY MAE OVER R²?
    ----------------
    For a business sales forecast, the practical question is:
    "By how many dollars is my forecast off on average?"
    MAE directly answers this. R² measures explained variance, which can
    be misleading when variance is dominated by trend rather than model skill.
    RMSE is also reported but penalises outlier months more heavily than
    may be appropriate for operational planning.
    """
    best_name = eval_df[primary_metric].idxmin()
    logger.info(
        f"Selected model: '{best_name}' "
        f"(lowest {primary_metric} = {eval_df.loc[best_name, primary_metric]:,.2f})"
    )
    return best_name, models[best_name]


def save_model(
    model,
    model_name: str,
    eval_df: pd.DataFrame,
    feature_cols: list,
    monthly: pd.DataFrame,
    filepath: str = MODEL_PATH,
) -> None:
    """Save the selected model and associated metadata using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "model":         model,
        "model_name":    model_name,
        "eval_df":       eval_df,
        "feature_cols":  feature_cols,
        "monthly_shape": monthly.shape,
        "trained_at":    datetime.now().isoformat(),
    }
    joblib.dump(payload, filepath)
    logger.info(f"Model saved: {filepath}")


def load_model(filepath: str = MODEL_PATH) -> dict:
    """Load a previously saved model payload. Returns None if not found."""
    if not os.path.exists(filepath):
        logger.warning(f"Model file not found: {filepath}")
        return None
    payload = joblib.load(filepath)
    logger.info(f"Model loaded: {payload.get('model_name')} (trained {payload.get('trained_at')})")
    return payload


# =============================================================================
# 9. FORECAST GENERATION
# =============================================================================

def generate_forecast(
    model,
    feature_df: pd.DataFrame,
    monthly: pd.DataFrame,
    horizon: int = 6,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Generate future-period sales forecasts by iteratively predicting one
    month ahead and feeding predictions back as lag features.

    Parameters
    ----------
    model        : trained pipeline
    feature_df   : DataFrame with engineered features (for back-test predictions)
    monthly      : full monthly series (used to anchor future predictions)
    horizon      : number of future months to forecast (default 6)
    feature_cols : feature column names to use

    Returns
    -------
    pd.DataFrame with columns:
        month_dt, actual (NaN for future), predicted, type
        type ∈ {'Actual', 'Backtest', 'Forecast'}

    NOTE: Future forecast values are estimates subject to uncertainty.
          They are produced by the trained model and should not be treated
          as guaranteed business outcomes.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feature_df.columns]

    # Back-test predictions on all feature rows
    X_all = feature_df[feature_cols]
    backtest_pred = model.predict(X_all)

    backtest_df = pd.DataFrame({
        "month_dt":  feature_df["month_dt"].values,
        "actual":    feature_df[TARGET_COL].values,
        "predicted": backtest_pred,
        "type":      "Backtest",
    })

    # Actual rows with no prediction (lag initialisation rows)
    first_feature_dt = feature_df["month_dt"].min()
    early_actual = monthly[monthly["month_dt"] < first_feature_dt][
        ["month_dt", "total_sales"]
    ].copy()
    early_actual.rename(columns={"total_sales": "actual"}, inplace=True)
    early_actual["predicted"] = np.nan
    early_actual["type"]      = "Actual"

    # Future forecasting via iterative one-step-ahead prediction
    # Build a rolling history from the full monthly series
    history = monthly["total_sales"].tolist()
    last_date = monthly["month_dt"].max()
    future_rows = []

    for i in range(1, horizon + 1):
        next_date     = last_date + pd.DateOffset(months=i)
        n_hist        = len(history)
        time_idx      = len(monthly) + i - 1   # continue the integer index

        def _lag(k):
            idx = n_hist - k
            return history[idx] if idx >= 0 else np.nan

        def _roll_mean(k):
            vals = history[max(0, n_hist - k):]
            return np.mean(vals) if vals else np.nan

        row = {
            "lag_1":          _lag(1),
            "lag_2":          _lag(2),
            "lag_3":          _lag(3),
            "lag_6":          _lag(6),
            "lag_12":         _lag(12),
            "rolling_mean_3": _roll_mean(3),
            "rolling_mean_6": _roll_mean(6),
            "rolling_mean_12":_roll_mean(12),
            "month_number":   next_date.month,
            "year":           next_date.year,
            "time_index":     time_idx,
        }
        row_df   = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])
        pred_val = max(0, float(model.predict(row_df)[0]))   # sales cannot be < 0

        future_rows.append({
            "month_dt":  next_date,
            "actual":    np.nan,
            "predicted": pred_val,
            "type":      "Forecast",
        })
        history.append(pred_val)

    forecast_df = pd.concat(
        [early_actual, backtest_df, pd.DataFrame(future_rows)],
        ignore_index=True,
    ).sort_values("month_dt").reset_index(drop=True)

    logger.info(f"Forecast generated: {horizon}-month horizon.")
    return forecast_df


# =============================================================================
# 10. BUSINESS INSIGHT GENERATION
# =============================================================================

def generate_business_insights(df: pd.DataFrame, kpis: dict, monthly: pd.DataFrame) -> dict:
    """
    Dynamically generate evidence-based business insights.

    All statements containing numbers are computed from the actual dataset.
    Associations are stated as observed patterns, NOT as causal claims.

    Returns
    -------
    dict with keys: findings, risks, opportunities, recommendations
    """
    findings      = []
    risks         = []
    opportunities = []
    recommendations = []

    # --- Revenue & Profit summary ---
    rev   = kpis["Total Revenue"]
    prof  = kpis["Total Profit"]
    margin = kpis["Profit Margin %"]

    findings.append(
        f"Total revenue across the analysis period is ${rev:,.0f} with a total profit of "
        f"${prof:,.0f}, representing an overall profit margin of {margin:.1f}%."
    )

    # --- YoY growth ---
    if kpis.get("YoY Growth %") is not None:
        direction = "grew" if kpis["YoY Growth %"] >= 0 else "declined"
        findings.append(
            f"Year-over-year revenue {direction} by {abs(kpis['YoY Growth %']):.1f}% "
            f"({kpis['YoY Comparison']})."
        )

    # --- Best performing category ---
    if "Category" in df.columns:
        cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        best_cat   = cat_profit.index[0]
        worst_cat  = cat_profit.index[-1]
        findings.append(
            f"'{best_cat}' is the highest-profit category (${cat_profit.iloc[0]:,.0f}). "
            f"'{worst_cat}' shows the weakest profit contribution (${cat_profit.iloc[-1]:,.0f})."
        )
        if cat_profit.iloc[-1] < 0:
            risks.append(
                f"'{worst_cat}' has NEGATIVE total profit (${cat_profit.iloc[-1]:,.0f}). "
                f"Products in this category may be discounted beyond their margin threshold."
            )
        opportunities.append(
            f"'{best_cat}' demonstrates strong profitability. Scaling inventory allocation "
            f"toward high-margin products in this category warrants investigation."
        )

    # --- Discount vs profit association ---
    if "Discount" in df.columns:
        high_disc = df[df["Discount"] >= 0.30]
        low_disc  = df[df["Discount"] < 0.10]
        if len(high_disc) > 0 and len(low_disc) > 0:
            hd_margin = (high_disc["Profit"].sum() / high_disc["Sales"].sum() * 100) if high_disc["Sales"].sum() != 0 else 0
            ld_margin = (low_disc["Profit"].sum()  / low_disc["Sales"].sum()  * 100) if low_disc["Sales"].sum()  != 0 else 0
            findings.append(
                f"Products with discounts ≥30% show an average profit margin of "
                f"{hd_margin:.1f}% in this dataset, compared with {ld_margin:.1f}% "
                f"for products with discounts below 10%. This association warrants further investigation."
            )
            if hd_margin < ld_margin:
                risks.append(
                    f"High-discount products (≥30%) are associated with a lower observed profit margin "
                    f"({hd_margin:.1f}% vs {ld_margin:.1f}% for low-discount products). "
                    f"This does not establish causation but suggests evaluating the discount policy."
                )

    # --- Top sub-category ---
    if "Sub-Category" in df.columns:
        sc_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        sc_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
        top_sales_sc  = sc_sales.index[0]
        top_profit_sc = sc_profit.index[0]
        worst_profit_sc = sc_profit.index[-1]
        findings.append(
            f"'{top_sales_sc}' is the top sub-category by revenue (${sc_sales.iloc[0]:,.0f}). "
            f"'{top_profit_sc}' leads on profit (${sc_profit.iloc[0]:,.0f})."
        )
        if sc_profit.iloc[-1] < 0:
            risks.append(
                f"'{worst_profit_sc}' sub-category has negative total profit "
                f"(${sc_profit.iloc[-1]:,.0f}), suggesting possible over-discounting "
                f"or cost structure issues."
            )
        opportunities.append(
            f"'{top_profit_sc}' sub-category shows the highest profit contribution. "
            f"Prioritising stock availability and targeted marketing in this segment could be explored."
        )

    # --- Regional insight ---
    if "Region" in df.columns:
        reg_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)
        reg_sales  = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        best_region  = reg_profit.index[0]
        worst_region = reg_profit.index[-1]
        opportunities.append(
            f"'{best_region}' region leads in profit (${reg_profit.iloc[0]:,.0f}). "
            f"Understanding what drives performance here may yield learnable practices for other regions."
        )
        if reg_profit.iloc[-1] < 0 or reg_profit.iloc[-1] < reg_profit.mean() * 0.5:
            risks.append(
                f"'{worst_region}' region shows relatively weak profit performance "
                f"(${reg_profit.iloc[-1]:,.0f}). Root-cause investigation is advisable."
            )

    # --- Segment ---
    if "Segment" in df.columns:
        seg_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
        best_seg  = seg_sales.index[0]
        findings.append(
            f"'{best_seg}' segment contributes the highest revenue (${seg_sales.iloc[0]:,.0f}) "
            f"among all customer segments."
        )

    # --- Recommendations ---
    recommendations = [
        "Investigate the discount policy for product sub-categories with negative or below-average profit margins. "
        "Test whether a moderate discount reduction maintains sales volume while improving contribution margin.",

        "Evaluate inventory and marketing investment allocation toward the highest-profit category and sub-category "
        "identified in this analysis.",

        "Investigate the root cause of weak regional performance. Consider piloting targeted promotions or "
        "operational improvements in under-performing regions and monitor the impact on profit.",

        "Monitor the sales forecast for periods with predicted lower-than-baseline sales. "
        "Prepare inventory and cash-flow planning accordingly.",

        "Consider segmenting discount experiments by customer type (Segment) to understand "
        "whether discount sensitivity differs across Consumer, Corporate, and Home Office customers.",
    ]

    return {
        "findings":        findings,
        "risks":           risks,
        "opportunities":   opportunities,
        "recommendations": recommendations,
    }


# =============================================================================
# 11. RISK & OPPORTUNITY ANALYSIS
# =============================================================================

def identify_risks(df: pd.DataFrame, monthly: pd.DataFrame) -> list:
    """
    Identify evidence-based business risks.

    Risk detection rules (defined explicitly below):
    1. Category with negative total profit.
    2. Sub-category with negative total profit.
    3. High-discount / low-profit-margin products.
    4. Declining monthly revenue in the most recent N months.
    5. Region with below-average profit.
    """
    risks = []

    # Risk 1: Negative-profit categories
    if "Category" in df.columns:
        cat_profit = df.groupby("Category")["Profit"].sum()
        neg_cats = cat_profit[cat_profit < 0]
        for cat, p in neg_cats.items():
            risks.append({
                "type":        "Negative Profit — Category",
                "entity":      cat,
                "detail":      f"Total profit: ${p:,.0f}",
                "risk_level":  "High",
            })

    # Risk 2: Negative-profit sub-categories
    if "Sub-Category" in df.columns:
        sc_profit = df.groupby("Sub-Category")["Profit"].sum()
        neg_scs = sc_profit[sc_profit < 0]
        for sc, p in neg_scs.items():
            risks.append({
                "type":        "Negative Profit — Sub-Category",
                "entity":      sc,
                "detail":      f"Total profit: ${p:,.0f}",
                "risk_level":  "High",
            })

    # Risk 3: High-discount products with negative margin
    if "Discount" in df.columns and "Product Name" in df.columns:
        prod_df = df.groupby("Product Name").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            avg_discount=("Discount", "mean"),
        ).reset_index()
        prod_df["margin"] = np.where(
            prod_df["total_sales"] != 0,
            prod_df["total_profit"] / prod_df["total_sales"] * 100, 0
        )
        risky_prods = prod_df[
            (prod_df["avg_discount"] > 0.25) & (prod_df["margin"] < 0)
        ].sort_values("total_profit").head(5)
        for _, row in risky_prods.iterrows():
            risks.append({
                "type":       "High Discount + Negative Margin — Product",
                "entity":     row["Product Name"][:50],
                "detail":     f"Avg discount: {row['avg_discount']*100:.1f}%  |  Margin: {row['margin']:.1f}%",
                "risk_level": "Medium",
            })

    # Risk 4: Declining recent revenue trend
    if len(monthly) >= 6:
        recent = monthly.tail(6)["total_sales"]
        if recent.iloc[-1] < recent.mean() * 0.85:
            risks.append({
                "type":       "Declining Recent Revenue",
                "entity":     "Most recent month vs 6-month average",
                "detail":     (
                    f"Latest month sales ${recent.iloc[-1]:,.0f} is more than 15% "
                    f"below the 6-month average of ${recent.mean():,.0f}."
                ),
                "risk_level": "Medium",
            })

    # Risk 5: Below-average-profit regions
    if "Region" in df.columns:
        reg_profit = df.groupby("Region")["Profit"].sum()
        avg_reg_profit = reg_profit.mean()
        weak_regions = reg_profit[reg_profit < avg_reg_profit * 0.5]
        for reg, p in weak_regions.items():
            risks.append({
                "type":       "Below-Average Profit — Region",
                "entity":     reg,
                "detail":     f"Profit ${p:,.0f} vs region average ${avg_reg_profit:,.0f}",
                "risk_level": "Low",
            })

    logger.info(f"Identified {len(risks)} risks.")
    return risks


def identify_opportunities(df: pd.DataFrame, monthly: pd.DataFrame) -> list:
    """
    Identify evidence-based business opportunities.

    Detection rules:
    1. Categories with both high sales and high profit.
    2. Fastest-growing sub-categories (year-over-year).
    3. Regions with above-average profit.
    4. Customer segments with high and growing revenue.
    5. Products with high margin and meaningful sales volume.
    """
    opps = []

    # Opportunity 1: High-sales + high-profit categories
    if "Category" in df.columns:
        cat_agg = df.groupby("Category").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        cat_agg["margin"] = cat_agg["total_profit"] / cat_agg["total_sales"] * 100
        top_cats = cat_agg.sort_values("total_profit", ascending=False).head(2)
        for _, row in top_cats.iterrows():
            if row["total_profit"] > 0:
                opps.append({
                    "type":   "High-Profit Category",
                    "entity": row["Category"],
                    "detail": (
                        f"Sales: ${row['total_sales']:,.0f}  |  "
                        f"Profit: ${row['total_profit']:,.0f}  |  "
                        f"Margin: {row['margin']:.1f}%"
                    ),
                })

    # Opportunity 2: Top-profit sub-categories
    if "Sub-Category" in df.columns:
        sc_agg = df.groupby("Sub-Category").agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
        ).reset_index()
        sc_agg["margin"] = np.where(
            sc_agg["total_sales"] != 0,
            sc_agg["total_profit"] / sc_agg["total_sales"] * 100, 0
        )
        top_scs = sc_agg[sc_agg["total_profit"] > 0].sort_values("margin", ascending=False).head(3)
        for _, row in top_scs.iterrows():
            opps.append({
                "type":   "High-Margin Sub-Category",
                "entity": row["Sub-Category"],
                "detail": (
                    f"Margin: {row['margin']:.1f}%  |  "
                    f"Profit: ${row['total_profit']:,.0f}"
                ),
            })

    # Opportunity 3: Above-average-profit regions
    if "Region" in df.columns:
        reg_profit = df.groupby("Region")["Profit"].sum()
        avg = reg_profit.mean()
        for reg, p in reg_profit[reg_profit > avg].items():
            opps.append({
                "type":   "Above-Average Profit — Region",
                "entity": reg,
                "detail": f"Profit ${p:,.0f} (region average ${avg:,.0f})",
            })

    # Opportunity 4: Improving revenue trend
    if len(monthly) >= 3:
        recent_3 = monthly.tail(3)["total_sales"].mean()
        prior_3  = monthly.iloc[-6:-3]["total_sales"].mean() if len(monthly) >= 6 else None
        if prior_3 and prior_3 > 0:
            trend_pct = (recent_3 - prior_3) / prior_3 * 100
            if trend_pct > 5:
                opps.append({
                    "type":   "Improving Revenue Trend",
                    "entity": "Overall business",
                    "detail": (
                        f"Recent 3-month average sales (${recent_3:,.0f}) "
                        f"is {trend_pct:.1f}% above the prior 3-month average (${prior_3:,.0f})."
                    ),
                })

    logger.info(f"Identified {len(opps)} opportunities.")
    return opps


# =============================================================================
# 12. PROFIT SCENARIO ANALYSIS
# =============================================================================

def analyze_profitability(df: pd.DataFrame) -> dict:
    """
    Discount vs Profit scenario analysis.

    IMPORTANT DISCLAIMER:
    These scenarios are exploratory and based on historical associations in
    the dataset. They do NOT establish causal relationships. The assumption
    that changing the discount rate will produce the estimated profit change
    is HYPOTHETICAL and subject to many real-world factors not captured here.

    Method: For each discount band, compute average profit margin.
    Then estimate profit under hypothetical discount scenarios using
    the historical relationship as a guide only.

    Returns
    -------
    dict with:
        discount_band_summary : DataFrame
        scenario_estimates    : DataFrame
        caveats               : str
    """
    if "Discount" not in df.columns:
        return {"error": "Discount column not available."}

    # Historical analysis by discount band
    band_summary = df.groupby("Discount Band").agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        order_count=("Sales", "count"),
        avg_discount=("Discount", "mean"),
    ).reset_index()
    band_summary["profit_margin"] = np.where(
        band_summary["total_sales"] != 0,
        band_summary["total_profit"] / band_summary["total_sales"] * 100, 0
    )
    band_summary = band_summary.sort_values("avg_discount")

    # Simple OLS relationship: discount → profit_margin at row level
    from sklearn.linear_model import LinearRegression as _LR
    disc_vals = df["Discount"].values.reshape(-1, 1)
    margin_vals = df["Profit Margin"].values if "Profit Margin" in df.columns else (
        np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, 0)
    )
    _lr = _LR()
    _lr.fit(disc_vals, margin_vals)
    baseline_disc  = df["Discount"].mean()
    total_sales    = df["Sales"].sum()

    # Hypothetical scenario estimates
    scenario_discounts = {
        "Current (as-is)":     baseline_disc,
        "Reduce by 5 pp":      max(0, baseline_disc - 0.05),
        "Reduce by 10 pp":     max(0, baseline_disc - 0.10),
        "Increase by 5 pp":    min(0.80, baseline_disc + 0.05),
    }
    scenarios = []
    for label, disc in scenario_discounts.items():
        est_margin = float(_lr.predict([[disc]])[0])
        est_profit = total_sales * (est_margin / 100)
        scenarios.append({
            "Scenario":           label,
            "Assumed Avg Discount": f"{disc*100:.1f}%",
            "Est. Profit Margin": f"{est_margin:.1f}%",
            "Est. Profit ($)":    f"${est_profit:,.0f}",
            "Note":               "Hypothetical — see caveats",
        })

    caveats = (
        "DISCLAIMER: These scenario estimates are derived from a simple linear regression "
        "between average discount rate and profit margin across transaction records. "
        "This is an ASSOCIATION model only. It does NOT establish that changing the discount "
        "rate will cause the estimated profit outcome. Real outcomes depend on customer "
        "price sensitivity, competitive dynamics, cost structure, product mix, and many "
        "other factors not modelled here. These estimates are intended as a starting point "
        "for management discussion and hypothesis formation — NOT as guaranteed projections."
    )

    return {
        "discount_band_summary": band_summary,
        "scenario_estimates":    pd.DataFrame(scenarios),
        "caveats":               caveats,
        "baseline_discount":     baseline_disc,
    }


# =============================================================================
# 13. MAIN PIPELINE (run from command line or notebook import)
# =============================================================================

def run_full_pipeline(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    forecast_horizon: int = 6,
    retrain: bool = False,
) -> dict:
    """
    Execute the full analytics and ML pipeline end-to-end.

    Returns a results dict with all computed artefacts for use in the
    dashboard and notebook.
    """
    print("\n" + "="*70)
    print("  AI-Powered E-Commerce Sales Forecasting & Profit Optimization")
    print("="*70)

    # 1. Load
    print("\n[1/9] Loading data …")
    raw_df = load_data(data_path)

    # 2. Clean
    print("[2/9] Cleaning data …")
    clean_df, cleaning_log = clean_data(raw_df)

    # 3. Features
    print("[3/9] Engineering features …")
    feat_df = create_features(clean_df)

    # 4. KPIs
    print("[4/9] Calculating KPIs …")
    kpis = calculate_kpis(feat_df)

    # 5. Monthly series
    print("[5/9] Building monthly time series …")
    monthly = create_monthly_series(feat_df)

    # 6. Forecast features
    print("[6/9] Creating forecast features …")
    forecast_feat_df = create_forecast_features(monthly)

    feature_cols = [c for c in FEATURE_COLS if c in forecast_feat_df.columns]

    # 7. Train / evaluate / select model
    trained_models = None
    eval_df_models = None
    best_name      = None
    best_model     = None

    existing_payload = load_model(model_path) if not retrain else None

    if existing_payload and not retrain:
        print("[7/9] Loading existing trained model …")
        best_model = existing_payload["model"]
        best_name  = existing_payload["model_name"]
        eval_df_models = existing_payload.get("eval_df")
        print(f"       → Loaded: {best_name}")
    else:
        print("[7/9] Training forecasting models …")
        if len(forecast_feat_df) < 10:
            raise ValueError(
                "Insufficient monthly data for forecasting. "
                f"Only {len(forecast_feat_df)} usable rows after lag feature creation. "
                "At least ~24 months of transaction data are recommended."
            )
        train_df, test_df = chronological_split(forecast_feat_df)
        trained_models    = train_forecast_models(train_df, feature_cols)
        eval_df_models    = evaluate_models(trained_models, test_df, feature_cols)
        best_name, best_model = select_best_model(eval_df_models, trained_models)
        save_model(best_model, best_name, eval_df_models, feature_cols, monthly, model_path)

    # 8. Forecast
    print("[8/9] Generating forecast …")
    forecast_df = generate_forecast(
        best_model, forecast_feat_df, monthly, horizon=forecast_horizon, feature_cols=feature_cols
    )

    # 9. Insights
    print("[9/9] Generating business insights …")
    insights   = generate_business_insights(feat_df, kpis, monthly)
    risks_list = identify_risks(feat_df, monthly)
    opps_list  = identify_opportunities(feat_df, monthly)
    profit_analysis = analyze_profitability(feat_df)

    print("\n" + "="*70)
    print("  Pipeline complete.")
    print(f"  Final dataset : {cleaning_log['final_row_count']:,} rows")
    print(f"  Date range    : {cleaning_log['date_range_start']} – {cleaning_log['date_range_end']}")
    print(f"  KPI Revenue   : ${kpis['Total Revenue']:,.0f}")
    print(f"  KPI Profit    : ${kpis['Total Profit']:,.0f}  ({kpis['Profit Margin %']:.1f}%)")
    print(f"  Selected model: {best_name}")
    if eval_df_models is not None and best_name in eval_df_models.index:
        m = eval_df_models.loc[best_name]
        print(f"  MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  R²={m['R2']:.4f}")
    print("="*70 + "\n")

    return {
        "raw_df":          raw_df,
        "clean_df":        clean_df,
        "feat_df":         feat_df,
        "cleaning_log":    cleaning_log,
        "kpis":            kpis,
        "monthly":         monthly,
        "forecast_feat_df":forecast_feat_df,
        "feature_cols":    feature_cols,
        "eval_df":         eval_df_models,
        "best_model_name": best_name,
        "best_model":      best_model,
        "forecast_df":     forecast_df,
        "insights":        insights,
        "risks":           risks_list,
        "opportunities":   opps_list,
        "profit_analysis": profit_analysis,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    results = run_full_pipeline()
    print("\nKPI Summary:")
    for k, v in results["kpis"].items():
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:,.2f}")
            else:
                print(f"  {k}: {v}")
