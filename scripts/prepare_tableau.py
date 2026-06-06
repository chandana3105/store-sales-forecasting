"""
Prepare a single flat CSV for Tableau visualizations.

Usage:
    python scripts/prepare_tableau.py --data-dir data/raw --output data/tableau_ready.csv

What it produces (one row per store-family-date):
    - Date dimensions  : date, year, month, month_name, quarter, week_of_year,
                         day_of_week, day_name, is_weekend
    - Store dimensions : store_nbr, city, state, store_type, cluster
    - Product          : family
    - Sales metrics    : sales, transactions, onpromotion
    - Oil signal       : oil_price, oil_7d_ma, oil_price_change_pct
    - Holiday flags    : is_national_holiday, is_local_holiday, is_holiday,
                         holiday_description
    - Rolling KPIs     : sales_7d_avg, sales_28d_avg, sales_mom_pct (month-over-month)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def load_and_merge(data_dir: Path) -> pd.DataFrame:
    logger.info("Loading raw tables...")
    train        = pd.read_csv(data_dir / "train.csv",          parse_dates=["date"])
    stores       = pd.read_csv(data_dir / "stores.csv")
    oil          = pd.read_csv(data_dir / "oil.csv",            parse_dates=["date"])
    holidays     = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(data_dir / "transactions.csv",   parse_dates=["date"])

    logger.info(f"  train: {train.shape}")

    # --- stores ---
    stores = stores.rename(columns={"type": "store_type"})
    df = train.merge(stores, on="store_nbr", how="left")

    # --- oil (fill gaps for weekends/holidays) ---
    date_range = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})
    oil_full = date_range.merge(oil.rename(columns={"dcoilwtico": "oil_price"}), on="date", how="left")
    oil_full["oil_price"] = oil_full["oil_price"].ffill().bfill()
    df = df.merge(oil_full, on="date", how="left")

    # --- holidays ---
    national = (
        holidays[holidays["locale"] == "National"]
        [["date", "type", "description", "transferred"]]
        .drop_duplicates("date")
        .rename(columns={"type": "nat_holiday_type", "description": "nat_holiday_desc"})
    )
    local = (
        holidays[holidays["locale"] == "Local"]
        [["date", "locale_name", "type", "description"]]
        .drop_duplicates(["date", "locale_name"])
        .rename(columns={
            "type": "local_holiday_type",
            "description": "local_holiday_desc",
            "locale_name": "city",
        })
    )
    df = df.merge(national, on="date", how="left")
    df = df.merge(local,    on=["date", "city"], how="left")

    df["is_national_holiday"] = df["nat_holiday_type"].notna().astype(int)
    df["is_local_holiday"]    = df["local_holiday_type"].notna().astype(int)
    df["is_holiday"]          = ((df["is_national_holiday"] == 1) | (df["is_local_holiday"] == 1)).astype(int)
    df["holiday_description"] = df["nat_holiday_desc"].fillna(df["local_holiday_desc"]).fillna("None")

    # --- transactions ---
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    logger.info(f"Merged shape: {df.shape}")
    return df


def add_date_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]         = df["date"].dt.year
    df["quarter"]      = df["date"].dt.quarter
    df["month"]        = df["date"].dt.month
    df["month_name"]   = df["date"].dt.month.map(MONTH_NAMES)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Mon
    df["day_name"]     = df["day_of_week"].map(DAY_NAMES)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    return df


def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")
    df["oil_7d_ma"] = df["oil_price"].rolling(7, min_periods=1).mean()
    df["oil_price_change_pct"] = df["oil_price"].pct_change().fillna(0).round(4)
    return df


def add_sales_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling averages and month-over-month growth per store-family."""
    df = df.sort_values(["store_nbr", "family", "date"])
    grp = df.groupby(["store_nbr", "family"])["sales"]

    df["sales_7d_avg"]  = grp.transform(lambda s: s.rolling(7,  min_periods=1).mean()).round(2)
    df["sales_28d_avg"] = grp.transform(lambda s: s.rolling(28, min_periods=1).mean()).round(2)

    # Month-over-month: compare current month avg to prior month avg
    monthly = (
        df.groupby(["store_nbr", "family", "year", "month"])["sales"]
        .mean()
        .rename("monthly_avg")
        .reset_index()
    )
    monthly["prev_monthly_avg"] = monthly.groupby(["store_nbr", "family"])["monthly_avg"].shift(1)
    monthly["sales_mom_pct"] = (
        (monthly["monthly_avg"] - monthly["prev_monthly_avg"])
        / monthly["prev_monthly_avg"].replace(0, np.nan)
        * 100
    ).round(2)
    df = df.merge(monthly[["store_nbr", "family", "year", "month", "sales_mom_pct"]],
                  on=["store_nbr", "family", "year", "month"], how="left")
    return df


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        # Time
        "date", "year", "quarter", "month", "month_name",
        "week_of_year", "day_of_week", "day_name", "is_weekend",
        # Store
        "store_nbr", "city", "state", "store_type", "cluster",
        # Product
        "family",
        # Core metrics
        "sales", "onpromotion", "transactions",
        # Oil
        "oil_price", "oil_7d_ma", "oil_price_change_pct",
        # Holiday
        "is_national_holiday", "is_local_holiday", "is_holiday", "holiday_description",
        # KPIs
        "sales_7d_avg", "sales_28d_avg", "sales_mom_pct",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def main():
    parser = argparse.ArgumentParser(description="Prepare Tableau-ready CSV")
    parser.add_argument("--data-dir", default="data/raw",            help="Raw Kaggle CSVs")
    parser.add_argument("--output",   default="data/tableau_ready.csv", help="Output CSV path")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Optional: limit to N rows for quick testing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(data_dir)
    df = add_date_dimensions(df)
    df = add_oil_features(df)
    df = add_sales_kpis(df)
    df = select_and_rename(df)

    if args.sample:
        df = df.sample(args.sample, random_state=42)
        logger.info(f"Sampled to {len(df)} rows")

    df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}  ({len(df):,} rows × {df.shape[1]} cols)")
    logger.info(f"Columns: {list(df.columns)}")

    # Quick summary
    print("\n--- Preview ---")
    print(df.head(3).to_string())
    print(f"\nDate range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Stores     : {df['store_nbr'].nunique()}")
    print(f"Families   : {df['family'].nunique()}")
    print(f"Total rows : {len(df):,}")


if __name__ == "__main__":
    main()
