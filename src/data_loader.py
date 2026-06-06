"""Load and merge all Kaggle Store Sales tables into a single DataFrame."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """
    Merge train.csv, stores.csv, oil.csv, holidays_events.csv, and
    transactions.csv into one flat frame ready for feature engineering.

    Args:
        data_dir: Directory containing the raw Kaggle CSVs.

    Returns:
        Merged DataFrame indexed by (date, store_nbr, family).
    """
    data_dir = Path(data_dir)
    _check_files(data_dir)

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])

    logger.info(f"Loaded train: {train.shape}, stores: {stores.shape}, oil: {oil.shape}")

    df = train.merge(stores, on="store_nbr", how="left")
    df = _merge_oil(df, oil)
    df = _merge_holidays(df, holidays)
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    logger.info(f"Final merged shape: {df.shape}")
    return df


def load_test(data_dir: str | Path) -> pd.DataFrame:
    """Load and enrich test.csv with store/oil/holiday info."""
    data_dir = Path(data_dir)
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    oil = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])

    df = test.merge(stores, on="store_nbr", how="left")
    df = _merge_oil(df, oil)
    df = _merge_holidays(df, holidays)
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_files(data_dir: Path) -> None:
    required = ["train.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}\n"
            "Download from: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data"
        )


def _merge_oil(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """Merge oil prices and forward-fill gaps (weekends/holidays have no price)."""
    date_range = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})
    oil_full = date_range.merge(oil, on="date", how="left")
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()
    oil_full = oil_full.rename(columns={"dcoilwtico": "oil_price"})
    return df.merge(oil_full, on="date", how="left")


def _merge_holidays(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Encode national and local holidays as binary flags."""
    national = (
        holidays[holidays["locale"] == "National"][["date", "type"]]
        .drop_duplicates("date")
        .rename(columns={"type": "holiday_type_national"})
    )
    local = (
        holidays[holidays["locale"] == "Local"][["date", "locale_name", "type"]]
        .drop_duplicates(["date", "locale_name"])
        .rename(columns={"type": "holiday_type_local", "locale_name": "city"})
    )
    df = df.merge(national, on="date", how="left")
    df = df.merge(local, on=["date", "city"], how="left")
    df["is_national_holiday"] = df["holiday_type_national"].notna().astype(np.int8)
    df["is_local_holiday"] = df["holiday_type_local"].notna().astype(np.int8)
    df = df.drop(columns=["holiday_type_national", "holiday_type_local"])
    return df
