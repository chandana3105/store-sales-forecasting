"""Feature engineering for the Store Sales forecasting pipeline."""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Add all engineered features to the merged DataFrame.

    Must be called after data_loader.load_raw() / load_test().
    The frame is sorted by (store_nbr, family, date) before calling.

    Args:
        df: Merged frame from data_loader.
        is_train: If True, also applies log1p transform to the target column.

    Returns:
        DataFrame with new feature columns appended.
    """
    df = df.copy()
    df = _calendar_features(df)
    df = _oil_features(df)
    df = _promotion_features(df)
    df = _holiday_features(df)
    df = _lag_features(df)
    df = _rolling_features(df)
    df = _store_features(df)

    if is_train and "sales" in df.columns:
        df["sales"] = np.log1p(df["sales"])

    return df


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------

def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["quarter"] = df["date"].dt.quarter
    return df


def _oil_features(df: pd.DataFrame) -> pd.DataFrame:
    if "oil_price" not in df.columns:
        return df
    df["oil_7d_ma"] = (
        df.groupby(["store_nbr", "family"])["oil_price"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    df["oil_price_diff"] = df.groupby(["store_nbr", "family"])["oil_price"].diff().fillna(0)
    return df


def _promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    if "onpromotion" not in df.columns:
        return df
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(np.int16)
    df["promo_rolling_7d"] = (
        df.groupby(["store_nbr", "family"])["onpromotion"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    return df


def _holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_national_holiday"] = df.get("is_national_holiday", pd.Series(0, index=df.index))
    df["is_local_holiday"] = df.get("is_local_holiday", pd.Series(0, index=df.index))
    df["is_holiday"] = (
        (df["is_national_holiday"] == 1) | (df["is_local_holiday"] == 1)
    ).astype(np.int8)

    # Days until next national holiday (forward-looking calendar signal, no leakage)
    holiday_dates = df.loc[df["is_national_holiday"] == 1, "date"].drop_duplicates().sort_values()
    def days_to_next(date):
        future = holiday_dates[holiday_dates > date]
        return (future.iloc[0] - date).days if len(future) else 999

    date_to_next = {d: days_to_next(d) for d in df["date"].unique()}
    df["days_to_next_holiday"] = df["date"].map(date_to_next).astype(np.int16)
    return df


def _lag_features(df: pd.DataFrame, lags: list[int] = [7, 14, 28]) -> pd.DataFrame:
    if "sales" not in df.columns:
        return df
    grp = df.groupby(["store_nbr", "family"])["sales"]
    for lag in lags:
        df[f"sales_lag_{lag}"] = grp.shift(lag)
    return df


def _rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [7, 28],
    lag: int = 7,
) -> pd.DataFrame:
    if "sales" not in df.columns:
        return df
    grp = df.groupby(["store_nbr", "family"])["sales"]
    for w in windows:
        shifted = grp.shift(lag)
        df[f"sales_roll_mean_{w}d"] = shifted.transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"sales_roll_std_{w}d"] = shifted.transform(
            lambda s: s.rolling(w, min_periods=1).std().fillna(0)
        )
    return df


def _store_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in ["type", "cluster", "city", "state", "family"] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns to pass to the model."""
    drop = {"id", "date", "sales", "store_nbr"}
    return [c for c in df.columns if c not in drop]
