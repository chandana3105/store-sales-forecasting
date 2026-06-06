"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_feature_cols, _calendar_features


@pytest.fixture
def sample_df():
    """Minimal DataFrame that mimics the merged loader output."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    n = len(dates)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "store_nbr": 1,
        "family": "BEVERAGES",
        "sales": rng.uniform(50, 300, n),
        "onpromotion": rng.integers(0, 10, n),
        "oil_price": rng.uniform(70, 90, n),
        "is_national_holiday": 0,
        "is_local_holiday": 0,
        "type": "A",
        "cluster": 1,
        "city": "Quito",
        "state": "Pichincha",
    })


def test_calendar_features_added(sample_df):
    out = _calendar_features(sample_df.copy())
    for col in ["day_of_week", "month", "year", "is_weekend", "quarter"]:
        assert col in out.columns, f"Missing column: {col}"


def test_build_features_shape(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    assert out.shape[0] == len(sample_df)
    assert out.shape[1] > sample_df.shape[1]


def test_log1p_applied_to_sales(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    original_sales = sample_df["sales"].values
    expected = np.log1p(original_sales)
    np.testing.assert_allclose(out["sales"].values, expected, rtol=1e-5)


def test_lag_features_present(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    for lag in [7, 14, 28]:
        assert f"sales_lag_{lag}" in out.columns


def test_rolling_features_present(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    for w in [7, 28]:
        assert f"sales_roll_mean_{w}d" in out.columns
        assert f"sales_roll_std_{w}d" in out.columns


def test_no_negative_predictions_after_expm1(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    assert (out["sales"] >= 0).all(), "log1p-transformed sales should be non-negative"


def test_get_feature_cols_excludes_target(sample_df):
    out = build_features(sample_df.copy(), is_train=True)
    cols = get_feature_cols(out)
    assert "sales" not in cols
    assert "date" not in cols
    assert "id" not in cols


def test_is_weekend_correctness(sample_df):
    out = _calendar_features(sample_df.copy())
    for _, row in out.iterrows():
        expected = 1 if row["date"].dayofweek >= 5 else 0
        assert row["is_weekend"] == expected
