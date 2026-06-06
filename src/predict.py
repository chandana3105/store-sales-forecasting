"""Inference utilities for Store Sales forecasting."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_features, get_feature_cols

logger = logging.getLogger(__name__)


class Forecaster:
    """Load a trained model and generate sales forecasts."""

    def __init__(self, artifact_dir: str | Path = "artifacts"):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model_lgbm.joblib")
        with open(artifact_dir / "schema.json") as f:
            self.schema = json.load(f)
        self.feature_cols = self.schema["feature_cols"]
        logger.info(f"Loaded model. CV RMSLE: {self.schema.get('cv_rmsle', 'N/A')}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for a prepared (feature-engineered) DataFrame.

        Returns the input frame with a `sales_pred` column added.
        Applies inverse log1p transform automatically.
        """
        available = [c for c in self.feature_cols if c in df.columns]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"{len(missing)} feature(s) missing — filling with 0: {missing[:5]}")
            for c in missing:
                df[c] = 0

        log_preds = self.model.predict(df[available + [c for c in missing]])
        df = df.copy()
        df["sales_pred"] = np.expm1(np.clip(log_preds, 0, None))
        return df

    def predict_store_family(
        self,
        store_nbr: int,
        family: str,
        start_date: str,
        horizon: int = 15,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Convenience method: forecast `horizon` days for one store-family pair.

        Args:
            store_nbr: Store number (1–54).
            family: Product family string (e.g. "BEVERAGES").
            start_date: First forecast date as "YYYY-MM-DD".
            horizon: Number of days to forecast.
            history: Optional historical sales DataFrame to compute lag features.

        Returns:
            DataFrame with columns [date, store_nbr, family, sales_pred].
        """
        dates = pd.date_range(start_date, periods=horizon, freq="D")
        future = pd.DataFrame({
            "date": dates,
            "store_nbr": store_nbr,
            "family": family,
            "onpromotion": 0,
            "sales": np.nan,
        })
        if history is not None:
            combined = pd.concat([history, future], ignore_index=True)
            combined = build_features(combined, is_train=False)
            future_feats = combined[combined["date"].isin(dates)].copy()
        else:
            future_feats = build_features(future, is_train=False)

        result = self.predict(future_feats)
        return result[["date", "store_nbr", "family", "sales_pred"]].reset_index(drop=True)
