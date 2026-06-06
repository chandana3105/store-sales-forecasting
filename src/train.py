"""Training entry point for the Store Sales forecasting pipeline."""

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import load_raw
from src.features import build_features, get_feature_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE on log1p-transformed targets (equivalent to RMSE in log space)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cross_validate(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> float:
    """Blocked time-series cross-validation. Returns mean RMSLE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dates = df["date"].values
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(dates)):
        X_tr = df.iloc[train_idx][feature_cols]
        y_tr = df.iloc[train_idx]["sales"]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx]["sales"]

        model = _default_lgbm()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        preds = model.predict(X_val)
        score = rmsle(y_val.values, preds)
        scores.append(score)
        logger.info(f"Fold {fold + 1}/{n_splits} — RMSLE: {score:.4f}")

    mean_score = float(np.mean(scores))
    logger.info(f"CV mean RMSLE: {mean_score:.4f}")
    return mean_score


def tune(df: pd.DataFrame, feature_cols: list[str], n_trials: int = 50) -> dict:
    """Optuna hyperparameter search. Returns best LightGBM params."""
    tscv = TimeSeriesSplit(n_splits=3)
    dates = df["date"].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        fold_scores = []
        for train_idx, val_idx in tscv.split(dates):
            X_tr = df.iloc[train_idx][feature_cols]
            y_tr = df.iloc[train_idx]["sales"]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx]["sales"]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
            fold_scores.append(rmsle(y_val.values, model.predict(X_val)))
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best params: {study.best_params} — RMSLE: {study.best_value:.4f}")
    return study.best_params


def train_final(df: pd.DataFrame, feature_cols: list[str], params: dict | None = None) -> lgb.LGBMRegressor:
    """Train final model on the full dataset."""
    model = lgb.LGBMRegressor(**(params or _default_lgbm_params()))
    model.fit(df[feature_cols], df["sales"])
    return model


def _default_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**_default_lgbm_params())


def _default_lgbm_params() -> dict:
    return {
        "n_estimators": 800,
        "num_leaves": 127,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Store Sales forecasting model")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw Kaggle CSVs")
    parser.add_argument("--output", default="artifacts", help="Output directory for model & schema")
    parser.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_dir}")
    df = load_raw(args.data_dir)
    df = build_features(df, is_train=True)
    df = df.dropna(subset=["sales"])

    feature_cols = get_feature_cols(df)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]} ...")

    cv_score = cross_validate(df, feature_cols)

    best_params = None
    if args.tune:
        logger.info("Running Optuna tuning...")
        best_params = tune(df, feature_cols, n_trials=args.n_trials)

    logger.info("Training final model on full dataset...")
    model = train_final(df, feature_cols, params=best_params)

    os.makedirs(args.output, exist_ok=True)
    model_path = Path(args.output) / "model_lgbm.joblib"
    joblib.dump(model, model_path)

    schema = {
        "feature_cols": feature_cols,
        "target": "sales",
        "target_transform": "log1p",
        "cv_rmsle": cv_score,
    }
    with open(Path(args.output) / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Saved model → {model_path}")
    logger.info(f"CV RMSLE: {cv_score:.4f}")


if __name__ == "__main__":
    main()
