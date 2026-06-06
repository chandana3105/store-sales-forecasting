"""FastAPI serving endpoint for Store Sales forecasting."""

from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import Forecaster

_forecaster: Optional[Forecaster] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _forecaster
    _forecaster = Forecaster(artifact_dir="artifacts")
    yield
    _forecaster = None


app = FastAPI(
    title="Store Sales Forecasting API",
    description="Predict daily unit sales for Corporación Favorita stores.",
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=54, description="Store number (1–54)")
    family: str = Field(..., description="Product family, e.g. BEVERAGES")
    start_date: str = Field(..., description="First forecast date in YYYY-MM-DD format")
    horizon: int = Field(15, ge=1, le=30, description="Number of days to forecast")


class ForecastPoint(BaseModel):
    date: str
    store_nbr: int
    family: str
    sales_pred: float


class ForecastResponse(BaseModel):
    store_nbr: int
    family: str
    forecast: list[ForecastPoint]
    model_cv_rmsle: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _forecaster is not None}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = _forecaster.predict_store_family(
            store_nbr=req.store_nbr,
            family=req.family,
            start_date=req.start_date,
            horizon=req.horizon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    points = [
        ForecastPoint(
            date=str(row.date.date()),
            store_nbr=int(row.store_nbr),
            family=str(row.family),
            sales_pred=round(float(row.sales_pred), 4),
        )
        for row in result.itertuples()
    ]

    return ForecastResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        forecast=points,
        model_cv_rmsle=_forecaster.schema.get("cv_rmsle"),
    )


@app.get("/families")
def list_families():
    """Return all valid product family names."""
    families = [
        "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
        "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
        "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
        "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
        "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
        "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
        "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS",
        "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
    ]
    return {"families": families}
