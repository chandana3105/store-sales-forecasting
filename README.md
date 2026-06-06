# Store Sales Time Series Forecasting

Tableau Dashboard: https://public.tableau.com/app/profile/chandana.priya.srinivasa/viz/StoreSalesForecasting_17793051184020/StoreSalesAnalysis

A machine learning project to predict daily unit sales for Corporacion Favorita, Ecuador's largest grocery chain, across 54 stores and 33 product families.

This is based on the Kaggle competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting


## Overview

Grocery stores need accurate demand forecasts to avoid overstock (waste) and understock (lost revenue). This project builds a forecasting pipeline that predicts sales 15 days ahead using historical sales data, oil prices, holidays, and store promotions.

- Metric: Root Mean Squared Logarithmic Error (RMSLE)
- Data: 54 stores, 33 product families, 1684 days (2013 to 2017)


## Results

Model               | CV RMSLE
--------------------|---------
Baseline (mean)     | 0.610
Ridge Regression    | 0.481
Random Forest       | 0.423
LightGBM (tuned)    | 0.376

LightGBM with lag features and holiday encoding achieves a 38% improvement over the naive baseline.


## Key Findings

- Oil price is the strongest external signal. A drop in oil prices correlates with lower store sales within 2 weeks.
- Promotions lift sales by around 23% on average, but the effect varies by product family.
- National holidays tend to suppress sales, while local holidays spike sales the day before.
- Grocery I and Beverages are the top two product families by sales volume across all years.
- Produce shows a clear seasonal peak in July each year.


## Project Structure

    store-sales-forecasting/
    ├── data/                   Raw CSVs from Kaggle (not committed)
    ├── src/
    │   ├── data_loader.py      Merges all Kaggle tables
    │   ├── features.py         Lag, rolling, holiday, oil features
    │   ├── train.py            Training pipeline
    │   └── predict.py          Inference utilities
    ├── api/
    │   └── main.py             FastAPI serving endpoint
    ├── app/
    │   └── streamlit_app.py    Interactive forecast demo
    ├── scripts/
    │   └── prepare_tableau.py  Exports data for Tableau dashboard
    ├── tests/
    │   └── test_features.py    Unit tests
    ├── Dockerfile
    └── requirements.txt


## How to Run

1. Clone the repo and install dependencies

    git clone https://github.com/your-username/store-sales-forecasting.git
    cd store-sales-forecasting
    pip install -r requirements.txt

2. Download data from Kaggle and place CSVs in the data folder

    kaggle competitions download -c store-sales-time-series-forecasting -p data/

3. Train the model

    python src/train.py --data-dir data --output artifacts/

4. Run the Streamlit demo

    streamlit run app/streamlit_app.py

5. Run the API

    uvicorn api.main:app --reload


## Tech Stack

- Python, pandas, numpy
- LightGBM, scikit-learn, Optuna
- FastAPI, Streamlit, Plotly
- Docker, GitHub Actions
- Tableau (dashboard)


## Setup for Development

    pip install -r requirements.txt
    pytest tests/ -v
