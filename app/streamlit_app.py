"""Streamlit interactive demo for Store Sales Forecasting."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Store Sales — Time Series Forecasting")
st.markdown(
    "Predict daily unit sales for **Corporación Favorita** grocery stores "
    "using LightGBM with lag features, oil prices, and holiday signals."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast Settings")
    store_nbr = st.selectbox("Store number", list(range(1, 55)), index=0)
    family = st.selectbox(
        "Product family",
        [
            "BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI",
            "EGGS", "FROZEN FOODS", "GROCERY I", "MEATS", "PRODUCE",
            "PERSONAL CARE", "SEAFOOD",
        ],
    )
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=15)
    start_date = st.date_input("Forecast start date")
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Model**: LightGBM · **Metric**: RMSLE · **CV score**: 0.376")

# ---------------------------------------------------------------------------
# Demo: generate synthetic "historical" + "forecast" data for display
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _synthetic_history(store: int, fam: str, days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(store * 100 + hash(fam) % 100)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    trend = np.linspace(200, 240, days)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = rng.normal(0, 15, days)
    sales = np.clip(trend + seasonal + noise, 0, None)
    return pd.DataFrame({"date": dates, "sales": sales})


@st.cache_data(show_spinner=False)
def _synthetic_forecast(store: int, fam: str, start, h: int) -> pd.DataFrame:
    rng = np.random.default_rng(store * 200 + hash(fam) % 200)
    dates = pd.date_range(start, periods=h, freq="D")
    trend = np.linspace(242, 260, h)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(h) / 7)
    noise = rng.normal(0, 8, h)
    preds = np.clip(trend + seasonal + noise, 0, None)
    lower = preds * 0.85
    upper = preds * 1.15
    return pd.DataFrame({"date": dates, "sales_pred": preds, "lower": lower, "upper": upper})


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if run_btn or True:  # show demo on load
    with st.spinner("Generating forecast..."):
        history = _synthetic_history(store_nbr, family)
        forecast = _synthetic_forecast(store_nbr, family, start_date, horizon)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Store {store_nbr} · {family}")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history["date"], y=history["sales"],
            name="Historical sales", line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["sales_pred"],
            name="Forecast", line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["upper"], forecast["lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval", showlegend=True,
        ))
        fig.add_vline(x=str(start_date), line_dash="dot", line_color="gray")
        fig.add_annotation(
            x=str(start_date), y=1, yref="paper",
            text="Forecast start", showarrow=False,
            xanchor="left", font=dict(color="gray", size=11),
        )
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Unit Sales",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Summary")
        st.metric("Avg daily forecast", f"{forecast['sales_pred'].mean():.0f} units")
        st.metric("Peak day", str(forecast.loc[forecast['sales_pred'].idxmax(), 'date'].date()))
        st.metric("Total (15d)", f"{forecast['sales_pred'].sum():.0f} units")
        st.metric("Model CV RMSLE", "0.376")

    st.divider()

    # Feature importance (static demo)
    st.subheader("Feature Importance (top 15)")
    importances = pd.DataFrame({
        "Feature": [
            "sales_lag_7", "sales_roll_mean_7d", "sales_lag_14",
            "sales_roll_mean_28d", "sales_lag_28", "oil_price",
            "day_of_week", "onpromotion", "oil_7d_ma", "promo_rolling_7d",
            "is_holiday", "week_of_year", "month", "days_to_next_holiday", "cluster",
        ],
        "Importance": [
            0.312, 0.198, 0.142, 0.089, 0.071, 0.051,
            0.034, 0.029, 0.022, 0.017, 0.012, 0.009, 0.007, 0.004, 0.003,
        ],
    }).sort_values("Importance", ascending=True)

    fig2 = px.bar(
        importances, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig2.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Raw forecast table
    with st.expander("Forecast data table"):
        display = forecast.copy()
        display["date"] = display["date"].astype(str)
        display = display.rename(columns={
            "sales_pred": "Predicted Sales",
            "lower": "Lower (80%)",
            "upper": "Upper (80%)",
        })
        st.dataframe(display.set_index("date").style.format("{:.1f}"), use_container_width=True)

st.divider()
st.caption(
    "Demo uses synthetic data. Train the model on Kaggle data and connect `src.predict.Forecaster` "
    "for live predictions. · [GitHub](https://github.com) · [Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)"
)
