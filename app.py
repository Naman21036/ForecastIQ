from fastapi import FastAPI
from keras.models import load_model

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(
    title="Sales Forecasting API",
    description="LSTM Based Time Series Forecasting API",
    version="1.0"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model('best_lstm_sales_forecast.keras')

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/processed_sales_data.csv")
# ==========================================
# PREPROCESS DATA
# ==========================================

df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    format='mixed'
)

df = df.sort_values(['State', 'Date'])

# ==========================================
# SCALER
# ==========================================

scaler = MinMaxScaler()

data = df['Total'].values.reshape(-1,1)

scaled_data = scaler.fit_transform(data)

# ==========================================
# HOME ROUTE
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Sales Forecasting API is running"
    }

# ==========================================
# FORECAST ROUTE
# ==========================================

@app.get("/forecast")
def forecast_sales():

    # --------------------------------------
    # LAST 30 VALUES
    # --------------------------------------

    last_sequence = scaled_data[-30:]

    future_predictions = []

    current_sequence = last_sequence.copy()

    # --------------------------------------
    # PREDICT NEXT 56 DAYS
    # --------------------------------------

    for _ in range(56):

        pred = model.predict(
            current_sequence.reshape(1,30,1),
            verbose=0
        )

        future_predictions.append(pred[0,0])

        current_sequence = np.append(
            current_sequence[1:],
            pred
        )

    # --------------------------------------
    # INVERSE TRANSFORM
    # --------------------------------------

    future_predictions = np.array(
        future_predictions
    ).reshape(-1,1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    # --------------------------------------
    # REMOVE NEGATIVE VALUES
    # --------------------------------------

    future_predictions = np.maximum(
        future_predictions,
        0
    )

    # --------------------------------------
    # FUTURE DATES
    # --------------------------------------

    future_dates = pd.date_range(
        start=df['Date'].max() + pd.Timedelta(days=1),
        periods=56,
        freq='D'
    )

    # --------------------------------------
    # CREATE RESPONSE
    # --------------------------------------

    results = []

    for date, prediction in zip(
        future_dates,
        future_predictions
    ):

        results.append({
            "date": str(date.date()),
            "forecast_sales": float(prediction[0])
        })

    return {
        "model": "LSTM",
        "forecast_horizon": "8 Weeks",
        "predictions": results
    }