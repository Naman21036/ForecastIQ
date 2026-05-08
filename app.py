from fastapi import FastAPI
from keras.models import load_model

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

app = FastAPI(
    title="Sales Forecasting API",
    description="LSTM Based Time Series Forecasting API",
    version="1.0"
)
model = load_model('best_lstm_sales_forecast.keras')
df = pd.read_csv("data/processed_sales_data.csv")
df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    format='mixed'
)

df = df.sort_values(['State', 'Date'])

scaler = MinMaxScaler()

data = df['Total'].values.reshape(-1,1)

scaled_data = scaler.fit_transform(data)

@app.get("/")
def home():

    return {
        "message": "Sales Forecasting API is running"
    }

@app.get("/forecast")
def forecast_sales():
    last_sequence = scaled_data[-30:]

    future_predictions = []

    current_sequence = last_sequence.copy()

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
    future_predictions = np.array(
        future_predictions
    ).reshape(-1,1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )
    future_predictions = np.maximum(
        future_predictions,
        0
    )
    future_dates = pd.date_range(
        start=df['Date'].max() + pd.Timedelta(days=1),
        periods=56,
        freq='D'
    )
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