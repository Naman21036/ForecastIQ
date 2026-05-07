# ForecastIQ 📈

ForecastIQ is an end to end sales forecasting system built using multiple time series forecasting techniques including SARIMA, Facebook Prophet, XGBoost, and LSTM Deep Learning models.

The project performs:
- Data preprocessing and feature engineering
- Time series forecasting
- Model comparison using RMSE and MAE
- Best model selection
- Future 8 week sales prediction
- FastAPI based REST API deployment

## Tech Stack

- Python
- Pandas
- Scikit Learn
- XGBoost
- TensorFlow / Keras
- FastAPI
- Matplotlib

## Best Performing Model

✅ LSTM Deep Learning Model

## Run FastAPI Server

```bash
uvicorn app:app --reload