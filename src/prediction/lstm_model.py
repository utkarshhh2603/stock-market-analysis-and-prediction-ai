from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def _create_sequences(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(window_size, len(values)):
        X.append(values[idx - window_size : idx])
        y.append(values[idx])
    return np.array(X), np.array(y)


def run_lstm_forecast(
    df: pd.DataFrame, forecast_horizon: int = 14, window_size: int = 30, epochs: int = 8
) -> dict[str, object]:
    close_values = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_values)
    X, y = _create_sequences(scaled, window_size)
    if len(X) < 20:
        raise ValueError("Not enough data points to train the LSTM model.")

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Sequential(
        [
            Input(shape=(window_size, 1)),
            LSTM(48, return_sequences=True),
            Dropout(0.15),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    test_predictions = model.predict(X_test, verbose=0)
    rmse = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(test_predictions)))

    rolling_window = scaled[-window_size:].copy()
    forecast: list[float] = []
    for _ in range(forecast_horizon):
        next_step = model.predict(rolling_window.reshape(1, window_size, 1), verbose=0)[0][0]
        forecast.append(next_step)
        rolling_window = np.append(rolling_window[1:], [[next_step]], axis=0)

    inverted_forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_dates = pd.bdate_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Close": inverted_forecast})
    return {"rmse": float(rmse), "forecast": forecast_df}
