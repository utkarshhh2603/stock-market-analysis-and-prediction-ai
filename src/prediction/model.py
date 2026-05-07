from __future__ import annotations

import math

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "Volatility_20",
]


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    model_df = df.copy().dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    model_df["Target"] = model_df["Close"].shift(-1)
    model_df = model_df.dropna(subset=["Target"])
    X = model_df[FEATURE_COLUMNS]
    y = model_df["Target"]
    latest_features = df[FEATURE_COLUMNS].dropna().tail(1)
    return X, y, latest_features


def _evaluate_model(model, X: pd.DataFrame, y: pd.Series, latest_features: pd.DataFrame) -> dict[str, object]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    next_prediction = model.predict(latest_features)[0] if not latest_features.empty else float("nan")
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(predictions),
        "next_prediction": float(next_prediction),
    }


def run_ml_prediction_suite(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    X, y, latest_features = _prepare_features(df)
    if len(X) < 50:
        raise ValueError("Not enough rows after feature engineering for ML prediction.")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

    return {name: _evaluate_model(model, X, y, latest_features) for name, model in models.items()}
