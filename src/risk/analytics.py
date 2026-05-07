from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict[str, object]:
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        raise ValueError("Risk metrics require at least two price observations.")

    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std(ddof=0) * np.sqrt(252)
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    cumulative_returns = (1 + returns).cumprod()
    rolling_peak = cumulative_returns.cummax()
    drawdown_curve = (cumulative_returns / rolling_peak) - 1
    max_drawdown = drawdown_curve.min()

    return {
        "returns": returns,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "drawdown_curve": drawdown_curve,
        "max_drawdown": float(max_drawdown),
    }
