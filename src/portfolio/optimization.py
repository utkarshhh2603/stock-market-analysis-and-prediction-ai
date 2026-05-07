from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_portfolio_performance(
    price_frame: pd.DataFrame, weights: dict[str, float], risk_free_rate: float = 0.02
) -> dict[str, float]:
    returns = price_frame.pct_change().dropna()
    weight_array = np.array([weights[ticker] for ticker in returns.columns])
    annualized_return = np.sum(returns.mean() * weight_array) * 252
    covariance = returns.cov() * 252
    annualized_volatility = np.sqrt(weight_array.T @ covariance.values @ weight_array)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
    }


def optimize_portfolio(price_frame: pd.DataFrame, risk_free_rate: float = 0.02) -> dict[str, dict[str, float]]:
    returns = price_frame.pct_change().dropna()
    asset_count = len(returns.columns)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)
    initial_weights = np.repeat(1 / asset_count, asset_count)
    covariance = returns.cov() * 252
    mean_returns = returns.mean() * 252

    def objective(weights: np.ndarray) -> float:
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ covariance.values @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        return -sharpe_ratio

    result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_weights = result.x if result.success else initial_weights
    normalized = optimal_weights / optimal_weights.sum()
    return {"weights": {ticker: float(weight) for ticker, weight in zip(returns.columns, normalized)}}
