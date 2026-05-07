from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["SMA_20"] = SMAIndicator(close=enriched["Close"], window=20).sma_indicator()
    enriched["EMA_20"] = EMAIndicator(close=enriched["Close"], window=20).ema_indicator()
    enriched["RSI"] = RSIIndicator(close=enriched["Close"], window=14).rsi()

    macd = MACD(close=enriched["Close"])
    enriched["MACD"] = macd.macd()
    enriched["MACD_Signal"] = macd.macd_signal()
    enriched["MACD_Hist"] = macd.macd_diff()

    bands = BollingerBands(close=enriched["Close"], window=20, window_dev=2)
    enriched["BB_High"] = bands.bollinger_hband()
    enriched["BB_Low"] = bands.bollinger_lband()
    enriched["BB_Mid"] = bands.bollinger_mavg()

    enriched["OBV"] = OnBalanceVolumeIndicator(close=enriched["Close"], volume=enriched["Volume"]).on_balance_volume()
    enriched["Volatility_20"] = enriched["Daily Return"].rolling(20).std() * np.sqrt(252)

    patterns = detect_candlestick_patterns(enriched)
    for column, values in patterns.items():
        enriched[column] = values

    enriched["Signal"] = generate_trade_signals(enriched)
    return enriched


def detect_candlestick_patterns(df: pd.DataFrame) -> dict[str, pd.Series]:
    body = (df["Close"] - df["Open"]).abs()
    full_range = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)

    doji = (body / full_range) < 0.1
    hammer = (lower_shadow > body * 2) & (upper_shadow < body)
    shooting_star = (upper_shadow > body * 2) & (lower_shadow < body)
    bullish_engulfing = (prev_close < prev_open) & (df["Close"] > df["Open"]) & (df["Close"] >= prev_open) & (df["Open"] <= prev_close)
    bearish_engulfing = (prev_close > prev_open) & (df["Close"] < df["Open"]) & (df["Open"] >= prev_close) & (df["Close"] <= prev_open)
    morning_star = (
        (prev_close.shift(1) < prev_open.shift(1))
        & (body.shift(1) < body.shift(2) * 0.5)
        & (df["Close"] > df["Open"])
        & (df["Close"] > ((prev_open.shift(1) + prev_close.shift(1)) / 2))
    )

    return {
        "Doji": doji.fillna(False),
        "Hammer": hammer.fillna(False),
        "Shooting_Star": shooting_star.fillna(False),
        "Bullish_Engulfing": bullish_engulfing.fillna(False),
        "Bearish_Engulfing": bearish_engulfing.fillna(False),
        "Morning_Star": morning_star.fillna(False),
    }


def generate_trade_signals(df: pd.DataFrame) -> pd.Series:
    buy_signal = (df["RSI"] < 30) & (df["MACD"] > df["MACD_Signal"])
    sell_signal = (df["RSI"] > 70) & (df["MACD"] < df["MACD_Signal"])
    return pd.Series(np.select([buy_signal, sell_signal], ["BUY", "SELL"], default="HOLD"), index=df.index)


def build_signal_summary(df: pd.DataFrame) -> dict[str, str]:
    signal = df.iloc[-1]["Signal"]
    if signal == "BUY":
        reason = "RSI is oversold and MACD crossed above signal."
    elif signal == "SELL":
        reason = "RSI is overbought and MACD crossed below signal."
    else:
        reason = "Momentum is mixed, so the engine stays neutral."
    return {"signal": signal, "reason": reason}
