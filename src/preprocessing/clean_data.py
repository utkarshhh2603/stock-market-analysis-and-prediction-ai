from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    # Yahoo Finance CSV exports sometimes include a second header row with ticker symbols.
    first_date = str(cleaned.iloc[0].get("Date", "")).strip() if not cleaned.empty else ""
    if first_date and pd.isna(pd.to_datetime(first_date, errors="coerce")):
        cleaned = cleaned.iloc[1:].copy()

    if "Date" not in cleaned.columns:
        raise ValueError("Expected a Date column in the stock dataset.")

    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates(subset=["Date"]).sort_values("Date")
    cleaned = cleaned.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    cleaned["Daily Return"] = cleaned["Close"].pct_change().fillna(0.0)

    std = cleaned["Daily Return"].std(ddof=0)
    z_scores = (cleaned["Daily Return"] - cleaned["Daily Return"].mean()) / (std if std else np.nan)
    cleaned["Is Outlier"] = z_scores.abs().fillna(0.0) > 3

    price_range = cleaned["Close"].max() - cleaned["Close"].min()
    cleaned["Close Normalized"] = (
        (cleaned["Close"] - cleaned["Close"].min()) / max(price_range, np.finfo(float).eps)
    )
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned
