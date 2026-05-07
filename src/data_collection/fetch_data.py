from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import re

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_downloaded_frame(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.reset_index()


def _cache_path_for_ticker(ticker: str) -> Path:
    safe_name = re.sub(r"[^a-z0-9]+", "_", ticker.lower()).strip("_")
    return DATA_DIR / f"{safe_name}_market.csv"


def fetch_stock_data(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    cache_path = _cache_path_for_ticker(ticker)

    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    if yf is None:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise RuntimeError("yfinance is not installed and no cached data is available.")

    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return pd.DataFrame()
    if data.empty:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return pd.DataFrame()

    normalized = _normalize_downloaded_frame(data)
    normalized.to_csv(cache_path, index=False)
    return normalized


def get_stock_news(ticker: str) -> list[dict[str, str]]:
    if yf is None:
        return []

    try:
        news_items = yf.Ticker(ticker.upper().strip()).news or []
    except Exception:
        return []

    return [
        {
            "title": item.get("title", ""),
            "publisher": item.get("publisher", "Unknown"),
            "link": item.get("link", ""),
        }
        for item in news_items[:15]
    ]


def get_available_tickers(seed_tickers: Iterable[str]) -> list[str]:
    seed_map = {ticker.upper(): ticker for ticker in seed_tickers}
    return sorted(seed_map.values(), key=str.upper)
