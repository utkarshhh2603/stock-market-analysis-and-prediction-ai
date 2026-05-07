from __future__ import annotations

import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/processed/market_platform.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_close REAL NOT NULL,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weights_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(ticker: str, model_name: str, predicted_close: float, score: float) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions (ticker, model_name, predicted_close, score) VALUES (?, ?, ?, ?)",
            (ticker, model_name, predicted_close, score),
        )
        connection.commit()


def save_portfolio_snapshot(weights: list[dict[str, float]], metrics: dict[str, float]) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO portfolio_snapshots (weights_json, metrics_json) VALUES (?, ?)",
            (json.dumps(weights), json.dumps(metrics)),
        )
        connection.commit()
