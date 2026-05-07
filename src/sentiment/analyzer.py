from __future__ import annotations

import re

import pandas as pd


POSITIVE_WORDS = {"beat", "bullish", "growth", "gain", "rally", "upgrade", "profit", "optimistic", "strong", "surge"}
NEGATIVE_WORDS = {"miss", "bearish", "drop", "downgrade", "loss", "lawsuit", "weak", "cut", "fall", "decline"}


def _score_text(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0
    positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
    negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
    return (positive_hits - negative_hits) / max(len(tokens), 1)


def analyze_headlines(headlines: list[dict[str, str]]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for item in headlines:
        title = item.get("title", "").strip()
        if not title:
            continue
        score = _score_text(title)
        label = "positive" if score > 0.02 else "negative" if score < -0.02 else "neutral"
        rows.append(
            {
                "headline": title,
                "publisher": item.get("publisher", "Unknown"),
                "sentiment_score": score,
                "sentiment_label": label,
                "link": item.get("link", ""),
            }
        )

    sentiment_df = pd.DataFrame(rows)
    if sentiment_df.empty:
        return sentiment_df, {"positive": 0, "neutral": 0, "negative": 0, "average_score": 0.0}

    summary = {
        "positive": int((sentiment_df["sentiment_label"] == "positive").sum()),
        "neutral": int((sentiment_df["sentiment_label"] == "neutral").sum()),
        "negative": int((sentiment_df["sentiment_label"] == "negative").sum()),
        "average_score": float(sentiment_df["sentiment_score"].mean()),
    }
    return sentiment_df.sort_values("sentiment_score", ascending=False), summary
