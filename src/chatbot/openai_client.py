from __future__ import annotations

import json
import os
from typing import Any

import requests

from src.chatbot.advisor import AssistantContext


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def get_openai_api_key(session_api_key: str | None = None) -> str | None:
    return session_api_key or os.getenv("OPENAI_API_KEY")


def get_default_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def build_market_system_prompt(context: AssistantContext) -> str:
    peer_lines = []
    for ticker, metrics in context.peer_summary.items():
        peer_lines.append(
            f"{ticker}: signal={metrics.get('signal')}, "
            f"annualized_return_pct={float(metrics.get('annualized_return_pct', 0.0)):.2f}, "
            f"annualized_volatility_pct={float(metrics.get('annualized_volatility_pct', 0.0)):.2f}, "
            f"sharpe_ratio={float(metrics.get('sharpe_ratio', 0.0)):.2f}"
        )

    weights_line = ", ".join(f"{ticker}={weight * 100:.1f}%" for ticker, weight in context.optimized_weights.items()) or "N/A"

    return (
        "You are a market analytics assistant inside a stock dashboard for general users. "
        "Use only the provided dashboard context. Be helpful, concise, and practical. "
        "Do not claim certainty. Do not present personalized financial advice. "
        "When users ask what to invest in, frame the answer as educational screening guidance based on the supplied metrics. "
        "Always mention key evidence like signal, RSI, MACD, volatility, drawdown, sentiment, prediction context, or peer comparison when relevant. "
        "If the user asks beyond the available context, say what data is missing.\n\n"
        f"Primary ticker: {context.ticker}\n"
        f"Latest price: {context.latest_price:.2f}\n"
        f"Signal: {context.signal}\n"
        f"Signal reason: {context.signal_reason}\n"
        f"RSI: {context.rsi:.2f}\n"
        f"MACD: {context.macd:.2f}\n"
        f"Daily return pct: {context.daily_return_pct:.2f}\n"
        f"Annualized return pct: {context.annualized_return_pct:.2f}\n"
        f"Annualized volatility pct: {context.annualized_volatility_pct:.2f}\n"
        f"Sharpe ratio: {context.sharpe_ratio:.2f}\n"
        f"Max drawdown pct: {context.max_drawdown_pct:.2f}\n"
        f"Predicted price: {context.predicted_price if context.predicted_price is not None else 'N/A'}\n"
        f"Predicted move pct: {context.predicted_move_pct if context.predicted_move_pct is not None else 'N/A'}\n"
        f"Best model name: {context.best_model_name or 'N/A'}\n"
        f"Sentiment label: {context.sentiment_label}\n"
        f"Sentiment score: {context.sentiment_score:.2f}\n"
        f"Optimized weights: {weights_line}\n"
        f"Peer summary:\n" + "\n".join(peer_lines)
    )


def generate_openai_market_reply(
    messages: list[dict[str, str]],
    context: AssistantContext,
    api_key: str,
    model: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model or get_default_openai_model(),
        "reasoning": {"effort": "low"},
        "max_output_tokens": 500,
        "input": [
            {"role": "developer", "content": build_market_system_prompt(context)},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in messages
                if message["role"] in {"user", "assistant"}
            ],
        ],
    }

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=45,
    )
    response.raise_for_status()
    body = response.json()
    output_text = body.get("output_text", "").strip()
    if output_text:
        return output_text

    output = body.get("output", [])
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                text = content.get("text", "").strip()
                if text:
                    return text
    raise ValueError("OpenAI response did not contain assistant text.")
