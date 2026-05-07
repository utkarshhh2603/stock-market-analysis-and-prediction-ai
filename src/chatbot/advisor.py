from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssistantContext:
    ticker: str
    latest_price: float
    signal: str
    signal_reason: str
    rsi: float
    macd: float
    daily_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    predicted_price: float | None
    predicted_move_pct: float | None
    best_model_name: str | None
    sentiment_label: str
    sentiment_score: float
    peer_summary: dict[str, dict[str, float | str]]
    optimized_weights: dict[str, float]


def default_prompts() -> list[str]:
    return [
        "What does the market scenario look like for this stock?",
        "Is this ticker looking risky right now?",
        "What should I put money on if I want lower risk?",
        "Compare the selected stock with the peer group.",
        "How should I think about portfolio allocation?",
    ]


def answer_market_question(question: str, context: AssistantContext) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask about trend, risk, prediction, sentiment, comparison, or portfolio allocation."

    if _matches(q, ["hello", "hi", "help", "what can you do"]):
        return (
            "I can summarize the selected stock, explain risk, compare peer tickers, discuss sentiment, "
            "and show a data-based portfolio view. I keep the guidance educational, not personal financial advice."
        )

    if _matches(q, ["sentiment", "news", "headline"]):
        return (
            f"{context.ticker} sentiment is currently `{context.sentiment_label}` with an average score of "
            f"{context.sentiment_score:.2f}. Treat this as a mood signal, not a standalone investment trigger."
        )

    if _matches(q, ["predict", "forecast", "target price", "next price"]):
        if context.predicted_price is None or context.predicted_move_pct is None or context.best_model_name is None:
            return (
                f"I do not have a fresh prediction for {context.ticker} yet. The current technical signal is "
                f"`{context.signal}`, with RSI {context.rsi:.1f} and MACD {context.macd:.2f}."
            )
        direction = "upside" if context.predicted_move_pct >= 0 else "downside"
        return (
            f"The best short-horizon model right now is `{context.best_model_name}`, which estimates the next close near "
            f"${context.predicted_price:.2f}. That implies about {abs(context.predicted_move_pct):.2f}% {direction} "
            f"from the latest price of ${context.latest_price:.2f}. This is a statistical estimate, not a guarantee."
        )

    if _matches(q, ["risk", "volatile", "safe", "safer", "drawdown"]):
        lower_risk = _best_peer(context.peer_summary, "annualized_volatility_pct", reverse=False)
        return (
            f"{context.ticker} has annualized volatility near {context.annualized_volatility_pct:.2f}% and a max drawdown "
            f"around {context.max_drawdown_pct:.2f}%, with a Sharpe ratio of {context.sharpe_ratio:.2f}. "
            f"Among the tracked peers, `{lower_risk}` currently looks lower-volatility. Lower volatility still does not mean low risk."
        )

    if _matches(q, ["compare", "better than", "best stock", "which stock", "what to put money on", "invest in"]):
        if "safe" in q or "low risk" in q or "less risk" in q:
            best = _best_peer(context.peer_summary, "annualized_volatility_pct", reverse=False)
            reason = f"it currently shows the lowest annualized volatility in the tracked peer set."
        elif "growth" in q or "upside" in q or "return" in q:
            best = _best_peer(context.peer_summary, "annualized_return_pct", reverse=True)
            reason = f"it currently shows the strongest annualized return profile in the tracked peer set."
        else:
            best = _best_peer(context.peer_summary, "sharpe_ratio", reverse=True)
            reason = f"it currently has the strongest risk-adjusted return, measured by Sharpe ratio, in the tracked peer set."

        summary = _format_peer_summary(context.peer_summary)
        return (
            f"If you want a data-based starting point rather than a personal recommendation, `{best}` stands out because {reason} "
            f"Peer snapshot: {summary} Use this as a screening aid and match it to your time horizon, risk tolerance, and diversification plan."
        )

    if _matches(q, ["portfolio", "allocation", "diversify", "weights"]):
        weights = ", ".join(f"{ticker} {weight * 100:.1f}%" for ticker, weight in context.optimized_weights.items())
        return (
            f"The current optimized peer allocation is: {weights}. This favors the best historical risk-adjusted mix in the tracked data, "
            f"but it is still backward-looking and should be sanity-checked against concentration risk and your goals."
        )

    if _matches(q, ["signal", "buy", "sell", "hold", "market scenario", "trend"]):
        prediction_text = ""
        if context.predicted_price is not None and context.predicted_move_pct is not None:
            move_word = "up" if context.predicted_move_pct >= 0 else "down"
            prediction_text = (
                f" The short-horizon model also points {move_word} toward ${context.predicted_price:.2f} "
                f"({context.predicted_move_pct:.2f}% vs current price)."
            )
        return (
            f"For {context.ticker}, the dashboard currently reads `{context.signal}`. {context.signal_reason} "
            f"RSI is {context.rsi:.1f}, MACD is {context.macd:.2f}, daily move is {context.daily_return_pct:.2f}%, "
            f"and sentiment is `{context.sentiment_label}`.{prediction_text} This is useful as market context, not a personal buy call."
        )

    return (
        f"I can help with {context.ticker}'s trend, signal, forecast, sentiment, risk, peer comparison, or portfolio mix. "
        f"Try asking: 'Is this stock risky?', 'Which tracked stock looks strongest?', or 'How should I diversify?'"
    )


def _matches(question: str, phrases: list[str]) -> bool:
    return any(phrase in question for phrase in phrases)


def _best_peer(peer_summary: dict[str, dict[str, float | str]], metric: str, reverse: bool) -> str:
    ranked = sorted(
        peer_summary.items(),
        key=lambda item: float(item[1].get(metric, 0.0)),
        reverse=reverse,
    )
    return ranked[0][0] if ranked else "N/A"


def _format_peer_summary(peer_summary: dict[str, dict[str, float | str]]) -> str:
    parts = []
    for ticker, metrics in peer_summary.items():
        parts.append(
            f"{ticker}: return {float(metrics['annualized_return_pct']):.1f}%, "
            f"volatility {float(metrics['annualized_volatility_pct']):.1f}%, "
            f"Sharpe {float(metrics['sharpe_ratio']):.2f}"
        )
    return " | ".join(parts)
