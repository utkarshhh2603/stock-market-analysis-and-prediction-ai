from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.chatbot.advisor import AssistantContext, answer_market_question, default_prompts
from src.chatbot.openai_client import (
    generate_openai_market_reply,
    get_default_openai_model,
    get_openai_api_key,
)
from src.data_collection.fetch_data import fetch_stock_data, get_available_tickers, get_stock_news
from src.database.storage import init_database, save_portfolio_snapshot, save_prediction
from src.indicators.indicators import add_technical_indicators, build_signal_summary
from src.portfolio.optimization import compute_portfolio_performance, optimize_portfolio
from src.prediction.lstm_model import run_lstm_forecast
from src.prediction.model import run_ml_prediction_suite
from src.preprocessing.clean_data import clean_stock_data
from src.risk.analytics import calculate_risk_metrics
from src.sentiment.analyzer import analyze_headlines
from src.visualization.charts import (
    create_candlestick_chart,
    create_correlation_heatmap,
    create_forecast_chart,
    create_indicator_chart,
    create_prediction_comparison_chart,
    create_returns_histogram,
    create_volume_chart,
)

st.set_page_config(page_title="Stock Market AI Platform", page_icon="chart_with_upwards_trend", layout="wide")

EQUITY_UNIVERSE = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "NFLX",
    "AMD",
    "JPM",
    "BAC",
    "WMT",
    "DIS",
    "BABA",
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
]

FOREX_UNIVERSE = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    "USDINR=X",
]

MARKET_UNIVERSES = {
    "Equities": EQUITY_UNIVERSE,
    "Forex": FOREX_UNIVERSE,
}


def render_app_shell() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

        :root {
            --bg: #120f1d;
            --bg-2: #1a1530;
            --panel: #221a3f;
            --panel-2: #2c2252;
            --ink: #f8f5ec;
            --muted: #b8aed6;
            --line: #4f4680;
            --accent: #2dd4bf;
            --accent-2: #f59e0b;
            --accent-3: #fb7185;
            --positive: #4ade80;
            --negative: #fb7185;
            --warning: #facc15;
            --shadow: 8px 8px 0px #0b0814;
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Mono", monospace;
            color: var(--ink);
        }

        .stApp {
            background:
                linear-gradient(0deg, rgba(255,255,255,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
                linear-gradient(180deg, #0f0b19 0%, var(--bg) 55%, #140f23 100%);
            background-size: 24px 24px, 24px 24px, auto;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        [data-testid="stHeader"] {
            background: rgba(18, 15, 29, 0.92);
            border-bottom: 3px solid var(--line);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #171128 0%, #100d1e 100%);
            border-right: 4px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: #f6f1e8 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] .stDateInput,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stMultiSelect {
            background: rgba(255,255,255,0.05);
            border-radius: 0;
        }

        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background: transparent;
            border: 3px solid var(--line);
            border-radius: 0;
            box-shadow: 4px 4px 0 #0b0814;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            font-family: "Press Start 2P", monospace;
            color: var(--ink);
            letter-spacing: 0.03em;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border: 4px solid var(--line);
            border-radius: 0;
            background:
                linear-gradient(135deg, rgba(45, 212, 191, 0.18), transparent 35%),
                linear-gradient(225deg, rgba(245, 158, 11, 0.16), transparent 35%),
                linear-gradient(180deg, var(--panel-2), var(--panel));
            box-shadow: var(--shadow);
            padding: 1.5rem;
            margin-bottom: 1.4rem;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.55rem 0.8rem;
            border-radius: 0;
            border: 3px solid var(--accent);
            background: rgba(45, 212, 191, 0.10);
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-title {
            margin: 0.9rem 0 0.35rem 0;
            font-size: clamp(1.35rem, 2.2vw, 2.35rem);
            line-height: 1.35;
            max-width: 15ch;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 0.92rem;
            max-width: 68ch;
            margin-bottom: 0;
        }

        .hero-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .hero-stat {
            border-radius: 0;
            padding: 1rem 1.05rem;
            background: rgba(16, 13, 30, 0.68);
            border: 3px solid var(--line);
            box-shadow: 4px 4px 0 #0b0814;
        }

        .hero-stat-label {
            color: var(--muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-stat-value {
            margin-top: 0.3rem;
            font-family: "Press Start 2P", monospace;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.5;
        }

        .section-label {
            margin: 0.4rem 0 0.8rem 0;
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .kpi-card {
            border-radius: 0;
            background: linear-gradient(180deg, var(--panel), #1b1433);
            border: 3px solid var(--line);
            box-shadow: var(--shadow);
            padding: 1rem 1.1rem;
            min-height: 120px;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .kpi-value {
            margin-top: 0.35rem;
            font-family: "Press Start 2P", monospace;
            font-size: 1rem;
            line-height: 1.45;
            font-weight: 700;
        }

        .kpi-delta {
            margin-top: 0.55rem;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .signal-banner {
            padding: 1.1rem 1.2rem;
            border-radius: 0;
            background: linear-gradient(135deg, #171128, #221a3f);
            border: 3px solid var(--line);
            box-shadow: var(--shadow);
            margin: 1rem 0 1.25rem 0;
        }

        .signal-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 0;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 2px solid currentColor;
        }

        .signal-title {
            margin-top: 0.7rem;
            font-family: "Press Start 2P", monospace;
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1.6;
        }

        .signal-copy {
            color: var(--muted);
            margin-top: 0.2rem;
            margin-bottom: 0;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: rgba(34, 26, 63, 0.78);
            padding: 0.45rem;
            border: 3px solid var(--line);
            border-radius: 0;
            box-shadow: var(--shadow);
            overflow-x: auto;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            height: 46px;
            border-radius: 0;
            padding: 0 1rem;
            background: transparent;
            color: var(--muted);
            font-weight: 700;
            font-size: 0.72rem;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, var(--accent), #14b8a6) !important;
            color: #0e1117 !important;
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        .stAlert,
        .stExpander,
        [data-testid="stMetric"],
        .stChatMessage {
            border-radius: 0;
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        .stExpander,
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, var(--panel), #1b1433);
            border: 3px solid var(--line);
            box-shadow: var(--shadow);
            padding: 0.35rem;
        }

        [data-testid="stMetric"] {
            padding: 1rem 1.1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.68rem;
        }

        [data-testid="stMetricValue"] {
            font-family: "Press Start 2P", monospace;
        }

        .stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            border-radius: 0;
            border: 3px solid #0b0814;
            background: linear-gradient(135deg, var(--accent-2), #facc15);
            color: #18121d;
            font-weight: 700;
            box-shadow: 4px 4px 0 #0b0814;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, var(--accent), #67e8f9);
            color: #18121d;
        }

        .stChatMessage {
            background: linear-gradient(180deg, var(--panel), #1b1433) !important;
            border: 3px solid var(--line);
            box-shadow: var(--shadow);
            padding: 0.35rem 0.4rem;
        }

        [data-testid="stChatInput"] {
            background: rgba(34, 26, 63, 0.92);
            border: 3px solid var(--line);
            border-radius: 0;
            box-shadow: var(--shadow);
        }

        .stTextInput > div > div,
        .stTextArea textarea,
        .stMultiSelect [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"],
        .stDateInput [data-baseweb="input"] {
            border-radius: 0 !important;
            border: 3px solid var(--line) !important;
            background: rgba(34, 26, 63, 0.92) !important;
            color: var(--ink) !important;
            box-shadow: 4px 4px 0 #0b0814;
        }

        .pixel-window {
            border: 4px solid var(--line);
            background: linear-gradient(180deg, var(--panel), #1b1433);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .pixel-window-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            background: #100d1e;
            padding: 0.8rem 1rem;
            border-bottom: 4px solid var(--line);
            color: var(--accent);
            font-family: "Press Start 2P", monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            line-height: 1.6;
        }

        .pixel-window-body {
            padding: 1rem;
        }

        .pixel-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.8rem 0 0.2rem 0;
        }

        .pixel-chip {
            padding: 0.5rem 0.65rem;
            border: 2px solid var(--line);
            background: rgba(45, 212, 191, 0.08);
            color: var(--ink);
            font-size: 0.72rem;
            text-transform: uppercase;
        }

        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin: 0.9rem 0 0.2rem 0;
        }

        .status-node {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.55rem 0.7rem;
            border: 2px solid var(--line);
            background: rgba(255,255,255,0.04);
            box-shadow: 4px 4px 0 #0b0814;
            text-transform: uppercase;
            font-size: 0.68rem;
        }

        .status-light {
            width: 12px;
            height: 12px;
            border: 2px solid #0b0814;
            box-shadow: 0 0 0 2px currentColor inset;
            animation: blink 1.35s steps(2, end) infinite;
        }

        .sprite-icon {
            width: 14px;
            height: 14px;
            background:
                linear-gradient(90deg, currentColor 0 100%);
            clip-path: polygon(0 30%, 30% 30%, 30% 0, 70% 0, 70% 30%, 100% 30%, 100% 70%, 70% 70%, 70% 100%, 30% 100%, 30% 70%, 0 70%);
        }

        .hud-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.25rem 0;
        }

        .hud-card {
            border: 3px solid var(--line);
            background: linear-gradient(180deg, rgba(17, 13, 33, 0.95), rgba(34, 26, 63, 0.95));
            box-shadow: 4px 4px 0 #0b0814;
            padding: 0.85rem;
        }

        .hud-label {
            color: var(--muted);
            font-size: 0.62rem;
            text-transform: uppercase;
            line-height: 1.5;
        }

        .hud-value {
            margin-top: 0.55rem;
            font-family: "Press Start 2P", monospace;
            font-size: 0.84rem;
            line-height: 1.55;
            color: var(--ink);
        }

        .scanbar {
            height: 18px;
            border: 3px solid var(--line);
            background: #120f1d;
            position: relative;
            overflow: hidden;
            margin-top: 0.6rem;
        }

        .scanbar-fill {
            position: absolute;
            inset: 0 auto 0 0;
            height: 100%;
            background: repeating-linear-gradient(
                90deg,
                var(--accent) 0px,
                var(--accent) 10px,
                #67e8f9 10px,
                #67e8f9 20px
            );
        }

        .radar-box {
            border: 3px solid var(--line);
            background: rgba(17, 13, 33, 0.86);
            box-shadow: 4px 4px 0 #0b0814;
            padding: 1rem;
        }

        .radar-line {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.45rem 0;
            border-bottom: 1px dashed rgba(184, 174, 214, 0.25);
            font-size: 0.78rem;
        }

        .radar-line:last-child {
            border-bottom: 0;
        }

        .terminal-note {
            border: 3px solid var(--accent-2);
            background: rgba(245, 158, 11, 0.08);
            color: var(--ink);
            padding: 0.8rem 1rem;
            margin: 0.8rem 0;
            box-shadow: 4px 4px 0 #0b0814;
            font-size: 0.78rem;
            line-height: 1.7;
        }

        @keyframes blink {
            0%, 49% { opacity: 1; }
            50%, 100% { opacity: 0.45; }
        }

        @media (max-width: 1100px) {
            .hud-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 900px) {
            .hero-stat-grid {
                grid-template-columns: 1fr;
            }
            .hud-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(context: AssistantContext) -> None:
    move_text = "No live forecast"
    if context.predicted_move_pct is not None:
        direction = "Upside bias" if context.predicted_move_pct >= 0 else "Downside bias"
        move_text = f"{direction} {context.predicted_move_pct:+.2f}%"

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Market Intelligence Layer</div>
            <div class="hero-title">{context.ticker} Research Terminal</div>
            <p class="hero-copy">
                Fresh visual shell for technical analytics, forecasting, risk diagnostics, sentiment context,
                portfolio screening, and an AI market assistant for public-facing questions.
            </p>
            <div class="hero-stat-grid">
                <div class="hero-stat">
                    <div class="hero-stat-label">Current Signal</div>
                    <div class="hero-stat-value">{context.signal}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Price + Daily Move</div>
                    <div class="hero-stat-value">${context.latest_price:.2f} · {context.daily_return_pct:+.2f}%</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Forecast Tone</div>
                    <div class="hero-stat-value">{move_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def open_pixel_window(title: str, status: str) -> None:
    st.markdown(
        f"""
        <div class="pixel-window">
            <div class="pixel-window-head">
                <span>{title}</span>
                <span>{status}</span>
            </div>
            <div class="pixel-window-body">
        """,
        unsafe_allow_html=True,
    )


def close_pixel_window() -> None:
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta: str = "", tone: str = "neutral") -> None:
    colors = {
        "positive": "var(--positive)",
        "negative": "var(--negative)",
        "warning": "var(--warning)",
        "neutral": "var(--muted)",
    }
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color: {colors.get(tone, 'var(--muted)')};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_banner(signal: str, reason: str) -> None:
    palette = {
        "BUY": ("rgba(21, 128, 61, 0.14)", "var(--positive)"),
        "SELL": ("rgba(185, 28, 28, 0.12)", "var(--negative)"),
        "HOLD": ("rgba(180, 83, 9, 0.12)", "var(--warning)"),
    }
    bg, fg = palette.get(signal, ("rgba(13, 148, 136, 0.1)", "var(--accent)"))
    st.markdown(
        f"""
        <div class="signal-banner">
            <span class="signal-pill" style="background:{bg}; color:{fg};">{signal}</span>
            <div class="signal-title">Current market posture: {signal}</div>
            <p class="signal-copy">{reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation_hint(current_page: str) -> None:
    st.markdown(
        f"""
        <div class="pixel-chip-row">
            <div class="pixel-chip">Mode: {current_page}</div>
            <div class="pixel-chip">Theme: Pixel Terminal</div>
            <div class="pixel-chip">Chatbot: Dedicated Screen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hud_row(items: list[tuple[str, str]]) -> None:
    cards = []
    for label, value in items:
        cards.append(
            f"""
            <div class="hud-card">
                <div class="hud-label">{label}</div>
                <div class="hud-value">{value}</div>
            </div>
            """
        )
    st.markdown(f'<div class="hud-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_status_strip(items: list[tuple[str, str, str]]) -> None:
    nodes = []
    for label, tone, icon in items:
        color = {
            "positive": "var(--positive)",
            "negative": "var(--negative)",
            "warning": "var(--warning)",
            "accent": "var(--accent)",
        }.get(tone, "var(--muted)")
        nodes.append(
            f"""
            <div class="status-node" style="color:{color};">
                <span class="status-light" style="background:{color}; color:{color};"></span>
                <span class="sprite-icon"></span>
                <span>{icon} {label}</span>
            </div>
            """
        )
    st.markdown(f'<div class="status-strip">{"".join(nodes)}</div>', unsafe_allow_html=True)


def render_scanbar(label: str, percent: float, color: str = "var(--accent)") -> None:
    width = max(0.0, min(100.0, percent))
    st.markdown(
        f"""
        <div class="hud-label" style="margin-top:0.65rem;">{label} // {width:.0f}%</div>
        <div class="scanbar">
            <div class="scanbar-fill" style="width:{width:.0f}%; background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radar_panel(lines: list[tuple[str, str]]) -> None:
    markup = "".join(
        f'<div class="radar-line"><span>{left}</span><strong>{right}</strong></div>' for left, right in lines
    )
    st.markdown(f'<div class="radar-box">{markup}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_market_data(ticker: str, start_date: date, end_date: date, refresh: bool) -> pd.DataFrame:
    raw = fetch_stock_data(
        ticker=ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        force_refresh=refresh,
    )
    return add_technical_indicators(clean_stock_data(raw))


@st.cache_data(show_spinner=False)
def load_multi_asset_prices(
    tickers: tuple[str, ...], start_date: date, end_date: date, refresh: bool
) -> pd.DataFrame:
    close_frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = load_market_data(ticker, start_date, end_date, refresh)
        if not df.empty:
            close_frames[ticker] = df.set_index("Date")["Close"]
    if not close_frames:
        return pd.DataFrame()
    return pd.DataFrame(close_frames).dropna(how="all")


@st.cache_data(show_spinner=False)
def load_sentiment_summary(ticker: str) -> tuple[pd.DataFrame, dict[str, float]]:
    return analyze_headlines(get_stock_news(ticker))


@st.cache_data(show_spinner=False)
def load_ml_results(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return run_ml_prediction_suite(df)


@st.cache_data(show_spinner=False)
def load_peer_summary(
    tickers: tuple[str, ...], start_date: date, end_date: date, refresh: bool
) -> dict[str, dict[str, float | str]]:
    summary: dict[str, dict[str, float | str]] = {}
    for ticker in tickers:
        df = load_market_data(ticker, start_date, end_date, refresh)
        if df.empty:
            continue
        risk = calculate_risk_metrics(df)
        summary[ticker] = {
            "signal": str(df.iloc[-1]["Signal"]),
            "annualized_return_pct": float(risk["annualized_return"] * 100),
            "annualized_volatility_pct": float(risk["annualized_volatility"] * 100),
            "sharpe_ratio": float(risk["sharpe_ratio"]),
        }
    return summary


def get_market_tickers(market_type: str) -> list[str]:
    return get_available_tickers(MARKET_UNIVERSES.get(market_type, EQUITY_UNIVERSE))


def build_chatbot_context(
    ticker: str,
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    refresh: bool,
    peer_tickers: tuple[str, ...],
) -> AssistantContext:
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    daily_return_pct = ((latest["Close"] - previous["Close"]) / previous["Close"] * 100) if previous["Close"] else 0.0
    risk = calculate_risk_metrics(df)
    signal = build_signal_summary(df)
    sentiment_df, sentiment_summary = load_sentiment_summary(ticker)

    predicted_price = None
    predicted_move_pct = None
    best_model_name = None
    try:
        ml_results = load_ml_results(df)
        leaderboard = sorted(ml_results.items(), key=lambda item: item[1]["r2"], reverse=True)
        if leaderboard:
            best_model_name, best_metrics = leaderboard[0]
            predicted_price = float(best_metrics["next_prediction"])
            predicted_move_pct = ((predicted_price - float(latest["Close"])) / float(latest["Close"])) * 100
    except Exception:
        pass

    peer_summary = load_peer_summary(peer_tickers, start_date, end_date, refresh)

    price_frame = load_multi_asset_prices(peer_tickers, start_date, end_date, refresh)
    optimized_weights = {}
    if not price_frame.empty and len(price_frame.columns) >= 2:
        optimized_weights = optimize_portfolio(price_frame)["weights"]
    elif ticker in peer_summary:
        optimized_weights = {ticker: 1.0}

    sentiment_label = "neutral"
    if not sentiment_df.empty:
        sentiment_label = (
            "positive"
            if sentiment_summary["average_score"] > 0.02
            else "negative"
            if sentiment_summary["average_score"] < -0.02
            else "neutral"
        )

    return AssistantContext(
        ticker=ticker,
        latest_price=float(latest["Close"]),
        signal=str(signal["signal"]),
        signal_reason=str(signal["reason"]),
        rsi=float(latest["RSI"]),
        macd=float(latest["MACD"]),
        daily_return_pct=float(daily_return_pct),
        annualized_return_pct=float(risk["annualized_return"] * 100),
        annualized_volatility_pct=float(risk["annualized_volatility"] * 100),
        sharpe_ratio=float(risk["sharpe_ratio"]),
        max_drawdown_pct=float(risk["max_drawdown"] * 100),
        predicted_price=predicted_price,
        predicted_move_pct=predicted_move_pct,
        best_model_name=best_model_name,
        sentiment_label=sentiment_label,
        sentiment_score=float(sentiment_summary["average_score"]),
        peer_summary=peer_summary,
        optimized_weights=optimized_weights,
    )


def render_home_page(df: pd.DataFrame, context: AssistantContext) -> None:
    open_pixel_window("Mission Control", "Online")
    render_hero(context)
    render_navigation_hint("Mission Control")
    render_status_strip(
        [
            ("Data Feed", "accent", "01"),
            ("Signal Engine", "warning" if context.signal == "HOLD" else "positive" if context.signal == "BUY" else "negative", "02"),
            ("Forecast Core", "positive" if context.predicted_move_pct is None or context.predicted_move_pct >= 0 else "negative", "03"),
            ("Risk Grid", "warning", "04"),
        ]
    )
    render_hud_row(
        [
            ("Ticker Lock", context.ticker),
            ("Signal State", context.signal),
            ("Sentiment", context.sentiment_label.upper()),
            ("Model Core", context.best_model_name or "LOCAL"),
        ]
    )
    close_pixel_window()

    latest = df.iloc[-1]
    metrics = calculate_risk_metrics(df)

    left, right = st.columns((1.2, 1))
    with left:
        open_pixel_window("Quick Scan", "Live Feed")
        render_section_label("Primary Readout")
        c1, c2 = st.columns(2)
        with c1:
            render_kpi_card("Current Price", f"${latest['Close']:.2f}", f"{context.daily_return_pct:+.2f}% today", "positive" if context.daily_return_pct >= 0 else "negative")
        with c2:
            render_kpi_card("Signal", context.signal, context.signal_reason, "warning")
        c3, c4 = st.columns(2)
        with c3:
            render_kpi_card("Sharpe", f"{metrics['sharpe_ratio']:.2f}", "Risk-adjusted", "neutral")
        with c4:
            render_kpi_card("Forecast", f"${context.predicted_price:.2f}" if context.predicted_price is not None else "N/A", context.best_model_name or "No model", "positive" if (context.predicted_move_pct or 0) >= 0 else "negative")
        render_scanbar("Momentum Meter", max(0.0, min(100.0, context.rsi)), "linear-gradient(90deg, #2dd4bf, #67e8f9)")
        render_scanbar("Risk Load", max(0.0, min(100.0, context.annualized_volatility_pct)), "linear-gradient(90deg, #f59e0b, #fb7185)")
        close_pixel_window()

    with right:
        open_pixel_window("Market Notes", "Public Brief")
        render_section_label("What This Screen Does")
        st.markdown(
            """
            This version is structured like a retro trading console:

            - `Mission Control` gives the top-line market snapshot.
            - `Market Lab` focuses on charts and raw trading behavior.
            - `Forecast Engine` isolates ML and LSTM prediction work.
            - `Chatbot Terminal` is now a full-screen assistant experience.
            - `Market Type` lets you switch between a larger equities watchlist and forex pairs.
            """,
        )
        st.markdown('<div class="terminal-note">Boot note: this dashboard now behaves like a retro analyst console. Use the left navigation to switch stations.</div>', unsafe_allow_html=True)
        close_pixel_window()

    open_pixel_window("Market Radar", "Chart Feed")
    st.plotly_chart(create_candlestick_chart(df, context.ticker), use_container_width=True)
    close_pixel_window()

    open_pixel_window("Signal Radar", "Scanner")
    render_radar_panel(
        [
            ("Daily Return", f"{context.daily_return_pct:+.2f}%"),
            ("Volatility", f"{context.annualized_volatility_pct:.2f}%"),
            ("Sharpe", f"{context.sharpe_ratio:.2f}"),
            ("Drawdown", f"{context.max_drawdown_pct:.2f}%"),
            ("RSI", f"{context.rsi:.2f}"),
            ("MACD", f"{context.macd:.2f}"),
        ]
    )
    close_pixel_window()


def render_metric_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    price_change = latest["Close"] - previous["Close"]
    daily_return_pct = (price_change / previous["Close"] * 100) if previous["Close"] else 0.0
    metrics = calculate_risk_metrics(df)
    signal = build_signal_summary(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Current Price", f"${latest['Close']:.2f}", f"{price_change:+.2f}", "positive" if price_change >= 0 else "negative")
    with col2:
        render_kpi_card("Daily Return", f"{daily_return_pct:.2f}%", "Session move", "positive" if daily_return_pct >= 0 else "negative")
    with col3:
        render_kpi_card("RSI", f"{latest['RSI']:.2f}", "Momentum oscillator", "warning" if latest["RSI"] > 70 or latest["RSI"] < 30 else "neutral")
    with col4:
        render_kpi_card("MACD", f"{latest['MACD']:.2f}", "Trend acceleration", "positive" if latest["MACD"] >= 0 else "negative")
    with col5:
        render_kpi_card("Volatility", f"{metrics['annualized_volatility']:.2%}", "Annualized", "warning")

    render_signal_banner(signal["signal"], signal["reason"])


def render_overview_tab(df: pd.DataFrame, ticker: str) -> None:
    open_pixel_window("Market Lab", "Active")
    latest = df.iloc[-1]
    render_status_strip(
        [
            ("OHLC Feed", "accent", "11"),
            ("Indicator Stack", "positive", "12"),
            ("Volume Tape", "warning", "13"),
        ]
    )
    render_hud_row(
        [
            ("Open", f"${latest['Open']:.2f}"),
            ("High", f"${latest['High']:.2f}"),
            ("Low", f"${latest['Low']:.2f}"),
            ("Close", f"${latest['Close']:.2f}"),
        ]
    )
    render_section_label("Instant Snapshot")
    render_metric_cards(df)
    close_pixel_window()

    open_pixel_window("Price Action", ticker)
    render_section_label("Price Action")
    left, right = st.columns((2, 1))
    with left:
        st.plotly_chart(create_candlestick_chart(df, ticker), use_container_width=True)
    with right:
        st.plotly_chart(create_indicator_chart(df), use_container_width=True)
    close_pixel_window()

    open_pixel_window("Flow And Distribution", "Batch A")
    render_section_label("Flow And Distribution")
    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.plotly_chart(create_volume_chart(df, ticker), use_container_width=True)
    with lower_right:
        st.plotly_chart(create_returns_histogram(df), use_container_width=True)
    close_pixel_window()

    open_pixel_window("Recent Market Data", "Table View")
    render_section_label("Recent Market Data")
    st.dataframe(df.tail(20), use_container_width=True)
    close_pixel_window()


def render_prediction_tab(df: pd.DataFrame, ticker: str) -> None:
    open_pixel_window("Forecast Engine", "ML Suite")
    render_status_strip(
        [
            ("Model Training", "accent", "21"),
            ("Regression Core", "positive", "22"),
            ("Sequence Net", "warning", "23"),
        ]
    )
    render_section_label("Model Leaderboard")
    ml_results = load_ml_results(df)
    leaderboard = pd.DataFrame(
        [
            {
                "Model": name,
                "R2": metrics["r2"],
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "Next Close Prediction": metrics["next_prediction"],
            }
            for name, metrics in ml_results.items()
        ]
    ).sort_values(by="R2", ascending=False)
    st.dataframe(leaderboard, use_container_width=True)

    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = ml_results[best_model_name]
    best_r2 = float(leaderboard.iloc[0]["R2"])
    render_hud_row(
        [
            ("Top Model", best_model_name),
            ("R2 Score", f"{best_r2:.3f}"),
            ("Next Close", f"${float(best_model['next_prediction']):.2f}"),
            ("Mode", "SHORT HORIZON"),
        ]
    )
    render_scanbar("Confidence Approx", max(0.0, min(100.0, (best_r2 + 1) * 50)), "linear-gradient(90deg, #2dd4bf, #facc15)")
    st.plotly_chart(
        create_prediction_comparison_chart(best_model["actual"], best_model["predicted"], best_model_name),
        use_container_width=True,
    )
    save_prediction(ticker, best_model_name, float(best_model["next_prediction"]), float(best_model["r2"]))
    close_pixel_window()

    open_pixel_window("LSTM Forecast", "Deep Mode")
    render_section_label("Deep Learning Forecast")
    with st.spinner("Training compact LSTM model..."):
        lstm_result = run_lstm_forecast(df, forecast_horizon=14)
    st.write(
        f"Validation RMSE: `{lstm_result['rmse']:.4f}` | Forecast horizon: `{len(lstm_result['forecast'])}` trading steps"
    )
    st.plotly_chart(
        create_forecast_chart(df, lstm_result["forecast"], ticker, "LSTM Forecast"),
        use_container_width=True,
    )
    close_pixel_window()


def render_risk_tab(df: pd.DataFrame) -> None:
    open_pixel_window("Risk Grid", "Diagnostics")
    render_status_strip(
        [
            ("Volatility Monitor", "warning", "31"),
            ("Drawdown Watch", "negative", "32"),
            ("Sharpe Scanner", "accent", "33"),
        ]
    )
    render_section_label("Risk Analytics")
    metrics = calculate_risk_metrics(df)
    render_hud_row(
        [
            ("Return", f"{metrics['annualized_return']:.2%}"),
            ("Volatility", f"{metrics['annualized_volatility']:.2%}"),
            ("Sharpe", f"{metrics['sharpe_ratio']:.2f}"),
            ("Drawdown", f"{metrics['max_drawdown']:.2%}"),
        ]
    )
    cols = st.columns(4)
    cols[0].metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    cols[1].metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2%}")
    cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    cols[3].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    st.line_chart(metrics["drawdown_curve"], use_container_width=True)
    close_pixel_window()

    open_pixel_window("Correlation Matrix", "Scanner")
    render_section_label("Correlation Diagnostics")
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    close_pixel_window()


def render_sentiment_tab(ticker: str) -> None:
    open_pixel_window("Sentiment Wire", ticker)
    render_status_strip(
        [
            ("News Parser", "accent", "41"),
            ("Mood Gauge", "warning", "42"),
            ("Signal Overlay", "positive", "43"),
        ]
    )
    render_section_label("News Sentiment")
    headlines = get_stock_news(ticker)
    custom_text = st.text_area(
        "Add headlines or research notes (one per line) to analyze custom sentiment:",
        height=140,
    )
    if custom_text.strip():
        headlines.extend(
            {"title": line.strip(), "publisher": "Custom Input", "link": ""}
            for line in custom_text.splitlines()
            if line.strip()
        )

    sentiment_df, summary = analyze_headlines(headlines)
    if sentiment_df.empty:
        st.info("No headlines available yet. Connect a news source or add custom headlines above.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", int(summary["positive"]))
    c2.metric("Neutral", int(summary["neutral"]))
    c3.metric("Negative", int(summary["negative"]))
    total = max(1, int(summary["positive"]) + int(summary["neutral"]) + int(summary["negative"]))
    render_scanbar("Positive Signal Density", summary["positive"] / total * 100, "linear-gradient(90deg, #4ade80, #2dd4bf)")
    render_scanbar("Negative Signal Density", summary["negative"] / total * 100, "linear-gradient(90deg, #f59e0b, #fb7185)")
    st.metric("Average Sentiment Score", f"{summary['average_score']:.2f}")
    st.dataframe(sentiment_df, use_container_width=True)
    close_pixel_window()


def render_chatbot_tab(context: AssistantContext) -> None:
    open_pixel_window("Chatbot Terminal", "Interactive")
    st.markdown(
        f"""
        <div class="hero-shell" style="margin-bottom: 1rem;">
            <div class="hero-kicker">Dedicated Assistant Console</div>
            <div class="hero-title">Ask The Market Bot</div>
            <p class="hero-copy">
                This is now a full-screen destination for public Q&amp;A. Ask about market scenario, lower-risk names,
                peer comparisons, forecast tone, or how the tracked portfolio mix is behaving for <strong>{context.ticker}</strong>.
            </p>
            <div class="pixel-chip-row">
                <div class="pixel-chip">Ticker: {context.ticker}</div>
                <div class="pixel-chip">Signal: {context.signal}</div>
                <div class="pixel-chip">Sentiment: {context.sentiment_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_strip(
        [
            ("Prompt Link", "accent", "51"),
            ("Advisor Core", "positive", "52"),
            ("Safety Rail", "warning", "53"),
            ("Public Mode", "accent", "54"),
        ]
    )
    render_hud_row(
        [
            ("Assist Mode", "OPENAI" if st.session_state.get("market_chat_use_llm", False) else "LOCAL"),
            ("Ticker", context.ticker),
            ("Signal", context.signal),
            ("Forecast", f"${context.predicted_price:.2f}" if context.predicted_price is not None else "N/A"),
        ]
    )
    st.session_state.setdefault("market_chat_use_llm", False)
    st.session_state.setdefault("market_chat_model", get_default_openai_model())
    st.session_state.setdefault("market_chat_api_key", "")

    if st.session_state.get("market_chat_ticker") != context.ticker:
        st.session_state.market_chat_ticker = context.ticker
        st.session_state.market_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"I'm ready to discuss `{context.ticker}` and the tracked peer group. "
                    "Ask what the market scenario looks like, whether this instrument seems risky, or how to think about diversification."
                ),
            }
        ]

    settings_col, helper_col = st.columns((1.1, 1))
    with settings_col:
        open_pixel_window("Assistant Settings", "Config")
        st.toggle("Use OpenAI assistant", key="market_chat_use_llm")
        st.text_input(
            "OpenAI API key",
            key="market_chat_api_key",
            type="password",
            help="Stored only in this Streamlit session. If empty, the app will use OPENAI_API_KEY from the environment if available.",
        )
        st.text_input(
            "Model",
            key="market_chat_model",
            help="Example: gpt-5.4-mini or another Responses API text model available on your account.",
        )

        active_api_key = get_openai_api_key(st.session_state.market_chat_api_key)
        if st.session_state.market_chat_use_llm and active_api_key:
            st.success(f"LLM mode enabled with model `{st.session_state.market_chat_model}`.")
        elif st.session_state.market_chat_use_llm:
            st.warning("LLM mode is enabled, but no API key is available yet. The assistant will fall back to local mode.")
        else:
            st.info("Local data-based assistant mode is active.")
        close_pixel_window()

    with helper_col:
        open_pixel_window("Suggested Queries", "Ready")
        st.markdown(
            """
            Ask things like:

            - `What does the market scenario look like right now?`
            - `Which tracked stock looks stronger on a risk-adjusted basis?`
            - `What should I put money on if I want lower risk?`
            - `How does the portfolio mix look today?`
            """,
        )
        close_pixel_window()

    if "market_chat_messages" not in st.session_state:
        st.session_state.market_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"I'm ready to discuss `{context.ticker}` and the tracked peer group. "
                    "Ask what the market scenario looks like, whether this instrument seems risky, or how to think about diversification."
                ),
            }
        ]

    def respond(prompt: str) -> str:
        use_llm = st.session_state.market_chat_use_llm
        api_key = get_openai_api_key(st.session_state.market_chat_api_key)
        model = st.session_state.market_chat_model
        if use_llm and api_key:
            try:
                return generate_openai_market_reply(st.session_state.market_chat_messages, context, api_key, model)
            except Exception as exc:
                return (
                    f"OpenAI assistant is temporarily unavailable, so I switched to local mode. "
                    f"Reason: {exc}. "
                    + answer_market_question(prompt, context)
                )
        return answer_market_question(prompt, context)

    open_pixel_window("Prompt Deck", "Hotkeys")
    prompt_cols = st.columns(len(default_prompts()))
    for idx, prompt in enumerate(default_prompts()):
        if prompt_cols[idx].button(prompt, key=f"chat_prompt_{idx}", use_container_width=True):
            st.session_state.market_chat_messages.append({"role": "user", "content": prompt})
            reply = respond(prompt)
            st.session_state.market_chat_messages.append({"role": "assistant", "content": reply})
    close_pixel_window()

    open_pixel_window("Conversation Feed", "Streaming")
    for message in st.session_state.market_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Ask about the market scenario, risk, or where the tracked data looks strongest.")
    if user_prompt:
        st.session_state.market_chat_messages.append({"role": "user", "content": user_prompt})
        reply = respond(user_prompt)
        st.session_state.market_chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
    close_pixel_window()
    close_pixel_window()


def render_portfolio_tab(start_date: date, end_date: date, refresh: bool, market_type: str) -> None:
    open_pixel_window("Portfolio Builder", "Optimizer")
    render_status_strip(
        [
            ("Allocation Solver", "accent", "61"),
            ("Diversification Map", "positive", "62"),
            ("Risk Balancer", "warning", "63"),
        ]
    )
    render_section_label("Portfolio Construction")
    available_tickers = get_market_tickers(market_type)
    default_selection = available_tickers[: min(3, len(available_tickers))]
    selected = st.multiselect(
        "Portfolio instruments",
        options=available_tickers,
        default=default_selection,
    )
    if not selected:
        st.info("Choose at least one asset to analyze a portfolio.")
        return

    price_frame = load_multi_asset_prices(tuple(selected), start_date, end_date, refresh)
    if price_frame.empty:
        st.warning("No pricing data available for the selected portfolio.")
        return

    optimization = optimize_portfolio(price_frame)
    performance = compute_portfolio_performance(price_frame, optimization["weights"])
    weights_df = pd.DataFrame(
        {"Ticker": list(optimization["weights"].keys()), "Weight": list(optimization["weights"].values())}
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return", f"{performance['annualized_return']:.2%}")
    c2.metric("Expected Volatility", f"{performance['annualized_volatility']:.2%}")
    c3.metric("Portfolio Sharpe", f"{performance['sharpe_ratio']:.2f}")
    render_scanbar("Return Profile", performance["annualized_return"] * 100, "linear-gradient(90deg, #2dd4bf, #4ade80)")
    render_scanbar("Risk Profile", performance["annualized_volatility"] * 100, "linear-gradient(90deg, #f59e0b, #fb7185)")

    st.dataframe(weights_df, use_container_width=True)
    st.line_chart((price_frame / price_frame.iloc[0]) * 100, use_container_width=True)
    save_portfolio_snapshot(weights_df.to_dict(orient="records"), performance)
    close_pixel_window()


def main() -> None:
    init_database()
    render_app_shell()

    with st.sidebar:
        st.markdown("## Pixel Command Deck")
        st.caption("Control the instrument, time horizon, refresh behavior, and assistant mode.")
        page = st.radio(
            "Navigation",
            [
                "Mission Control",
                "Market Lab",
                "Forecast Engine",
                "Risk Grid",
                "Sentiment Wire",
                "Portfolio Builder",
                "Chatbot Terminal",
            ],
        )
        market_type = st.radio("Market Type", ["Equities", "Forex"], horizontal=False)
        universe = get_market_tickers(market_type)
        ticker = st.selectbox("Primary instrument", universe, index=0)
        end_date = st.date_input("End date", value=date.today())
        start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 3))
        refresh = st.toggle("Refresh from API", value=False)
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            st.stop()

    with st.spinner(f"Loading {ticker} market data..."):
        df = load_market_data(ticker, start_date, end_date, refresh)

    if df.empty:
        st.error("No data could be loaded for the selected ticker and date range.")
        st.stop()

    peer_tickers = tuple(dict.fromkeys(universe[: min(len(universe), 8)]))
    if ticker not in peer_tickers:
        peer_tickers = tuple(dict.fromkeys([ticker, *peer_tickers]))

    chatbot_context = build_chatbot_context(ticker, df, start_date, end_date, refresh, peer_tickers)

    if page == "Mission Control":
        render_home_page(df, chatbot_context)
    elif page == "Market Lab":
        render_overview_tab(df, ticker)
    elif page == "Forecast Engine":
        render_prediction_tab(df, ticker)
    elif page == "Risk Grid":
        render_risk_tab(df)
    elif page == "Sentiment Wire":
        render_sentiment_tab(ticker)
    elif page == "Portfolio Builder":
        render_portfolio_tab(start_date, end_date, refresh, market_type)
    elif page == "Chatbot Terminal":
        render_chatbot_tab(chatbot_context)


if __name__ == "__main__":
    main()
