from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _apply_market_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,252,247,0.45)",
        font=dict(family="DM Sans, sans-serif", color="#172126"),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=22, color="#172126")),
        margin=dict(l=18, r=18, t=56, b=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(23,33,38,0.08)", zeroline=False),
    )
    return fig


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                increasing_line_color="#0d9488",
                decreasing_line_color="#f97316",
                increasing_fillcolor="#0d9488",
                decreasing_fillcolor="#f97316",
            )
        ]
    )
    fig.update_layout(title=f"{ticker} Candlestick Analysis", xaxis_rangeslider_visible=False)
    return _apply_market_theme(fig)


def create_indicator_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#172126", width=2.6)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(color="#0d9488", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_20"], name="EMA 20", line=dict(color="#f97316", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_High"], name="Bollinger High", line=dict(color="#94a3b8", dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Low"], name="Bollinger Low", line=dict(color="#94a3b8", dash="dot")))
    fig.update_layout(title="Trend Indicators")
    return _apply_market_theme(fig)


def create_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = px.bar(df, x="Date", y="Volume", title=f"{ticker} Volume Profile")
    fig.update_traces(marker_color="#0d9488")
    return _apply_market_theme(fig)


def create_returns_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Daily Return", nbins=50, title="Daily Return Distribution")
    fig.update_traces(marker_color="#f97316")
    return _apply_market_theme(fig)


def create_prediction_comparison_chart(actual: pd.Series, predicted: pd.Series, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color="#172126", width=2.5)))
    fig.add_trace(go.Scatter(y=predicted, name="Predicted", line=dict(color="#0d9488", width=2.2)))
    fig.update_layout(title=f"{model_name}: Actual vs Predicted Next-Day Close")
    return _apply_market_theme(fig)


def create_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Close"], name="Historical Close", line=dict(color="#172126", width=2.4)))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Close"], name="Forecast", line=dict(color="#0d9488", width=2.4)))
    fig.update_layout(title=f"{ticker} {title}")
    return _apply_market_theme(fig)


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    correlation_columns = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Volatility_20"]
    correlation = df[correlation_columns].corr(numeric_only=True)
    fig = px.imshow(correlation, text_auto=".2f", title="Feature Correlation Matrix", color_continuous_scale="RdBu_r")
    return _apply_market_theme(fig)
