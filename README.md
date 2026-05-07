# Stock Market AI Platform

An AI-powered stock market analytics platform built with Python and Streamlit for real-time analytics, technical indicators, forecasting, sentiment analysis, risk intelligence, and portfolio optimization.

## Features

- Historical and near real-time market data using `yfinance`
- Data cleaning pipeline for Yahoo Finance CSV exports
- Technical indicators: RSI, MACD, SMA, EMA, Bollinger Bands, OBV
- Candlestick analytics and rule-based buy/sell/hold signals
- Machine learning forecasts with Linear Regression, Random Forest, Decision Tree, and optional XGBoost
- LSTM-based time-series forecasting
- Risk analytics: returns, volatility, Sharpe ratio, drawdown
- Portfolio optimization using a Markowitz-style Sharpe maximization workflow
- News sentiment scoring with a lightweight analyzer and room for FinBERT/VADER upgrades
- Optional OpenAI-powered market chatbot using the Responses API
- SQLite persistence for predictions and portfolio snapshots

## Project Structure

```text
stock-market-ai/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_collection/
│   ├── database/
│   ├── indicators/
│   ├── portfolio/
│   ├── prediction/
│   ├── preprocessing/
│   ├── risk/
│   ├── sentiment/
│   └── visualization/
└── requirements.txt
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If you already have the included `venv`, use:

```bash
.\venv\Scripts\streamlit run app.py
```

## OpenAI Chatbot Setup

The assistant tab works in two modes:

- Local mode: uses dashboard metrics and rule-based market reasoning
- OpenAI mode: uses the OpenAI Responses API for richer natural-language answers grounded in your dashboard context

Recommended setup:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Then start the app and enable `Use OpenAI assistant` inside the `Assistant` tab settings.

You can also paste the API key directly into the app for the current session only.

## Notes

- The app can read cached CSV files from `data/raw/`.
- Turning on `Refresh from API` in the sidebar will try to fetch fresh market data with `yfinance`.
- `XGBoost`, `TextBlob`, `VADER`, and transformer-based sentiment models are optional upgrades. The current app degrades gracefully if they are not installed.
- OpenAI chatbot mode requires network access and a valid OpenAI API key.
