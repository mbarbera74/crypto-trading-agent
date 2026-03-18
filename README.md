---
title: Crypto Trading Agent
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: dashboard/app.py
pinned: false
---

# 🤖 Crypto Trading Agent

Agente di trading automatico per criptovalute con strategia **Momentum / Trend Following**, machine learning e dashboard interattiva.

## 🏗️ Architettura

```
crypto_trading_agent/
├── config/
│   └── settings.py          # Configurazione centralizzata
├── data/
│   └── fetcher.py            # Recupero dati OHLCV via ccxt
├── indicators/
│   └── technical.py          # Indicatori tecnici (RSI, MACD, BB, ADX...)
├── strategy/
│   └── momentum.py           # Strategia Momentum / Trend Following
├── ml/
│   ├── features.py           # Feature engineering
│   └── model.py              # Modello XGBoost per predizione
├── engine/
│   ├── backtester.py         # Motore di backtesting
│   └── live_trader.py        # Motore di trading live
├── notifications/
│   └── notifier.py           # Notifiche Telegram
├── dashboard/
│   └── app.py                # Dashboard Streamlit
├── utils/
│   └── logger.py             # Logging con loguru
├── main.py                   # Entry point CLI
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Installazione

```bash
cd crypto_trading_agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Configurazione

Copia il file di esempio e configura le tue API keys:

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
```

Modifica `.env` con le tue credenziali:

```env
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=la_tua_api_key
EXCHANGE_API_SECRET=il_tuo_secret
USE_SANDBOX=true

TELEGRAM_BOT_TOKEN=il_tuo_bot_token
TELEGRAM_CHAT_ID=il_tuo_chat_id
```

### 3. Utilizzo

```bash
# Analisi del segnale corrente
python main.py analyze

# Backtest su 90 giorni
python main.py backtest --days 90

# Backtest con Machine Learning
python main.py backtest --days 180 --ml

# Trading live (sandbox)
python main.py live

# Dashboard interattiva
python main.py dashboard
```

## 📊 Strategia

La strategia **Momentum / Trend Following** combina diversi indicatori per generare segnali:

| Indicatore | Peso | Segnale BUY | Segnale SELL |
|---|---|---|---|
| **EMA 9/21 Crossover** | 25% | EMA9 > EMA21 | EMA9 < EMA21 |
| **MACD** | 20% | Histogram > 0 | Histogram < 0 |
| **RSI** | 15% | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **ADX** | 15% | ADX > 20 + DI+ > DI- | ADX > 20 + DI- > DI+ |
| **Supertrend** | 10% | Direction = 1 | Direction = -1 |
| **Bollinger Bands** | 5% | Prezzo < 20% BB | Prezzo > 80% BB |
| **Volume** | 5% | Conferma segnale dominante | Conferma segnale dominante |
| **ML (opzionale)** | 15% | Prob. rialzo > 0.6 | Prob. rialzo < 0.4 |

## 🧠 Machine Learning

Il modello XGBoost viene addestrato su feature derivate dagli indicatori tecnici:

- **Feature**: 30+ feature tra returns, volatilità, momentum, indicatori
- **Target**: Prezzo sale di almeno 1% nei prossimi 5 periodi
- **Validazione**: Time Series Cross-Validation (5 fold)
- **Retrain**: Automatico ogni 24 ore

## 🛡️ Risk Management

- **Stop Loss**: 2% (configurabile)
- **Take Profit**: 4% (configurabile)
- **Max Position Size**: 5% del capitale
- **Max Daily Trades**: 10
- **Commissioni**: 0.1% per trade (simulate nel backtest)

## 📱 Notifiche Telegram

Per configurare le notifiche:

1. Crea un bot Telegram con [@BotFather](https://t.me/BotFather)
2. Ottieni il `BOT_TOKEN`
3. Ottieni il tuo `CHAT_ID` da [@userinfobot](https://t.me/userinfobot)
4. Inserisci i valori nel file `.env`

Le notifiche includono:
- 📈 Apertura trade
- ✅/❌ Chiusura trade (con PnL)
- 📊 Report giornaliero
- ⚠️ Errori

## 📈 Dashboard

La dashboard Streamlit include:
- Grafico candlestick con EMA, Bollinger Bands
- Indicatori RSI, MACD, Volume
- Segnale corrente della strategia
- Backtesting con equity curve
- Metriche: Sharpe Ratio, Win Rate, Max Drawdown, Profit Factor
- Feature importance del modello ML

```bash
python main.py dashboard
# Apri http://localhost:8501
```

## ⚠️ Disclaimer

> Questo software è fornito solo a scopo educativo e di ricerca. Il trading di criptovalute comporta rischi significativi. Non investire denaro che non puoi permetterti di perdere. L'autore non si assume alcuna responsabilità per eventuali perdite finanziarie.

## 📝 Licenza

MIT License
