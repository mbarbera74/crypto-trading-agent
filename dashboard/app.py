"""
Dashboard interattiva con Streamlit per monitorare il trading agent.
Include grafici dei prezzi, indicatori, equity curve, metriche,
analisi multi-asset (NDX, SWDA.MI, CSNDX), CAPE per-asset, liquidità e livelli di ingresso.
"""

import sys
from pathlib import Path

# Aggiungi il root del progetto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import config
from data.fetcher import DataFetcher
from data.stock_fetcher import StockFetcher, TICKERS
from indicators.technical import TechnicalIndicators
from strategy.momentum import MomentumStrategy
from engine.backtester import Backtester
from ml.model import MLPredictor
from analysis.market_analyzer import MarketAnalyzer
from analysis.valuation import CapeAnalyzer
from analysis.liquidity import LiquidityAnalyzer

# ============================
# CONFIGURAZIONE PAGINA
# ============================
st.set_page_config(
    page_title="Trading Agent - Multi Asset",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# NAVIGAZIONE TABS
# ============================
tab_futures, tab_monitor, tab_markets, tab_news, tab_crypto, tab_cape, tab_liquidity = st.tabs([
    "📡 Futures & Apertura",
    "📱 Segnali Accumulo",
    "🌍 NDX / SWDA.MI / CSNDX",
    "📰 News & Calendario",
    "🪙 Crypto Trading",
    "📊 CAPE per Asset",
    "💧 Liquidità",
])

# ============================
# SIDEBAR
# ============================
# Configurazione backtest per asset
ASSET_CONFIG = {
    "BTC (Crypto)": {"key": "btc", "name": "BTC/USDT", "currency": "$", "long_only": False,
                     "commission": 0.001, "sl": 0.02, "tp": 4.0, "min_conf": 0.5},
    "NASDAQ 100 (NDX)": {"key": "ndx", "name": "NASDAQ 100 (^NDX)", "currency": "$", "long_only": False,
                         "commission": 0.0005, "ticker": "^NDX", "fallbacks": ["QQQ"],
                         "sl": 0.04, "tp": 4.0, "min_conf": 0.50},
    "SWDA.MI (MSCI World)": {"key": "swda", "name": "SWDA.MI (iShares MSCI World)", "currency": "€",
                              "long_only": False, "commission": 0.001, "ticker": "SWDA.MI",
                              "fallbacks": ["SWDA.L", "IWDA.AS"], "sl": 0.04, "tp": 4.0, "min_conf": 0.50},
    "CSNDX (iShares NDX)": {"key": "csndx", "name": "CSNDX (iShares NASDAQ 100)", "currency": "€",
                             "long_only": False, "commission": 0.001, "ticker": "CNDX.MI",
                             "fallbacks": ["CSNDX.MI", "SXRV.DE"], "sl": 0.04, "tp": 4.0, "min_conf": 0.50},
}

with st.sidebar:
    st.header("⚙️ Configurazione")

    # Asset selector per backtest
    backtest_asset = st.selectbox(
        "🎯 Asset per Backtest",
        list(ASSET_CONFIG.keys()),
        index=3,  # CSNDX default
    )
    bt_cfg = ASSET_CONFIG[backtest_asset]

    symbol = st.text_input("Simbolo Crypto", value=config.trading.symbol)
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=4 if bt_cfg["key"] == "btc" else 6,
    )
    default_days = 90 if bt_cfg["key"] == "btc" else 1825  # 5 anni default
    days = st.slider("Giorni di storico", 7, 7300, default_days)
    years_label = f"({days / 365:.1f} anni)" if days >= 365 else f"({days} giorni)"
    st.caption(f"📅 Periodo: **{days} giorni** {years_label}")
    initial_capital = st.number_input("Capitale iniziale (€)", value=130000, step=1000)

    st.divider()
    st.subheader("🎯 Strategia")
    trading_mode = st.selectbox(
        "📊 Modalità Trading",
        ["Solo Long", "Long + Short"],
        index=0,  # Solo Long default
    )
    long_only = (trading_mode == "Solo Long")
    position_size_pct = st.slider(
        "💰 % Capitale per Trade",
        5, 100, 100, 5,
        help="Percentuale del capitale investita in ogni singolo trade",
    )
    use_stop_loss = st.checkbox("🛑 Usa Stop Loss", value=False)
    if use_stop_loss:
        stop_loss = st.slider("Stop Loss %", 0.5, 50.0, bt_cfg["sl"] * 100, 0.5)
    else:
        stop_loss = 0.0
    use_take_profit = st.checkbox("🎯 Usa Take Profit", value=False)
    if use_take_profit:
        take_profit = st.slider("Take Profit %", 1.0, 400.0, bt_cfg["tp"] * 100, 1.0)
    else:
        take_profit = 0.0
    min_confidence = st.slider("Confidenza minima", 0.1, 0.9, bt_cfg["min_conf"], 0.05)
    no_exit_signal = st.checkbox(
        "🔒 No uscita su segnale (hold fino a fine periodo)",
        value=True,
        help="Se attivo, una volta entrati non si esce mai. Il trade si chiude solo a fine backtest.",
    )

    # Info asset selezionato
    st.caption(
        f"**{bt_cfg['name']}** | {'Solo Long' if long_only else 'Long + Short'} | "
        f"Comm: {bt_cfg['commission']*100:.2f}% | {bt_cfg['currency']}"
    )

    st.divider()
    use_ml = st.checkbox("🧠 Usa ML", value=True)

    run_backtest = st.button("🚀 Esegui Backtest", type="primary", use_container_width=True)
    refresh_data = st.button("🔄 Aggiorna Dati", use_container_width=True)

    st.divider()
    st.subheader("🌍 Mercati Tradizionali")
    market_period = st.selectbox("Periodo analisi", ["3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "max"], index=4)
    run_market_analysis = st.button("📊 Analizza Mercati", use_container_width=True)


# ============================
# HELPER: Render backtest results
# ============================
def _render_backtest_results(result, initial_capital, currency="$"):
    """Mostra i risultati del backtest in modo uniforme."""
    if result.total_trades > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("PnL Totale", f"{currency}{result.total_pnl:,.2f}", f"{result.total_pnl_pct:+.2f}%")
        with col2:
            st.metric("Win Rate", f"{result.win_rate:.1f}%", f"{result.total_trades} trade")
        with col3:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{result.max_drawdown_pct:.1f}%")

        # Metriche aggiuntive
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Capitale Finale", f"{currency}{result.equity_curve[-1]:,.2f}" if result.equity_curve else "N/A")
        with col6:
            st.metric("Media Trade", f"{currency}{result.avg_trade_pnl:,.2f}" if hasattr(result, 'avg_trade_pnl') else "N/A")
        with col7:
            st.metric("Profit Factor", f"{result.profit_factor:.2f}" if hasattr(result, 'profit_factor') and result.profit_factor else "N/A")
        with col8:
            mode = "Long Only" if hasattr(result, 'asset_name') and result.asset_name else "Long + Short"
            st.metric("Modalità", mode)

        # Equity Curve
        st.subheader("💰 Equity Curve")
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=result.timestamps, y=result.equity_curve, mode="lines",
            name="Equity", line=dict(color="cyan", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,255,0.1)",
        ))
        eq_fig.add_hline(y=initial_capital, line_dash="dash", line_color="white",
                         annotation_text="Capitale Iniziale")
        eq_fig.update_layout(height=400, template="plotly_dark", yaxis_title=f"Equity ({currency})")
        st.plotly_chart(eq_fig, use_container_width=True)

        # Lista trade
        st.subheader("📋 Lista Trade")
        trade_data = []
        for t in result.trades:
            # Calcola durata trade
            duration = ""
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                days_held = delta.days
                if days_held >= 365:
                    duration = f"{days_held/365:.1f} anni"
                elif days_held >= 30:
                    duration = f"{days_held/30:.1f} mesi"
                else:
                    duration = f"{days_held} giorni"

            entry_reasons_str = ", ".join(t.entry_reasons[:3]) if hasattr(t, 'entry_reasons') and t.entry_reasons else ""

            trade_data.append({
                "Side": t.side.upper(),
                "Entrata": t.entry_time.strftime("%Y-%m-%d") if t.entry_time else "",
                "Uscita": t.exit_time.strftime("%Y-%m-%d") if t.exit_time else "",
                "Durata": duration,
                f"Entry ({currency})": f"{currency}{t.entry_price:,.2f}",
                f"Exit ({currency})": f"{currency}{t.exit_price:,.2f}" if t.exit_price else "",
                f"PnL ({currency})": f"{currency}{t.pnl:,.2f}",
                "PnL (%)": f"{t.pnl_pct:+.2f}%",
                "Uscita": t.exit_reason,
                "Motivi Entrata": entry_reasons_str,
            })
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
    else:
        st.warning("Nessun trade eseguito. Prova ad aumentare i giorni di storico o ridurre la confidenza minima.")


# ============================
# TAB 1: CRYPTO TRADING
# ============================
with tab_crypto:
    st.title("📈 Crypto Trading Agent")

    # CARICAMENTO DATI
    @st.cache_data(ttl=300)
    def load_data(symbol: str, timeframe: str, days: int):
        """Carica i dati dall'exchange con caching."""
        fetcher = DataFetcher()
        try:
            df = fetcher.fetch_historical(symbol=symbol, timeframe=timeframe, days=days)
            if not df.empty:
                df_indicators = TechnicalIndicators.add_all(df)
                return df, df_indicators
        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if refresh_data:
        st.cache_data.clear()

    df_raw, df = load_data(symbol, timeframe, days)

    if df.empty:
        st.warning("Nessun dato disponibile. Verifica la configurazione dell'exchange.")
    else:
        # METRICHE ATTUALI
        st.header("📊 Situazione Attuale")

        current = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = (current["close"] - prev["close"]) / prev["close"] * 100

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Prezzo", f"${current['close']:,.2f}", f"{price_change:+.2f}%")
        with col2:
            rsi_val = current.get("rsi", 50)
            st.metric("RSI", f"{rsi_val:.1f}", delta=None)
        with col3:
            macd_val = current.get("macd_histogram", 0)
            st.metric("MACD Hist", f"{macd_val:.4f}")
        with col4:
            adx_val = current.get("adx", 0)
            st.metric("ADX", f"{adx_val:.1f}")
        with col5:
            bb_val = current.get("bb_pct", 0.5)
            st.metric("BB %", f"{bb_val:.2f}")
        with col6:
            vol_ratio = current.get("volume_ratio", 1)
            st.metric("Vol Ratio", f"{vol_ratio:.1f}x")

        # Segnali attuali
        signals = TechnicalIndicators.get_current_signals(df)
        strategy = MomentumStrategy(
            stop_loss_pct=stop_loss / 100,
            take_profit_pct=take_profit / 100,
            min_confidence=min_confidence,
        )

        ml_pred = None
        if use_ml:
            try:
                ml_model = MLPredictor()
                ml_pred = ml_model.predict(df)
            except Exception:
                pass

        current_signal = strategy.analyze(df, ml_prediction=ml_pred)

        signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

        st.info(
            f"{signal_color.get(current_signal.signal.value, '⚪')} "
            f"**Segnale: {current_signal.signal.value}** | "
            f"Confidenza: {current_signal.confidence:.0%} | "
            f"Motivi: {', '.join(current_signal.reasons[:3])}"
        )

        # GRAFICO PRINCIPALE
        st.header("📈 Grafico")

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=("Prezzo + Indicatori", "RSI", "MACD", "Volume"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="OHLCV",
            ), row=1, col=1,
        )

        for ema_col, color in [("ema_9", "orange"), ("ema_21", "blue"), ("ema_50", "purple")]:
            if ema_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[ema_col], name=ema_col.upper(),
                               line=dict(color=color, width=1)),
                    row=1, col=1,
                )

        if "bb_upper" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                           line=dict(color="gray", dash="dot", width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                           line=dict(color="gray", dash="dot", width=1),
                           fill="tonexty", fillcolor="rgba(128,128,128,0.1)"),
                row=1, col=1,
            )

        if "rsi" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                           line=dict(color="purple", width=1.5)),
                row=2, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if "macd" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["macd"], name="MACD",
                           line=dict(color="blue", width=1)),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                           line=dict(color="orange", width=1)),
                row=3, col=1,
            )
            colors_macd = ["green" if v >= 0 else "red" for v in df["macd_histogram"]]
            fig.add_trace(
                go.Bar(x=df.index, y=df["macd_histogram"], name="Histogram",
                       marker_color=colors_macd),
                row=3, col=1,
            )

        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="Volume",
                   marker_color="rgba(100,100,200,0.5)"),
            row=4, col=1,
        )

        fig.update_layout(
            height=900, showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)

        # BACKTEST (BTC only - crypto tab)
        if run_backtest and bt_cfg["key"] == "btc":
            st.header("🧪 Risultati Backtest - BTC")

            with st.spinner("Esecuzione backtest BTC in corso..."):
                backtester = Backtester(
                    strategy=strategy,
                    initial_capital=initial_capital,
                    commission_pct=bt_cfg["commission"],
                    use_ml=use_ml,
                    long_only=long_only,
                    asset_name=bt_cfg["name"],
                    currency=bt_cfg["currency"],
                    position_size=position_size_pct / 100,
                    no_exit_signal=no_exit_signal,
                )
                result = backtester.run(df_raw)

            _render_backtest_results(result, initial_capital, bt_cfg["currency"])

        elif run_backtest and bt_cfg["key"] != "btc":
            st.info(f"Hai selezionato **{bt_cfg['name']}** — vai al tab '🌍 NDX / SWDA.MI / CSNDX' per vedere i risultati del backtest.")

# ============================
# TAB FUTURES & APERTURA
# ============================
with tab_futures:
    st.title("📡 Futures & Previsione Apertura Mercati")

    st.markdown("""
    Monitoraggio in tempo reale dei futures **NASDAQ 100 (NQ)**, **S&P 500 (ES)** e **Dow Jones (YM)**.
    Analisi della direzione attesa all'apertura basata su futures, momentum e VIX.
    """)

    run_futures = st.button("📡 Aggiorna Futures", use_container_width=True)

    if run_futures:
        from analysis.futures_monitor import FuturesMonitor
        with st.spinner("Recupero dati futures..."):
            st.session_state["futures_data"] = FuturesMonitor().analyze()

    fa = st.session_state.get("futures_data")
    if fa:

        # ── OUTLOOK PRINCIPALE ──
        if fa.outlook:
            ol = fa.outlook
            dir_colors = {
                "RIALZISTA": "green", "LEGGERMENTE RIALZISTA": "green",
                "RIBASSISTA": "red", "LEGGERMENTE RIBASSISTA": "red",
                "NEUTRO": "orange"
            }
            dir_emojis = {
                "RIALZISTA": "🟢📈", "LEGGERMENTE RIALZISTA": "🟢",
                "RIBASSISTA": "🔴📉", "LEGGERMENTE RIBASSISTA": "🔴",
                "NEUTRO": "🟡"
            }
            color = dir_colors.get(ol.direction, "gray")
            emoji = dir_emojis.get(ol.direction, "⚪")

            st.markdown(f"## {emoji} Direzione attesa: **:{color}[{ol.direction}]**")

            col_o1, col_o2, col_o3, col_o4 = st.columns(4)
            with col_o1:
                st.metric("Confidenza", f"{ol.confidence:.0%}")
            with col_o2:
                gap_color = "green" if ol.gap_pct > 0 else "red" if ol.gap_pct < 0 else "gray"
                st.metric("Gap Atteso", ol.gap_expected, f"{ol.gap_pct:+.2f}%")
            with col_o3:
                st.metric("Rischio", ol.risk_level)
            with col_o4:
                st.metric("VIX", f"{fa.vix_current:.1f}", f"{fa.vix_change:+.1f}%")

        # ── FUTURES DETTAGLIO ──
        st.divider()
        st.header("📊 Dettaglio Futures")

        futures_list = [
            ("NQ=F", "NASDAQ 100", fa.nasdaq_futures, "cyan"),
            ("ES=F", "S&P 500", fa.sp500_futures, "lime"),
            ("YM=F", "Dow Jones", fa.dow_futures, "#ffaa00"),
        ]

        cols_f = st.columns(3)
        for i, (ticker, name, snap, color) in enumerate(futures_list):
            with cols_f[i]:
                if snap:
                    trend_emoji = {"UP": "📈", "DOWN": "📉", "FLAT": "➡️"}.get(snap.intraday_trend, "")
                    chg_color = "green" if snap.change_pct > 0 else "red" if snap.change_pct < 0 else "gray"

                    st.markdown(f"### {trend_emoji} {name}")
                    st.metric("Prezzo", f"${snap.last_price:,.2f}", f"{snap.change_pct:+.2f}%")
                    st.write(f"Chiusura prec.: ${snap.prev_close:,.2f}")
                    st.write(f"Mom. 1h: **{snap.momentum_1h:+.2f}%** | 4h: **{snap.momentum_4h:+.2f}%**")
                    st.write(f"Range: ${snap.low_today:,.2f} — ${snap.high_today:,.2f} ({snap.range_pct:.2f}%)")
                    st.write(f"Trend: **{snap.intraday_trend}**")

                    # Mini chart
                    try:
                        tk = yf.Ticker(ticker)
                        h = tk.history(period="2d", interval="15m", prepost=True)
                        if not h.empty:
                            chart = go.Figure()
                            chart.add_trace(go.Scatter(
                                x=h.index, y=h["Close"], mode="lines",
                                line=dict(color=color, width=2),
                                name=name,
                            ))
                            chart.update_layout(
                                height=250, template="plotly_dark",
                                margin=dict(l=20, r=20, t=10, b=20),
                                xaxis_title="", yaxis_title="",
                            )
                            st.plotly_chart(chart, use_container_width=True)
                    except Exception:
                        pass
                else:
                    st.warning(f"Dati {name} non disponibili")

        # ── SEGNALI ──
        if fa.outlook and fa.outlook.signals:
            st.divider()
            st.header("🔍 Analisi Segnali")
            for sig in fa.outlook.signals:
                st.write(sig)

        # ── SOMMARIO ──
        if fa.outlook:
            st.divider()
            st.header("📋 Riepilogo Apertura")

            ol = fa.outlook
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"**Direzione:** {ol.direction}")
                st.markdown(f"**Confidenza:** {ol.confidence:.0%}")
                st.markdown(f"**Gap atteso:** {ol.gap_expected} ({ol.gap_pct:+.2f}%)")
            with col_s2:
                st.markdown(f"**VIX:** {fa.vix_current:.1f} ({fa.vix_change:+.1f}%)")
                st.markdown(f"**Rischio:** {ol.risk_level}")
                nq_str = f"NQ {fa.nasdaq_futures.change_pct:+.2f}%" if fa.nasdaq_futures else "N/A"
                es_str = f"ES {fa.sp500_futures.change_pct:+.2f}%" if fa.sp500_futures else "N/A"
                st.markdown(f"**Futures:** {nq_str} | {es_str}")

            st.caption(f"Ultimo aggiornamento: {fa.last_updated}")

# ============================
# TAB 2: NDX / SWDA.MI / CSNDX
# ============================
with tab_markets:
    st.title("🌍 Analisi NDX / SWDA.MI / CSNDX")

    @st.cache_data(ttl=600)
    def load_market_data(period: str):
        fetcher = StockFetcher()
        ndx = fetcher.fetch_ohlcv(TICKERS["NASDAQ100"], period=period)
        swda = fetcher.fetch_ohlcv(TICKERS["SWDA"], period=period)
        if swda.empty:
            swda = fetcher.fetch_ohlcv(TICKERS["SWDA_ALT"], period=period)
        if swda.empty:
            swda = fetcher.fetch_ohlcv(TICKERS["SWDA_ALT2"], period=period)
        csndx = fetcher.fetch_ohlcv(TICKERS["CSNDX"], period=period)
        if csndx.empty:
            csndx = fetcher.fetch_ohlcv(TICKERS["CSNDX_ALT"], period=period)
        if csndx.empty:
            csndx = fetcher.fetch_ohlcv(TICKERS["CSNDX_ALT2"], period=period)
        return ndx, swda, csndx

    ndx_data, swda_data, csndx_data = load_market_data(market_period)

    col_n, col_s, col_c = st.columns(3)

    # Helper per creare chart
    def render_asset_chart(col, data, name, ticker, color, currency="$"):
        with col:
            st.subheader(f"📊 {name}")
            if not data.empty:
                price_now = data["close"].iloc[-1]
                price_prev = data["close"].iloc[-2] if len(data) > 1 else price_now
                chg = (price_now / price_prev - 1) * 100
                st.metric("Prezzo", f"{currency}{price_now:,.2f}", f"{chg:+.2f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["close"], mode="lines",
                    name=ticker, line=dict(color=color, width=2),
                ))
                if len(data) >= 50:
                    sma50 = data["close"].rolling(50).mean()
                    fig.add_trace(go.Scatter(
                        x=data.index, y=sma50, mode="lines",
                        name="SMA 50", line=dict(color="orange", width=1, dash="dash"),
                    ))
                if len(data) >= 200:
                    sma200 = data["close"].rolling(200).mean()
                    fig.add_trace(go.Scatter(
                        x=data.index, y=sma200, mode="lines",
                        name="SMA 200", line=dict(color="red", width=1, dash="dash"),
                    ))
                fig.update_layout(height=350, template="plotly_dark", title=f"{ticker} - {name}")
                st.plotly_chart(fig, use_container_width=True)

                perf = {}
                for label, n_days in [("1W", 5), ("1M", 22), ("3M", 66), ("6M", 132), ("1Y", 252)]:
                    if len(data) > n_days:
                        perf[label] = f"{(data['close'].iloc[-1] / data['close'].iloc[-n_days] - 1) * 100:+.1f}%"
                if perf:
                    st.dataframe(pd.DataFrame([perf], index=["Performance"]), use_container_width=True)
            else:
                st.warning(f"Dati {ticker} non disponibili")

    render_asset_chart(col_n, ndx_data, "NASDAQ 100", "^NDX", "cyan", "$")
    render_asset_chart(col_s, swda_data, "SWDA.MI", "SWDA.MI", "lime", "€")
    render_asset_chart(col_c, csndx_data, "CSNDX", "CNDX.MI", "#ff88ff", "€")

    # ============================
    # BACKTEST ASSET TRADIZIONALI
    # ============================
    if run_backtest and bt_cfg["key"] != "btc":
        st.divider()
        st.header(f"🧪 Backtest — {bt_cfg['name']}")

        with st.spinner(f"Esecuzione backtest {bt_cfg['name']} in corso..."):
            bt_fetcher = StockFetcher()
            ticker = bt_cfg["ticker"]
            bt_df = bt_fetcher.fetch_historical(ticker, days=days)

            # Prova fallback
            if bt_df.empty and "fallbacks" in bt_cfg:
                for fb in bt_cfg["fallbacks"]:
                    bt_df = bt_fetcher.fetch_historical(fb, days=days)
                    if not bt_df.empty:
                        break

            if bt_df.empty:
                st.error(f"Nessun dato disponibile per {bt_cfg['name']}")
            else:
                st.success(f"Scaricate {len(bt_df)} candele per {bt_cfg['name']}")

                # Volume sintetico se mancante
                if "volume" not in bt_df.columns or bt_df["volume"].sum() == 0:
                    bt_df["volume"] = ((bt_df["high"] - bt_df["low"]) / bt_df["close"] * 1_000_000).clip(lower=100)

                bt_strategy = MomentumStrategy(
                    stop_loss_pct=stop_loss / 100,
                    take_profit_pct=take_profit / 100,
                    min_confidence=min_confidence,
                )

                bt_engine = Backtester(
                    strategy=bt_strategy,
                    initial_capital=initial_capital,
                    commission_pct=bt_cfg["commission"],
                    use_ml=use_ml,
                    long_only=long_only,
                    asset_name=bt_cfg["name"],
                    currency=bt_cfg["currency"],
                    position_size=position_size_pct / 100,
                    no_exit_signal=no_exit_signal,
                )

                bt_result = bt_engine.run(bt_df)
                _render_backtest_results(bt_result, initial_capital, bt_cfg["currency"])

    # Full analysis button
    if run_market_analysis:
        st.divider()
        st.header("🎯 Analisi Completa con CAPE per-asset + Liquidità + Livelli Ingresso")
        with st.spinner("Analisi multi-fattore in corso..."):
            analyzer = MarketAnalyzer()
            report = analyzer.full_analysis(period=market_period)

        # Raccomandazioni per ciascun asset
        col1, col2, col3 = st.columns(3)
        for col, name, score, rec, entry_type, asset in [
            (col1, "NDX (NASDAQ 100)", report.nasdaq100_score, report.nasdaq100_recommendation,
             report.nasdaq100_entry_type, report.nasdaq100),
            (col2, "SWDA.MI", report.swda_score, report.swda_recommendation,
             report.swda_entry_type, report.swda),
            (col3, "CSNDX", report.csndx_score, report.csndx_recommendation,
             report.csndx_entry_type, report.csndx),
        ]:
            with col:
                st.subheader(f"{name}")
                st.metric("Score Composito", f"{score:+.2f}")
                st.write(f"**{rec}**")
                st.write(f"Tipo ingresso: {entry_type}")
                if asset and asset.cape_analysis:
                    ca = asset.cape_analysis
                    st.write(f"CAPE: {ca.cape_value:.1f} ({ca.valuation_level})")
                if asset:
                    for sig in asset.technical_signals:
                        st.write(f"• {sig}")

        # Livelli di ingresso per ciascun asset
        st.divider()
        st.header("🎯 Livelli di Ingresso con Probabilità")
        for name, asset, currency in [
            ("NDX (NASDAQ 100)", report.nasdaq100, "$"),
            ("SWDA.MI", report.swda, "€"),
            ("CSNDX", report.csndx, "€"),
        ]:
            if asset and asset.entry_levels:
                st.subheader(f"{name} — Prezzo attuale: {currency}{asset.current_price:,.2f}")
                levels_data = []
                for lvl in asset.entry_levels:
                    levels_data.append({
                        "Livello": lvl.level,
                        "Prezzo": f"{currency}{lvl.price:,.2f}",
                        "Distanza": f"{lvl.distance_pct:+.1f}%",
                        "Prob. 30gg": f"{lvl.prob_30d:.0%}",
                        "Prob. 90gg": f"{lvl.prob_90d:.0%}",
                        "Tipo": lvl.level_type.replace('_', ' ').title(),
                    })
                st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)

        # Asset allocation chart
        if report.suggested_allocation:
            st.subheader("💼 Asset Allocation Consigliata")
            alloc_fig = go.Figure(go.Pie(
                labels=list(report.suggested_allocation.keys()),
                values=list(report.suggested_allocation.values()),
                hole=0.4,
                marker_colors=["#00d4ff", "#ff88ff", "#00ff88", "#4488ff", "#ffcc00"],
            ))
            alloc_fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(alloc_fig, use_container_width=True)

        if report.summary:
            st.divider()
            st.header("📋 Riepilogo Mercati")

            # CAPE e Liquidità
            if report.cape_sp500:
                c = report.cape_sp500
                st.markdown(f"- 📊 **CAPE S&P 500:** {c.cape_value:.1f} — {c.valuation_level} (rendimento 10Y atteso: ~{c.expected_10y_return:.1f}%)")
            if report.liquidity_analysis:
                l = report.liquidity_analysis
                st.markdown(f"- 💧 **Liquidità:** {l.liquidity_level} (score {l.overall_score:+.2f}) — {l.entry_recommendation}")

            st.markdown("---")

            # Per ogni asset
            for name_label, asset_obj, asset_score, asset_rec in [
                ("NDX (NASDAQ 100)", report.nasdaq100, report.nasdaq100_score, report.nasdaq100_recommendation),
                ("SWDA.MI (MSCI World)", report.swda, report.swda_score, report.swda_recommendation),
                ("CSNDX (iShares NDX)", report.csndx, report.csndx_score, report.csndx_recommendation),
            ]:
                if asset_obj and asset_obj.current_price > 0:
                    currency = "$" if "NDX" == name_label[:3] else "€"
                    cape_txt = ""
                    if asset_obj.cape_analysis:
                        cape_txt = f" | CAPE: {asset_obj.cape_analysis.cape_value:.0f}"
                    dd_txt = ""
                    if asset_obj.drawdown_from_ath != 0:
                        dd_txt = f" | 📉 **{asset_obj.drawdown_from_ath:.1f}% dal max** ({currency}{asset_obj.ath_price:,.2f})"

                    st.markdown(
                        f"- **{name_label}:** {currency}{asset_obj.current_price:,.2f} "
                        f"({asset_obj.price_change_1m:+.1f}% 1M) | "
                        f"Trend: {asset_obj.trend} | RSI: {asset_obj.rsi:.0f}"
                        f"{cape_txt}{dd_txt}"
                    )
                    st.markdown(f"  - Score: {asset_score:+.2f} → _{asset_rec}_")

# ============================
# TAB 3: CAPE RATIO
# ============================
with tab_cape:
    st.title("📊 Analisi CAPE Ratio per Asset")

    st.markdown("""
    Il **CAPE Ratio** (Cyclically Adjusted Price-to-Earnings) è il rapporto prezzo/utili
    medio degli ultimi 10 anni, aggiustato per l'inflazione.

    Il CAPE viene stimato **per ciascun asset**:
    - **NASDAQ 100**: CAPE ~1.45x rispetto all'S&P 500 (settore tech con multipli più alti)
    - **SWDA (MSCI World)**: CAPE ~1.05x rispetto all'S&P 500 (diversificazione globale)
    - **CSNDX**: Stesso CAPE del NASDAQ 100 (replica lo stesso indice)
    """)

    cape_analyzer = CapeAnalyzer()

    with st.spinner("Recupero CAPE Ratio corrente..."):
        sp500_cape_val = cape_analyzer.fetch_current_cape()
        sp500_analysis = cape_analyzer.analyze(cape_value=sp500_cape_val, region="US")
        ndx_analysis = cape_analyzer.analyze_asset("NDX", sp500_cape_val)
        swda_cape_analysis = cape_analyzer.analyze_asset("SWDA", sp500_cape_val)
        csndx_cape_analysis = cape_analyzer.analyze_asset("CSNDX", sp500_cape_val)

    # Metriche S&P 500 base
    st.subheader("🏛️ CAPE S&P 500 (Base)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CAPE S&P 500", f"{sp500_analysis.cape_value:.1f}")
    with col2:
        st.metric("Media Storica", f"{sp500_analysis.historical_mean:.1f}",
                   f"{sp500_analysis.deviation_from_mean:+.0f}%")
    with col3:
        st.metric("Percentile", f"{sp500_analysis.percentile:.0f}°")
    with col4:
        st.metric("Rend. 10Y Atteso", f"~{sp500_analysis.expected_10y_return:.1f}%")

    # CAPE per-asset comparison
    st.subheader("📊 CAPE per Asset")
    cape_assets = [
        ("NDX (NASDAQ 100)", ndx_analysis),
        ("SWDA.MI (MSCI World)", swda_cape_analysis),
        ("CSNDX (iShares NDX)", csndx_cape_analysis),
    ]

    cape_cols = st.columns(3)
    for i, (name, ca) in enumerate(cape_assets):
        with cape_cols[i]:
            st.metric(f"CAPE {name}", f"{ca.cape_value:.1f}")
            st.write(f"**{ca.valuation_level}**")
            st.write(f"Segnale: {ca.entry_signal}")
            st.write(f"Media storica: {ca.historical_mean:.1f} (dev: {ca.deviation_from_mean:+.0f}%)")
            st.write(f"Rend. 10Y atteso: ~{ca.expected_10y_return:.1f}%")

    # Gauge chart (usa NASDAQ 100 CAPE come gauge principale)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ndx_analysis.cape_value,
        title={"text": "CAPE NASDAQ 100 (stimato)"},
        gauge={
            "axis": {"range": [10, 70]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [10, 24], "color": "#00cc00"},
                {"range": [24, 30], "color": "#88cc00"},
                {"range": [30, 38], "color": "#cccc00"},
                {"range": [38, 45], "color": "#ff8800"},
                {"range": [45, 52], "color": "#ff4400"},
                {"range": [52, 70], "color": "#cc0000"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": 28.0,  # Media storica NASDAQ CAPE
            },
        },
    ))
    gauge_fig.update_layout(height=350, template="plotly_dark")
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Allocazione basata su CAPE
    alloc = cape_analyzer.get_cape_adjusted_allocation(sp500_analysis.cape_value)
    st.subheader("💼 Allocazione Basata sul CAPE")
    alloc_cols = st.columns(3)
    with alloc_cols[0]:
        st.metric("Azionario", f"{alloc['azionario']}%")
    with alloc_cols[1]:
        st.metric("Obbligazionario", f"{alloc['obbligazionario']}%")
    with alloc_cols[2]:
        st.metric("Liquidità", f"{alloc['liquidità']}%")

# ============================
# TAB 4: LIQUIDITÀ
# ============================
with tab_liquidity:
    st.title("💧 Analisi della Liquidità di Mercato")

    st.markdown("""
    L'analisi della liquidità combina 7 indicatori macro per determinare
    se le condizioni di mercato sono favorevoli per l'ingresso su asset rischiosi.
    """)

    if run_market_analysis or st.button("🔄 Aggiorna Analisi Liquidità", key="liq_refresh"):
        with st.spinner("Analisi indicatori di liquidità..."):
            liq_analyzer = LiquidityAnalyzer()
            liq_result = liq_analyzer.analyze(period="6mo")

        # Score complessivo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score Liquidità", f"{liq_result.overall_score:+.2f}")
        with col2:
            st.metric("Livello", liq_result.liquidity_level)
        with col3:
            st.metric("Raccomandazione", liq_result.entry_recommendation)

        st.write(liq_result.description)

        # Gauge
        liq_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=liq_result.overall_score,
            title={"text": "Indice di Liquidità"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "cyan"},
                "steps": [
                    {"range": [-1, -0.5], "color": "#cc0000"},
                    {"range": [-0.5, -0.2], "color": "#ff8800"},
                    {"range": [-0.2, 0.2], "color": "#cccc00"},
                    {"range": [0.2, 0.5], "color": "#88cc00"},
                    {"range": [0.5, 1], "color": "#00cc00"},
                ],
            },
        ))
        liq_gauge.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(liq_gauge, use_container_width=True)

        # Dettaglio indicatori
        st.subheader("📊 Dettaglio Indicatori")
        for ind in liq_result.indicators:
            color = "🟢" if ind.signal == "POSITIVO" else "🔴" if ind.signal == "NEGATIVO" else "🟡"
            with st.expander(f"{color} {ind.name} — Score: {ind.score:+.2f} (peso: {ind.weight:.0%})"):
                st.write(ind.description)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Valore", f"{ind.value:.2f}")
                with col_b:
                    st.metric("Segnale", ind.signal)

        # Risk factors
        if liq_result.risk_factors:
            st.subheader("⚠️ Fattori di Rischio")
            for f in liq_result.risk_factors:
                st.error(f)

        if liq_result.positive_factors:
            st.subheader("✅ Fattori Positivi")
            for f in liq_result.positive_factors:
                st.success(f)
    else:
        st.info("Clicca '📊 Analizza Mercati' nella sidebar o il bottone qui sopra per avviare l'analisi.")

# ============================
# TAB 3: SEGNALI ACCUMULO
# ============================
with tab_monitor:
    st.title("📱 Segnali di Accumulo — CSNDX / SWDA.MI")

    st.markdown("""
    Analisi automatica per la strategia di **accumulo (PAC)** su CSNDX e SWDA.MI.
    Combina analisi tecnica, CAPE, liquidità e livelli di ingresso per generare
    un segnale operativo: **COMPRA**, **ATTENDI** o **EVITA**.
    """)

    run_monitor = st.button("📱 Aggiorna Segnali", use_container_width=True)

    if run_monitor:
        with st.spinner("Analisi segnali accumulo..."):
            analyzer = MarketAnalyzer()
            st.session_state["monitor_report"] = analyzer.full_analysis(period="1y")

    report = st.session_state.get("monitor_report")
    if report:
        monitor_assets = []
        for asset, score, rec, entry_type, currency, name in [
            (report.swda, report.swda_score, report.swda_recommendation,
             report.swda_entry_type, "€", "SWDA.MI (MSCI World)"),
            (report.csndx, report.csndx_score, report.csndx_recommendation,
             report.csndx_entry_type, "€", "CSNDX (iShares NDX)"),
        ]:
            if asset and asset.current_price > 0:
                if score > 0.2:
                    action = "COMPRA"
                elif score > -0.2:
                    action = "ATTENDI"
                else:
                    action = "EVITA"

                # Miglior livello raggiungibile (prob_90d > 60%, sotto il prezzo)
                best_level = None
                for lvl in asset.entry_levels:
                    if lvl.prob_90d > 0.60 and lvl.distance_pct < 0:
                        if best_level is None or lvl.price < best_level.price:
                            best_level = lvl

                monitor_assets.append({
                    "name": name,
                    "asset": asset,
                    "score": score,
                    "action": action,
                    "rec": rec,
                    "entry_type": entry_type,
                    "currency": currency,
                    "best_level": best_level,
                })

        # ── SEGNALI PRINCIPALI ──
        cols = st.columns(len(monitor_assets))
        for i, ma in enumerate(monitor_assets):
            with cols[i]:
                action_colors = {"COMPRA": "green", "ATTENDI": "orange", "EVITA": "red"}
                action_emojis = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}
                color = action_colors.get(ma["action"], "gray")
                emoji = action_emojis.get(ma["action"], "⚪")

                st.markdown(f"### {emoji} {ma['name']}")
                st.metric("Prezzo", f"{ma['currency']}{ma['asset'].current_price:,.2f}",
                           f"{ma['asset'].price_change_1m:+.1f}% (1M)")
                st.metric("Score Composito", f"{ma['score']:+.2f}")
                st.markdown(f"**Segnale: :{color}[{ma['action']}]**")

                # Drawdown da massimi
                if ma['asset'].drawdown_from_ath != 0:
                    dd_color = "green" if ma['asset'].drawdown_from_ath > -5 else "orange" if ma['asset'].drawdown_from_ath > -15 else "red"
                    st.metric("📉 Drawdown da Max", f"{ma['asset'].drawdown_from_ath:.1f}%",
                              f"Max: {ma['currency']}{ma['asset'].ath_price:,.2f} ({ma['asset'].ath_date})")
                    if ma['asset'].drawdown_from_52w != 0 and ma['asset'].drawdown_from_52w != ma['asset'].drawdown_from_ath:
                        st.write(f"📊 Da max 52 sett.: **{ma['asset'].drawdown_from_52w:.1f}%** (max: {ma['currency']}{ma['asset'].high_52w:,.2f})")

                    # Suggerimento basato su drawdown
                    dd = ma['asset'].drawdown_from_ath
                    if dd < -30:
                        st.error("🔴 Crollo significativo (>30%). Opportunità se fondamentali solidi.")
                    elif dd < -20:
                        st.warning("🟠 Correzione forte (>20%). Zona di accumulo aggressivo.")
                    elif dd < -10:
                        st.info("🟡 Correzione moderata (>10%). Buon punto per DCA.")
                    elif dd < -5:
                        st.info("🔵 Leggero ritracciamento (>5%). Ingresso ragionevole.")
                    else:
                        st.success("🟢 Vicino ai massimi (<5%). Attendere pullback o usare DCA stretto.")

                st.write(f"📈 Trend: **{ma['asset'].trend}** | RSI: {ma['asset'].rsi:.0f}")
                if ma['asset'].cape_analysis:
                    ca = ma['asset'].cape_analysis
                    st.write(f"📉 CAPE: {ca.cape_value:.1f} ({ca.valuation_level})")
                    st.write(f"📊 Rend. 10Y atteso: ~{ca.expected_10y_return:.1f}%/anno")

                if ma["best_level"]:
                    bl = ma["best_level"]
                    st.success(
                        f"🎯 **Target:** {ma['currency']}{bl.price:,.2f} ({bl.level})\n\n"
                        f"Distanza: {bl.distance_pct:+.1f}% | Prob. 90gg: {bl.prob_90d:.0%}"
                    )
                else:
                    st.info("Nessun livello raggiungibile con prob > 60%")

                st.caption(f"💡 {ma['entry_type']}")

        # ── RACCOMANDAZIONE DETTAGLIATA ──
        st.divider()
        st.header("💡 Raccomandazione Dettagliata")
        for ma in monitor_assets:
            with st.expander(f"{'🟢' if ma['action']=='COMPRA' else '🟡' if ma['action']=='ATTENDI' else '🔴'} {ma['name']} — {ma['action']}", expanded=True):
                st.write(f"**{ma['rec']}**")
                st.write(f"Strategia consigliata: **{ma['entry_type']}**")

                # Segnali tecnici
                if ma["asset"].technical_signals:
                    st.write("**Segnali tecnici:**")
                    for sig in ma["asset"].technical_signals:
                        st.write(f"  • {sig}")

        # ── PREVISIONE PROSSIMO INGRESSO ──
        st.divider()
        st.header("🔮 Previsione Prossimo Ingresso")

        for ma in monitor_assets:
            asset = ma["asset"]
            score = ma["score"]
            currency = ma["currency"]
            name = ma["name"]

            with st.expander(f"🔮 {name}", expanded=True):
                # Raccogli tutti i fattori
                factors = []
                factor_scores = {}

                # 1. CAPE
                if asset.cape_analysis:
                    ca = asset.cape_analysis
                    cape_score = ca.valuation_score
                    factor_scores["CAPE"] = cape_score
                    if cape_score < -0.5:
                        factors.append(("🔴", "CAPE", f"CARO ({ca.cape_value:.0f}) — mercato sopravvalutato, attendere correzione"))
                    elif cape_score < 0:
                        factors.append(("🟡", "CAPE", f"Moderatamente caro ({ca.cape_value:.0f}) — ingresso graduale"))
                    else:
                        factors.append(("🟢", "CAPE", f"Valutazione nella norma ({ca.cape_value:.0f}) — buon momento"))

                # 2. Liquidità
                if report.liquidity_analysis:
                    liq = report.liquidity_analysis
                    factor_scores["Liquidità"] = liq.overall_score
                    if liq.overall_score > 0.3:
                        factors.append(("🟢", "Liquidità", f"Favorevole ({liq.overall_score:+.2f}) — condizioni buone per entrare"))
                    elif liq.overall_score > -0.2:
                        factors.append(("🟡", "Liquidità", f"Neutrale ({liq.overall_score:+.2f}) — cautela"))
                    else:
                        factors.append(("🔴", "Liquidità", f"Sfavorevole ({liq.overall_score:+.2f}) — evitare grandi ingressi"))

                # 3. Trend
                factor_scores["Trend"] = asset.technical_score
                if asset.trend in ["FORTE RIALZISTA", "RIALZISTA"]:
                    factors.append(("🟢", "Trend", f"{asset.trend} — momentum positivo"))
                elif asset.trend in ["FORTE RIBASSISTA", "RIBASSISTA"]:
                    factors.append(("🔴", "Trend", f"{asset.trend} — aspettare inversione"))
                else:
                    factors.append(("🟡", "Trend", f"{asset.trend} — direzione incerta"))

                # 4. RSI
                if asset.rsi < 30:
                    factors.append(("🟢", "RSI", f"Ipervenduto ({asset.rsi:.0f}) — 🎯 MOMENTO IDEALE per entrare"))
                    factor_scores["RSI"] = 0.5
                elif asset.rsi < 40:
                    factors.append(("🟢", "RSI", f"Zona favorevole ({asset.rsi:.0f}) — buon punto di ingresso"))
                    factor_scores["RSI"] = 0.3
                elif asset.rsi > 70:
                    factors.append(("🔴", "RSI", f"Ipercomprato ({asset.rsi:.0f}) — attendere ritracciamento"))
                    factor_scores["RSI"] = -0.5
                elif asset.rsi > 60:
                    factors.append(("🟡", "RSI", f"Alto ({asset.rsi:.0f}) — rischio pullback"))
                    factor_scores["RSI"] = -0.2
                else:
                    factors.append(("🟡", "RSI", f"Neutro ({asset.rsi:.0f})"))
                    factor_scores["RSI"] = 0

                # 5. Drawdown
                if asset.drawdown_from_ath < -20:
                    factors.append(("🟢", "Drawdown", f"{asset.drawdown_from_ath:.1f}% dai massimi — forte sconto"))
                    factor_scores["Drawdown"] = 0.5
                elif asset.drawdown_from_ath < -10:
                    factors.append(("🟢", "Drawdown", f"{asset.drawdown_from_ath:.1f}% dai massimi — correzione interessante"))
                    factor_scores["Drawdown"] = 0.3
                elif asset.drawdown_from_ath < -5:
                    factors.append(("🟡", "Drawdown", f"{asset.drawdown_from_ath:.1f}% dai massimi — leggero sconto"))
                    factor_scores["Drawdown"] = 0.1
                else:
                    factors.append(("🔴", "Drawdown", f"{asset.drawdown_from_ath:.1f}% dai massimi — vicino ai max"))
                    factor_scores["Drawdown"] = -0.2

                # 6. Golden Cross / Death Cross
                if asset.golden_cross:
                    factors.append(("🟢", "SMA", "Golden Cross (SMA50 > SMA200) — trend rialzista confermato"))
                else:
                    factors.append(("🔴", "SMA", "Death Cross (SMA50 < SMA200) — trend ribassista"))

                # Mostra tabella fattori
                for emoji, name_f, desc in factors:
                    st.write(f"{emoji} **{name_f}:** {desc}")

                # ── VERDETTO PREVISIONE ──
                st.divider()
                avg_factor = sum(factor_scores.values()) / len(factor_scores) if factor_scores else 0

                # Prezzo target consigliato
                best = ma.get("best_level")
                if best:
                    target_price = best.price
                    target_name = best.level
                    target_prob = best.prob_90d
                else:
                    target_price = asset.current_price * 0.95  # -5% come fallback
                    target_name = "Prezzo corrente -5%"
                    target_prob = 0.5

                if avg_factor > 0.2:
                    st.success(
                        f"✅ **CONDIZIONI FAVOREVOLI per ingresso**\n\n"
                        f"Score medio fattori: {avg_factor:+.2f}\n\n"
                        f"🎯 Prezzo consigliato: **{currency}{target_price:,.2f}** ({target_name})\n\n"
                        f"📈 Probabilità di raggiungerlo in 90gg: **{target_prob:.0%}**\n\n"
                        f"💡 **Suggerimento:** Entra ora o piazza un ordine limite a {currency}{target_price:,.2f}"
                    )
                elif avg_factor > -0.1:
                    st.warning(
                        f"⚠️ **CONDIZIONI MISTE — ingresso graduale (DCA)**\n\n"
                        f"Score medio fattori: {avg_factor:+.2f}\n\n"
                        f"🎯 Attendi il prezzo: **{currency}{target_price:,.2f}** ({target_name})\n\n"
                        f"📈 Probabilità in 90gg: **{target_prob:.0%}**\n\n"
                        f"💡 **Suggerimento:** Accumula in 3-4 tranche mensili. Non entrare tutto in una volta."
                    )
                else:
                    st.error(
                        f"🛑 **CONDIZIONI SFAVOREVOLI — attendere**\n\n"
                        f"Score medio fattori: {avg_factor:+.2f}\n\n"
                        f"🎯 Livello migliore: **{currency}{target_price:,.2f}** ({target_name})\n\n"
                        f"💡 **Suggerimento:** Mantenere liquidità. Aspettare che RSI scenda sotto 40 o drawdown >10%."
                    )

        # ── LIVELLI DI INGRESSO ──
        st.divider()
        st.header("🎯 Livelli di Ingresso con Probabilità")
        for ma in monitor_assets:
            asset = ma["asset"]
            if asset.entry_levels:
                st.subheader(f"{ma['name']} — Prezzo: {ma['currency']}{asset.current_price:,.2f}")
                levels_data = []
                for lvl in asset.entry_levels:
                    levels_data.append({
                        "Livello": lvl.level,
                        "Prezzo": f"{ma['currency']}{lvl.price:,.2f}",
                        "Distanza": f"{lvl.distance_pct:+.1f}%",
                        "Prob. 30gg": f"{lvl.prob_30d:.0%}",
                        "Prob. 90gg": f"{lvl.prob_90d:.0%}",
                        "Tipo": lvl.level_type.replace('_', ' ').title(),
                    })
                st.dataframe(pd.DataFrame(levels_data), use_container_width=True, hide_index=True)

        # ── ASSET ALLOCATION ──
        if report.suggested_allocation:
            st.divider()
            st.subheader("💼 Asset Allocation Consigliata")
            alloc_fig = go.Figure(go.Pie(
                labels=list(report.suggested_allocation.keys()),
                values=list(report.suggested_allocation.values()),
                hole=0.4,
                marker_colors=["#00d4ff", "#ff88ff", "#00ff88", "#4488ff", "#ffcc00"],
            ))
            alloc_fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(alloc_fig, use_container_width=True)

        # ── STATO WHATSAPP ──
        st.divider()
        from config.settings import config as app_config
        if app_config.whatsapp.enabled:
            st.success(f"✅ Notifiche WhatsApp attive → {app_config.whatsapp.to_number}")
            st.caption("Per attivare il monitor automatico: `python main.py monitor --schedule`")
        else:
            st.warning("⚠️ Notifiche WhatsApp non configurate. Configura TWILIO_ACCOUNT_SID nel .env")


# ============================
# TAB 4: NEWS & CALENDARIO
# ============================
with tab_news:
    st.title("📰 News, Earnings & Calendario Macro")

    st.markdown("""
    Calendario utili trimestrali delle principali aziende NASDAQ 100,
    prossimi eventi macroeconomici (FOMC, CPI, NFP, GDP, BCE) e
    ultime news di mercato.
    """)

    run_calendar = st.button("🔄 Aggiorna News & Calendario", use_container_width=True)

    if run_calendar:
        from analysis.news_calendar import NewsCalendarProvider
        with st.spinner("Recupero earnings e news..."):
            st.session_state["news_calendar"] = NewsCalendarProvider().get_full_calendar()

    calendar = st.session_state.get("news_calendar")
    if calendar:
        # ── PROSSIMI EARNINGS ──
        st.header("📅 Prossimi Utili Trimestrali")
        if calendar.upcoming_earnings:
            upcoming_data = []
            now_dt = pd.Timestamp.now()
            for e in calendar.upcoming_earnings[:20]:
                days_to = (e.date - datetime.now()).days
                # Revenue in miliardi
                rev_str = f"${e.revenue_estimate / 1e9:.1f}B" if e.revenue_estimate and e.revenue_estimate > 1e6 else "—"
                # Range EPS
                eps_range = ""
                if e.eps_low and e.eps_high:
                    eps_range = f"${e.eps_low:.2f} - ${e.eps_high:.2f}"
                elif e.eps_estimate:
                    eps_range = f"${e.eps_estimate:.2f}"
                # Crescita YoY
                growth_str = f"{e.eps_growth:+.1%}" if e.eps_growth is not None else "—"
                # Analisti
                analysts_str = str(e.num_analysts) if e.num_analysts else "—"

                upcoming_data.append({
                    "📆 Data": e.date.strftime("%Y-%m-%d"),
                    "⏳ Giorni": f"{days_to}gg",
                    "Ticker": e.ticker,
                    "Azienda": e.company,
                    "EPS Stimato": eps_range,
                    "Crescita YoY": growth_str,
                    "Revenue Stimato": rev_str,
                    "# Analisti": analysts_str,
                })
            st.dataframe(pd.DataFrame(upcoming_data), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun earnings futuro trovato")

        # ── EARNINGS RECENTI (con sorpresa EPS + Revenue) ──
        st.header("✅ Earnings Recenti (Ultimi 45 giorni)")
        if calendar.recent_earnings:
            recent_data = []
            for e in calendar.recent_earnings[:20]:
                # EPS surprise
                eps_surprise_str = ""
                if e.surprise_pct is not None:
                    emoji = "🟢" if e.surprise_pct > 0 else "🔴"
                    eps_surprise_str = f"{emoji} {e.surprise_pct:+.1f}%"
                # Revenue surprise
                rev_surprise_str = ""
                if e.revenue_surprise_pct is not None:
                    emoji_r = "🟢" if e.revenue_surprise_pct > 0 else "🔴"
                    rev_surprise_str = f"{emoji_r} {e.revenue_surprise_pct:+.1f}%"
                # Revenue stimato vs effettivo
                rev_est = f"${e.revenue_estimate / 1e9:.1f}B" if e.revenue_estimate and e.revenue_estimate > 1e6 else "—"
                rev_act = f"${e.reported_revenue / 1e9:.1f}B" if e.reported_revenue and e.reported_revenue > 1e6 else "—"

                recent_data.append({
                    "Data": e.date.strftime("%Y-%m-%d"),
                    "Ticker": e.ticker,
                    "Azienda": e.company,
                    "EPS Stima": f"${e.eps_estimate:.2f}" if e.eps_estimate else "—",
                    "EPS Reale": f"${e.reported_eps:.2f}" if e.reported_eps else "—",
                    "Sorpr. EPS": eps_surprise_str,
                    "Rev. Stima": rev_est,
                    "Rev. Reale": rev_act,
                    "Sorpr. Rev.": rev_surprise_str,
                })
            st.dataframe(pd.DataFrame(recent_data), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun earnings recente trovato")

        # ── CALENDARIO MACRO ──
        st.divider()
        st.header("🏛️ Calendario Macroeconomico")

        # Indicatori attuali in colonne
        macro_values = [e for e in calendar.macro_events if e.latest_value]
        if macro_values:
            macro_cols = st.columns(len(macro_values))
            for i, m in enumerate(macro_values):
                with macro_cols[i]:
                    st.metric(m.name, m.latest_value)
            st.divider()

        # Eventi per importanza
        high_events = [e for e in calendar.macro_events if e.importance == "ALTA" and not e.latest_value]
        medium_events = [e for e in calendar.macro_events if e.importance == "MEDIA" and not e.latest_value]

        if high_events:
            st.subheader("🔴 Eventi ad Alta Importanza")
            high_data = []
            for e in high_events:
                now_dt = pd.Timestamp.now()
                try:
                    evt_dt = pd.Timestamp(e.date)
                    days_to = (evt_dt - now_dt).days
                    timing = f"tra {days_to}gg" if days_to > 0 else ("✅ Passato" if days_to < 0 else "🟠 Oggi")
                except Exception:
                    timing = ""
                high_data.append({
                    "Data": e.date,
                    "Evento": e.name,
                    "Categoria": e.category,
                    "Timing": timing,
                    "Descrizione": e.description,
                })
            st.dataframe(pd.DataFrame(high_data), use_container_width=True, hide_index=True)

        if medium_events:
            st.subheader("🟡 Eventi Importanza Media")
            med_data = []
            for e in medium_events:
                med_data.append({
                    "Data": e.date,
                    "Evento": e.name,
                    "Categoria": e.category,
                    "Descrizione": e.description,
                })
            st.dataframe(pd.DataFrame(med_data), use_container_width=True, hide_index=True)

        # ── CALENDARIO ECONOMICO INVESTING.COM ──
        st.divider()
        st.header("📅 Calendario Economico (investing.com)")
        if calendar.economic_calendar:
            # Filtro per importanza
            imp_filter = st.multiselect(
                "Filtra per importanza",
                ["🔴 ALTA", "🟡 MEDIA", "🟢 BASSA"],
                default=["🔴 ALTA"],
                key="eco_filter",
            )

            filtered = [e for e in calendar.economic_calendar if e.importance in imp_filter]

            if filtered:
                # Raggruppa per data (ordinamento corretto dd/mm/yyyy → datetime)
                def _parse_date(d):
                    try:
                        return datetime.strptime(d, "%d/%m/%Y")
                    except Exception:
                        return datetime.max
                dates = sorted(set(e.date for e in filtered), key=_parse_date)
                for date_str in dates:
                    day_events = [e for e in filtered if e.date == date_str]
                    st.subheader(f"📆 {date_str}")

                    eco_data = []
                    for e in day_events:
                        eco_data.append({
                            "Ora": e.time,
                            "Paese": e.country.title(),
                            "💲": e.currency,
                            "Importanza": e.importance,
                            "Evento": e.event,
                            "Attuale": e.actual if e.actual else "—",
                            "Previsto": e.forecast if e.forecast else "—",
                            "Precedente": e.previous if e.previous else "—",
                        })
                    st.dataframe(pd.DataFrame(eco_data), use_container_width=True, hide_index=True)
            else:
                st.info("Nessun evento con il filtro selezionato")
        else:
            st.warning("Calendario economico non disponibile (investpy non installato o errore di rete)")

        # ── NEWS ──
        st.divider()
        st.header("📰 Ultime News di Mercato")
        if calendar.news:
            # Filtro: solo news importanti (market-moving) di default
            important_keywords = [
                "fed", "fomc", "interest rate", "inflation", "cpi", "gdp",
                "tariff", "trade war", "recession", "crash", "rally", "surge",
                "earnings", "layoff", "bankruptcy", "merger", "acquisition",
                "nvidia", "apple", "microsoft", "amazon", "google", "meta", "tesla",
                "nasdaq", "s&p", "dow", "market", "economy", "jobs", "unemployment",
                "bank", "treasury", "bond", "yield", "dollar", "euro",
                "war", "sanctions", "crisis", "default", "stimulus",
                "ai", "regulation", "antitrust", "supreme court",
            ]
            show_all = st.checkbox("📋 Mostra tutte le news (non solo market-moving)", value=False, key="news_filter")

            for n in calendar.news:
                text_lower = (n.title + " " + n.summary).lower()
                is_important = any(kw in text_lower for kw in important_keywords)

                if not show_all and not is_important:
                    continue

                marker = "🔴" if is_important else "📄"
                with st.expander(f"{marker} {n.title}", expanded=False):
                    if is_important:
                        st.markdown("**⚡ Market-moving**")
                    st.caption(f"{n.source} | {n.date}")
                    st.write(n.summary)
                    if n.url:
                        st.markdown(f"[🔗 Leggi articolo completo]({n.url})")
        else:
            st.info("Nessuna news disponibile")

        st.caption(f"Ultimo aggiornamento: {calendar.last_updated}")


# ============================
# FOOTER
# ============================
st.divider()
st.caption("Trading Agent Multi-Asset | Crypto + NDX + SWDA.MI + CSNDX + CAPE + Liquidità + Segnali + News")