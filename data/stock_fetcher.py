"""
Modulo per il recupero dati di mercato tradizionale (azioni, ETF, indici) tramite yfinance.
Supporta NASDAQ 100 (^NDX), SWDA.MI, CSNDX.MI, VIX e dati macro.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("data.stock_fetcher")

# Ticker mappings
TICKERS = {
    "NASDAQ100": "^NDX",        # NASDAQ 100 Index (prezzo indice)
    "NASDAQ100_ETF": "QQQ",    # Invesco QQQ Trust (per volumi e liquidità)
    "SWDA": "SWDA.MI",          # iShares Core MSCI World UCITS ETF (Milano)
    "SWDA_ALT": "SWDA.L",      # Alternativa London
    "SWDA_ALT2": "IWDA.AS",    # Alternativa Amsterdam
    "CSNDX": "CNDX.MI",         # iShares NASDAQ 100 UCITS ETF Acc (Milano)
    "CSNDX_ALT": "CSNDX.MI",   # Alternativa ticker Milano
    "CSNDX_ALT2": "SXRV.DE",   # Alternativa Xetra
    "SP500": "SPY",             # S&P 500 ETF
    "VIX": "^VIX",             # CBOE Volatility Index
    "TNX": "^TNX",             # 10-Year Treasury Yield
    "IRX": "^IRX",             # 13-Week Treasury Bill
    "DXY": "DX-Y.NYB",        # US Dollar Index
    "GOLD": "GC=F",            # Gold Futures
    "TLT": "TLT",             # 20+ Year Treasury Bond ETF (proxy liquidità)
    "HYG": "HYG",             # High Yield Corporate Bond ETF (risk appetite)
    "LQD": "LQD",             # Investment Grade Corporate Bond ETF
}


class StockFetcher:
    """Classe per il recupero dei dati di mercato tradizionale via yfinance."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Recupera dati OHLCV per un ticker.

        Args:
            ticker: Simbolo Yahoo Finance (es. 'QQQ', 'SWDA.L')
            period: Periodo ('1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
            interval: Intervallo ('1d', '1wk', '1mo')

        Returns:
            DataFrame con colonne: open, high, low, close, volume
        """
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                logger.warning(f"Nessun dato per {ticker}")
                return pd.DataFrame()

            # Normalizza nomi colonne (yfinance usa maiuscole)
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            data.index.name = "timestamp"

            # Rimuovi colonne multi-level se presenti
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            logger.info(f"Recuperate {len(data)} candele per {ticker} [{interval}] "
                       f"da {data.index[0]} a {data.index[-1]}")

            self._cache[ticker] = data
            return data

        except Exception as e:
            logger.error(f"Errore nel recupero dati per {ticker}: {e}")
            return pd.DataFrame()

    def fetch_historical(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Recupera dati storici per un numero specifico di giorni.

        Args:
            ticker: Simbolo Yahoo Finance
            days: Numero di giorni di storico
            interval: Intervallo temporale

        Returns:
            DataFrame OHLCV
        """
        start = datetime.now() - timedelta(days=days)
        end = datetime.now()

        try:
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                logger.warning(f"Nessun dato storico per {ticker}")
                return pd.DataFrame()

            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            data.index.name = "timestamp"

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            logger.info(f"Recuperate {len(data)} candele storiche per {ticker}")
            self._cache[ticker] = data
            return data

        except Exception as e:
            logger.error(f"Errore nel recupero storico per {ticker}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Recupera dati per più ticker contemporaneamente.

        Returns:
            Dizionario ticker -> DataFrame
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, period=period, interval=interval)
            if not df.empty:
                results[ticker] = df
        return results

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Recupera il prezzo corrente di un ticker."""
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            return info.get("lastPrice", None) or info.get("last_price", None)
        except Exception as e:
            logger.error(f"Errore prezzo corrente {ticker}: {e}")
            return None

    def get_ticker_info(self, ticker: str) -> dict:
        """Recupera informazioni dettagliate su un ticker."""
        try:
            t = yf.Ticker(ticker)
            return t.info
        except Exception as e:
            logger.error(f"Errore info {ticker}: {e}")
            return {}

    def fetch_pe_ratio(self, ticker: str = "SPY") -> Optional[float]:
        """Recupera il P/E ratio di un ETF/indice."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            pe = info.get("trailingPE", None) or info.get("forwardPE", None)
            return pe
        except Exception as e:
            logger.error(f"Errore P/E per {ticker}: {e}")
            return None

    def get_cached(self, ticker: str) -> Optional[pd.DataFrame]:
        """Ritorna dati dalla cache se disponibili."""
        return self._cache.get(ticker)

    def calc_entry_levels(self, df: pd.DataFrame, ticker: str = "") -> list[dict]:
        """
        Calcola livelli di ingresso ottimali con probabilità di raggiungerli.

        Livelli calcolati:
        - SMA 200 (supporto di lungo periodo)
        - SMA 50 (supporto di medio periodo)
        - Fibonacci retracements (dal max/min degli ultimi 6 mesi)
        - Bollinger Band inferiore (1.5 std dev)
        - Supporti storici recenti (minimi locali)

        Per ogni livello calcola la probabilità di essere raggiunto
        nei prossimi 30-90 giorni basandosi sulla volatilità storica.

        Args:
            df: DataFrame OHLCV con almeno 200 righe
            ticker: Nome del ticker per il log

        Returns:
            Lista di dict con: level, price, distance_pct, prob_30d, prob_90d, description
        """
        if df is None or df.empty or len(df) < 50:
            return []

        current = df["close"].iloc[-1]
        levels = []

        # --- SMA 200 ---
        if len(df) >= 200:
            sma200 = df["close"].tail(200).mean()
            levels.append({
                "level": "SMA 200",
                "price": sma200,
                "type": "supporto_dinamico",
                "description": "Media mobile 200 giorni - supporto di lungo termine",
            })

        # --- SMA 50 ---
        if len(df) >= 50:
            sma50 = df["close"].tail(50).mean()
            levels.append({
                "level": "SMA 50",
                "price": sma50,
                "type": "supporto_dinamico",
                "description": "Media mobile 50 giorni - supporto di medio termine",
            })

        # --- Fibonacci retracements ---
        lookback = min(len(df), 132)  # ~6 mesi
        recent = df["close"].tail(lookback)
        high = recent.max()
        low = recent.min()
        fib_range = high - low

        if fib_range > 0:
            for fib_pct, fib_name in [
                (0.236, "Fib 23.6%"),
                (0.382, "Fib 38.2%"),
                (0.500, "Fib 50.0%"),
                (0.618, "Fib 61.8%"),
            ]:
                fib_price = high - fib_range * fib_pct
                levels.append({
                    "level": fib_name,
                    "price": fib_price,
                    "type": "fibonacci",
                    "description": f"Fibonacci {fib_pct:.1%} dal max {high:,.0f} al min {low:,.0f}",
                })

        # --- Bollinger Band inferiore (20 periodi, 1.5 std) ---
        if len(df) >= 20:
            sma20 = df["close"].tail(20).mean()
            std20 = df["close"].tail(20).std()
            bb_lower = sma20 - 1.5 * std20
            levels.append({
                "level": "BB Lower (1.5σ)",
                "price": bb_lower,
                "type": "volatilita",
                "description": "Banda di Bollinger inferiore (1.5 deviazioni standard)",
            })

        # --- Minimi locali recenti (supporti) ---
        lookback_sup = min(len(df), 66)  # ~3 mesi
        recent_df = df.tail(lookback_sup)
        local_mins = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df["low"].iloc[i] <= recent_df["low"].iloc[i-1] and
                recent_df["low"].iloc[i] <= recent_df["low"].iloc[i-2] and
                recent_df["low"].iloc[i] <= recent_df["low"].iloc[i+1] and
                recent_df["low"].iloc[i] <= recent_df["low"].iloc[i+2]):
                local_mins.append(recent_df["low"].iloc[i])

        if local_mins:
            # Prendi i 2 minimi più significativi (diversi da current)
            local_mins = sorted(set([round(m, 2) for m in local_mins]))
            for i, sup in enumerate(local_mins[:2]):
                if abs(sup - current) / current > 0.005:  # Almeno 0.5% diverso
                    levels.append({
                        "level": f"Supporto {i+1}",
                        "price": sup,
                        "type": "supporto_statico",
                        "description": f"Minimo locale recente (3 mesi)",
                    })

        # --- Calcola distanza % e probabilità per ogni livello ---
        # Volatilità annualizzata
        if len(df) >= 22:
            returns = df["close"].pct_change().dropna()
            daily_vol = returns.tail(66).std()  # vol ultimi 3 mesi
            annual_vol = daily_vol * np.sqrt(252)
        else:
            daily_vol = 0.015  # fallback 1.5% al giorno
            annual_vol = daily_vol * np.sqrt(252)

        result = []
        for lvl in levels:
            price = lvl["price"]
            dist_pct = (price / current - 1) * 100

            # Probabilità di raggiungere il livello
            # Modello: probabilità che il prezzo tocchi un livello in N giorni
            # Usa la formula del minimo di un random walk geometrico
            prob_30d = self._calc_touch_probability(dist_pct / 100, daily_vol, 30)
            prob_90d = self._calc_touch_probability(dist_pct / 100, daily_vol, 90)

            result.append({
                "level": lvl["level"],
                "price": price,
                "distance_pct": dist_pct,
                "prob_30d": prob_30d,
                "prob_90d": prob_90d,
                "type": lvl["type"],
                "description": lvl["description"],
            })

        # Ordina per prezzo decrescente (dal più vicino al più lontano sotto)
        result.sort(key=lambda x: -x["price"])

        logger.info(f"Calcolati {len(result)} livelli di ingresso per {ticker}")
        return result

    @staticmethod
    def _calc_touch_probability(target_return: float, daily_vol: float, days: int) -> float:
        """
        Calcola la probabilità che il prezzo tocchi un livello target in N giorni.

        Usa la formula del primo passaggio per un moto browniano geometrico:
        P(min(S_t) <= K) per t in [0, T]

        Per livelli sotto il prezzo corrente (target_return < 0):
        la formula del barrier crossing approssimata.

        Per livelli sopra il prezzo corrente (target_return > 0):
        probabilità alta su pullback in bull market.

        Args:
            target_return: Return necessario (negativo = sotto prezzo attuale)
            daily_vol: Volatilità giornaliera
            days: Orizzonte in giorni

        Returns:
            Probabilità 0.0 → 1.0
        """
        from scipy.stats import norm

        if abs(target_return) < 0.001:
            return 0.95  # Già al livello

        vol_t = daily_vol * np.sqrt(days)
        drift = -0.0001 * days  # drift leggermente negativo (conservativo)

        if target_return < 0:
            # Livello sotto il prezzo corrente
            # Probabilità di toccare un livello inferiore
            # P(min S_t <= K) ≈ 2 * N(target / vol_sqrt_T) per drift ~0
            z = (target_return - drift) / vol_t
            prob = 2 * norm.cdf(z)
        else:
            # Livello sopra il prezzo corrente
            z = (target_return - drift) / vol_t
            prob = 2 * (1 - norm.cdf(z))

        return max(0.01, min(0.99, prob))
