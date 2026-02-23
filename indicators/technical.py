"""
Modulo per il calcolo degli indicatori tecnici.
Usa la libreria 'ta' (Technical Analysis Library in Python).
Include indicatori di trend, momentum, volatilità e volume.
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from utils.logger import get_logger

logger = get_logger("indicators.technical")


class TechnicalIndicators:
    """Calcola e aggiunge indicatori tecnici a un DataFrame OHLCV."""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge tutti gli indicatori tecnici al DataFrame.
        Il DataFrame deve avere colonne: open, high, low, close, volume.
        """
        df = df.copy()

        # --- Indicatori di Trend ---
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_adx(df)
        df = TechnicalIndicators.add_supertrend(df)

        # --- Indicatori di Momentum ---
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_cci(df)

        # --- Indicatori di Volatilità ---
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)

        # --- Indicatori di Volume ---
        df = TechnicalIndicators.add_volume_indicators(df)

        # --- Segnali derivati ---
        df = TechnicalIndicators.add_crossover_signals(df)

        df.dropna(inplace=True)
        logger.debug(f"Aggiunti {len(df.columns)} indicatori tecnici, {len(df)} righe valide")
        return df

    # ========================
    # INDICATORI DI TREND
    # ========================

    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge medie mobili esponenziali e semplici."""
        # EMA (Exponential Moving Average)
        df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
        df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()

        # SMA (Simple Moving Average)
        df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()

        # Distanza dal prezzo alle medie mobili (in %)
        df["price_vs_ema_21"] = (df["close"] - df["ema_21"]) / df["ema_21"] * 100
        df["price_vs_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100

        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge MACD (Moving Average Convergence Divergence)."""
        macd_indicator = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_indicator.macd()
        df["macd_histogram"] = macd_indicator.macd_diff()
        df["macd_signal"] = macd_indicator.macd_signal()
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge ADX (Average Directional Index) per la forza del trend."""
        adx_indicator = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx_indicator.adx()
        df["di_plus"] = adx_indicator.adx_pos()
        df["di_minus"] = adx_indicator.adx_neg()
        return df

    @staticmethod
    def add_supertrend(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge Supertrend indicator (implementazione custom basata su ATR)."""
        period = 10
        multiplier = 3.0

        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period).average_true_range()

        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        df["supertrend"] = supertrend
        df["supertrend_direction"] = direction
        return df

    # ========================
    # INDICATORI DI MOMENTUM
    # ========================

    @staticmethod
    def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Aggiunge RSI (Relative Strength Index)."""
        df["rsi"] = RSIIndicator(close=df["close"], window=length).rsi()
        # Livelli RSI
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge Stochastic Oscillator."""
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        return df

    @staticmethod
    def add_cci(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge CCI (Commodity Channel Index)."""
        df["cci"] = CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20).cci()
        return df

    # ========================
    # INDICATORI DI VOLATILITÀ
    # ========================

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge Bollinger Bands."""
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_bandwidth"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Aggiunge ATR (Average True Range)."""
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=length).average_true_range()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    # ========================
    # INDICATORI DI VOLUME
    # ========================

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge indicatori basati sul volume."""
        # OBV (On-Balance Volume)
        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

        # Volume SMA (per confrontare il volume attuale con la media)
        df["volume_sma_20"] = SMAIndicator(close=df["volume"], window=20).sma_indicator()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # VWAP approssimato (su singola sessione)
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        return df

    # ========================
    # SEGNALI DI CROSSOVER
    # ========================

    @staticmethod
    def add_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge segnali derivati dai crossover degli indicatori."""

        # EMA 9/21 crossover
        df["ema_cross_bull"] = (
            (df["ema_9"] > df["ema_21"]) &
            (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
        ).astype(int)
        df["ema_cross_bear"] = (
            (df["ema_9"] < df["ema_21"]) &
            (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
        ).astype(int)

        # MACD crossover
        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_cross_bull"] = (
                (df["macd"] > df["macd_signal"]) &
                (df["macd"].shift(1) <= df["macd_signal"].shift(1))
            ).astype(int)
            df["macd_cross_bear"] = (
                (df["macd"] < df["macd_signal"]) &
                (df["macd"].shift(1) >= df["macd_signal"].shift(1))
            ).astype(int)

        # Golden Cross / Death Cross (SMA 50/200)
        if "ema_50" in df.columns and "ema_200" in df.columns:
            df["golden_cross"] = (
                (df["ema_50"] > df["ema_200"]) &
                (df["ema_50"].shift(1) <= df["ema_200"].shift(1))
            ).astype(int)
            df["death_cross"] = (
                (df["ema_50"] < df["ema_200"]) &
                (df["ema_50"].shift(1) >= df["ema_200"].shift(1))
            ).astype(int)

        return df

    # ========================
    # UTILITY
    # ========================

    @staticmethod
    def get_current_signals(df: pd.DataFrame) -> dict:
        """
        Analizza l'ultima riga del DataFrame e restituisce
        un riepilogo dei segnali attuali.
        """
        if df.empty:
            return {}

        last = df.iloc[-1]

        signals = {
            "rsi": round(last.get("rsi", 0), 2),
            "rsi_signal": "overbought" if last.get("rsi", 50) > 70
                          else "oversold" if last.get("rsi", 50) < 30
                          else "neutral",
            "macd_histogram": round(last.get("macd_histogram", 0), 4),
            "macd_signal": "bullish" if last.get("macd_histogram", 0) > 0 else "bearish",
            "adx": round(last.get("adx", 0), 2),
            "trend_strength": "strong" if last.get("adx", 0) > 25 else "weak",
            "bb_position": round(last.get("bb_pct", 0.5), 2),
            "ema_trend": "bullish" if last.get("ema_9", 0) > last.get("ema_21", 0) else "bearish",
            "supertrend": "bullish" if last.get("supertrend_direction", 1) == 1 else "bearish",
            "volume_ratio": round(last.get("volume_ratio", 1), 2),
        }

        return signals
