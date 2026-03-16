"""
Strategia Momentum / Trend Following.

Logica:
- LONG quando EMA veloce > EMA lenta + conferma MACD bullish + RSI non ipercomprato + ADX forte
- SHORT/EXIT quando EMA veloce < EMA lenta + conferma MACD bearish + RSI non ipervenduto
- Filtri: Supertrend direction, volume sopra la media, Bollinger Band position
- Score combinato ML (opzionale) per filtrare i falsi segnali
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

from indicators.technical import TechnicalIndicators
from utils.logger import get_logger

logger = get_logger("strategy.momentum")


class Signal(Enum):
    """Tipi di segnale di trading."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Rappresenta un segnale di trading con i dettagli."""
    signal: Signal
    confidence: float          # 0.0 - 1.0
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: list[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class MomentumStrategy:
    """
    Strategia basata sul momentum e trend following.
    Combina più indicatori tecnici per generare segnali di trading.
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        adx_threshold: float = 20,
        volume_threshold: float = 1.0,
        min_confidence: float = 0.5,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_threshold = adx_threshold
        self.volume_threshold = volume_threshold
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def analyze(self, df: pd.DataFrame, ml_prediction: Optional[float] = None,
                regime: Optional[str] = None) -> TradeSignal:
        """
        Analizza il DataFrame con indicatori tecnici e genera un segnale.

        Args:
            df: DataFrame con indicatori tecnici già calcolati
            ml_prediction: Predizione ML opzionale (0=bearish, 0.5=neutral, 1=bullish)
            regime: Regime di mercato HMM opzionale (es. "Bear Market", "Bull Trend")

        Returns:
            TradeSignal con segnale, confidenza e dettagli
        """
        if len(df) < 2:
            return TradeSignal(signal=Signal.HOLD, confidence=0.0, price=0.0, reasons=["Dati insufficienti"])

        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current["close"]

        buy_score = 0.0
        sell_score = 0.0
        reasons_buy: list[str] = []
        reasons_sell: list[str] = []

        # ============================================================
        # 1. EMA CROSSOVER (peso: 25%)
        # ============================================================
        ema_fast_val = current.get(f"ema_{self.ema_fast}", None)
        ema_slow_val = current.get(f"ema_{self.ema_slow}", None)

        if ema_fast_val is not None and ema_slow_val is not None:
            if ema_fast_val > ema_slow_val:
                buy_score += 0.25
                reasons_buy.append(f"EMA{self.ema_fast} > EMA{self.ema_slow} (trend rialzista)")
            else:
                sell_score += 0.25
                reasons_sell.append(f"EMA{self.ema_fast} < EMA{self.ema_slow} (trend ribassista)")

            # Bonus per crossover appena avvenuto
            ema_cross_bull = current.get("ema_cross_bull", 0)
            ema_cross_bear = current.get("ema_cross_bear", 0)
            if ema_cross_bull:
                buy_score += 0.10
                reasons_buy.append("Crossover EMA bullish appena avvenuto")
            elif ema_cross_bear:
                sell_score += 0.10
                reasons_sell.append("Crossover EMA bearish appena avvenuto")

        # ============================================================
        # 2. MACD (peso: 20%)
        # ============================================================
        macd_hist = current.get("macd_histogram", None)
        prev_macd_hist = prev.get("macd_histogram", None)

        if macd_hist is not None:
            if macd_hist > 0:
                buy_score += 0.15
                reasons_buy.append(f"MACD histogram positivo ({macd_hist:.4f})")
            else:
                sell_score += 0.15
                reasons_sell.append(f"MACD histogram negativo ({macd_hist:.4f})")

            # MACD momentum crescente
            if prev_macd_hist is not None:
                if macd_hist > prev_macd_hist and macd_hist > 0:
                    buy_score += 0.05
                    reasons_buy.append("MACD momentum in aumento")
                elif macd_hist < prev_macd_hist and macd_hist < 0:
                    sell_score += 0.05
                    reasons_sell.append("MACD momentum in diminuzione")

        # ============================================================
        # 3. RSI (peso: 15%)
        # ============================================================
        rsi = current.get("rsi", 50)

        if rsi < self.rsi_oversold:
            buy_score += 0.15
            reasons_buy.append(f"RSI ipervenduto ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            sell_score += 0.15
            reasons_sell.append(f"RSI ipercomprato ({rsi:.1f})")
        elif rsi < 50:
            buy_score += 0.05
            reasons_buy.append(f"RSI sotto 50 ({rsi:.1f})")
        else:
            sell_score += 0.05
            reasons_sell.append(f"RSI sopra 50 ({rsi:.1f})")

        # ============================================================
        # 4. ADX - Forza del trend (peso: 15%)
        # ============================================================
        adx = current.get("adx", 0)
        di_plus = current.get("di_plus", 0)
        di_minus = current.get("di_minus", 0)

        if adx > self.adx_threshold:
            if di_plus > di_minus:
                buy_score += 0.15
                reasons_buy.append(f"ADX forte ({adx:.1f}), DI+ > DI- (trend rialzista)")
            else:
                sell_score += 0.15
                reasons_sell.append(f"ADX forte ({adx:.1f}), DI- > DI+ (trend ribassista)")
        else:
            # Trend debole, segnale meno affidabile
            reasons_buy.append(f"ADX debole ({adx:.1f}), trend non definito")

        # ============================================================
        # 5. SUPERTREND (peso: 10%)
        # ============================================================
        st_direction = current.get("supertrend_direction", 0)
        if st_direction == 1:
            buy_score += 0.10
            reasons_buy.append("Supertrend bullish")
        elif st_direction == -1:
            sell_score += 0.10
            reasons_sell.append("Supertrend bearish")

        # ============================================================
        # 6. BOLLINGER BANDS (peso: 5%)
        # ============================================================
        bb_pct = current.get("bb_pct", 0.5)
        if bb_pct < 0.2:
            buy_score += 0.05
            reasons_buy.append(f"Prezzo vicino alla BB inferiore ({bb_pct:.2f})")
        elif bb_pct > 0.8:
            sell_score += 0.05
            reasons_sell.append(f"Prezzo vicino alla BB superiore ({bb_pct:.2f})")

        # ============================================================
        # 7. VOLUME (peso: 5%)
        # ============================================================
        volume_ratio = current.get("volume_ratio", 1.0)
        if volume_ratio > self.volume_threshold:
            # Volume alto conferma il segnale dominante
            if buy_score > sell_score:
                buy_score += 0.05
                reasons_buy.append(f"Volume confermato ({volume_ratio:.1f}x media)")
            else:
                sell_score += 0.05
                reasons_sell.append(f"Volume confermato ({volume_ratio:.1f}x media)")

        # ============================================================
        # 8. ML PREDICTION (peso: fino al 15%)
        # ============================================================
        if ml_prediction is not None:
            if ml_prediction > 0.6:
                ml_bonus = 0.15 * (ml_prediction - 0.5) * 2  # Scala 0-0.15
                buy_score += ml_bonus
                reasons_buy.append(f"ML predice rialzo (prob: {ml_prediction:.2f})")
            elif ml_prediction < 0.4:
                ml_bonus = 0.15 * (0.5 - ml_prediction) * 2
                sell_score += ml_bonus
                reasons_sell.append(f"ML predice ribasso (prob: {ml_prediction:.2f})")

        # ============================================================
        # 9. REGIME FILTER (HMM) - Modula il segnale se disponibile
        # ============================================================
        if regime:
            regime_adjustments = {
                "Strong Bull": (0.05, 0.0, "Regime Strong Bull: conferma BUY"),
                "Bull Trend": (0.03, 0.0, "Regime Bull Trend: lieve supporto BUY"),
                "Recovery": (0.08, 0.0, "Regime Recovery: forte supporto BUY"),
                "Consolidation": (0.0, 0.0, "Regime Consolidation: segnale neutro"),
                "High Volatility": (0.0, 0.03, "Regime High Volatility: cautela"),
                "Correction": (-0.05, 0.10, "Regime Correction: contrarian BUY"),
                "Bear Market": (-0.08, 0.15, "Regime Bear Market: forte contrarian BUY"),
            }
            buy_adj, sell_adj, reason = regime_adjustments.get(regime, (0, 0, ""))
            if reason:
                # In regime bearish, per trading attivo (non accumulo) riduciamo buy e aumentiamo sell
                # Ma in accumulo il bonus è positivo — qui siamo nel trading attivo
                buy_score += buy_adj
                sell_score += sell_adj
                if buy_adj > 0:
                    reasons_buy.append(reason)
                elif sell_adj > 0:
                    reasons_sell.append(reason)
                else:
                    reasons_buy.append(reason)

        # ============================================================
        # DECISIONE FINALE
        # ============================================================
        confidence = abs(buy_score - sell_score)

        if buy_score > sell_score and confidence >= self.min_confidence:
            signal = Signal.BUY
            stop_loss = price * (1 - self.stop_loss_pct) if self.stop_loss_pct > 0 else None
            take_profit = price * (1 + self.take_profit_pct) if self.take_profit_pct > 0 else None
            reasons = reasons_buy

        elif sell_score > buy_score and confidence >= self.min_confidence:
            signal = Signal.SELL
            stop_loss = price * (1 + self.stop_loss_pct) if self.stop_loss_pct > 0 else None
            take_profit = price * (1 - self.take_profit_pct) if self.take_profit_pct > 0 else None
            reasons = reasons_sell

        else:
            signal = Signal.HOLD
            stop_loss = None
            take_profit = None
            reasons = [f"Confidenza insufficiente (buy: {buy_score:.2f}, sell: {sell_score:.2f})"]

        trade_signal = TradeSignal(
            signal=signal,
            confidence=min(confidence, 1.0),
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
        )

        logger.info(
            f"Segnale: {signal.value} | Confidenza: {confidence:.2f} | "
            f"Prezzo: {price:.2f} | Motivi: {', '.join(reasons[:3])}"
        )

        return trade_signal

    def generate_backtest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali per ogni riga del DataFrame (per backtesting).
        Aggiunge colonne 'signal', 'confidence', 'stop_loss', 'take_profit'.
        """
        signals = []
        for i in range(1, len(df)):
            window = df.iloc[:i + 1]
            trade_signal = self.analyze(window)
            signals.append({
                "signal": trade_signal.signal.value,
                "confidence": trade_signal.confidence,
                "stop_loss": trade_signal.stop_loss,
                "take_profit": trade_signal.take_profit,
            })

        # Prima riga senza segnale
        signals.insert(0, {
            "signal": "HOLD",
            "confidence": 0.0,
            "stop_loss": None,
            "take_profit": None,
        })

        signals_df = pd.DataFrame(signals, index=df.index)
        return pd.concat([df, signals_df], axis=1)
