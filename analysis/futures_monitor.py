"""
Futures Monitor - Monitoraggio futures NASDAQ 100, S&P 500 e Dow Jones.

Analizza la direzione pre-market dei futures per prevedere l'apertura dei mercati.
Combina:
1. Variazione futures vs chiusura precedente
2. Momentum intraday (trend delle ultime ore)
3. VIX (paura/avidità)
4. Correlazione con news recenti
5. Gap analysis (gap up/down atteso all'apertura)

Tickers futures:
- NQ=F → E-mini NASDAQ 100 Futures
- ES=F → E-mini S&P 500 Futures
- YM=F → E-mini Dow Jones Futures
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("analysis.futures_monitor")


@dataclass
class FuturesSnapshot:
    """Snapshot di un singolo contratto futures."""
    ticker: str
    name: str
    last_price: float = 0.0
    prev_close: float = 0.0
    change_pct: float = 0.0         # Variazione vs chiusura precedente
    intraday_trend: str = "FLAT"     # UP, DOWN, FLAT
    momentum_1h: float = 0.0        # Variazione ultima ora
    momentum_4h: float = 0.0        # Variazione ultime 4 ore
    high_today: float = 0.0
    low_today: float = 0.0
    range_pct: float = 0.0          # Range giornaliero %
    volume: int = 0
    last_update: str = ""


@dataclass
class MarketOutlook:
    """Previsione della direzione di mercato all'apertura."""
    direction: str = "NEUTRO"       # RIALZISTA, RIBASSISTA, NEUTRO
    confidence: float = 0.0         # 0.0 → 1.0
    gap_expected: str = "FLAT"      # GAP UP, GAP DOWN, FLAT
    gap_pct: float = 0.0
    summary: str = ""
    signals: list[str] = field(default_factory=list)
    risk_level: str = "MEDIO"       # BASSO, MEDIO, ALTO


@dataclass
class FuturesAnalysis:
    """Analisi completa dei futures."""
    nasdaq_futures: Optional[FuturesSnapshot] = None
    sp500_futures: Optional[FuturesSnapshot] = None
    dow_futures: Optional[FuturesSnapshot] = None
    vix_current: float = 0.0
    vix_change: float = 0.0
    outlook: Optional[MarketOutlook] = None
    last_updated: str = ""


class FuturesMonitor:
    """Monitora i futures e genera previsioni sulla direzione dei mercati."""

    FUTURES = {
        "NQ=F": "NASDAQ 100 Futures (E-mini)",
        "ES=F": "S&P 500 Futures (E-mini)",
        "YM=F": "Dow Jones Futures (E-mini)",
    }

    def analyze(self) -> FuturesAnalysis:
        """Esegue l'analisi completa dei futures."""
        logger.info("Analisi futures in corso...")

        result = FuturesAnalysis(
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Recupera snapshot di ogni futures
        snapshots = {}
        for ticker, name in self.FUTURES.items():
            snap = self._get_snapshot(ticker, name)
            if snap:
                snapshots[ticker] = snap

        result.nasdaq_futures = snapshots.get("NQ=F")
        result.sp500_futures = snapshots.get("ES=F")
        result.dow_futures = snapshots.get("YM=F")

        # VIX
        result.vix_current, result.vix_change = self._get_vix()

        # Genera outlook
        result.outlook = self._generate_outlook(result)

        logger.info(f"Futures analizzati: {result.outlook.direction} "
                    f"(confidenza {result.outlook.confidence:.0%})")

        return result

    def _get_snapshot(self, ticker: str, name: str) -> Optional[FuturesSnapshot]:
        """Recupera lo snapshot di un singolo contratto futures."""
        try:
            tk = yf.Ticker(ticker)

            # Dati orari per momentum
            hourly = tk.history(period="5d", interval="1h")
            if hourly.empty:
                return None

            # Dati 15 min per dettaglio intraday
            intraday = tk.history(period="1d", interval="15m", prepost=True)

            last_price = hourly["Close"].iloc[-1]

            # Previous close: prezzo di chiusura della sessione regolare precedente
            # Usa il close di 2 giorni fa (il futures potrebbe essere già in sessione)
            daily = tk.history(period="5d", interval="1d")
            if daily is not None and len(daily) >= 2:
                prev_close = daily["Close"].iloc[-2]
            else:
                prev_close = hourly["Close"].iloc[0]

            change_pct = (last_price / prev_close - 1) * 100 if prev_close > 0 else 0

            # Momentum 1h e 4h
            momentum_1h = 0.0
            if len(hourly) >= 2:
                momentum_1h = (hourly["Close"].iloc[-1] / hourly["Close"].iloc[-2] - 1) * 100
            momentum_4h = 0.0
            if len(hourly) >= 5:
                momentum_4h = (hourly["Close"].iloc[-1] / hourly["Close"].iloc[-5] - 1) * 100

            # Trend intraday
            if momentum_4h > 0.2:
                trend = "UP"
            elif momentum_4h < -0.2:
                trend = "DOWN"
            else:
                trend = "FLAT"

            # Range giornaliero
            if not intraday.empty:
                high_today = intraday["High"].max()
                low_today = intraday["Low"].min()
            else:
                high_today = hourly["High"].iloc[-1]
                low_today = hourly["Low"].iloc[-1]
            range_pct = (high_today / low_today - 1) * 100 if low_today > 0 else 0

            # Volume
            vol = int(hourly["Volume"].tail(5).sum()) if "Volume" in hourly.columns else 0

            snap = FuturesSnapshot(
                ticker=ticker,
                name=name,
                last_price=last_price,
                prev_close=prev_close,
                change_pct=change_pct,
                intraday_trend=trend,
                momentum_1h=momentum_1h,
                momentum_4h=momentum_4h,
                high_today=high_today,
                low_today=low_today,
                range_pct=range_pct,
                volume=vol,
                last_update=datetime.now().strftime("%H:%M"),
            )

            return snap

        except Exception as e:
            logger.error(f"Errore futures {ticker}: {e}")
            return None

    def _get_vix(self) -> tuple[float, float]:
        """Recupera VIX corrente e variazione."""
        try:
            vix = yf.Ticker("^VIX")
            h = vix.history(period="5d")
            if h.empty:
                return 0.0, 0.0
            current = h["Close"].iloc[-1]
            prev = h["Close"].iloc[-2] if len(h) >= 2 else current
            change = (current / prev - 1) * 100
            return current, change
        except Exception:
            return 0.0, 0.0

    def _generate_outlook(self, analysis: FuturesAnalysis) -> MarketOutlook:
        """Genera la previsione di direzione basata su tutti i fattori."""
        outlook = MarketOutlook()
        score = 0.0  # Positivo = rialzista, negativo = ribassista
        signals = []

        # ── 1. Direzione futures (peso 40%) ──
        futures_changes = []
        for snap in [analysis.nasdaq_futures, analysis.sp500_futures, analysis.dow_futures]:
            if snap:
                futures_changes.append(snap.change_pct)

        if futures_changes:
            avg_change = sum(futures_changes) / len(futures_changes)
            if avg_change > 0.5:
                score += 0.4
                signals.append(f"🟢 Futures in rialzo ({avg_change:+.2f}%): apertura positiva attesa")
            elif avg_change > 0.1:
                score += 0.2
                signals.append(f"🟢 Futures leggermente positivi ({avg_change:+.2f}%)")
            elif avg_change < -0.5:
                score -= 0.4
                signals.append(f"🔴 Futures in ribasso ({avg_change:+.2f}%): apertura negativa attesa")
            elif avg_change < -0.1:
                score -= 0.2
                signals.append(f"🔴 Futures leggermente negativi ({avg_change:+.2f}%)")
            else:
                signals.append(f"🟡 Futures piatti ({avg_change:+.2f}%): apertura invariata")

            # Gap analysis
            outlook.gap_pct = avg_change
            if avg_change > 0.3:
                outlook.gap_expected = "GAP UP"
            elif avg_change < -0.3:
                outlook.gap_expected = "GAP DOWN"
            else:
                outlook.gap_expected = "FLAT"

        # ── 2. Momentum intraday (peso 25%) ──
        if analysis.nasdaq_futures:
            m4h = analysis.nasdaq_futures.momentum_4h
            if m4h > 0.3:
                score += 0.25
                signals.append(f"🟢 Momentum NQ 4h positivo ({m4h:+.2f}%): trend rialzista in atto")
            elif m4h < -0.3:
                score -= 0.25
                signals.append(f"🔴 Momentum NQ 4h negativo ({m4h:+.2f}%): trend ribassista in atto")

        # ── 3. Concordanza tra futures (peso 15%) ──
        if len(futures_changes) >= 3:
            all_positive = all(c > 0 for c in futures_changes)
            all_negative = all(c < 0 for c in futures_changes)
            if all_positive:
                score += 0.15
                signals.append("🟢 Tutti i futures concordi al rialzo (NQ, ES, YM)")
            elif all_negative:
                score -= 0.15
                signals.append("🔴 Tutti i futures concordi al ribasso (NQ, ES, YM)")
            else:
                signals.append("🟡 Futures discordanti: segnale misto")

        # ── 4. VIX (peso 20%) ──
        if analysis.vix_current > 0:
            if analysis.vix_current > 30:
                score -= 0.2
                signals.append(f"🔴 VIX molto alto ({analysis.vix_current:.1f}): mercato in panico")
                outlook.risk_level = "ALTO"
            elif analysis.vix_current > 25:
                score -= 0.1
                signals.append(f"🟡 VIX alto ({analysis.vix_current:.1f}): nervosismo elevato")
                outlook.risk_level = "ALTO"
            elif analysis.vix_current > 20:
                signals.append(f"🟡 VIX moderato ({analysis.vix_current:.1f}): cautela")
                outlook.risk_level = "MEDIO"
            else:
                score += 0.1
                signals.append(f"🟢 VIX basso ({analysis.vix_current:.1f}): mercato calmo")
                outlook.risk_level = "BASSO"

            # Variazione VIX
            if analysis.vix_change > 10:
                score -= 0.1
                signals.append(f"🔴 VIX in forte aumento ({analysis.vix_change:+.1f}%): paura crescente")
            elif analysis.vix_change < -10:
                score += 0.1
                signals.append(f"🟢 VIX in forte calo ({analysis.vix_change:+.1f}%): paura in diminuzione")

        # ── VERDETTO FINALE ──
        outlook.signals = signals

        if score > 0.3:
            outlook.direction = "RIALZISTA"
            outlook.confidence = min(0.9, 0.5 + score)
        elif score > 0.1:
            outlook.direction = "LEGGERMENTE RIALZISTA"
            outlook.confidence = 0.4 + score
        elif score < -0.3:
            outlook.direction = "RIBASSISTA"
            outlook.confidence = min(0.9, 0.5 + abs(score))
        elif score < -0.1:
            outlook.direction = "LEGGERMENTE RIBASSISTA"
            outlook.confidence = 0.4 + abs(score)
        else:
            outlook.direction = "NEUTRO"
            outlook.confidence = 0.3

        # Summary
        nq_str = f"NQ {analysis.nasdaq_futures.change_pct:+.2f}%" if analysis.nasdaq_futures else "N/A"
        es_str = f"ES {analysis.sp500_futures.change_pct:+.2f}%" if analysis.sp500_futures else "N/A"
        outlook.summary = (
            f"Direzione attesa: {outlook.direction} (confidenza {outlook.confidence:.0%}). "
            f"Futures: {nq_str}, {es_str}. "
            f"VIX: {analysis.vix_current:.1f} ({analysis.vix_change:+.1f}%). "
            f"Gap atteso: {outlook.gap_expected} ({outlook.gap_pct:+.2f}%). "
            f"Rischio: {outlook.risk_level}."
        )

        return outlook
