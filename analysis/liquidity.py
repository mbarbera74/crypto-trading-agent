"""
Modulo di analisi della liquidità di mercato.

La liquidità di mercato è un indicatore fondamentale per determinare il timing
di ingresso su asset rischiosi (azioni, ETF). Quando la liquidità è abbondante,
il denaro fluisce verso gli asset rischiosi e i prezzi tendono a salire.

Indicatori di liquidità analizzati:
1. VIX (Fear Index) → Volatilità implicita, misura la paura del mercato
2. Yield Curve (10Y-2Y) → Pendenza della curva dei rendimenti
3. Credit Spread (HYG vs LQD) → Appetito per il rischio nel mercato obbligazionario
4. Dollar Index (DXY) → Forza del dollaro (dollaro forte = meno liquidità globale)
5. Volume Analysis → Volumi di scambio su QQQ/SWDA
6. Gold vs SPY → Risk-on vs Risk-off
7. TLT momentum → Flussi verso i bond (flight to quality)
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from data.stock_fetcher import StockFetcher, TICKERS
from utils.logger import get_logger

logger = get_logger("analysis.liquidity")


@dataclass
class LiquidityIndicator:
    """Singolo indicatore di liquidità."""
    name: str
    value: float
    signal: str  # "POSITIVO", "NEUTRO", "NEGATIVO"
    score: float  # -1.0 (illiquido/rischio) → +1.0 (liquido/favorevole)
    weight: float  # Peso nel punteggio totale
    description: str


@dataclass
class LiquidityAnalysis:
    """Risultato completo dell'analisi di liquidità."""
    indicators: list[LiquidityIndicator] = field(default_factory=list)
    overall_score: float = 0.0  # -1.0 → +1.0
    liquidity_level: str = ""  # "ALTA", "MEDIA", "BASSA", "CRITICA"
    entry_recommendation: str = ""
    entry_color: str = "yellow"
    description: str = ""
    risk_factors: list[str] = field(default_factory=list)
    positive_factors: list[str] = field(default_factory=list)


class LiquidityAnalyzer:
    """
    Analizzatore della liquidità di mercato.
    Combina più indicatori per determinare il timing di ingresso.
    """

    def __init__(self):
        self.fetcher = StockFetcher()

    def analyze(self, period: str = "6mo") -> LiquidityAnalysis:
        """
        Esegue l'analisi completa della liquidità di mercato.

        Args:
            period: Periodo di analisi per lo storico

        Returns:
            LiquidityAnalysis con tutti gli indicatori e la raccomandazione
        """
        result = LiquidityAnalysis()
        indicators = []

        # 1. VIX Analysis
        vix_indicator = self._analyze_vix(period)
        if vix_indicator:
            indicators.append(vix_indicator)

        # 2. Yield Curve (10Y - 3M proxy)
        yield_indicator = self._analyze_yield_curve(period)
        if yield_indicator:
            indicators.append(yield_indicator)

        # 3. Credit Spread (HYG vs LQD)
        credit_indicator = self._analyze_credit_spread(period)
        if credit_indicator:
            indicators.append(credit_indicator)

        # 4. Dollar Strength (DXY)
        dxy_indicator = self._analyze_dollar(period)
        if dxy_indicator:
            indicators.append(dxy_indicator)

        # 5. Volume Analysis (QQQ)
        vol_indicator = self._analyze_volume(period)
        if vol_indicator:
            indicators.append(vol_indicator)

        # 6. Risk Appetite (Gold vs SPY)
        risk_indicator = self._analyze_risk_appetite(period)
        if risk_indicator:
            indicators.append(risk_indicator)

        # 7. Treasury Momentum (TLT)
        tlt_indicator = self._analyze_treasury_momentum(period)
        if tlt_indicator:
            indicators.append(tlt_indicator)

        result.indicators = indicators

        # Calcola punteggio complessivo
        if indicators:
            total_weight = sum(ind.weight for ind in indicators)
            if total_weight > 0:
                result.overall_score = sum(
                    ind.score * ind.weight for ind in indicators
                ) / total_weight

        # Categorizza
        score = result.overall_score
        if score > 0.5:
            result.liquidity_level = "ALTA"
            result.entry_recommendation = "FAVOREVOLE ALL'INGRESSO"
            result.entry_color = "green"
            result.description = (
                "Condizioni di liquidità molto favorevoli. Il mercato è in modalità risk-on "
                "con flussi positivi verso gli asset rischiosi. Buon momento per entrare."
            )
        elif score > 0.2:
            result.liquidity_level = "BUONA"
            result.entry_recommendation = "INGRESSO CON CAUTELA"
            result.entry_color = "lime"
            result.description = (
                "Liquidità adeguata. Condizioni generalmente favorevoli per l'ingresso, "
                "ma alcuni indicatori suggeriscono prudenza. Consigliato ingresso graduale (DCA)."
            )
        elif score > -0.2:
            result.liquidity_level = "MEDIA"
            result.entry_recommendation = "NEUTRO - DCA CONSIGLIATO"
            result.entry_color = "yellow"
            result.description = (
                "Condizioni di liquidità miste. Non ci sono segnali chiari. "
                "Utilizzare un piano di accumulo (DCA) per mediare il rischio."
            )
        elif score > -0.5:
            result.liquidity_level = "BASSA"
            result.entry_recommendation = "ATTENDERE"
            result.entry_color = "orange"
            result.description = (
                "Liquidità in contrazione. Il mercato potrebbe essere sotto stress. "
                "Meglio aspettare condizioni migliori o ridurre l'esposizione."
            )
        else:
            result.liquidity_level = "CRITICA"
            result.entry_recommendation = "NON ENTRARE"
            result.entry_color = "red"
            result.description = (
                "Condizioni di liquidità critiche. Forte stress di mercato in corso. "
                "Flight to quality in atto. Mantenere liquidità e posizioni difensive."
            )

        # Raccogli fattori positivi e negativi
        for ind in indicators:
            if ind.score > 0.2:
                result.positive_factors.append(f"{ind.name}: {ind.description}")
            elif ind.score < -0.2:
                result.risk_factors.append(f"{ind.name}: {ind.description}")

        logger.info(
            f"Analisi liquidità completata | Score: {result.overall_score:.2f} | "
            f"Livello: {result.liquidity_level} | {result.entry_recommendation}"
        )

        return result

    def _analyze_vix(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza il VIX (indice della paura)."""
        try:
            df = self.fetcher.fetch_ohlcv(TICKERS["VIX"], period=period)
            if df.empty:
                return None

            current_vix = df["close"].iloc[-1]
            avg_vix = df["close"].mean()
            vix_20d = df["close"].tail(20).mean()

            # VIX < 15: Mercato molto calmo (risk-on)
            # VIX 15-20: Normale
            # VIX 20-25: Nervosismo
            # VIX 25-30: Paura
            # VIX > 30: Panico
            if current_vix < 15:
                score = 0.8
                signal = "POSITIVO"
                desc = f"VIX molto basso ({current_vix:.1f}): mercato calmo, risk-on"
            elif current_vix < 20:
                score = 0.4
                signal = "POSITIVO"
                desc = f"VIX nella norma ({current_vix:.1f}): condizioni stabili"
            elif current_vix < 25:
                score = 0.0
                signal = "NEUTRO"
                desc = f"VIX elevato ({current_vix:.1f}): nervosismo crescente"
            elif current_vix < 30:
                score = -0.5
                signal = "NEGATIVO"
                desc = f"VIX alto ({current_vix:.1f}): paura nel mercato"
            else:
                score = -0.8
                signal = "NEGATIVO"
                desc = f"VIX in panico ({current_vix:.1f}): forte avversione al rischio"

            # Bonus se VIX sta scendendo (trend favorevole)
            if current_vix < vix_20d:
                score += 0.1
                desc += " (in calo)"

            return LiquidityIndicator(
                name="VIX (Fear Index)",
                value=current_vix,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.20,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi VIX: {e}")
            return None

    def _analyze_yield_curve(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza la curva dei rendimenti (10Y - 3M Treasury)."""
        try:
            tnx = self.fetcher.fetch_ohlcv(TICKERS["TNX"], period=period)
            irx = self.fetcher.fetch_ohlcv(TICKERS["IRX"], period=period)

            if tnx.empty or irx.empty:
                return None

            # Allinea le date
            tnx_last = tnx["close"].iloc[-1]
            irx_last = irx["close"].iloc[-1]

            # Spread 10Y - 3M
            spread = tnx_last - irx_last

            if spread > 1.5:
                score = 0.7
                signal = "POSITIVO"
                desc = f"Curva ripida ({spread:.2f}%): condizioni monetarie espansive"
            elif spread > 0.5:
                score = 0.3
                signal = "POSITIVO"
                desc = f"Curva positiva ({spread:.2f}%): condizioni normali"
            elif spread > 0:
                score = 0.0
                signal = "NEUTRO"
                desc = f"Curva piatta ({spread:.2f}%): attenzione al rallentamento"
            elif spread > -0.5:
                score = -0.4
                signal = "NEGATIVO"
                desc = f"Curva leggermente invertita ({spread:.2f}%): segnale recessione"
            else:
                score = -0.8
                signal = "NEGATIVO"
                desc = f"Curva fortemente invertita ({spread:.2f}%): recessione probabile"

            return LiquidityIndicator(
                name="Yield Curve (10Y-3M)",
                value=spread,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.20,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi yield curve: {e}")
            return None

    def _analyze_credit_spread(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza il credit spread (HYG/LQD ratio)."""
        try:
            hyg = self.fetcher.fetch_ohlcv(TICKERS["HYG"], period=period)
            lqd = self.fetcher.fetch_ohlcv(TICKERS["LQD"], period=period)

            if hyg.empty or lqd.empty:
                return None

            # Ratio HYG/LQD: alto = risk-on, basso = risk-off
            current_ratio = hyg["close"].iloc[-1] / lqd["close"].iloc[-1]
            avg_ratio = (hyg["close"] / lqd["close"].reindex(hyg.index, method="ffill")).mean()

            ratio_change = ((current_ratio - avg_ratio) / avg_ratio) * 100

            if ratio_change > 2:
                score = 0.6
                signal = "POSITIVO"
                desc = f"Credit spread compresso: forte appetito per il rischio"
            elif ratio_change > 0:
                score = 0.3
                signal = "POSITIVO"
                desc = f"Credit spread stabile: condizioni normali"
            elif ratio_change > -2:
                score = -0.2
                signal = "NEUTRO"
                desc = f"Credit spread in leggero allargamento: cautela"
            else:
                score = -0.7
                signal = "NEGATIVO"
                desc = f"Credit spread in forte allargamento: stress creditizio"

            return LiquidityIndicator(
                name="Credit Spread (HYG/LQD)",
                value=current_ratio,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.15,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi credit spread: {e}")
            return None

    def _analyze_dollar(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza la forza del dollaro (DXY) - dollaro forte = meno liquidità globale."""
        try:
            df = self.fetcher.fetch_ohlcv(TICKERS["DXY"], period=period)
            if df.empty:
                return None

            current = df["close"].iloc[-1]
            sma_50 = df["close"].tail(50).mean()
            sma_200 = df["close"].tail(200).mean() if len(df) >= 200 else df["close"].mean()

            # Dollaro forte = negativo per liquidità globale
            if current < sma_50 and current < sma_200:
                score = 0.6
                signal = "POSITIVO"
                desc = f"Dollaro debole (DXY {current:.1f}): liquidità globale in espansione"
            elif current < sma_50:
                score = 0.3
                signal = "POSITIVO"
                desc = f"Dollaro in calo (DXY {current:.1f}): condizioni favorevoli"
            elif current > sma_50 and current > sma_200:
                score = -0.5
                signal = "NEGATIVO"
                desc = f"Dollaro forte (DXY {current:.1f}): liquidità globale in contrazione"
            else:
                score = 0.0
                signal = "NEUTRO"
                desc = f"Dollaro stabile (DXY {current:.1f}): condizioni miste"

            return LiquidityIndicator(
                name="US Dollar Index (DXY)",
                value=current,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.15,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi DXY: {e}")
            return None

    def _analyze_volume(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza i volumi di scambio su QQQ (ETF NASDAQ 100)."""
        try:
            # Usa QQQ ETF per i volumi (^NDX è un indice senza volume)
            df = self.fetcher.fetch_ohlcv(TICKERS.get("NASDAQ100_ETF", "QQQ"), period=period)
            if df.empty:
                return None

            avg_volume = df["volume"].mean()
            recent_vol = df["volume"].tail(5).mean()
            vol_ratio = recent_vol / avg_volume if avg_volume > 0 else 1.0

            # Volume in aumento con trend positivo = buon segno
            price_trend = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100

            if vol_ratio > 1.3 and price_trend > 0:
                score = 0.6
                signal = "POSITIVO"
                desc = f"Volumi alti ({vol_ratio:.1f}x) con trend positivo: forte partecipazione"
            elif vol_ratio > 1.0:
                score = 0.2
                signal = "POSITIVO"
                desc = f"Volumi sopra la media ({vol_ratio:.1f}x): partecipazione adeguata"
            elif vol_ratio > 0.7:
                score = 0.0
                signal = "NEUTRO"
                desc = f"Volumi nella media ({vol_ratio:.1f}x): partecipazione normale"
            else:
                score = -0.3
                signal = "NEGATIVO"
                desc = f"Volumi bassi ({vol_ratio:.1f}x): scarsa partecipazione"

            return LiquidityIndicator(
                name="Volume QQQ",
                value=vol_ratio,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.10,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi volumi: {e}")
            return None

    def _analyze_risk_appetite(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza risk appetite tramite Gold/SPY ratio."""
        try:
            gold = self.fetcher.fetch_ohlcv(TICKERS["GOLD"], period=period)
            spy = self.fetcher.fetch_ohlcv(TICKERS["SP500"], period=period)

            if gold.empty or spy.empty:
                return None

            # Ratio Gold/SPY: alto = risk-off (oro preferito), basso = risk-on
            # Calcola la variazione recente del ratio
            min_len = min(len(gold), len(spy))
            gold_recent = gold["close"].iloc[-1]
            gold_old = gold["close"].iloc[-min(20, min_len)]
            spy_recent = spy["close"].iloc[-1]
            spy_old = spy["close"].iloc[-min(20, min_len)]

            ratio_now = gold_recent / spy_recent
            ratio_before = gold_old / spy_old
            ratio_change = ((ratio_now - ratio_before) / ratio_before) * 100

            if ratio_change < -3:
                score = 0.6
                signal = "POSITIVO"
                desc = f"SPY supera Gold: forte risk-on, denaro verso azioni"
            elif ratio_change < 0:
                score = 0.2
                signal = "POSITIVO"
                desc = f"SPY leggermente meglio di Gold: lieve risk-on"
            elif ratio_change < 3:
                score = -0.1
                signal = "NEUTRO"
                desc = f"Gold leggermente meglio di SPY: cautela"
            else:
                score = -0.6
                signal = "NEGATIVO"
                desc = f"Gold forte vs SPY: flight to safety in corso"

            return LiquidityIndicator(
                name="Risk Appetite (Gold/SPY)",
                value=ratio_change,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.10,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi risk appetite: {e}")
            return None

    def _analyze_treasury_momentum(self, period: str) -> Optional[LiquidityIndicator]:
        """Analizza il momentum dei Treasury (TLT) - proxy flight to quality."""
        try:
            df = self.fetcher.fetch_ohlcv(TICKERS["TLT"], period=period)
            if df.empty:
                return None

            current = df["close"].iloc[-1]
            sma_20 = df["close"].tail(20).mean()
            change_1m = (current / df["close"].iloc[-min(20, len(df))] - 1) * 100

            # TLT in forte salita = flight to quality = negativo per azioni
            if change_1m > 5:
                score = -0.6
                signal = "NEGATIVO"
                desc = f"TLT in forte rialzo ({change_1m:+.1f}%): flight to quality"
            elif change_1m > 2:
                score = -0.2
                signal = "NEUTRO"
                desc = f"TLT in rialzo ({change_1m:+.1f}%): domanda bond in aumento"
            elif change_1m > -2:
                score = 0.2
                signal = "POSITIVO"
                desc = f"TLT stabile ({change_1m:+.1f}%): condizioni normali"
            else:
                score = 0.5
                signal = "POSITIVO"
                desc = f"TLT in calo ({change_1m:+.1f}%): denaro esce dai bond verso azioni"

            return LiquidityIndicator(
                name="Treasury Momentum (TLT)",
                value=change_1m,
                signal=signal,
                score=max(-1, min(1, score)),
                weight=0.10,
                description=desc,
            )
        except Exception as e:
            logger.error(f"Errore analisi TLT: {e}")
            return None
