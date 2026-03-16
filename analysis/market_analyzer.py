"""
Market Analyzer - Analisi combinata multi-asset.

Combina:
1. Analisi tecnica su NASDAQ 100 (^NDX), SWDA.MI e CSNDX (iShares NASDAQ 100)
2. Valutazione CAPE per-asset per timing di mercato
3. Analisi della liquidità per condizioni macro
4. Livelli di ingresso ottimali con probabilità di raggiungerli
5. Rendimento atteso basato su valutazione + liquidità

Produce una raccomandazione finale per ogni asset con livelli di ingresso.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from data.stock_fetcher import StockFetcher, TICKERS
from analysis.valuation import CapeAnalyzer, CapeAnalysis
from analysis.liquidity import LiquidityAnalyzer, LiquidityAnalysis
from analysis.regime_detector import RegimeDetector, RegimeResult, MultiAssetRegimeReport
from utils.logger import get_logger

logger = get_logger("analysis.market_analyzer")


@dataclass
class EntryLevel:
    """Un singolo livello di ingresso con probabilità."""
    level: str           # Nome del livello (es. "SMA 200", "Fib 38.2%")
    price: float         # Prezzo del livello
    distance_pct: float  # Distanza % dal prezzo corrente
    prob_30d: float      # Probabilità di raggiungerlo in 30 giorni
    prob_90d: float      # Probabilità di raggiungerlo in 90 giorni
    level_type: str      # Tipo (supporto_dinamico, fibonacci, etc.)
    description: str


@dataclass
class AssetAnalysis:
    """Analisi completa di un singolo asset."""
    ticker: str
    name: str
    current_price: float = 0.0
    price_change_1d: float = 0.0
    price_change_1m: float = 0.0
    price_change_3m: float = 0.0
    price_change_1y: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    rsi: float = 50.0
    trend: str = "NEUTRO"
    above_sma200: bool = False
    golden_cross: bool = False
    technical_score: float = 0.0
    technical_signals: list[str] = field(default_factory=list)
    # CAPE per-asset
    cape_analysis: Optional[CapeAnalysis] = None
    # Livelli di ingresso
    entry_levels: list[EntryLevel] = field(default_factory=list)
    # Drawdown da massimi
    ath_price: float = 0.0              # Massimo storico (o massimo periodo)
    ath_date: str = ""                  # Data del massimo
    drawdown_from_ath: float = 0.0      # % discesa dal massimo
    high_52w: float = 0.0               # Massimo 52 settimane
    drawdown_from_52w: float = 0.0      # % discesa dal max 52 settimane


@dataclass
class MarketReport:
    """Report completo dell'analisi di mercato."""
    timestamp: str = ""
    # Asset individuali
    nasdaq100: Optional[AssetAnalysis] = None
    swda: Optional[AssetAnalysis] = None
    csndx: Optional[AssetAnalysis] = None
    # Analisi macro (CAPE S&P 500 come riferimento base)
    cape_sp500: Optional[CapeAnalysis] = None
    liquidity_analysis: Optional[LiquidityAnalysis] = None
    # Regime di mercato (HMM)
    regime_report: Optional[MultiAssetRegimeReport] = None
    # Raccomandazioni finali
    nasdaq100_recommendation: str = ""
    nasdaq100_score: float = 0.0
    nasdaq100_entry_type: str = ""
    swda_recommendation: str = ""
    swda_score: float = 0.0
    swda_entry_type: str = ""
    csndx_recommendation: str = ""
    csndx_score: float = 0.0
    csndx_entry_type: str = ""
    # Asset allocation consigliata
    suggested_allocation: dict = field(default_factory=dict)
    # Sommario
    summary: str = ""


class MarketAnalyzer:
    """
    Analizzatore di mercato multi-asset.
    Combina analisi tecnica, CAPE ratio per-asset e liquidità per generare
    raccomandazioni di investimento su NASDAQ 100 (^NDX), SWDA.MI e CSNDX.
    """

    def __init__(self):
        self.stock_fetcher = StockFetcher()
        self.cape_analyzer = CapeAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.regime_detector = RegimeDetector()

    def full_analysis(self, period: str = "1y") -> MarketReport:
        """
        Esegue l'analisi completa di mercato.

        Args:
            period: Periodo di storico per l'analisi tecnica

        Returns:
            MarketReport con tutte le analisi e raccomandazioni
        """
        from datetime import datetime
        report = MarketReport(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))

        logger.info("=" * 60)
        logger.info("INIZIO ANALISI MERCATO MULTI-ASSET")
        logger.info("=" * 60)

        # 0. Recupera CAPE S&P 500 come base per le stime per-asset
        logger.info("Recupero CAPE S&P 500 come riferimento base...")
        sp500_cape = self.cape_analyzer.fetch_current_cape()
        report.cape_sp500 = self.cape_analyzer.analyze(cape_value=sp500_cape, region="US")

        # 1. Analisi NASDAQ 100 (^NDX)
        logger.info("Analisi NASDAQ 100 (^NDX)...")
        report.nasdaq100 = self._analyze_asset(
            TICKERS["NASDAQ100"], "NASDAQ 100 (^NDX)", period
        )
        report.nasdaq100.cape_analysis = self.cape_analyzer.analyze_asset("NDX", sp500_cape)

        # 2. Analisi SWDA.MI
        logger.info("Analisi SWDA (iShares MSCI World - Milano)...")
        report.swda = self._analyze_asset(
            TICKERS["SWDA"], "SWDA.MI (iShares MSCI World)", period
        )
        # Fallback se SWDA.MI non funziona
        if report.swda and report.swda.current_price == 0:
            logger.info("Tentativo con SWDA London (SWDA.L)...")
            report.swda = self._analyze_asset(
                TICKERS["SWDA_ALT"], "SWDA (iShares MSCI World)", period
            )
        if report.swda and report.swda.current_price == 0:
            logger.info("Tentativo con IWDA Amsterdam (IWDA.AS)...")
            report.swda = self._analyze_asset(
                TICKERS["SWDA_ALT2"], "SWDA (iShares MSCI World)", period
            )
        report.swda.cape_analysis = self.cape_analyzer.analyze_asset("SWDA", sp500_cape)

        # 3. Analisi CSNDX (iShares NASDAQ 100 UCITS ETF - Milano)
        logger.info("Analisi CSNDX (iShares NASDAQ 100 UCITS ETF - Milano)...")
        report.csndx = self._analyze_asset(
            TICKERS["CSNDX"], "CSNDX (iShares NASDAQ 100 ETF)", period
        )
        if report.csndx and report.csndx.current_price == 0:
            logger.info("Tentativo con CSNDX.MI...")
            report.csndx = self._analyze_asset(
                TICKERS["CSNDX_ALT"], "CSNDX (iShares NASDAQ 100 ETF)", period
            )
        if report.csndx and report.csndx.current_price == 0:
            logger.info("Tentativo con SXRV.DE (Xetra)...")
            report.csndx = self._analyze_asset(
                TICKERS["CSNDX_ALT2"], "CSNDX (iShares NASDAQ 100 ETF)", period
            )
        report.csndx.cape_analysis = self.cape_analyzer.analyze_asset("CSNDX", sp500_cape)

        # 4. Analisi liquidità
        logger.info("Analisi liquidità di mercato...")
        report.liquidity_analysis = self.liquidity_analyzer.analyze(period="6mo")

        # 5. Regime Detection (HMM)
        logger.info("Analisi regime di mercato (HMM)...")
        try:
            report.regime_report = self.regime_detector.full_analysis()
        except Exception as e:
            logger.warning(f"Regime detection non disponibile: {e}")
            report.regime_report = None

        # 6. Genera raccomandazioni finali (con regime)
        self._generate_recommendations(report)

        logger.info("=" * 60)
        logger.info("ANALISI COMPLETATA")
        logger.info("=" * 60)

        return report

    def _analyze_asset(self, ticker: str, name: str, period: str) -> AssetAnalysis:
        """Analizza un singolo asset con indicatori tecnici e livelli di ingresso."""
        analysis = AssetAnalysis(ticker=ticker, name=name)

        try:
            df = self.stock_fetcher.fetch_ohlcv(ticker, period=period, interval="1d")
            if df.empty:
                logger.warning(f"Nessun dato per {ticker}")
                return analysis

            # Prezzo corrente e variazioni
            analysis.current_price = df["close"].iloc[-1]

            if len(df) >= 2:
                analysis.price_change_1d = (
                    (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
                )
            if len(df) >= 22:
                analysis.price_change_1m = (
                    (df["close"].iloc[-1] / df["close"].iloc[-22] - 1) * 100
                )
            if len(df) >= 66:
                analysis.price_change_3m = (
                    (df["close"].iloc[-1] / df["close"].iloc[-66] - 1) * 100
                )
            if len(df) >= 252:
                analysis.price_change_1y = (
                    (df["close"].iloc[-1] / df["close"].iloc[-252] - 1) * 100
                )

            # Medie mobili
            if len(df) >= 50:
                analysis.sma_50 = df["close"].tail(50).mean()
            if len(df) >= 200:
                analysis.sma_200 = df["close"].tail(200).mean()

            analysis.above_sma200 = analysis.current_price > analysis.sma_200 if analysis.sma_200 > 0 else True
            analysis.golden_cross = analysis.sma_50 > analysis.sma_200 if (analysis.sma_50 > 0 and analysis.sma_200 > 0) else False

            # RSI
            if len(df) >= 15:
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                analysis.rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            # Trend
            if analysis.sma_50 > 0 and analysis.sma_200 > 0:
                if analysis.current_price > analysis.sma_50 > analysis.sma_200:
                    analysis.trend = "FORTE RIALZISTA"
                elif analysis.current_price > analysis.sma_200:
                    analysis.trend = "RIALZISTA"
                elif analysis.current_price < analysis.sma_50 < analysis.sma_200:
                    analysis.trend = "FORTE RIBASSISTA"
                elif analysis.current_price < analysis.sma_200:
                    analysis.trend = "RIBASSISTA"
                else:
                    analysis.trend = "LATERALE"

            # Score tecnico complessivo (-1 → +1)
            tech_score = 0.0
            signals = []

            if analysis.golden_cross:
                tech_score += 0.25
                signals.append("Golden Cross (SMA50 > SMA200)")
            else:
                tech_score -= 0.25
                signals.append("Death Cross (SMA50 < SMA200)")

            if analysis.above_sma200:
                tech_score += 0.20
                signals.append("Prezzo sopra SMA200 (trend di lungo positivo)")
            else:
                tech_score -= 0.20
                signals.append("Prezzo sotto SMA200 (trend di lungo negativo)")

            if analysis.rsi < 30:
                tech_score += 0.20
                signals.append(f"RSI ipervenduto ({analysis.rsi:.1f})")
            elif analysis.rsi < 45:
                tech_score += 0.10
                signals.append(f"RSI in zona favorevole ({analysis.rsi:.1f})")
            elif analysis.rsi > 70:
                tech_score -= 0.20
                signals.append(f"RSI ipercomprato ({analysis.rsi:.1f})")
            elif analysis.rsi > 60:
                tech_score -= 0.05
                signals.append(f"RSI alto ({analysis.rsi:.1f})")

            if analysis.price_change_1m > 5:
                tech_score += 0.15
                signals.append(f"Forte momentum 1M ({analysis.price_change_1m:+.1f}%)")
            elif analysis.price_change_1m > 0:
                tech_score += 0.05
                signals.append(f"Momentum 1M positivo ({analysis.price_change_1m:+.1f}%)")
            elif analysis.price_change_1m < -5:
                tech_score -= 0.15
                signals.append(f"Momentum 1M negativo ({analysis.price_change_1m:+.1f}%)")

            if analysis.price_change_3m > 10:
                tech_score += 0.10
                signals.append(f"Forte trend 3M ({analysis.price_change_3m:+.1f}%)")
            elif analysis.price_change_3m < -10:
                tech_score -= 0.10
                signals.append(f"Trend 3M negativo ({analysis.price_change_3m:+.1f}%)")

            analysis.technical_score = max(-1, min(1, tech_score))
            analysis.technical_signals = signals

            # ============================================
            # DRAWDOWN DA MASSIMI
            # ============================================
            # Massimo del periodo disponibile (ATH proxy)
            ath_price = df["high"].max()
            ath_idx = df["high"].idxmax()
            analysis.ath_price = ath_price
            analysis.ath_date = str(ath_idx.date()) if hasattr(ath_idx, 'date') else str(ath_idx)
            analysis.drawdown_from_ath = ((analysis.current_price / ath_price) - 1) * 100 if ath_price > 0 else 0

            # Massimo 52 settimane (252 giorni)
            if len(df) >= 252:
                high_52w = df["high"].tail(252).max()
            else:
                high_52w = ath_price
            analysis.high_52w = high_52w
            analysis.drawdown_from_52w = ((analysis.current_price / high_52w) - 1) * 100 if high_52w > 0 else 0

            # ============================================
            # LIVELLI DI INGRESSO CON PROBABILITÀ
            # ============================================
            raw_levels = self.stock_fetcher.calc_entry_levels(df, ticker)
            analysis.entry_levels = [
                EntryLevel(
                    level=lvl["level"],
                    price=lvl["price"],
                    distance_pct=lvl["distance_pct"],
                    prob_30d=lvl["prob_30d"],
                    prob_90d=lvl["prob_90d"],
                    level_type=lvl["type"],
                    description=lvl["description"],
                )
                for lvl in raw_levels
            ]

        except Exception as e:
            logger.error(f"Errore nell'analisi di {ticker}: {e}")

        return analysis

    def _generate_recommendations(self, report: MarketReport):
        """Genera raccomandazioni finali combinando tutti i fattori (incluso regime HMM)."""

        liquidity_score = report.liquidity_analysis.overall_score if report.liquidity_analysis else 0

        # Regime bonus per asset (da HMM)
        regime_bonus_ndx = 0.0
        regime_bonus_swda = 0.0
        regime_bonus_csndx = 0.0

        if report.regime_report:
            rr = report.regime_report
            # NASDAQ 100 usa il suo regime
            ndx_regime = rr.nasdaq100 or rr.sp500
            if ndx_regime:
                regime_bonus_ndx = ndx_regime.accumulation_bonus

            # CSNDX ha il suo regime HMM dedicato (con fallback a NDX)
            csndx_regime = getattr(rr, 'csndx', None) or rr.nasdaq100 or rr.sp500
            if csndx_regime:
                regime_bonus_csndx = csndx_regime.accumulation_bonus

            # SWDA ha il suo regime HMM dedicato (con fallback a MSCI World)
            swda_regime = getattr(rr, 'swda', None) or rr.msci_world or rr.sp500
            if swda_regime:
                regime_bonus_swda = swda_regime.accumulation_bonus

        # NASDAQ 100 (^NDX)
        if report.nasdaq100:
            tech_score = report.nasdaq100.technical_score
            cape_score = report.nasdaq100.cape_analysis.valuation_score if report.nasdaq100.cape_analysis else 0
            composite = tech_score * 0.25 + cape_score * 0.30 + liquidity_score * 0.30 + regime_bonus_ndx * 0.15
            report.nasdaq100_score = composite
            report.nasdaq100_recommendation = self._score_to_recommendation(composite)
            report.nasdaq100_entry_type = self._score_to_entry_type(composite)

        # SWDA.MI
        if report.swda:
            tech_score = report.swda.technical_score
            cape_score = report.swda.cape_analysis.valuation_score if report.swda.cape_analysis else 0
            composite = tech_score * 0.25 + cape_score * 0.20 + liquidity_score * 0.40 + regime_bonus_swda * 0.15
            report.swda_score = composite
            report.swda_recommendation = self._score_to_recommendation(composite)
            report.swda_entry_type = self._score_to_entry_type(composite)

        # CSNDX
        if report.csndx:
            tech_score = report.csndx.technical_score
            cape_score = report.csndx.cape_analysis.valuation_score if report.csndx.cape_analysis else 0
            composite = tech_score * 0.25 + cape_score * 0.30 + liquidity_score * 0.30 + regime_bonus_csndx * 0.15
            report.csndx_score = composite
            report.csndx_recommendation = self._score_to_recommendation(composite)
            report.csndx_entry_type = self._score_to_entry_type(composite)

        # ASSET ALLOCATION
        scores = [s for s in [report.nasdaq100_score, report.swda_score, report.csndx_score] if s != 0]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score > 0.4:
            report.suggested_allocation = {
                "NASDAQ 100 (^NDX)": 20,
                "CSNDX (iShares NASDAQ 100)": 20,
                "SWDA.MI (MSCI World)": 35,
                "Bond/Obbligazioni": 15,
                "Liquidità": 10,
            }
        elif avg_score > 0.1:
            report.suggested_allocation = {
                "NASDAQ 100 (^NDX)": 15,
                "CSNDX (iShares NASDAQ 100)": 15,
                "SWDA.MI (MSCI World)": 30,
                "Bond/Obbligazioni": 25,
                "Liquidità": 15,
            }
        elif avg_score > -0.2:
            report.suggested_allocation = {
                "NASDAQ 100 (^NDX)": 10,
                "CSNDX (iShares NASDAQ 100)": 10,
                "SWDA.MI (MSCI World)": 25,
                "Bond/Obbligazioni": 30,
                "Liquidità": 25,
            }
        else:
            report.suggested_allocation = {
                "NASDAQ 100 (^NDX)": 5,
                "CSNDX (iShares NASDAQ 100)": 5,
                "SWDA.MI (MSCI World)": 15,
                "Bond/Obbligazioni": 35,
                "Liquidità": 40,
            }

        report.summary = self._generate_summary(report)

    def _score_to_recommendation(self, score: float) -> str:
        if score > 0.5:
            return "FORTE ACQUISTO - Condizioni ottimali per l'ingresso"
        elif score > 0.2:
            return "ACQUISTO - Buon momento per accumulare"
        elif score > 0:
            return "ACQUISTO MODERATO - Ingresso graduale consigliato (DCA)"
        elif score > -0.2:
            return "NEUTRO - Attendere un pullback o usare DCA stretto"
        elif score > -0.5:
            return "CAUTELA - Ridurre esposizione, mantenere liquidità"
        else:
            return "EVITARE - Condizioni sfavorevoli, proteggere il capitale"

    def _score_to_entry_type(self, score: float) -> str:
        if score > 0.4:
            return "LUMP SUM - Ingresso con importo pieno"
        elif score > 0.1:
            return "DCA ACCELERATO - Ingresso in 2-3 tranche ravvicinate"
        elif score > -0.1:
            return "DCA STANDARD - Piano di accumulo mensile regolare"
        elif score > -0.4:
            return "DCA DILUITO - Piano di accumulo trimestrale"
        else:
            return "ATTENDERE - Non entrare, accumulare liquidità"

    def _generate_summary(self, report: MarketReport) -> str:
        lines = []

        # Regime di mercato
        if report.regime_report and report.regime_report.dominant_regime:
            rr = report.regime_report
            lines.append(
                f"REGIME: {rr.dominant_regime_emoji} {rr.dominant_regime} "
                f"(concordanza asset: {rr.concordance:.0%}). "
                f"{rr.market_phase_description[:150]}"
            )

        if report.cape_sp500:
            c = report.cape_sp500
            lines.append(
                f"CAPE S&P 500: {c.cape_value:.1f} ({c.valuation_level}). "
                f"Rendimento 10Y: ~{c.expected_10y_return:.1f}%."
            )

        if report.liquidity_analysis:
            l = report.liquidity_analysis
            lines.append(
                f"LIQUIDITÀ: {l.liquidity_level} (score {l.overall_score:+.2f}). "
                f"{l.entry_recommendation}."
            )

        if report.nasdaq100 and report.nasdaq100.current_price > 0:
            n = report.nasdaq100
            cape_txt = ""
            if n.cape_analysis:
                cape_txt = f" CAPE NDX: {n.cape_analysis.cape_value:.1f} ({n.cape_analysis.valuation_level})."
            lines.append(
                f"NDX: {n.current_price:,.2f} ({n.price_change_1m:+.1f}% 1M). "
                f"Trend: {n.trend}. RSI: {n.rsi:.0f}.{cape_txt} "
                f"→ {report.nasdaq100_recommendation}"
            )

        if report.swda and report.swda.current_price > 0:
            s = report.swda
            cape_txt = ""
            if s.cape_analysis:
                cape_txt = f" CAPE World: {s.cape_analysis.cape_value:.1f} ({s.cape_analysis.valuation_level})."
            lines.append(
                f"SWDA.MI: €{s.current_price:,.2f} ({s.price_change_1m:+.1f}% 1M). "
                f"Trend: {s.trend}. RSI: {s.rsi:.0f}.{cape_txt} "
                f"→ {report.swda_recommendation}"
            )

        if report.csndx and report.csndx.current_price > 0:
            cs = report.csndx
            cape_txt = ""
            if cs.cape_analysis:
                cape_txt = f" CAPE NDX: {cs.cape_analysis.cape_value:.1f}."
            lines.append(
                f"CSNDX: €{cs.current_price:,.2f} ({cs.price_change_1m:+.1f}% 1M). "
                f"Trend: {cs.trend}. RSI: {cs.rsi:.0f}.{cape_txt} "
                f"→ {report.csndx_recommendation}"
            )

        return "\n".join(lines)
