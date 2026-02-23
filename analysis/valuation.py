"""
Modulo di analisi della valutazione di mercato basata sul CAPE Ratio (Shiller P/E).

Il CAPE (Cyclically Adjusted Price-to-Earnings) è il rapporto prezzo/utili
medio degli ultimi 10 anni, aggiustato per l'inflazione. Indica se il mercato
è sopravvalutato o sottovalutato rispetto alla media storica.

Livelli di riferimento (S&P 500):
    CAPE < 15  → Mercato molto sottovalutato (forte opportunità di acquisto)
    CAPE 15-20 → Sottovalutato / fair value basso
    CAPE 20-25 → Fair value (valutazione ragionevole)
    CAPE 25-30 → Sopravvalutato (cautela)
    CAPE 30-35 → Molto sopravvalutato (alto rischio)
    CAPE > 35  → Bolla / estrema sopravvalutazione

NASDAQ 100 CAPE è tipicamente 1.3-1.6x rispetto all'S&P 500
(P/E medio NASDAQ 100 ~30-35 vs S&P 500 ~20-25 negli ultimi anni)

Media storica CAPE S&P 500: ~17
"""

from dataclasses import dataclass
from typing import Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("analysis.valuation")


# Medie storiche CAPE per regione/indice
CAPE_HISTORICAL = {
    "US": {"mean": 17.0, "median": 16.0, "current_high": 35.0},
    "NASDAQ100": {"mean": 28.0, "median": 26.0, "current_high": 55.0},
    "WORLD": {"mean": 18.5, "median": 17.0, "current_high": 30.0},
    "EUROPE": {"mean": 15.0, "median": 14.5, "current_high": 25.0},
}

# Fattore di conversione CAPE S&P → CAPE stimato per indice
# NASDAQ P/E è storicamente ~1.4x S&P P/E
CAPE_MULTIPLIERS = {
    "NDX": 1.45,       # NASDAQ 100 (tech-heavy, P/E più alto)
    "CSNDX": 1.45,     # iShares NASDAQ 100 (stessa esposizione)
    "SWDA": 1.05,      # MSCI World (simile a S&P 500, un po' più diversificato)
    "SP500": 1.00,     # Base
}


@dataclass
class CapeAnalysis:
    """Risultato dell'analisi CAPE."""
    cape_value: float
    cape_region: str  # "US", "WORLD", etc.
    historical_mean: float
    percentile: float  # Percentile rispetto alla storia (0-100)
    deviation_from_mean: float  # % di deviazione dalla media
    valuation_level: str  # "Molto Sottovalutato" → "Bolla"
    valuation_score: float  # -1.0 (molto caro) → +1.0 (molto economico)
    expected_10y_return: float  # Rendimento atteso annuo a 10 anni
    entry_signal: str  # "FORTE ACQUISTO", "ACQUISTO", "NEUTRO", "CAUTELA", "EVITA"
    entry_signal_color: str
    description: str


class CapeAnalyzer:
    """
    Analizzatore della valutazione di mercato basata sul CAPE Ratio.
    Scraping dei dati dal sito multpl.com e analisi personalizzata.
    """

    def __init__(self):
        self._cape_cache: dict[str, float] = {}

    def fetch_current_cape(self) -> Optional[float]:
        """
        Recupera il CAPE Ratio corrente dell'S&P 500 da multpl.com.

        Returns:
            Valore CAPE corrente o None se non disponibile
        """
        if "US" in self._cape_cache:
            return self._cape_cache["US"]

        try:
            url = "https://www.multpl.com/shiller-pe"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Cerca il valore CAPE nella pagina
            cape_element = soup.find("div", {"id": "current"})
            if cape_element:
                cape_text = cape_element.get_text(strip=True)
                # Estrai il numero
                import re
                match = re.search(r"([\d.]+)", cape_text)
                if match:
                    cape_value = float(match.group(1))
                    self._cape_cache["US"] = cape_value
                    logger.info(f"CAPE Ratio S&P 500 corrente: {cape_value:.2f}")
                    return cape_value

            logger.warning("Non è stato possibile estrarre il CAPE da multpl.com")
            return None

        except requests.RequestException as e:
            logger.error(f"Errore nel recupero CAPE da multpl.com: {e}")
            return None

    def fetch_cape_history(self) -> Optional[pd.DataFrame]:
        """
        Recupera lo storico del CAPE ratio.

        Returns:
            DataFrame con date e valori CAPE storici
        """
        try:
            url = "https://www.multpl.com/shiller-pe/table/by-month"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "datatable"})

            if table is None:
                logger.warning("Tabella CAPE storica non trovata")
                return None

            rows = table.find_all("tr")[1:]  # Salta header
            data = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    try:
                        date_str = cols[0].get_text(strip=True)
                        value_str = cols[1].get_text(strip=True)
                        date = pd.to_datetime(date_str)
                        value = float(value_str)
                        data.append({"date": date, "cape": value})
                    except (ValueError, TypeError):
                        continue

            if data:
                df = pd.DataFrame(data).set_index("date").sort_index()
                logger.info(f"Recuperati {len(df)} valori CAPE storici")
                return df

            return None

        except requests.RequestException as e:
            logger.error(f"Errore nel recupero storico CAPE: {e}")
            return None

    def estimate_cape_from_pe(self, pe_ratio: float, adjustment: float = 1.15) -> float:
        """
        Stima il CAPE a partire dal P/E trailing.
        Il CAPE è tipicamente più alto del P/E di circa 10-20%.

        Args:
            pe_ratio: P/E trailing
            adjustment: Fattore di aggiustamento (default 1.15)

        Returns:
            CAPE stimato
        """
        return pe_ratio * adjustment

    def estimate_asset_cape(self, asset_key: str, sp500_cape: Optional[float] = None) -> float:
        """
        Stima il CAPE per un asset specifico a partire dal CAPE S&P 500.

        Il NASDAQ 100 ha storicamente un P/E ~1.4-1.5x rispetto all'S&P 500
        a causa della dominanza tech con multipli più elevati.

        Args:
            asset_key: Chiave dell'asset ("NDX", "CSNDX", "SWDA", "SP500")
            sp500_cape: CAPE S&P 500 corrente (se None, lo recupera)

        Returns:
            CAPE stimato per l'asset
        """
        if sp500_cape is None:
            sp500_cape = self.fetch_current_cape()
            if sp500_cape is None:
                sp500_cape = 33.0

        multiplier = CAPE_MULTIPLIERS.get(asset_key, 1.0)
        estimated_cape = sp500_cape * multiplier

        logger.info(f"CAPE stimato per {asset_key}: {estimated_cape:.1f} "
                    f"(S&P 500: {sp500_cape:.1f} × {multiplier})")
        return estimated_cape

    def analyze_asset(self, asset_key: str, sp500_cape: Optional[float] = None) -> CapeAnalysis:
        """
        Analizza il CAPE per un asset specifico.

        Args:
            asset_key: "NDX", "CSNDX", "SWDA", "SP500"
            sp500_cape: CAPE S&P 500 (se None, lo recupera)

        Returns:
            CapeAnalysis con valutazione specifica per l'asset
        """
        if sp500_cape is None:
            sp500_cape = self.fetch_current_cape()
            if sp500_cape is None:
                sp500_cape = 33.0

        # Per asset diversi da SP500, stima il CAPE specifico
        if asset_key in ("NDX", "CSNDX"):
            cape_value = self.estimate_asset_cape(asset_key, sp500_cape)
            region = "NASDAQ100"
        elif asset_key == "SWDA":
            cape_value = self.estimate_asset_cape(asset_key, sp500_cape)
            region = "WORLD"
        else:
            cape_value = sp500_cape
            region = "US"

        return self.analyze(cape_value=cape_value, region=region)

    def analyze(self, cape_value: Optional[float] = None, region: str = "US") -> CapeAnalysis:
        """
        Analizza il CAPE ratio e genera raccomandazione.

        Args:
            cape_value: Valore CAPE (se None, lo recupera automaticamente)
            region: Regione ("US", "NASDAQ100", "WORLD", "EUROPE")

        Returns:
            CapeAnalysis con valutazione completa
        """
        if cape_value is None:
            cape_value = self.fetch_current_cape()

        if cape_value is None:
            cape_value = 33.0
            logger.warning(f"CAPE non disponibile, uso fallback: {cape_value}")

        hist = CAPE_HISTORICAL.get(region, CAPE_HISTORICAL["US"])
        historical_mean = hist["mean"]

        # Calcola deviazione dalla media
        deviation = ((cape_value - historical_mean) / historical_mean) * 100

        # Percentile (usa breakpoint specifici per NASDAQ se applicabile)
        if region == "NASDAQ100":
            percentile = self._estimate_percentile_ndx(cape_value)
        else:
            percentile = self._estimate_percentile(cape_value)

        # Soglie di valutazione aggiustate per regione
        # NASDAQ 100 ha strutturalmente P/E più alti → soglie più alte
        if region == "NASDAQ100":
            thresholds = [(18, 24, 30, 38, 45, 52)]  # Adattate per NASDAQ
            t_low, t_fair_low, t_fair, t_over, t_vover, t_bubble = 18, 24, 30, 38, 45, 52
        elif region == "WORLD":
            t_low, t_fair_low, t_fair, t_over, t_vover, t_bubble = 13, 17, 22, 27, 32, 38
        else:  # US (S&P 500)
            t_low, t_fair_low, t_fair, t_over, t_vover, t_bubble = 12, 16, 20, 25, 30, 35

        if cape_value < t_low:
            level = "Estremamente Sottovalutato"
            score = 1.0
            signal = "FORTE ACQUISTO"
            signal_color = "green"
            desc = "Valutazione ai minimi storici. Opportunità eccezionale di acquisto a lungo termine."
        elif cape_value < t_fair_low:
            level = "Molto Sottovalutato"
            score = 0.8
            signal = "FORTE ACQUISTO"
            signal_color = "green"
            desc = "Valutazione molto attraente. Ottimo momento per incrementare posizioni."
        elif cape_value < t_fair:
            level = "Sottovalutato"
            score = 0.5
            signal = "ACQUISTO"
            signal_color = "lime"
            desc = "Valutazione sotto la media storica. Buon momento per accumulare."
        elif cape_value < t_over:
            level = "Fair Value"
            score = 0.2
            signal = "ACQUISTO MODERATO"
            signal_color = "yellow"
            desc = "Valutazione nella norma. Ingresso ragionevole con piano di accumulo."
        elif cape_value < t_vover:
            level = "Sopravvalutato"
            score = -0.2
            signal = "CAUTELA"
            signal_color = "orange"
            desc = "Mercato sopravvalutato. Ridurre esposizione o usare DCA diluito nel tempo."
        elif cape_value < t_bubble:
            level = "Molto Sopravvalutato"
            score = -0.6
            signal = "FORTE CAUTELA"
            signal_color = "red"
            desc = "Rendimenti futuri attesi bassi. Considerare de-risking e aumento liquidità."
        else:
            level = "Bolla / Estrema Sopravvalutazione"
            score = -1.0
            signal = "RIDURRE ESPOSIZIONE"
            signal_color = "darkred"
            desc = "Valutazione ai massimi storici. Rischio di correzione significativa."

        # Stima rendimento atteso a 10 anni (basato sulla formula di Shiller)
        # Rendimento reale 10Y ≈ 1/CAPE (inverso)
        expected_return = (1 / cape_value) * 100  # % annuo reale
        # Aggiusta per inflazione attesa (~2.5%)
        expected_nominal_return = expected_return + 2.5

        return CapeAnalysis(
            cape_value=cape_value,
            cape_region=region,
            historical_mean=historical_mean,
            percentile=percentile,
            deviation_from_mean=deviation,
            valuation_level=level,
            valuation_score=score,
            expected_10y_return=expected_nominal_return,
            entry_signal=signal,
            entry_signal_color=signal_color,
            description=desc,
        )

    def _estimate_percentile(self, cape: float) -> float:
        """
        Stima il percentile storico del CAPE (S&P 500).
        Basato sulla distribuzione approssimativa dei dati dal 1881.
        """
        breakpoints = [
            (5, 8), (10, 10), (25, 12), (50, 16),
            (75, 22), (90, 28), (95, 33), (99, 44),
        ]

        if cape <= breakpoints[0][1]:
            return 1.0
        if cape >= breakpoints[-1][1]:
            return 99.0

        for i in range(len(breakpoints) - 1):
            p1, v1 = breakpoints[i]
            p2, v2 = breakpoints[i + 1]
            if v1 <= cape <= v2:
                frac = (cape - v1) / (v2 - v1)
                return p1 + frac * (p2 - p1)

        return 50.0

    def _estimate_percentile_ndx(self, cape: float) -> float:
        """
        Stima il percentile storico del CAPE per NASDAQ 100.
        Distribuzione diversa dall'S&P 500: strutturalmente più alto.
        NASDAQ 100 ha dati dal 1985, range CAPE stimato ~15 → ~70.
        """
        breakpoints = [
            (5, 15), (10, 18), (25, 22), (50, 28),
            (75, 38), (90, 48), (95, 55), (99, 65),
        ]

        if cape <= breakpoints[0][1]:
            return 1.0
        if cape >= breakpoints[-1][1]:
            return 99.0

        for i in range(len(breakpoints) - 1):
            p1, v1 = breakpoints[i]
            p2, v2 = breakpoints[i + 1]
            if v1 <= cape <= v2:
                frac = (cape - v1) / (v2 - v1)
                return p1 + frac * (p2 - p1)

        return 50.0

    def get_cape_adjusted_allocation(self, cape_value: float, base_allocation: float = 0.7) -> dict:
        """
        Calcola l'allocazione azionaria aggiustata per il CAPE.

        Args:
            cape_value: Valore CAPE corrente
            base_allocation: Allocazione base in azionario (default 70%)

        Returns:
            Dict con allocazione consigliata
        """
        analysis = self.analyze(cape_value)
        score = analysis.valuation_score

        # Aggiusta allocazione: -30% → +30% rispetto al base
        adjustment = score * 0.30
        equity_alloc = max(0.2, min(1.0, base_allocation + adjustment))
        bond_alloc = max(0.0, 1.0 - equity_alloc - 0.05)
        cash_alloc = max(0.05, 1.0 - equity_alloc - bond_alloc)

        return {
            "azionario": round(equity_alloc * 100, 1),
            "obbligazionario": round(bond_alloc * 100, 1),
            "liquidità": round(cash_alloc * 100, 1),
            "cape_score": score,
            "ragione": analysis.description,
        }
