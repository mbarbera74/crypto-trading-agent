"""
Regime Detector - Hidden Markov Model per rilevamento regimi di mercato.

Identifica 7 regimi di mercato usando un Gaussian HMM:
1. Strong Bull   - Uptrend forte, volatilità bassa → DCA standard
2. Bull Trend    - Uptrend moderato → DCA standard
3. Recovery      - Transizione bear→bull → Accelerare DCA
4. Consolidation - Laterale, bassa volatilità → DCA standard, pazienza
5. High Volatility - Choppy, vol alta → DCA ridotto
6. Correction    - Calo -10/-20% → DCA aggressivo (opportunità)
7. Bear Market   - Calo >20%, vol alta → Lump sum se possibile

Multi-asset: funziona su NDX, S&P 500, MSCI World (SWDA), CSNDX, BTC.
Usa 10-15 anni di dati per robustezza.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from data.stock_fetcher import StockFetcher
from utils.logger import get_logger

logger = get_logger("analysis.regime_detector")

# Directory per salvare modelli HMM
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "saved_models"


# ──────────────────────────────────────
# Definizione dei 7 regimi
# ──────────────────────────────────────

REGIME_LABELS = {
    0: "Strong Bull",
    1: "Bull Trend",
    2: "Recovery",
    3: "Consolidation",
    4: "High Volatility",
    5: "Correction",
    6: "Bear Market",
}

REGIME_COLORS = {
    "Strong Bull": "#00c853",
    "Bull Trend": "#76ff03",
    "Recovery": "#ffeb3b",
    "Consolidation": "#9e9e9e",
    "High Volatility": "#ff9800",
    "Correction": "#ff5722",
    "Bear Market": "#d50000",
}

REGIME_EMOJIS = {
    "Strong Bull": "🟢",
    "Bull Trend": "🟩",
    "Recovery": "🟡",
    "Consolidation": "⚪",
    "High Volatility": "🟠",
    "Correction": "🔴",
    "Bear Market": "⛔",
}

# Impatto del regime sullo score di accumulo
REGIME_ACCUMULATION_BONUS = {
    "Strong Bull": -0.10,       # Non inseguire rialzi estremi
    "Bull Trend": 0.00,         # Neutro
    "Recovery": +0.20,          # Fase molto favorevole per accumulare
    "Consolidation": 0.00,      # Neutro
    "High Volatility": -0.05,   # Riduce leggermente la confidenza
    "Correction": +0.25,        # Opportunità di acquisto
    "Bear Market": +0.30,       # Massima opportunità storica
}

# Strategia di accumulo consigliata per regime
REGIME_STRATEGY = {
    "Strong Bull": "DCA STANDARD - Non inseguire, mantenere piano regolare",
    "Bull Trend": "DCA STANDARD - Continuare accumulo regolare",
    "Recovery": "DCA ACCELERATO - Fase ideale, aumentare importi",
    "Consolidation": "DCA STANDARD - Pazienza, mercato in pausa",
    "High Volatility": "DCA RIDOTTO - Attendere direzione chiara",
    "Correction": "DCA AGGRESSIVO - Correzione = opportunità, aumentare importi",
    "Bear Market": "LUMP SUM / DCA AGGRESSIVO - Massima opportunità storica",
}


@dataclass
class RegimeResult:
    """Risultato del rilevamento regime per un singolo asset."""
    asset_name: str
    ticker: str
    current_regime: str           # Nome del regime corrente
    regime_id: int                # ID numerico (0-6)
    confidence: float             # Confidenza nella classificazione (0-1)
    regime_color: str             # Colore per visualizzazione
    regime_emoji: str
    accumulation_bonus: float     # Bonus/penalità sullo score accumulo
    strategy_suggestion: str      # Suggerimento strategico
    days_in_regime: int = 0       # Da quanti giorni siamo in questo regime
    previous_regime: str = ""     # Regime precedente
    transition_date: str = ""     # Data ultima transizione
    # Drawdown reale dal massimo
    drawdown_pct: float = 0.0     # Drawdown % dal massimo del periodo
    max_price: float = 0.0        # Prezzo massimo del periodo
    max_price_date: str = ""      # Data del massimo
    current_price: float = 0.0    # Prezzo corrente
    # Serie storica dei regimi (per grafici)
    regime_history: Optional[pd.DataFrame] = None
    # Probabilità di transizione prossimo regime
    transition_probs: dict = field(default_factory=dict)


@dataclass
class MultiAssetRegimeReport:
    """Report regime per tutti gli asset monitorati."""
    timestamp: str = ""
    sp500: Optional[RegimeResult] = None
    nasdaq100: Optional[RegimeResult] = None
    msci_world: Optional[RegimeResult] = None
    btc: Optional[RegimeResult] = None
    # Regime dominante (basato su S&P 500 come benchmark)
    dominant_regime: str = ""
    dominant_regime_emoji: str = ""
    market_phase_description: str = ""
    # Cross-regime concordanza
    concordance: float = 0.0     # 0-1: quanto gli asset concordano sul regime


# ──────────────────────────────────────
# Asset configurazione per HMM
# ──────────────────────────────────────

ASSET_CONFIG = {
    "sp500": {
        "ticker": "^GSPC",
        "name": "S&P 500",
        "years": 15,
        "fallbacks": ["SPY"],
    },
    "nasdaq100": {
        "ticker": "^NDX",
        "name": "NASDAQ 100",
        "years": 15,
        "fallbacks": ["QQQ"],
    },
    "msci_world": {
        "ticker": "URTH",
        "name": "MSCI World",
        "years": 12,
        "fallbacks": ["SWDA.MI", "SWDA.L", "IWDA.AS"],
    },
    "btc": {
        "ticker": "BTC-USD",
        "name": "Bitcoin",
        "years": 10,
        "fallbacks": [],
    },
}


class RegimeDetector:
    """
    Rileva il regime di mercato corrente usando un Hidden Markov Model a 7 stati.

    Usa features derivate dai prezzi:
    - Returns giornalieri
    - Volatilità rolling (20gg)
    - Momentum (ROC 20gg)
    - Volume ratio (vs media 50gg)
    """

    N_REGIMES = 7
    N_ITER = 200
    RANDOM_STATE = 42

    def __init__(self):
        self.fetcher = StockFetcher()
        self._models: dict[str, GaussianHMM] = {}
        self._load_models()

    # ────────────────────────────────────────
    # ANALISI COMPLETA MULTI-ASSET
    # ────────────────────────────────────────

    def full_analysis(self) -> MultiAssetRegimeReport:
        """Esegue l'analisi di regime su tutti gli asset configurati."""
        from datetime import datetime
        report = MultiAssetRegimeReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

        logger.info("=" * 60)
        logger.info("REGIME DETECTION - Analisi HMM multi-asset")
        logger.info("=" * 60)

        # Analizza ogni asset
        for key, cfg in ASSET_CONFIG.items():
            logger.info(f"Analisi regime {cfg['name']}...")
            result = self.detect_regime(
                ticker=cfg["ticker"],
                asset_name=cfg["name"],
                years=cfg["years"],
                fallbacks=cfg["fallbacks"],
            )
            setattr(report, key, result)

        # Regime dominante (basato su S&P 500)
        if report.sp500:
            report.dominant_regime = report.sp500.current_regime
            report.dominant_regime_emoji = report.sp500.regime_emoji
        elif report.nasdaq100:
            report.dominant_regime = report.nasdaq100.current_regime
            report.dominant_regime_emoji = report.nasdaq100.regime_emoji

        # Concordanza tra asset
        report.concordance = self._calc_concordance(report)

        # Descrizione fase di mercato
        report.market_phase_description = self._describe_market_phase(report)

        logger.info(f"Regime dominante: {report.dominant_regime}")
        logger.info(f"Concordanza asset: {report.concordance:.0%}")
        logger.info("=" * 60)

        return report

    # ────────────────────────────────────────
    # RILEVAMENTO REGIME SINGOLO ASSET
    # ────────────────────────────────────────

    def detect_regime(
        self,
        ticker: str,
        asset_name: str = "",
        years: int = 15,
        fallbacks: list[str] = None,
    ) -> Optional[RegimeResult]:
        """
        Rileva il regime corrente per un singolo asset.

        Args:
            ticker: Ticker Yahoo Finance
            asset_name: Nome leggibile dell'asset
            years: Anni di storico per il training
            fallbacks: Ticker alternativi se il primo fallisce

        Returns:
            RegimeResult con il regime corrente e la storia
        """
        # Fetch dati
        df = self._fetch_data(ticker, years, fallbacks or [])
        if df is None or len(df) < 252:
            logger.warning(f"Dati insufficienti per {ticker} (serve almeno 1 anno)")
            return None

        # Prepara features
        features, dates = self._prepare_features(df)
        if features is None or len(features) < 100:
            logger.warning(f"Feature insufficienti per {ticker}")
            return None

        # Fit o recupera modello
        model = self._fit_model(ticker, features)
        if model is None:
            return None

        # Predici regimi
        regimes = model.predict(features)

        # Mappa regimi in base alle caratteristiche (returns medi per stato)
        regime_map = self._map_regimes(model, features, regimes)
        mapped_regimes = np.array([regime_map[r] for r in regimes])

        # Regime corrente
        current_regime_id = mapped_regimes[-1]
        current_regime = REGIME_LABELS[current_regime_id]

        # Confidenza: probabilità dello stato corrente
        state_probs = model.predict_proba(features)
        raw_confidence = state_probs[-1][regimes[-1]]

        # Giorni nel regime corrente
        days_in_regime = 1
        for i in range(len(mapped_regimes) - 2, -1, -1):
            if mapped_regimes[i] == current_regime_id:
                days_in_regime += 1
            else:
                break

        # Regime precedente
        prev_regime = ""
        transition_date = ""
        for i in range(len(mapped_regimes) - days_in_regime - 1, -1, -1):
            if mapped_regimes[i] != current_regime_id:
                prev_regime = REGIME_LABELS[mapped_regimes[i]]
                transition_date = str(dates[len(mapped_regimes) - days_in_regime].date())
                break

        # Calcola probabilità di transizione dal regime corrente
        transition_probs = self._calc_transition_probs(model, regime_map, regimes[-1])

        # Costruisci serie storica per grafici
        regime_history = pd.DataFrame({
            "date": dates[len(dates) - len(mapped_regimes):],
            "regime_id": mapped_regimes,
            "regime": [REGIME_LABELS[r] for r in mapped_regimes],
            "close": df["close"].values[len(df) - len(mapped_regimes):],
        })
        regime_history.set_index("date", inplace=True)

        # Calcola drawdown reale dal massimo del periodo
        max_price = df["close"].max()
        max_price_idx = df["close"].idxmax()
        max_price_date = str(max_price_idx.date()) if hasattr(max_price_idx, 'date') else str(max_price_idx)
        current_price = df["close"].iloc[-1]
        drawdown_pct = (current_price / max_price - 1) * 100 if max_price > 0 else 0.0

        result = RegimeResult(
            asset_name=asset_name or ticker,
            ticker=ticker,
            current_regime=current_regime,
            regime_id=current_regime_id,
            confidence=raw_confidence,
            regime_color=REGIME_COLORS[current_regime],
            regime_emoji=REGIME_EMOJIS[current_regime],
            accumulation_bonus=REGIME_ACCUMULATION_BONUS[current_regime],
            strategy_suggestion=REGIME_STRATEGY[current_regime],
            days_in_regime=days_in_regime,
            previous_regime=prev_regime,
            transition_date=transition_date,
            drawdown_pct=drawdown_pct,
            max_price=max_price,
            max_price_date=max_price_date,
            current_price=current_price,
            transition_probs=transition_probs,
            regime_history=regime_history,
        )

        logger.info(
            f"{asset_name}: Regime={current_regime} "
            f"(conf={raw_confidence:.1%}, {days_in_regime}gg, "
            f"prev={prev_regime})"
        )

        return result

    # ────────────────────────────────────────
    # QUERY RAPIDA PER SINGOLO ASSET
    # ────────────────────────────────────────

    def get_regime_for_asset(self, asset_key: str) -> Optional[RegimeResult]:
        """
        Query rapida per ottenere il regime di un asset specifico.

        Args:
            asset_key: Chiave asset ('sp500', 'nasdaq100', 'msci_world', 'btc')

        Returns:
            RegimeResult o None
        """
        if asset_key not in ASSET_CONFIG:
            logger.error(f"Asset key '{asset_key}' non riconosciuto. Usa: {list(ASSET_CONFIG.keys())}")
            return None

        cfg = ASSET_CONFIG[asset_key]
        return self.detect_regime(
            ticker=cfg["ticker"],
            asset_name=cfg["name"],
            years=cfg["years"],
            fallbacks=cfg["fallbacks"],
        )

    # ────────────────────────────────────────
    # INTERNALS
    # ────────────────────────────────────────

    def _fetch_data(self, ticker: str, years: int, fallbacks: list[str]) -> Optional[pd.DataFrame]:
        """Fetch dati storici con fallback ticker."""
        days = years * 365

        df = self.fetcher.fetch_historical(ticker, days=days)
        if df is not None and not df.empty and len(df) > 100:
            return df

        for fb in fallbacks:
            logger.info(f"Fallback: tentativo con {fb}...")
            df = self.fetcher.fetch_historical(fb, days=days)
            if df is not None and not df.empty and len(df) > 100:
                return df

        return None

    def _prepare_features(self, df: pd.DataFrame) -> tuple[Optional[np.ndarray], pd.DatetimeIndex]:
        """
        Prepara le feature per l'HMM dai dati OHLCV.

        Features:
        1. Returns giornalieri (log returns)
        2. Volatilità rolling 20gg (std dei returns)
        3. Momentum / ROC 20gg
        4. Volume ratio vs media 50gg (se disponibile)
        """
        if df is None or df.empty:
            return None, pd.DatetimeIndex([])

        close = df["close"].values
        dates = df.index

        # 1. Log returns
        log_returns = np.diff(np.log(close))

        # 2. Volatilità rolling 20gg
        volatility = pd.Series(log_returns).rolling(20).std().values

        # 3. Momentum (ROC 20gg)
        momentum = np.zeros(len(log_returns))
        for i in range(20, len(log_returns)):
            if close[i - 20] > 0:
                momentum[i] = (close[i + 1] - close[i + 1 - 20]) / close[i + 1 - 20]

        # 4. Volume ratio (se disponibile)
        has_volume = "volume" in df.columns and df["volume"].sum() > 0
        if has_volume:
            volume = df["volume"].values[1:]  # Allinea con returns
            vol_sma50 = pd.Series(volume).rolling(50).mean().values
            vol_ratio = np.where(vol_sma50 > 0, volume / vol_sma50, 1.0)
        else:
            vol_ratio = np.ones(len(log_returns))

        # Trova primo indice valido (dopo warmup di 50 periodi)
        start_idx = 50
        valid_slice = slice(start_idx, None)

        features = np.column_stack([
            log_returns[valid_slice],
            volatility[valid_slice],
            momentum[valid_slice],
            vol_ratio[valid_slice],
        ])

        valid_dates = dates[start_idx + 1:]  # +1 per l'offset dei returns

        # Rimuovi righe con NaN
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        valid_dates = valid_dates[mask]

        logger.debug(f"Feature shape: {features.shape}, date range: {valid_dates[0]} → {valid_dates[-1]}")

        return features, valid_dates

    def _fit_model(self, ticker: str, features: np.ndarray) -> Optional[GaussianHMM]:
        """Fit dell'HMM sui dati. Riusa modello in memoria se disponibile."""
        cache_key = ticker.replace("^", "").replace("=", "").replace("-", "_")

        # Riusa modello già in memoria (stesso processo) per consistenza tra tab
        if cache_key in self._models:
            logger.debug(f"Riuso modello HMM in memoria per {ticker}")
            return self._models[cache_key]

        try:
            model = GaussianHMM(
                n_components=self.N_REGIMES,
                covariance_type="full",
                n_iter=self.N_ITER,
                random_state=self.RANDOM_STATE,
                verbose=False,
                init_params="stmc",
            )
            model.fit(features)
            logger.info(f"HMM fit per {ticker}: score={model.score(features):.2f}")

            # Salva modello
            self._save_model(cache_key, model)
            self._models[cache_key] = model

            return model

        except Exception as e:
            logger.error(f"Errore nel fit HMM per {ticker}: {e}")
            return None

    def _map_regimes(
        self, model: GaussianHMM, features: np.ndarray, raw_regimes: np.ndarray
    ) -> dict[int, int]:
        """
        Mappa gli stati raw dell'HMM ai 7 regimi interpretativi.

        L'HMM assegna numeri arbitrari agli stati. Li riordiniamo in base
        al rendimento medio e alla volatilità media di ogni stato.
        """
        n_states = model.n_components
        state_stats = []

        for s in range(n_states):
            mask = raw_regimes == s
            if mask.sum() == 0:
                state_stats.append({"state": s, "mean_return": 0, "mean_vol": 0, "count": 0})
                continue

            state_features = features[mask]
            state_stats.append({
                "state": s,
                "mean_return": np.mean(state_features[:, 0]),   # returns
                "mean_vol": np.mean(state_features[:, 1]),      # volatilità
                "count": mask.sum(),
            })

        # Ordina per rendimento medio (dal più alto al più basso)
        sorted_stats = sorted(state_stats, key=lambda x: x["mean_return"], reverse=True)

        # Separazione basata su rendimento e volatilità
        # Top rendimento + bassa vol → Strong Bull (0)
        # Top rendimento + alta vol → Recovery (2)
        # Rendimento positivo moderato → Bull Trend (1)
        # Rendimento ~0 + bassa vol → Consolidation (3)
        # Rendimento ~0 + alta vol → High Volatility (4)
        # Rendimento negativo moderato → Correction (5)
        # Rendimento molto negativo → Bear Market (6)

        mapping = {}
        median_vol = np.median([s["mean_vol"] for s in sorted_stats if s["count"] > 0])

        for rank, stats in enumerate(sorted_stats):
            raw_state = stats["state"]
            ret = stats["mean_return"]
            vol = stats["mean_vol"]

            if rank == 0:
                # Miglior rendimento
                if vol <= median_vol:
                    mapping[raw_state] = 0  # Strong Bull
                else:
                    mapping[raw_state] = 2  # Recovery (high return + high vol)
            elif rank == 1:
                if vol <= median_vol:
                    mapping[raw_state] = 1  # Bull Trend
                else:
                    mapping[raw_state] = 0 if 0 not in mapping.values() else 2  # Strong Bull o Recovery
            elif rank in (2, 3):
                # Rendimento medio
                if ret > 0:
                    if vol > median_vol:
                        mapping[raw_state] = 2 if 2 not in mapping.values() else 4  # Recovery o HighVol
                    else:
                        mapping[raw_state] = 1 if 1 not in mapping.values() else 3  # Bull o Consolidation
                else:
                    if vol > median_vol:
                        mapping[raw_state] = 4  # High Volatility
                    else:
                        mapping[raw_state] = 3  # Consolidation
            elif rank == 4:
                mapping[raw_state] = 4 if 4 not in mapping.values() else 3  # HighVol o Consolidation
            elif rank == 5:
                mapping[raw_state] = 5  # Correction
            else:
                mapping[raw_state] = 6  # Bear Market

        # Assicurati che tutti gli stati siano mappati
        used_targets = set(mapping.values())
        available_targets = [i for i in range(7) if i not in used_targets]
        for s in range(n_states):
            if s not in mapping:
                mapping[s] = available_targets.pop(0) if available_targets else 3

        return mapping

    def _calc_transition_probs(
        self, model: GaussianHMM, regime_map: dict[int, int], current_raw_state: int
    ) -> dict[str, float]:
        """Calcola le probabilità di transizione dal regime corrente."""
        trans_matrix = model.transmat_
        probs = {}

        for raw_state, prob in enumerate(trans_matrix[current_raw_state]):
            target_regime = REGIME_LABELS[regime_map[raw_state]]
            probs[target_regime] = probs.get(target_regime, 0) + prob

        # Ordina per probabilità decrescente
        return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    def _calc_concordance(self, report: MultiAssetRegimeReport) -> float:
        """Calcola quanto i vari asset concordano sul tipo di regime."""
        regimes = []
        for key in ["sp500", "nasdaq100", "msci_world", "btc"]:
            result = getattr(report, key, None)
            if result:
                regimes.append(result.regime_id)

        if len(regimes) < 2:
            return 0.0

        # Concordanza = % di coppie che hanno lo stesso regime "categoria"
        # Categorie: bullish (0,1,2), neutro (3,4), bearish (5,6)
        def regime_category(rid):
            if rid in (0, 1, 2):
                return "bullish"
            elif rid in (3, 4):
                return "neutral"
            else:
                return "bearish"

        categories = [regime_category(r) for r in regimes]
        n = len(categories)
        matches = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if categories[i] == categories[j]:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _describe_market_phase(self, report: MultiAssetRegimeReport) -> str:
        """Genera una descrizione testuale della fase di mercato."""
        if not report.dominant_regime:
            return "Dati insufficienti per determinare la fase di mercato."

        regime = report.dominant_regime
        concordance = report.concordance

        if concordance > 0.8:
            agreement = "Tutti gli asset confermano"
        elif concordance > 0.5:
            agreement = "La maggior parte degli asset concorda"
        else:
            agreement = "Gli asset mostrano regimi divergenti"

        descriptions = {
            "Strong Bull": (
                f"MERCATO IN FORTE RIALZO. {agreement} su un trend positivo con bassa volatilità. "
                "Fase matura del ciclo: mantenere DCA regolare senza inseguire i massimi. "
                "Storicamente segue una fase di consolidamento o correzione."
            ),
            "Bull Trend": (
                f"MERCATO IN TREND RIALZISTA. {agreement} su un uptrend moderato. "
                "Fase favorevole per l'accumulo: continuare DCA regolare."
            ),
            "Recovery": (
                f"MERCATO IN FASE DI RECOVERY. {agreement} su una ripresa dopo un periodo negativo. "
                "Fase IDEALE per accumulare: i prezzi salgono dai minimi con volatilità che si riduce. "
                "Consiglio: aumentare gli importi del DCA."
            ),
            "Consolidation": (
                f"MERCATO IN CONSOLIDAMENTO. {agreement} su un periodo laterale. "
                "Fase neutra: mantenere DCA standard e attendere una direzione chiara."
            ),
            "High Volatility": (
                f"MERCATO AD ALTA VOLATILITÀ. {agreement} su turbolenza di mercato. "
                "Fase rischiosa: ridurre importi DCA, mantenere disciplina. "
                "Evitare decisioni emotive, attendere stabilizzazione."
            ),
            "Correction": (
                f"MERCATO IN CORREZIONE (-10/-20%). {agreement} su un calo significativo. "
                "OPPORTUNITÀ: storicamente le correzioni sono i migliori momenti per accumulare. "
                "Consiglio: DCA aggressivo, aumentare importi del 50-100%."
            ),
            "Bear Market": (
                f"MERCATO ORSO (>-20%). {agreement} su un calo prolungato. "
                "MASSIMA OPPORTUNITÀ per un investitore di lungo periodo (5-15 anni). "
                "Storicamente chi ha accumulato in bear market ha ottenuto rendimenti superiori al +100% nei 3-5 anni successivi. "
                "Consiglio: massimizzare gli acquisti."
            ),
        }

        return descriptions.get(regime, f"Regime: {regime}. {agreement}.")

    # ────────────────────────────────────────
    # PERSISTENZA MODELLI
    # ────────────────────────────────────────

    def _save_model(self, key: str, model: GaussianHMM):
        """Salva il modello HMM su disco."""
        import joblib
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"hmm_{key}.joblib"
        joblib.dump(model, path)
        logger.debug(f"Modello HMM salvato: {path}")

    def _load_models(self):
        """Carica modelli HMM salvati."""
        import joblib
        if not MODEL_DIR.exists():
            return
        for path in MODEL_DIR.glob("hmm_*.joblib"):
            key = path.stem.replace("hmm_", "")
            try:
                self._models[key] = joblib.load(path)
                logger.debug(f"Modello HMM caricato: {key}")
            except Exception as e:
                logger.warning(f"Errore caricamento modello {path}: {e}")
