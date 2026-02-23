"""
Feature engineering per il modello ML.
Trasforma i dati OHLCV + indicatori in feature per il modello predittivo.
"""

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("ml.features")


class FeatureEngineer:
    """Crea feature per il modello di Machine Learning."""

    # Feature da utilizzare per il modello
    FEATURE_COLUMNS = [
        # Indicatori di trend
        "price_vs_ema_21", "price_vs_ema_50",
        "macd", "macd_histogram", "macd_signal",
        "adx", "di_plus", "di_minus",
        "supertrend_direction",
        # Momentum
        "rsi", "stoch_k", "stoch_d", "cci",
        # Volatilità
        "bb_pct", "bb_bandwidth", "atr_pct",
        # Volume
        "volume_ratio", "obv",
        # Derivati
        "returns_1", "returns_3", "returns_5", "returns_10",
        "volatility_5", "volatility_10", "volatility_20",
        "high_low_pct", "close_open_pct",
        "momentum_3", "momentum_5", "momentum_10",
        "roc_5", "roc_10",
    ]

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea feature aggiuntive a partire dal DataFrame con indicatori.

        Args:
            df: DataFrame con colonne OHLCV + indicatori tecnici

        Returns:
            DataFrame con feature aggiuntive
        """
        df = df.copy()

        # Returns (rendimenti)
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_3"] = df["close"].pct_change(3)
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)

        # Volatilità rolling
        df["volatility_5"] = df["returns_1"].rolling(5).std()
        df["volatility_10"] = df["returns_1"].rolling(10).std()
        df["volatility_20"] = df["returns_1"].rolling(20).std()

        # Range della candela
        df["high_low_pct"] = (df["high"] - df["low"]) / df["low"] * 100
        df["close_open_pct"] = (df["close"] - df["open"]) / df["open"] * 100

        # Momentum
        df["momentum_3"] = df["close"] - df["close"].shift(3)
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        # Rate of Change
        df["roc_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100
        df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100

        df.dropna(inplace=True)

        logger.debug(f"Create {len(FeatureEngineer.FEATURE_COLUMNS)} feature, {len(df)} righe valide")

        return df

    @staticmethod
    def create_target(df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.01) -> pd.DataFrame:
        """
        Crea la variabile target per il training del modello.

        Target:
            1 = il prezzo salirà di almeno `threshold`% nei prossimi `forward_periods`
            0 = il prezzo scenderà o resterà stabile

        Args:
            df: DataFrame con feature
            forward_periods: Periodi futuri da considerare
            threshold: Soglia di variazione % per la classe positiva

        Returns:
            DataFrame con colonna 'target'
        """
        df = df.copy()

        # Rendimento futuro
        df["future_return"] = df["close"].shift(-forward_periods) / df["close"] - 1

        # Target binario
        df["target"] = (df["future_return"] > threshold).astype(int)

        # Rimuovi le ultime righe senza target
        df.dropna(subset=["target"], inplace=True)

        positive_pct = df["target"].mean() * 100
        logger.info(
            f"Target creato: {len(df)} campioni, {positive_pct:.1f}% positivi "
            f"(forward={forward_periods}, threshold={threshold*100:.1f}%)"
        )

        return df

    @staticmethod
    def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Estrae la matrice delle feature (X) e il target (y).

        Returns:
            Tupla (X, y) dove y è None se non c'è la colonna target
        """
        available_features = [col for col in FeatureEngineer.FEATURE_COLUMNS if col in df.columns]

        if not available_features:
            raise ValueError("Nessuna feature disponibile nel DataFrame")

        X = df[available_features].copy()

        # Gestione valori mancanti
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        y = df["target"] if "target" in df.columns else None

        logger.debug(f"Feature matrix: {X.shape}, Features usate: {len(available_features)}")

        return X, y
