"""
Modello di Machine Learning per predire la direzione del prezzo.
Utilizza XGBoost per classificazione binaria (rialzo/ribasso).
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from config.settings import config
from ml.features import FeatureEngineer
from utils.logger import get_logger

logger = get_logger("ml.model")


class MLPredictor:
    """
    Modello predittivo basato su XGBoost per segnali di trading.
    Predice la probabilità di un movimento rialzista.
    """

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.feature_engineer = FeatureEngineer()
        self.model_path = Path(config.ml.model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.last_train_time: Optional[datetime] = None
        self.feature_importance: Optional[dict] = None
        self.metrics: dict = {}

        # Prova a caricare un modello salvato
        self._load_model()

    def train(self, df: pd.DataFrame) -> dict:
        """
        Addestra il modello sui dati storici.

        Args:
            df: DataFrame con indicatori tecnici già calcolati

        Returns:
            Dizionario con le metriche di valutazione
        """
        logger.info("Inizio training del modello ML...")

        # Crea feature e target
        df = self.feature_engineer.create_features(df)
        df = self.feature_engineer.create_target(df)
        X, y = self.feature_engineer.get_feature_matrix(df)

        if y is None or len(X) < 100:
            logger.warning("Dati insufficienti per il training")
            return {"error": "Dati insufficienti"}

        # Time Series Split (non random per rispettare l'ordine temporale)
        tscv = TimeSeriesSplit(n_splits=5)
        scores_accuracy = []
        scores_f1 = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBClassifier(
                n_estimators=config.ml.n_estimators,
                max_depth=config.ml.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            scores_accuracy.append(accuracy_score(y_test, y_pred))
            scores_f1.append(f1_score(y_test, y_pred, zero_division=0))

        # Addestra il modello finale su tutti i dati
        split_point = int(len(X) * config.ml.train_test_split)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        self.model = XGBClassifier(
            n_estimators=config.ml.n_estimators,
            max_depth=config.ml.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Valutazione finale
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "cv_accuracy_mean": float(np.mean(scores_accuracy)),
            "cv_accuracy_std": float(np.std(scores_accuracy)),
            "cv_f1_mean": float(np.mean(scores_f1)),
            "cv_f1_std": float(np.std(scores_f1)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate": float(y_test.mean()),
        }

        # Feature importance
        feature_names = X.columns.tolist()
        importances = self.model.feature_importances_
        self.feature_importance = dict(
            sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        )

        self.last_train_time = datetime.utcnow()

        logger.info(
            f"Training completato | "
            f"CV Accuracy: {self.metrics['cv_accuracy_mean']:.3f} ± {self.metrics['cv_accuracy_std']:.3f} | "
            f"CV F1: {self.metrics['cv_f1_mean']:.3f} ± {self.metrics['cv_f1_std']:.3f} | "
            f"Test Accuracy: {self.metrics['test_accuracy']:.3f}"
        )

        # Salva il modello
        self._save_model()

        return self.metrics

    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Predice la probabilità di rialzo dato il DataFrame corrente.

        Args:
            df: DataFrame con indicatori tecnici

        Returns:
            Probabilità di rialzo (0.0 - 1.0), None se il modello non è disponibile
        """
        if self.model is None:
            logger.warning("Modello non addestrato, skip predizione ML")
            return None

        try:
            df = self.feature_engineer.create_features(df)
            X, _ = self.feature_engineer.get_feature_matrix(df)

            if X.empty:
                return None

            # Predici sull'ultima riga
            last_features = X.iloc[[-1]]
            probability = self.model.predict_proba(last_features)[0][1]

            logger.debug(f"ML prediction: probabilità rialzo = {probability:.4f}")
            return float(probability)

        except Exception as e:
            logger.error(f"Errore nella predizione ML: {e}")
            return None

    def needs_retrain(self) -> bool:
        """Verifica se è necessario riaddestrare il modello."""
        if self.model is None or self.last_train_time is None:
            return True

        hours_since_train = (datetime.utcnow() - self.last_train_time).total_seconds() / 3600
        return hours_since_train >= config.ml.retrain_hours

    def _save_model(self):
        """Salva il modello su disco."""
        try:
            model_file = self.model_path / "xgb_model.joblib"
            meta_file = self.model_path / "model_meta.joblib"

            joblib.dump(self.model, model_file)
            joblib.dump({
                "last_train_time": self.last_train_time,
                "metrics": self.metrics,
                "feature_importance": self.feature_importance,
            }, meta_file)

            logger.info(f"Modello salvato in {model_file}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio del modello: {e}")

    def _load_model(self):
        """Carica il modello da disco se disponibile."""
        try:
            model_file = self.model_path / "xgb_model.joblib"
            meta_file = self.model_path / "model_meta.joblib"

            if model_file.exists():
                self.model = joblib.load(model_file)

                if meta_file.exists():
                    meta = joblib.load(meta_file)
                    self.last_train_time = meta.get("last_train_time")
                    self.metrics = meta.get("metrics", {})
                    self.feature_importance = meta.get("feature_importance")

                logger.info(f"Modello caricato da {model_file}")
            else:
                logger.info("Nessun modello salvato trovato")

        except Exception as e:
            logger.warning(f"Errore nel caricamento del modello: {e}")
            self.model = None

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Ritorna le top N feature più importanti."""
        if self.feature_importance is None:
            return []
        return list(self.feature_importance.items())[:n]
