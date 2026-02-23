"""
Modulo per il recupero dei dati di mercato tramite ccxt.
Supporta dati storici e in tempo reale da qualsiasi exchange supportato.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from config.settings import config
from utils.logger import get_logger

logger = get_logger("data.fetcher")


class DataFetcher:
    """Classe per il recupero dei dati di mercato."""

    def __init__(self):
        self.exchange = self._init_exchange()
        self.public_exchange = self._init_public_exchange()
        self._cache: dict[str, pd.DataFrame] = {}

    def _init_exchange(self) -> ccxt.Exchange:
        """Inizializza la connessione con l'exchange (con API key per ordini)."""
        exchange_class = getattr(ccxt, config.exchange.name)
        exchange = exchange_class({
            "apiKey": config.exchange.api_key,
            "secret": config.exchange.api_secret,
            "sandbox": config.exchange.sandbox,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if config.exchange.sandbox:
            exchange.set_sandbox_mode(True)
            logger.info(f"Exchange {config.exchange.name} inizializzato in modalità SANDBOX")
        else:
            logger.info(f"Exchange {config.exchange.name} inizializzato in modalità LIVE")

        return exchange

    def _init_public_exchange(self) -> ccxt.Exchange:
        """Inizializza connessione pubblica (senza sandbox) per dati OHLCV storici."""
        exchange_class = getattr(ccxt, config.exchange.name)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        logger.info(f"Exchange pubblico {config.exchange.name} inizializzato per dati storici")
        return exchange

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = None,
        limit: int = 500,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Recupera dati OHLCV (candlestick) dall'exchange.

        Args:
            symbol: Coppia di trading (es. 'BTC/USDT')
            timeframe: Timeframe (es. '1h', '4h', '1d')
            limit: Numero massimo di candele
            since: Timestamp di inizio in millisecondi

        Returns:
            DataFrame con colonne: timestamp, open, high, low, close, volume
        """
        symbol = symbol or config.trading.symbol
        timeframe = timeframe or config.trading.timeframe

        try:
            # Usa exchange pubblico per OHLCV (evita limiti sandbox)
            exchange = self.public_exchange if hasattr(self, 'public_exchange') else self.exchange
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since,
            )

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)

            logger.debug(
                f"Recuperate {len(df)} candele per {symbol} [{timeframe}] "
                f"da {df.index[0]} a {df.index[-1]}"
            )

            # Aggiorna la cache
            cache_key = f"{symbol}_{timeframe}"
            self._cache[cache_key] = df

            return df

        except ccxt.BaseError as e:
            logger.error(f"Errore nel recupero OHLCV per {symbol}: {e}")
            raise

    def fetch_historical(
        self,
        symbol: str = None,
        timeframe: str = None,
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Recupera una quantità estesa di dati storici paginando le richieste.

        Args:
            symbol: Coppia di trading
            timeframe: Timeframe
            days: Numero di giorni di storico da recuperare

        Returns:
            DataFrame completo con tutti i dati storici
        """
        symbol = symbol or config.trading.symbol
        timeframe = timeframe or config.trading.timeframe

        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_data: list[pd.DataFrame] = []

        logger.info(f"Recupero storico di {days} giorni per {symbol} [{timeframe}]...")

        while True:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1000,
                    since=since,
                )

                if df.empty:
                    break

                all_data.append(df)

                # Aggiorna il timestamp di partenza
                last_ts = int(df.index[-1].timestamp() * 1000)
                if last_ts == since:
                    break
                since = last_ts + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Errore durante il recupero storico: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="last")]
        result.sort_index(inplace=True)

        logger.info(f"Recuperate {len(result)} candele storiche totali")

        # Salva su disco
        self._save_to_csv(result, symbol, timeframe)

        return result

    def fetch_ticker(self, symbol: str = None) -> dict:
        """Recupera il ticker corrente (prezzo bid/ask, volume, ecc.)."""
        symbol = symbol or config.trading.symbol
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(
                f"Ticker {symbol}: last={ticker['last']}, "
                f"bid={ticker['bid']}, ask={ticker['ask']}"
            )
            return ticker
        except ccxt.BaseError as e:
            logger.error(f"Errore nel recupero ticker per {symbol}: {e}")
            raise

    def fetch_order_book(self, symbol: str = None, limit: int = 20) -> dict:
        """Recupera il book degli ordini."""
        symbol = symbol or config.trading.symbol
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except ccxt.BaseError as e:
            logger.error(f"Errore nel recupero order book per {symbol}: {e}")
            raise

    def fetch_balance(self) -> dict:
        """Recupera il bilancio dell'account."""
        try:
            balance = self.exchange.fetch_balance()
            free = {k: v for k, v in balance["free"].items() if v > 0}
            logger.info(f"Bilancio disponibile: {free}")
            return balance
        except ccxt.BaseError as e:
            logger.error(f"Errore nel recupero bilancio: {e}")
            raise

    def get_cached(self, symbol: str = None, timeframe: str = None) -> Optional[pd.DataFrame]:
        """Ritorna i dati dalla cache se disponibili."""
        symbol = symbol or config.trading.symbol
        timeframe = timeframe or config.trading.timeframe
        cache_key = f"{symbol}_{timeframe}"
        return self._cache.get(cache_key)

    def _save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Salva i dati su file CSV per uso futuro."""
        data_dir = Path(config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath)
        logger.debug(f"Dati salvati in {filepath}")

    def load_from_csv(self, symbol: str = None, timeframe: str = None) -> Optional[pd.DataFrame]:
        """Carica dati precedentemente salvati da CSV."""
        symbol = symbol or config.trading.symbol
        timeframe = timeframe or config.trading.timeframe
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = Path(config.data_dir) / filename

        if filepath.exists():
            df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
            logger.info(f"Caricati {len(df)} record da {filepath}")
            return df

        logger.warning(f"File non trovato: {filepath}")
        return None
