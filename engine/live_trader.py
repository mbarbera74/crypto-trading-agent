"""
Motore di trading live che esegue ordini reali sull'exchange.
Include gestione del rischio, trailing stop e monitoraggio delle posizioni.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import schedule

from config.settings import config
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from strategy.momentum import MomentumStrategy, Signal, TradeSignal
from ml.model import MLPredictor
from utils.logger import get_logger

logger = get_logger("engine.live_trader")


class LiveTrader:
    """
    Trader live che esegue ordini reali tramite l'exchange.
    Gestisce il ciclo completo: analisi → decisione → esecuzione → monitoraggio.
    """

    def __init__(
        self,
        strategy: MomentumStrategy = None,
        notifier=None,
        use_ml: bool = True,
    ):
        self.fetcher = DataFetcher()
        self.strategy = strategy or MomentumStrategy(
            stop_loss_pct=config.trading.stop_loss_pct,
            take_profit_pct=config.trading.take_profit_pct,
        )
        self.notifier = notifier
        self.use_ml = use_ml and config.ml.enabled
        self.ml_predictor = MLPredictor() if self.use_ml else None

        # Stato del trader
        self.is_running = False
        self.current_position: Optional[dict] = None
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time: Optional[datetime] = None
        self.trade_history: list[dict] = []

    def start(self):
        """Avvia il ciclo di trading live."""
        self.is_running = True
        logger.info(
            f"🚀 Trader live avviato | "
            f"Symbol: {config.trading.symbol} | "
            f"Timeframe: {config.trading.timeframe} | "
            f"ML: {'ON' if self.use_ml else 'OFF'} | "
            f"Sandbox: {config.exchange.sandbox}"
        )

        if self.notifier:
            self.notifier.send_sync(
                f"🚀 *Trading Agent Avviato*\n"
                f"Symbol: `{config.trading.symbol}`\n"
                f"Timeframe: `{config.trading.timeframe}`\n"
                f"ML: {'✅' if self.use_ml else '❌'}\n"
                f"Mode: {'🧪 Sandbox' if config.exchange.sandbox else '💰 LIVE'}"
            )

        # Training iniziale ML
        if self.use_ml and self.ml_predictor:
            self._train_ml()

        # Programma il ciclo
        interval_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        interval = interval_map.get(config.trading.timeframe, 60)

        # Esegui subito la prima analisi
        self._trading_cycle()

        # Programma le analisi successive
        schedule.every(interval).minutes.do(self._trading_cycle)

        # Reset giornaliero dei contatori
        schedule.every().day.at("00:00").do(self._daily_reset)

        # Retrain ML periodico
        if self.use_ml:
            schedule.every(config.ml.retrain_hours).hours.do(self._train_ml)

        logger.info(f"Ciclo di trading ogni {interval} minuti")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Ferma il trader."""
        self.is_running = False
        logger.info("Trader live fermato")

        if self.notifier:
            self.notifier.send_sync(
                f"🛑 *Trading Agent Fermato*\n"
                f"Trade oggi: {self.daily_trades}\n"
                f"PnL oggi: ${self.daily_pnl:,.2f}"
            )

    def _trading_cycle(self):
        """Esegue un ciclo completo di analisi e trading."""
        try:
            logger.info("=" * 60)
            logger.info("Inizio ciclo di trading...")

            # 1. Recupera dati
            df = self.fetcher.fetch_ohlcv(limit=300)
            if df.empty:
                logger.warning("Nessun dato disponibile, skip ciclo")
                return

            # 2. Calcola indicatori
            df_indicators = TechnicalIndicators.add_all(df)

            # 3. Predizione ML (opzionale)
            ml_prediction = None
            if self.use_ml and self.ml_predictor:
                ml_prediction = self.ml_predictor.predict(df_indicators)

            # 3.5 Regime HMM (opzionale)
            current_regime = None
            try:
                from analysis.regime_detector import RegimeDetector
                detector = RegimeDetector()
                btc_regime = detector.get_regime_for_asset("btc")
                if btc_regime:
                    current_regime = btc_regime.current_regime
                    logger.info(f"Regime BTC: {btc_regime.regime_emoji} {current_regime} "
                                f"({btc_regime.days_in_regime}gg, bonus={btc_regime.accumulation_bonus:+.2f})")
            except Exception as e:
                logger.debug(f"Regime HMM non disponibile: {e}")

            # 4. Genera segnale
            signal = self.strategy.analyze(df_indicators, ml_prediction=ml_prediction, regime=current_regime)

            # 5. Log del segnale
            current_signals = TechnicalIndicators.get_current_signals(df_indicators)
            logger.info(f"Indicatori: {current_signals}")
            logger.info(
                f"Segnale: {signal.signal.value} | "
                f"Confidenza: {signal.confidence:.2f} | "
                f"Prezzo: {signal.price:.2f}"
            )

            # 6. Esegui il segnale
            self._execute_signal(signal)

            # 7. Controlla posizione aperta
            if self.current_position:
                self._check_position(df_indicators.iloc[-1])

        except Exception as e:
            logger.error(f"Errore nel ciclo di trading: {e}")
            if self.notifier:
                self.notifier.send_sync(f"⚠️ *Errore*: {str(e)}")

    def _execute_signal(self, signal: TradeSignal):
        """Esegue un segnale di trading sull'exchange."""

        # Controlla limiti giornalieri
        if self.daily_trades >= config.trading.max_daily_trades:
            logger.warning("Limite giornaliero di trade raggiunto")
            return

        if signal.signal == Signal.BUY and self.current_position is None:
            self._open_position(signal)

        elif signal.signal == Signal.SELL and self.current_position is not None:
            self._close_position(signal.price, "Segnale SELL")

    def _open_position(self, signal: TradeSignal):
        """Apre una nuova posizione."""
        try:
            symbol = config.trading.symbol
            amount = config.trading.amount

            logger.info(f"Apertura posizione LONG: {amount} {symbol} @ {signal.price:.2f}")

            # Ordine market buy
            order = self.fetcher.exchange.create_market_buy_order(
                symbol=symbol,
                amount=amount,
            )

            self.current_position = {
                "order_id": order["id"],
                "symbol": symbol,
                "side": "long",
                "entry_price": order.get("average", signal.price),
                "amount": amount,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "entry_time": datetime.utcnow(),
                "reasons": signal.reasons,
            }

            self.daily_trades += 1
            self.last_trade_time = datetime.utcnow()

            logger.info(
                f"TRADE APERTO | BUY {amount} {symbol} @ "
                f"${self.current_position['entry_price']:.2f} | "
                f"SL: ${signal.stop_loss:.2f} | TP: ${signal.take_profit:.2f}"
            )

            if self.notifier:
                self.notifier.send_sync(
                    f"📈 *TRADE APERTO*\n"
                    f"BUY `{amount}` `{symbol}`\n"
                    f"Prezzo: `${self.current_position['entry_price']:,.2f}`\n"
                    f"Stop Loss: `${signal.stop_loss:,.2f}`\n"
                    f"Take Profit: `${signal.take_profit:,.2f}`\n"
                    f"Confidenza: `{signal.confidence:.0%}`\n"
                    f"Motivi: {', '.join(signal.reasons[:3])}"
                )

        except Exception as e:
            logger.error(f"Errore nell'apertura della posizione: {e}")
            if self.notifier:
                self.notifier.send_sync(f"❌ *Errore apertura trade*: {str(e)}")

    def _close_position(self, price: float, reason: str):
        """Chiude la posizione corrente."""
        if self.current_position is None:
            return

        try:
            symbol = self.current_position["symbol"]
            amount = self.current_position["amount"]

            logger.info(f"Chiusura posizione: SELL {amount} {symbol} @ {price:.2f}")

            # Ordine market sell
            order = self.fetcher.exchange.create_market_sell_order(
                symbol=symbol,
                amount=amount,
            )

            exit_price = order.get("average", price)
            entry_price = self.current_position["entry_price"]

            pnl = (exit_price - entry_price) * amount
            pnl_pct = (exit_price / entry_price - 1) * 100

            self.daily_pnl += pnl

            trade_record = {
                **self.current_position,
                "exit_price": exit_price,
                "exit_time": datetime.utcnow(),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
            }
            self.trade_history.append(trade_record)

            emoji = "✅" if pnl >= 0 else "❌"
            color = "green" if pnl >= 0 else "red"

            logger.info(
                f"TRADE CHIUSO | {emoji} SELL {amount} {symbol} @ ${exit_price:.2f} | "
                f"PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%) | Motivo: {reason}"
            )

            if self.notifier:
                self.notifier.send_sync(
                    f"{emoji} *TRADE CHIUSO*\n"
                    f"SELL `{amount}` `{symbol}`\n"
                    f"Entry: `${entry_price:,.2f}`\n"
                    f"Exit: `${exit_price:,.2f}`\n"
                    f"PnL: `${pnl:,.2f}` (`{pnl_pct:+.2f}%`)\n"
                    f"Motivo: {reason}\n"
                    f"PnL Giornaliero: `${self.daily_pnl:,.2f}`"
                )

            self.current_position = None

        except Exception as e:
            logger.error(f"Errore nella chiusura della posizione: {e}")
            if self.notifier:
                self.notifier.send_sync(f"❌ *Errore chiusura trade*: {str(e)}")

    def _check_position(self, current_data: pd.Series):
        """Controlla la posizione aperta per stop loss e take profit."""
        if self.current_position is None:
            return

        price = current_data["close"]
        low = current_data["low"]
        high = current_data["high"]

        sl = self.current_position.get("stop_loss")
        tp = self.current_position.get("take_profit")

        if sl and low <= sl:
            self._close_position(sl, "Stop Loss raggiunto")
        elif tp and high >= tp:
            self._close_position(tp, "Take Profit raggiunto")

    def _train_ml(self):
        """Addestra o riaddestra il modello ML."""
        if not self.ml_predictor:
            return

        try:
            logger.info("Training/retrain modello ML...")
            df = self.fetcher.fetch_historical(days=90)
            if not df.empty:
                df_indicators = TechnicalIndicators.add_all(df)
                metrics = self.ml_predictor.train(df_indicators)
                logger.info(f"ML training completato: {metrics}")

                if self.notifier:
                    self.notifier.send_sync(
                        f"🧠 *ML Model Aggiornato*\n"
                        f"Accuracy: `{metrics.get('test_accuracy', 0):.3f}`\n"
                        f"F1 Score: `{metrics.get('test_f1', 0):.3f}`"
                    )
        except Exception as e:
            logger.error(f"Errore nel training ML: {e}")

    def _daily_reset(self):
        """Reset giornaliero dei contatori."""
        logger.info(
            f"Reset giornaliero | Trade: {self.daily_trades} | PnL: ${self.daily_pnl:,.2f}"
        )

        if self.notifier:
            self.notifier.send_sync(
                f"📊 *Report Giornaliero*\n"
                f"Trade: {self.daily_trades}\n"
                f"PnL: `${self.daily_pnl:,.2f}`\n"
                f"Posizione aperta: {'Sì' if self.current_position else 'No'}"
            )

        self.daily_trades = 0
        self.daily_pnl = 0.0

    def get_status(self) -> dict:
        """Ritorna lo stato corrente del trader."""
        return {
            "is_running": self.is_running,
            "symbol": config.trading.symbol,
            "timeframe": config.trading.timeframe,
            "sandbox": config.exchange.sandbox,
            "ml_enabled": self.use_ml,
            "current_position": self.current_position,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "total_trades": len(self.trade_history),
            "last_trade_time": str(self.last_trade_time) if self.last_trade_time else None,
        }
