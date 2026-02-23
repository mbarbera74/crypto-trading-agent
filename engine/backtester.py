"""
Motore di backtesting per simulare la strategia su dati storici.
Calcola metriche di performance: PnL, Sharpe Ratio, Max Drawdown, Win Rate, ecc.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from indicators.technical import TechnicalIndicators
from strategy.momentum import MomentumStrategy, Signal, TradeSignal
from ml.model import MLPredictor
from config.settings import config
from utils.logger import get_logger

logger = get_logger("engine.backtester")


@dataclass
class Trade:
    """Rappresenta un singolo trade nel backtest."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str = "long"
    amount: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    entry_reasons: list[str] = field(default_factory=list)  # Motivi dell'entrata

    @property
    def is_open(self) -> bool:
        return self.exit_time is None


@dataclass
class BacktestResult:
    """Risultati del backtest."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    # Metriche
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    initial_capital: float = 0.0
    final_capital: float = 0.0
    asset_name: str = ""  # Nome dell'asset per il report
    currency: str = "$"  # Simbolo valuta


class Backtester:
    """
    Motore di backtesting per testare la strategia su dati storici.
    Supporta crypto (BTC), indici (NDX) e ETF (SWDA.MI, CSNDX) in modalità long-only.
    """

    def __init__(
        self,
        strategy: MomentumStrategy = None,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        use_ml: bool = False,
        long_only: bool = False,
        asset_name: str = "",
        currency: str = "$",
        position_size: float = 0.0,
        no_exit_signal: bool = False,  # Se True, non esce su segnale SELL (solo fine backtest)
    ):
        self.strategy = strategy or MomentumStrategy()
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.use_ml = use_ml
        self.long_only = long_only
        self.asset_name = asset_name
        self.currency = currency
        self.position_size = position_size if position_size > 0 else config.trading.max_position_size
        self.no_exit_signal = no_exit_signal
        self.ml_predictor = MLPredictor() if use_ml else None

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Esegue il backtest completo sulla serie storica.

        Args:
            df: DataFrame OHLCV grezzo (senza indicatori)

        Returns:
            BacktestResult con tutte le metriche e i trade
        """
        logger.info(
            f"Inizio backtest | Capitale: ${self.initial_capital:,.2f} | "
            f"Periodo: {df.index[0]} → {df.index[-1]} | Candele: {len(df)}"
        )

        # Aggiungi indicatori tecnici
        df_indicators = TechnicalIndicators.add_all(df)

        if len(df_indicators) < 50:
            logger.error("Dati insufficienti dopo il calcolo degli indicatori")
            return BacktestResult()

        # Addestra ML se abilitato
        if self.use_ml and self.ml_predictor:
            logger.info("Training modello ML per il backtest...")
            # Usa i primi 60% per il training
            train_size = int(len(df_indicators) * 0.6)
            train_data = df_indicators.iloc[:train_size]
            self.ml_predictor.train(train_data)

        # Simulazione
        capital = self.initial_capital
        position: Optional[Trade] = None
        trades: list[Trade] = []
        equity_curve: list[float] = []
        timestamps: list[datetime] = []

        for i in range(50, len(df_indicators)):
            current = df_indicators.iloc[i]
            timestamp = df_indicators.index[i]
            price = current["close"]
            high = current["high"]
            low = current["low"]

            # Controlla stop loss / take profit se posizione aperta
            if position is not None and position.is_open:
                closed = False

                if position.side == "long":
                    # Stop Loss long
                    if position.stop_loss and low <= position.stop_loss:
                        position.exit_time = timestamp
                        position.exit_price = position.stop_loss
                        position.exit_reason = "Stop Loss"
                        closed = True
                    # Take Profit long
                    elif position.take_profit and high >= position.take_profit:
                        position.exit_time = timestamp
                        position.exit_price = position.take_profit
                        position.exit_reason = "Take Profit"
                        closed = True
                elif position.side == "short":
                    # Stop Loss short (prezzo sale)
                    if position.stop_loss and high >= position.stop_loss:
                        position.exit_time = timestamp
                        position.exit_price = position.stop_loss
                        position.exit_reason = "Stop Loss"
                        closed = True
                    # Take Profit short (prezzo scende)
                    elif position.take_profit and low <= position.take_profit:
                        position.exit_time = timestamp
                        position.exit_price = position.take_profit
                        position.exit_reason = "Take Profit"
                        closed = True

                if closed:
                    # Calcola PnL
                    if position.side == "long":
                        position.pnl = (position.exit_price - position.entry_price) * position.amount
                        position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100
                    else:
                        position.pnl = (position.entry_price - position.exit_price) * position.amount
                        position.pnl_pct = (position.entry_price / position.exit_price - 1) * 100

                    # Commissioni
                    commission = abs(position.pnl) * self.commission_pct * 2  # entry + exit
                    position.pnl -= commission

                    capital += position.pnl
                    trades.append(position)
                    position = None

            # Genera segnale
            window = df_indicators.iloc[:i + 1]
            ml_pred = None
            if self.use_ml and self.ml_predictor and i > int(len(df_indicators) * 0.6):
                ml_pred = self.ml_predictor.predict(window)

            signal = self.strategy.analyze(window, ml_prediction=ml_pred)

            # Esegui segnale
            if signal.signal == Signal.BUY and position is None:
                # Apri posizione long
                trade_amount = capital * self.position_size / price
                position = Trade(
                    entry_time=timestamp,
                    entry_price=price,
                    side="long",
                    amount=trade_amount,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_reasons=signal.reasons[:5],
                )

            elif signal.signal == Signal.SELL and position is None and not self.long_only:
                # Apri posizione short (solo se non long_only)
                trade_amount = capital * self.position_size / price
                position = Trade(
                    entry_time=timestamp,
                    entry_price=price,
                    side="short",
                    amount=trade_amount,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_reasons=signal.reasons[:5],
                )

            elif signal.signal == Signal.SELL and position is not None and position.is_open and position.side == "long" and not self.no_exit_signal:
                # Chiudi posizione long (solo se no_exit_signal è False)
                position.exit_time = timestamp
                position.exit_price = price
                position.exit_reason = "Segnale SELL"
                position.pnl = (position.exit_price - position.entry_price) * position.amount
                position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100

                commission = abs(position.pnl) * self.commission_pct * 2
                position.pnl -= commission

                capital += position.pnl
                trades.append(position)
                position = None

            elif signal.signal == Signal.BUY and position is not None and position.is_open and position.side == "short" and not self.no_exit_signal:
                # Chiudi posizione short (solo se no_exit_signal è False)
                position.exit_time = timestamp
                position.exit_price = price
                position.exit_reason = "Segnale BUY"
                position.pnl = (position.entry_price - position.exit_price) * position.amount
                position.pnl_pct = (position.entry_price / position.exit_price - 1) * 100

                commission = abs(position.pnl) * self.commission_pct * 2
                position.pnl -= commission

                capital += position.pnl
                trades.append(position)
                position = None

            # Aggiorna equity curve
            unrealized = 0
            if position is not None and position.is_open:
                if position.side == "long":
                    unrealized = (price - position.entry_price) * position.amount
                elif position.side == "short":
                    unrealized = (position.entry_price - price) * position.amount

            equity_curve.append(capital + unrealized)
            timestamps.append(timestamp)

        # Chiudi posizione aperta alla fine
        if position is not None and position.is_open:
            last_price = df_indicators.iloc[-1]["close"]
            position.exit_time = df_indicators.index[-1]
            position.exit_price = last_price
            position.exit_reason = "Fine backtest"

            if position.side == "long":
                position.pnl = (position.exit_price - position.entry_price) * position.amount
                position.pnl_pct = (position.exit_price / position.entry_price - 1) * 100
            elif position.side == "short":
                position.pnl = (position.entry_price - position.exit_price) * position.amount
                position.pnl_pct = (position.entry_price / position.exit_price - 1) * 100

            commission = abs(position.pnl) * self.commission_pct * 2
            position.pnl -= commission
            capital += position.pnl
            trades.append(position)

        # Calcola metriche
        result = self._calculate_metrics(trades, equity_curve, timestamps, capital)
        result.asset_name = self.asset_name
        result.currency = self.currency

        logger.info(
            f"Backtest completato{' (' + self.asset_name + ')' if self.asset_name else ''} | "
            f"Trade: {result.total_trades} | "
            f"Win Rate: {result.win_rate:.1f}% | PnL: {self.currency}{result.total_pnl:,.2f} | "
            f"Sharpe: {result.sharpe_ratio:.2f} | Max DD: {result.max_drawdown_pct:.1f}%"
        )

        return result

    def _calculate_metrics(
        self,
        trades: list[Trade],
        equity_curve: list[float],
        timestamps: list[datetime],
        final_capital: float,
    ) -> BacktestResult:
        """Calcola le metriche di performance del backtest."""
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
        )

        if not trades:
            return result

        pnl_list = [t.pnl for t in trades]
        pnl_pct_list = [t.pnl_pct for t in trades]
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        result.total_trades = len(trades)
        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = len(winning) / len(trades) * 100 if trades else 0
        result.total_pnl = sum(pnl_list)
        result.total_pnl_pct = (final_capital / self.initial_capital - 1) * 100
        result.avg_pnl_pct = float(np.mean(pnl_pct_list)) if pnl_pct_list else 0

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max Drawdown
        if equity_curve:
            equity_arr = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            result.max_drawdown_pct = abs(float(np.min(drawdown)) * 100)
            result.max_drawdown = abs(float(np.min(equity_arr - peak)))

        # Sharpe Ratio (annualizzato, assumendo candele orarie)
        if len(pnl_pct_list) > 1:
            returns = np.array(pnl_pct_list) / 100
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                # ~8760 ore in un anno
                result.sharpe_ratio = float(avg_return / std_return * np.sqrt(8760 / len(trades)))

            # Sortino Ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    result.sortino_ratio = float(avg_return / downside_std * np.sqrt(8760 / len(trades)))

        # Best / Worst trade
        result.best_trade_pct = max(pnl_pct_list) if pnl_pct_list else 0
        result.worst_trade_pct = min(pnl_pct_list) if pnl_pct_list else 0

        # Durata media dei trade
        durations = []
        for t in trades:
            if t.entry_time and t.exit_time:
                duration_hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration_hours)
        result.avg_trade_duration = float(np.mean(durations)) if durations else 0

        return result

    def print_report(self, result: BacktestResult):
        """Stampa un report formattato dei risultati del backtest."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        cur = result.currency if result.currency else "$"
        asset_label = f" - {result.asset_name}" if result.asset_name else ""
        mode_label = " [Long Only]" if self.long_only else ""

        # Header
        console.print(Panel(
            f"[bold cyan]REPORT BACKTEST{asset_label}{mode_label}[/bold cyan]",
            style="bold white",
        ))

        # Metriche principali
        table = Table(title=f"Metriche di Performance{asset_label}", show_header=True)
        table.add_column("Metrica", style="cyan", width=25)
        table.add_column("Valore", style="green", justify="right")

        color_pnl = "green" if result.total_pnl >= 0 else "red"

        table.add_row("Asset", result.asset_name or "BTC/USDT")
        table.add_row("Modalità", "Long Only" if self.long_only else "Long + Short")
        table.add_row("Posizione per Trade", f"{self.position_size*100:.0f}%")
        table.add_row("─" * 25, "─" * 15)
        table.add_row("Capitale Iniziale", f"{cur}{result.initial_capital:,.2f}")
        table.add_row("Capitale Finale", f"[{color_pnl}]{cur}{result.final_capital:,.2f}[/{color_pnl}]")
        table.add_row("PnL Totale", f"[{color_pnl}]{cur}{result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)[/{color_pnl}]")
        table.add_row("─" * 25, "─" * 15)
        table.add_row("Trade Totali", str(result.total_trades))
        table.add_row("Trade Vincenti", f"{result.winning_trades} ({result.win_rate:.1f}%)")
        table.add_row("Trade Perdenti", str(result.losing_trades))
        table.add_row("PnL Medio per Trade", f"{result.avg_pnl_pct:.2f}%")
        table.add_row("Miglior Trade", f"{result.best_trade_pct:.2f}%")
        table.add_row("Peggior Trade", f"{result.worst_trade_pct:.2f}%")
        table.add_row("─" * 25, "─" * 15)
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
        table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
        table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.1f}%")
        table.add_row("Durata Media Trade", f"{result.avg_trade_duration:.1f}h")

        console.print(table)
