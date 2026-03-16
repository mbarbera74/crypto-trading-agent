"""
Sistema di notifiche tramite Telegram.
Invia alert per trade aperti/chiusi, errori e report giornalieri.
"""

import asyncio
from typing import Optional

from config.settings import config
from utils.logger import get_logger

logger = get_logger("notifications.notifier")


class TelegramNotifier:
    """Invia notifiche di trading tramite Telegram Bot."""

    def __init__(self):
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.enabled = config.telegram.enabled
        self._bot = None

        if self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
                logger.info("Telegram notifier inizializzato")
            except Exception as e:
                logger.warning(f"Impossibile inizializzare Telegram: {e}")
                self.enabled = False
        else:
            logger.info("Notifiche Telegram disabilitate (token non configurato)")

    async def send(self, message: str, parse_mode: str = "Markdown"):
        """Invia un messaggio asincrono tramite Telegram."""
        if not self.enabled or not self._bot:
            logger.debug(f"Notifica (non inviata): {message[:80]}...")
            return

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            logger.debug(f"Notifica Telegram inviata: {message[:80]}...")
        except Exception as e:
            logger.error(f"Errore invio Telegram: {e}")

    def send_sync(self, message: str, parse_mode: str = "Markdown"):
        """Invia un messaggio in modo sincrono (wrapper per contesti non-async)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send(message, parse_mode))
            else:
                loop.run_until_complete(self.send(message, parse_mode))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send(message, parse_mode))

    def send_trade_open(self, symbol: str, side: str, amount: float, price: float,
                         stop_loss: float, take_profit: float, confidence: float):
        """Invia notifica di apertura trade."""
        emoji = "📈" if side == "long" else "📉"
        msg = (
            f"{emoji} *TRADE APERTO*\n"
            f"{'BUY' if side == 'long' else 'SELL'} `{amount}` `{symbol}`\n"
            f"Prezzo: `${price:,.2f}`\n"
            f"Stop Loss: `${stop_loss:,.2f}`\n"
            f"Take Profit: `${take_profit:,.2f}`\n"
            f"Confidenza: `{confidence:.0%}`"
        )
        self.send_sync(msg)

    def send_trade_close(self, symbol: str, entry: float, exit_price: float,
                          pnl: float, pnl_pct: float, reason: str):
        """Invia notifica di chiusura trade."""
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} *TRADE CHIUSO*\n"
            f"`{symbol}`\n"
            f"Entry: `${entry:,.2f}` → Exit: `${exit_price:,.2f}`\n"
            f"PnL: `${pnl:,.2f}` (`{pnl_pct:+.2f}%`)\n"
            f"Motivo: {reason}"
        )
        self.send_sync(msg)

    def send_daily_report(self, trades: int, pnl: float, win_rate: float,
                           position_open: bool):
        """Invia il report giornaliero."""
        msg = (
            f"📊 *Report Giornaliero*\n"
            f"Trade: `{trades}`\n"
            f"PnL: `${pnl:,.2f}`\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"Posizione aperta: {'Sì ✅' if position_open else 'No'}"
        )
        self.send_sync(msg)

    def send_error(self, error: str):
        """Invia notifica di errore."""
        msg = f"⚠️ *ERRORE*\n```\n{error[:500]}\n```"
        self.send_sync(msg)

    def send_accumulation_signal(self, asset_name: str, action: str, price: float,
                                  currency: str, score: float, entry_level: str,
                                  entry_price: float, probability: float,
                                  recommendation: str, cape_info: str = "",
                                  regime_name: str = "", regime_emoji: str = "",
                                  regime_strategy: str = "", **kwargs):
        """Invia segnale di accumulo formattato per Telegram."""
        emoji = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}.get(action, "⚪")
        msg = (
            f"{emoji} *SEGNALE ACCUMULO \\- {asset_name}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Azione: *{action}*\n"
            f"💰 Prezzo: `{currency}{price:,.2f}`\n"
            f"🎯 Target: `{currency}{entry_price:,.2f}` \\({entry_level}\\)\n"
            f"📈 Prob\\. 90gg: `{probability:.0%}`\n"
            f"⚖️ Score: `{score:+.2f}`\n"
        )
        if regime_name:
            msg += f"🔮 Regime: {regime_emoji} {regime_name}\n"
            if regime_strategy:
                msg += f"   → {regime_strategy}\n"
        if cape_info:
            msg += f"📉 CAPE: {cape_info}\n"
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 {recommendation}\n"
        )
        self.send_sync(msg, parse_mode="MarkdownV2")

    def send_daily_accumulation_summary(self, assets_summary: list[dict]):
        """Invia riepilogo giornaliero accumulo via Telegram."""
        msg = "📋 *RIEPILOGO ACCUMULO*\n━━━━━━━━━━━━━━━━━━━\n"
        for a in assets_summary:
            emoji = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}.get(a.get("action", ""), "⚪")
            msg += (
                f"\n{emoji} *{a['name']}*\n"
                f"  Prezzo: `{a['currency']}{a['price']:,.2f}`\n"
                f"  Score: `{a['score']:+.2f}` → {a['recommendation']}\n"
            )
            if a.get("best_entry"):
                msg += f"  🎯 Target: `{a['currency']}{a['best_entry_price']:,.2f}` (prob {a['best_entry_prob']:.0%})\n"
        self.send_sync(msg)

    def send_price_alert(self, asset_name: str, currency: str, current_price: float,
                          target_price: float, level_name: str):
        """Notifica prezzo raggiunto via Telegram."""
        msg = (
            f"🎯 *PREZZO RAGGIUNTO\\!*\n"
            f"📊 *{asset_name}*\n"
            f"💰 Prezzo: `{currency}{current_price:,.2f}`\n"
            f"🎯 Livello: `{currency}{target_price:,.2f}` \\({level_name}\\)\n"
            f"⚡ Valuta l'acquisto\\!"
        )
        self.send_sync(msg, parse_mode="MarkdownV2")
