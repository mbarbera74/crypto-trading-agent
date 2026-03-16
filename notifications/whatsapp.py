"""
Notifiche WhatsApp tramite Twilio API.
Invia alert per segnali di accumulo su CSNDX e SWDA.
"""

from twilio.rest import Client
from config.settings import config
from utils.logger import get_logger

logger = get_logger("notifications.whatsapp")


class WhatsAppNotifier:
    """Invia notifiche WhatsApp tramite Twilio."""

    def __init__(self):
        self.account_sid = config.whatsapp.account_sid
        self.auth_token = config.whatsapp.auth_token
        self.from_number = config.whatsapp.from_number  # "whatsapp:+14155238886" (Twilio sandbox)
        self.to_number = config.whatsapp.to_number      # "whatsapp:+393316037980"
        self.enabled = config.whatsapp.enabled
        self._client = None

        if self.enabled:
            try:
                self._client = Client(self.account_sid, self.auth_token)
                logger.info(f"WhatsApp notifier inizializzato → {self.to_number}")
            except Exception as e:
                logger.warning(f"Impossibile inizializzare Twilio WhatsApp: {e}")
                self.enabled = False
        else:
            logger.info("Notifiche WhatsApp disabilitate (credenziali Twilio non configurate)")

    def send(self, message: str) -> bool:
        """Invia un messaggio WhatsApp."""
        if not self.enabled or not self._client:
            logger.debug(f"WhatsApp (non inviato): {message[:80]}...")
            return False

        try:
            msg = self._client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number,
            )
            logger.info(f"WhatsApp inviato (SID: {msg.sid}): {message[:80]}...")
            return True
        except Exception as e:
            logger.error(f"Errore invio WhatsApp: {e}")
            return False

    def send_accumulation_signal(self, asset_name: str, action: str, price: float,
                                  currency: str, score: float, entry_level: str,
                                  entry_price: float, probability: float,
                                  recommendation: str, cape_info: str = "",
                                  drawdown_from_ath: float = 0.0, ath_price: float = 0.0,
                                  drawdown_from_52w: float = 0.0,
                                  regime_name: str = "", regime_emoji: str = "",
                                  regime_strategy: str = ""):
        """Invia segnale di accumulo formattato per WhatsApp."""
        emoji = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}.get(action, "⚪")

        msg = (
            f"{emoji} *SEGNALE ACCUMULO - {asset_name}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Azione: *{action}*\n"
            f"💰 Prezzo attuale: {currency}{price:,.2f}\n"
            f"📉 Drawdown da max: {drawdown_from_ath:.1f}% (max: {currency}{ath_price:,.2f})\n"
            f"📉 Da max 52 sett.: {drawdown_from_52w:.1f}%\n"
            f"🎯 Livello target: {currency}{entry_price:,.2f} ({entry_level})\n"
            f"📈 Probabilità 90gg: {probability:.0%}\n"
            f"⚖️ Score composito: {score:+.2f}\n"
        )
        if regime_name:
            msg += f"🔮 Regime HMM: {regime_emoji} {regime_name}\n"
            if regime_strategy:
                msg += f"   → {regime_strategy}\n"
        if cape_info:
            msg += f"📊 CAPE: {cape_info}\n"
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 {recommendation}\n"
        )

        return self.send(msg)

    def send_daily_summary(self, assets_summary: list[dict]):
        """Invia riepilogo giornaliero con tutti gli asset monitorati."""
        msg = "📋 *RIEPILOGO GIORNALIERO ACCUMULO*\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━\n"

        for asset in assets_summary:
            emoji = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}.get(asset.get("action", ""), "⚪")
            msg += (
                f"\n{emoji} *{asset['name']}*\n"
                f"   Prezzo: {asset['currency']}{asset['price']:,.2f}\n"
                f"   Score: {asset['score']:+.2f}\n"
                f"   → {asset['recommendation']}\n"
            )
            if asset.get("best_entry"):
                msg += f"   🎯 Target: {asset['currency']}{asset['best_entry_price']:,.2f} (prob {asset['best_entry_prob']:.0%})\n"

        msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━"
        return self.send(msg)

    def send_price_alert(self, asset_name: str, currency: str, current_price: float,
                          target_price: float, level_name: str):
        """Notifica quando il prezzo raggiunge un livello di ingresso."""
        msg = (
            f"🎯 *PREZZO RAGGIUNTO!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *{asset_name}*\n"
            f"💰 Prezzo: {currency}{current_price:,.2f}\n"
            f"🎯 Livello: {currency}{target_price:,.2f} ({level_name})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Valuta l'acquisto immediato!"
        )
        return self.send(msg)
