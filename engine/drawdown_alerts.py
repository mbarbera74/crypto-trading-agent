"""
Drawdown Alert Engine — Monitora gli asset e invia alert su WhatsApp e Telegram
quando il prezzo scende oltre una soglia dal massimo rilevante.

Configurazione soglie:
    CSNDX: alert se drawdown >= 10%
    SWDA:  alert se drawdown >= 5%

L'engine usa un file di stato per evitare notifiche duplicate.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import yfinance as yf

from utils.logger import get_logger

logger = get_logger("engine.drawdown_alerts")

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "drawdown_alert_state.json"

# ══════════════════════════════════════════════════════════════
# CONFIGURAZIONE SOGLIE DRAWDOWN
# ══════════════════════════════════════════════════════════════
DRAWDOWN_RULES = {
    "CSNDX": {
        "yahoo": "CNDX.MI",
        "fallbacks": ["CSNDX.MI", "SXRV.DE"],
        "label": "CSNDX (iShares NASDAQ 100)",
        "currency": "€",
        "period": "1y",
        "threshold_pct": -10.0,  # Alert se drawdown >= 10%
    },
    "SWDA": {
        "yahoo": "SWDA.MI",
        "fallbacks": ["SWDA.L", "IWDA.AS"],
        "label": "SWDA.MI (iShares MSCI World)",
        "currency": "€",
        "period": "1y",
        "threshold_pct": -5.0,  # Alert se drawdown >= 5%
    },
}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _fetch_price_and_high(rule: dict) -> tuple[Optional[float], Optional[float]]:
    """Recupera prezzo attuale e massimo del periodo per un asset."""
    tickers_to_try = [rule["yahoo"]] + rule.get("fallbacks", [])
    for ticker in tickers_to_try:
        try:
            data = yf.Ticker(ticker).history(period=rule["period"])
            if not data.empty and len(data) >= 2:
                price = data["Close"].iloc[-1]
                high = data["High"].max()
                return float(price), float(high)
        except Exception:
            continue
    return None, None


def check_drawdown_alerts(notify: bool = True) -> list[dict]:
    """
    Controlla tutti gli asset configurati e invia alert se il drawdown
    supera la soglia. Evita notifiche duplicate (max 1 al giorno per asset).

    Args:
        notify: Se True, invia effettivamente le notifiche

    Returns:
        Lista di alert triggered (anche se non notificati)
    """
    state = _load_state()
    today = datetime.now().strftime("%Y-%m-%d")
    triggered = []

    for key, rule in DRAWDOWN_RULES.items():
        price, high = _fetch_price_and_high(rule)
        if price is None or high is None or high == 0:
            logger.warning(f"Impossibile recuperare dati per {key}")
            continue

        drawdown_pct = (price / high - 1) * 100

        logger.debug(
            f"{key}: prezzo={rule['currency']}{price:,.2f}, "
            f"max={rule['currency']}{high:,.2f}, drawdown={drawdown_pct:.1f}%"
        )

        if drawdown_pct <= rule["threshold_pct"]:
            alert = {
                "asset": key,
                "label": rule["label"],
                "currency": rule["currency"],
                "price": price,
                "high": high,
                "drawdown_pct": drawdown_pct,
                "threshold_pct": rule["threshold_pct"],
                "timestamp": datetime.now().isoformat(),
            }
            triggered.append(alert)

            # Controlla se già notificato oggi
            last_alert_key = f"{key}_last_alert"
            last_alert_date = state.get(last_alert_key, "")

            if last_alert_date != today and notify:
                _send_alert(alert)
                state[last_alert_key] = today
                logger.info(
                    f"🚨 ALERT INVIATO: {key} drawdown {drawdown_pct:.1f}% "
                    f"(soglia: {rule['threshold_pct']}%)"
                )
            elif last_alert_date == today:
                logger.debug(f"{key}: alert già inviato oggi, skip")

    _save_state(state)
    return triggered


def _send_alert(alert: dict):
    """Invia alert drawdown su WhatsApp + Telegram."""
    msg_wa = (
        f"🚨 *ALERT DRAWDOWN — {alert['label']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📉 Drawdown: *{alert['drawdown_pct']:.1f}%* dal massimo\n"
        f"💰 Prezzo attuale: {alert['currency']}{alert['price']:,.2f}\n"
        f"📈 Massimo periodo: {alert['currency']}{alert['high']:,.2f}\n"
        f"⚠️ Soglia alert: {alert['threshold_pct']:.0f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"💡 Valuta se è un'opportunità di acquisto!"
    )

    msg_tg = (
        f"🚨 *ALERT DRAWDOWN*\n"
        f"📊 *{alert['label']}*\n"
        f"📉 Drawdown: `{alert['drawdown_pct']:.1f}%` dal massimo\n"
        f"💰 Prezzo: `{alert['currency']}{alert['price']:,.2f}`\n"
        f"📈 Max: `{alert['currency']}{alert['high']:,.2f}`\n"
        f"⚠️ Soglia: `{alert['threshold_pct']:.0f}%`\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    # WhatsApp
    try:
        from notifications.whatsapp import WhatsAppNotifier
        wa = WhatsAppNotifier()
        if wa.enabled:
            wa.send(msg_wa)
            logger.info(f"Alert WhatsApp inviato per {alert['asset']}")
        else:
            logger.info(f"WhatsApp non configurato, alert non inviato")
    except Exception as e:
        logger.error(f"Errore WhatsApp alert: {e}")

    # Telegram
    try:
        from notifications.notifier import TelegramNotifier
        tg = TelegramNotifier()
        if tg.enabled:
            tg.send_sync(msg_tg)
            logger.info(f"Alert Telegram inviato per {alert['asset']}")
        else:
            logger.info(f"Telegram non configurato, alert non inviato")
    except Exception as e:
        logger.error(f"Errore Telegram alert: {e}")


def send_test_alert():
    """
    Invia un alert di test simulando un drawdown.
    Utile per verificare che WhatsApp e Telegram funzionino.
    """
    logger.info("Invio alert di TEST...")
    test_alert = {
        "asset": "TEST",
        "label": "CSNDX (TEST — simulato)",
        "currency": "€",
        "price": 1098.59,
        "high": 1220.66,
        "drawdown_pct": -10.0,
        "threshold_pct": -10.0,
        "timestamp": datetime.now().isoformat(),
    }
    _send_alert(test_alert)
    logger.info("Alert di TEST inviato su WhatsApp e Telegram")
    return test_alert
