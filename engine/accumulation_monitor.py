"""
Accumulation Monitor - Motore di segnali per strategia di accumulo (PAC).

Monitora CSNDX e SWDA.MI e genera segnali di acquisto basati su:
1. Livelli di ingresso con probabilità (Fibonacci, SMA, supporti)
2. Score composito (tecnico + CAPE + liquidità)
3. Alert quando il prezzo tocca livelli favorevoli
4. Notifiche WhatsApp in tempo reale

Uso:
    python main.py monitor               # Singola analisi + notifica
    python main.py monitor --schedule    # Loop continuo (ogni 4h di mercato aperto)
"""

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Optional
import json
from pathlib import Path

from analysis.market_analyzer import MarketAnalyzer, MarketReport, AssetAnalysis
from notifications.whatsapp import WhatsAppNotifier
from notifications.notifier import TelegramNotifier
from utils.logger import get_logger

logger = get_logger("engine.accumulation_monitor")

# Dove salvare lo stato per evitare notifiche duplicate
STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "monitor_state.json"


@dataclass
class AccumulationSignal:
    """Segnale di accumulo per un singolo asset."""
    asset_name: str
    ticker: str
    currency: str
    current_price: float
    score: float
    action: str          # COMPRA, ATTENDI, EVITA
    recommendation: str
    entry_type: str
    best_entry_level: str = ""
    best_entry_price: float = 0.0
    best_entry_prob: float = 0.0
    cape_info: str = ""
    rsi: float = 50.0
    trend: str = ""
    price_change_1m: float = 0.0
    drawdown_from_ath: float = 0.0
    ath_price: float = 0.0
    drawdown_from_52w: float = 0.0
    crossed_levels: list[dict] = field(default_factory=list)  # Livelli raggiunti


class AccumulationMonitor:
    """
    Monitora gli asset della strategia di accumulo e genera segnali.
    """

    def __init__(self):
        self.analyzer = MarketAnalyzer()
        self.notifier = WhatsAppNotifier()
        self.telegram = TelegramNotifier()
        self._state = self._load_state()

    # ────────────────────────────────────────
    # ANALISI PRINCIPALE
    # ────────────────────────────────────────

    def check_signals(self, notify: bool = True) -> list[AccumulationSignal]:
        """
        Esegue l'analisi di mercato e genera segnali di accumulo per CSNDX e SWDA.

        Args:
            notify: Se True, invia notifiche WhatsApp

        Returns:
            Lista di AccumulationSignal per ogni asset monitorato
        """
        logger.info("=" * 50)
        logger.info("MONITOR ACCUMULO - Analisi in corso...")
        logger.info("=" * 50)

        # Analisi completa di mercato
        report = self.analyzer.full_analysis(period="1y")

        # Recupera news importanti
        market_news = self._fetch_important_news()

        signals = []

        # Genera segnali per SWDA.MI e CSNDX (gli asset di accumulo)
        for asset, score, rec, entry_type, currency, name in [
            (report.swda, report.swda_score, report.swda_recommendation,
             report.swda_entry_type, "€", "SWDA.MI (MSCI World)"),
            (report.csndx, report.csndx_score, report.csndx_recommendation,
             report.csndx_entry_type, "€", "CSNDX (iShares NDX)"),
        ]:
            if asset and asset.current_price > 0:
                signal = self._build_signal(
                    asset, score, rec, entry_type, currency, name
                )
                signals.append(signal)

        # Controlla se ci sono livelli di prezzo raggiunti (alert)
        for sig in signals:
            sig.crossed_levels = self._check_price_levels(sig.asset_name, sig.current_price, report)

        # Notifiche
        if notify and (self.notifier.enabled or self.telegram.enabled):
            self._send_notifications(signals, report, market_news)

        # Salva stato per evitare notifiche duplicate
        self._save_state(signals)

        logger.info("=" * 50)
        logger.info("MONITOR ACCUMULO - Completato")
        logger.info("=" * 50)

        return signals

    def _build_signal(self, asset: AssetAnalysis, score: float, rec: str,
                       entry_type: str, currency: str, name: str) -> AccumulationSignal:
        """Costruisce un segnale di accumulo per un asset."""

        # Determina azione
        if score > 0.2:
            action = "COMPRA"
        elif score > -0.2:
            action = "ATTENDI"
        else:
            action = "EVITA"

        # Trova il miglior livello di ingresso raggiungibile (prob_90d > 60%)
        best_level = ""
        best_price = 0.0
        best_prob = 0.0

        reachable_levels = [
            lvl for lvl in asset.entry_levels
            if lvl.prob_90d > 0.60 and lvl.distance_pct < 0
        ]
        if reachable_levels:
            # Prendi il livello più profondo tra quelli raggiungibili
            best = min(reachable_levels, key=lambda x: x.price)
            best_level = best.level
            best_price = best.price
            best_prob = best.prob_90d

        # CAPE info
        cape_info = ""
        if asset.cape_analysis:
            ca = asset.cape_analysis
            cape_info = f"{ca.cape_value:.1f} ({ca.valuation_level})"

        return AccumulationSignal(
            asset_name=name,
            ticker=asset.ticker,
            currency=currency,
            current_price=asset.current_price,
            score=score,
            action=action,
            recommendation=rec,
            entry_type=entry_type,
            best_entry_level=best_level,
            best_entry_price=best_price,
            best_entry_prob=best_prob,
            cape_info=cape_info,
            rsi=asset.rsi,
            trend=asset.trend,
            price_change_1m=asset.price_change_1m,
            drawdown_from_ath=asset.drawdown_from_ath,
            ath_price=asset.ath_price,
            drawdown_from_52w=asset.drawdown_from_52w,
        )

    def _check_price_levels(self, asset_name: str, current_price: float,
                             report: MarketReport) -> list[dict]:
        """Controlla se il prezzo ha raggiunto dei livelli di ingresso dall'ultima analisi."""
        crossed = []

        # Recupera il prezzo precedente dallo stato
        prev_price = self._state.get(f"{asset_name}_price", 0)
        if prev_price == 0:
            return crossed

        # Trova l'asset nel report
        asset = None
        if "SWDA" in asset_name and report.swda:
            asset = report.swda
        elif "CSNDX" in asset_name and report.csndx:
            asset = report.csndx

        if not asset:
            return crossed

        # Controlla ogni livello: se il prezzo ha attraversato dall'alto
        for lvl in asset.entry_levels:
            if prev_price > lvl.price >= current_price:
                crossed.append({
                    "level": lvl.level,
                    "price": lvl.price,
                    "prob_90d": lvl.prob_90d,
                })
                logger.info(f"🎯 {asset_name} ha raggiunto il livello {lvl.level} ({lvl.price:.2f})")

        return crossed

    # ────────────────────────────────────────
    # NOTIFICHE
    # ────────────────────────────────────────

    def _send_notifications(self, signals: list[AccumulationSignal], report: MarketReport, market_news: list[dict] = None):
        """Decide quali notifiche inviare ed effettua l'invio su WhatsApp + Telegram."""

        now = datetime.now()

        for sig in signals:
            # 1. Alert per livelli di prezzo raggiunti (sempre, su entrambi i canali)
            for lvl in sig.crossed_levels:
                self.notifier.send_price_alert(
                    sig.asset_name, sig.currency, sig.current_price,
                    lvl["price"], lvl["level"]
                )
                if self.telegram.enabled:
                    self.telegram.send_price_alert(
                        sig.asset_name, sig.currency, sig.current_price,
                        lvl["price"], lvl["level"]
                    )

            # 2. Segnale COMPRA → notifica immediata (max 1 al giorno per asset)
            last_buy_key = f"{sig.asset_name}_last_buy_notify"
            last_buy_date = self._state.get(last_buy_key, "")
            today = now.strftime("%Y-%m-%d")

            if sig.action == "COMPRA" and last_buy_date != today:
                kwargs = dict(
                    asset_name=sig.asset_name,
                    action=sig.action,
                    price=sig.current_price,
                    currency=sig.currency,
                    score=sig.score,
                    entry_level=sig.best_entry_level,
                    entry_price=sig.best_entry_price,
                    probability=sig.best_entry_prob,
                    recommendation=sig.recommendation,
                    cape_info=sig.cape_info,
                    drawdown_from_ath=sig.drawdown_from_ath,
                    ath_price=sig.ath_price,
                    drawdown_from_52w=sig.drawdown_from_52w,
                )
                self.notifier.send_accumulation_signal(**kwargs)
                if self.telegram.enabled:
                    self.telegram.send_accumulation_signal(**kwargs)
                self._state[last_buy_key] = today

        # 3. Riepilogo giornaliero (una volta al giorno, dopo le 18:00)
        last_summary_date = self._state.get("last_daily_summary", "")
        if now.strftime("%Y-%m-%d") != last_summary_date and now.hour >= 18:
            assets_summary = []
            for sig in signals:
                assets_summary.append({
                    "name": sig.asset_name,
                    "currency": sig.currency,
                    "price": sig.current_price,
                    "score": sig.score,
                    "action": sig.action,
                    "recommendation": sig.recommendation,
                    "best_entry": bool(sig.best_entry_level),
                    "best_entry_price": sig.best_entry_price,
                    "best_entry_prob": sig.best_entry_prob,
                })
            self.notifier.send_daily_summary(assets_summary)
            if self.telegram.enabled:
                self.telegram.send_daily_accumulation_summary(assets_summary)
            self._state["last_daily_summary"] = now.strftime("%Y-%m-%d")

        # 4. News importanti (max 1 volta ogni 4 ore)
        if market_news:
            last_news_key = "last_news_notify"
            last_news_time = self._state.get(last_news_key, "")
            now_str = now.strftime("%Y-%m-%d %H")
            if last_news_time != now_str:
                news_msg = self._format_news_message(market_news)
                if news_msg:
                    if self.notifier.enabled:
                        self.notifier.send(news_msg)
                    if self.telegram.enabled:
                        self.telegram.send_sync(news_msg)
                    self._state[last_news_key] = now_str

    # ────────────────────────────────────────
    # NEWS IMPORTANTI
    # ────────────────────────────────────────

    def _fetch_important_news(self) -> list[dict]:
        """Recupera le news più importanti che possono muovere i mercati."""
        import yfinance as yf

        all_news = []
        seen = set()

        # Keywords che indicano notizie market-moving
        important_keywords = [
            "fed", "fomc", "interest rate", "inflation", "cpi", "gdp",
            "tariff", "trade war", "recession", "crash", "rally", "surge",
            "earnings", "layoff", "bankruptcy", "merger", "acquisition",
            "nvidia", "apple", "microsoft", "amazon", "google", "meta", "tesla",
            "nasdaq", "s&p", "dow", "market", "economy", "jobs", "unemployment",
            "bank", "treasury", "bond", "yield", "dollar", "euro",
            "war", "sanctions", "crisis", "default", "stimulus",
            "ai", "regulation", "antitrust", "supreme court",
        ]

        for ticker in ["^NDX", "^GSPC", "QQQ"]:
            try:
                t = yf.Ticker(ticker)
                news_data = t.news
                if not news_data:
                    continue

                for item in news_data[:8]:
                    content = item.get("content", {})
                    title = content.get("title", "")
                    summary = content.get("summary", "")

                    if not title or title in seen:
                        continue
                    seen.add(title)

                    # Filtra solo news importanti
                    text_lower = (title + " " + summary).lower()
                    is_important = any(kw in text_lower for kw in important_keywords)

                    if is_important:
                        pub_date = content.get("pubDate", "")
                        source = content.get("provider", {})
                        source_name = source.get("displayName", "") if isinstance(source, dict) else ""

                        all_news.append({
                            "title": title,
                            "summary": summary[:200],
                            "source": source_name,
                            "date": pub_date[:16] if pub_date else "",
                        })

            except Exception as e:
                logger.debug(f"Errore news {ticker}: {e}")

        # Max 5 news
        return all_news[:5]

    def _format_news_message(self, news: list[dict]) -> str:
        """Formatta le news in un messaggio per WhatsApp/Telegram."""
        if not news:
            return ""

        msg = "📰 *NEWS IMPORTANTI*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━\n"

        for n in news:
            msg += f"\n📌 *{n['title']}*\n"
            if n['summary']:
                msg += f"   {n['summary'][:150]}\n"
            if n['source']:
                msg += f"   _{n['source']}_\n"

        return msg

    def _load_state(self) -> dict:
        """Carica lo stato precedente dal file JSON."""
        try:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Impossibile caricare stato: {e}")
        return {}

    def _save_state(self, signals: list[AccumulationSignal]):
        """Salva lo stato corrente per la prossima analisi."""
        for sig in signals:
            self._state[f"{sig.asset_name}_price"] = sig.current_price
            self._state[f"{sig.asset_name}_score"] = sig.score
            self._state[f"{sig.asset_name}_action"] = sig.action

        self._state["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Impossibile salvare stato: {e}")

    # ────────────────────────────────────────
    # SCHEDULER
    # ────────────────────────────────────────

    def run_scheduled(self, interval_hours: int = 4):
        """
        Esegue il monitor su base periodica (durante orari di mercato).
        
        Il mercato europeo (Borsa Italiana) è aperto 9:00-17:30 CET.
        Le analisi vengono eseguite ogni `interval_hours` ore durante l'apertura,
        più un riepilogo dopo la chiusura (18:00).
        """
        import schedule
        import time as time_mod

        logger.info(f"Monitor accumulo schedulato ogni {interval_hours}h (orari mercato europeo)")
        logger.info("Ctrl+C per terminare")

        def job():
            now = datetime.now()
            hour = now.hour

            # Mercato europeo: 9:00 - 17:30 + riepilogo alle 18:00
            if 9 <= hour <= 18:
                logger.info(f"Esecuzione analisi schedulata ({now.strftime('%H:%M')})")
                try:
                    self.check_signals(notify=True)
                except Exception as e:
                    logger.error(f"Errore durante analisi schedulata: {e}")
                    if self.notifier.enabled:
                        self.notifier.send(f"⚠️ Errore monitor: {str(e)[:200]}")
            else:
                logger.debug(f"Fuori orario di mercato ({hour}:00), skip.")

        # Prima esecuzione immediata
        job()

        # Programma le successive
        schedule.every(interval_hours).hours.do(job)

        while True:
            schedule.run_pending()
            time_mod.sleep(60)  # Controlla ogni minuto
