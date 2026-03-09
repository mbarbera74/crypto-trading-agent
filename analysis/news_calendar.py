"""
News & Calendar - Calendario utili trimestrali, dati macro e news di mercato.

Recupera da Yahoo Finance:
1. Prossimi earnings delle principali aziende (Magnificent 7 + top NASDAQ)
2. News di mercato rilevanti per NASDAQ 100, MSCI World, S&P 500
3. Calendario macro: Fed meetings, CPI, NFP, GDP (date note + dati recenti)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd

from utils.logger import get_logger

logger = get_logger("analysis.news_calendar")

# Aziende monitorate per earnings (top NASDAQ 100 per peso)
EARNINGS_TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta (Facebook)",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "COST": "Costco",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
}


@dataclass
class EarningsEvent:
    """Un singolo evento di earnings."""
    ticker: str
    company: str
    date: datetime
    eps_estimate: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None
    reported_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    revenue_estimate: Optional[float] = None   # in dollari
    reported_revenue: Optional[float] = None   # in dollari
    revenue_surprise_pct: Optional[float] = None  # % sorpresa revenue
    eps_growth: Optional[float] = None          # % crescita YoY
    num_analysts: Optional[int] = None
    is_upcoming: bool = True


@dataclass
class MacroEvent:
    """Un evento macroeconomico."""
    name: str
    date: str           # Data o intervallo
    description: str
    importance: str      # ALTA, MEDIA, BASSA
    category: str        # FED, CPI, NFP, GDP, PMI, etc.
    latest_value: str = ""


@dataclass
class NewsItem:
    """Una notizia di mercato."""
    title: str
    summary: str
    source: str
    date: str
    url: str


@dataclass
class EconomicCalendarEvent:
    """Evento dal calendario economico investing.com."""
    date: str
    time: str
    country: str
    currency: str
    importance: str        # high, medium, low
    event: str
    actual: str = ""
    forecast: str = ""
    previous: str = ""


@dataclass
class MarketCalendar:
    """Calendario completo di mercato."""
    upcoming_earnings: list[EarningsEvent] = field(default_factory=list)
    recent_earnings: list[EarningsEvent] = field(default_factory=list)
    macro_events: list[MacroEvent] = field(default_factory=list)
    economic_calendar: list[EconomicCalendarEvent] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    last_updated: str = ""


class NewsCalendarProvider:
    """Recupera earnings, news e calendario macro."""

    def __init__(self):
        pass

    def get_full_calendar(self) -> MarketCalendar:
        """Recupera il calendario completo."""
        cal = MarketCalendar(last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"))

        logger.info("Recupero calendario earnings...")
        cal.upcoming_earnings, cal.recent_earnings = self._fetch_earnings()

        logger.info("Generazione calendario macro...")
        cal.macro_events = self._get_macro_calendar()

        logger.info("Recupero calendario economico da investing.com...")
        cal.economic_calendar = self._fetch_economic_calendar()

        logger.info("Recupero news di mercato...")
        cal.news = self._fetch_news()

        logger.info(f"Calendario completato: {len(cal.upcoming_earnings)} earnings futuri, "
                    f"{len(cal.recent_earnings)} recenti, {len(cal.macro_events)} eventi macro, "
                    f"{len(cal.economic_calendar)} eventi economici, {len(cal.news)} news")

        return cal

    # ────────────────────────────────────────
    # EARNINGS
    # ────────────────────────────────────────

    def _fetch_earnings(self) -> tuple[list[EarningsEvent], list[EarningsEvent]]:
        """Recupera le date degli utili per le principali aziende NASDAQ."""
        upcoming = []
        recent = []
        now = datetime.now()

        for ticker, company in EARNINGS_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                ed = t.earnings_dates
                if ed is None or ed.empty:
                    continue

                # Recupera stime aggiuntive (revenue, crescita, range EPS)
                extra = {}
                try:
                    cal = t.calendar
                    if cal:
                        extra["revenue_avg"] = cal.get("Revenue Average")
                        extra["eps_low"] = cal.get("Earnings Low")
                        extra["eps_high"] = cal.get("Earnings High")
                except Exception:
                    pass

                try:
                    ee = t.earnings_estimate
                    if ee is not None and not ee.empty:
                        row0 = ee.iloc[0]  # prossimo trimestre
                        extra["growth"] = float(row0.get("growth")) if pd.notna(row0.get("growth")) else None
                        extra["num_analysts"] = int(row0.get("numberOfAnalysts")) if pd.notna(row0.get("numberOfAnalysts")) else None
                except Exception:
                    pass

                try:
                    re_est = t.revenue_estimate
                    if re_est is not None and not re_est.empty:
                        rev_avg = re_est.iloc[0].get("avg")
                        if pd.notna(rev_avg):
                            extra["revenue_avg"] = float(rev_avg)
                except Exception:
                    pass

                # Recupera revenue effettivo dai quarterly financials (per earnings recenti)
                actual_revenues = {}
                try:
                    qf = t.quarterly_financials
                    if qf is not None and not qf.empty and "Total Revenue" in qf.index:
                        for col in qf.columns:
                            # col è un Timestamp del quarter end date
                            actual_revenues[col] = float(qf.loc["Total Revenue", col])
                except Exception:
                    pass

                for dt_idx, row in ed.iterrows():
                    # Converti il timestamp (con timezone) in datetime naive
                    if hasattr(dt_idx, 'tz_localize'):
                        dt = dt_idx.to_pydatetime().replace(tzinfo=None)
                    else:
                        dt = pd.Timestamp(dt_idx).to_pydatetime().replace(tzinfo=None)

                    eps_est = row.get("EPS Estimate")
                    eps_rep = row.get("Reported EPS")
                    surprise = row.get("Surprise(%)")

                    event = EarningsEvent(
                        ticker=ticker,
                        company=company,
                        date=dt,
                        eps_estimate=float(eps_est) if pd.notna(eps_est) else None,
                        eps_low=extra.get("eps_low"),
                        eps_high=extra.get("eps_high"),
                        reported_eps=float(eps_rep) if pd.notna(eps_rep) else None,
                        surprise_pct=float(surprise) if pd.notna(surprise) else None,
                        revenue_estimate=extra.get("revenue_avg"),
                        eps_growth=extra.get("growth"),
                        num_analysts=extra.get("num_analysts"),
                    )

                    # Per earnings recenti: cerca il revenue effettivo più vicino
                    if dt < now and actual_revenues:
                        # Trova il quarter più vicino alla data earnings
                        closest_q = min(actual_revenues.keys(),
                                        key=lambda q: abs((q.to_pydatetime().replace(tzinfo=None) - dt).days))
                        days_diff = abs((closest_q.to_pydatetime().replace(tzinfo=None) - dt).days)
                        if days_diff < 60:  # Entro 60 giorni
                            event.reported_revenue = actual_revenues[closest_q]
                            # Calcola revenue surprise se abbiamo la stima
                            if event.revenue_estimate and event.reported_revenue:
                                event.revenue_surprise_pct = (
                                    (event.reported_revenue / event.revenue_estimate - 1) * 100
                                )

                    if dt > now:
                        event.is_upcoming = True
                        upcoming.append(event)
                    elif dt > now - timedelta(days=45):
                        event.is_upcoming = False
                        recent.append(event)

            except Exception as e:
                logger.debug(f"Errore earnings {ticker}: {e}")

        # Ordina per data
        upcoming.sort(key=lambda x: x.date)
        recent.sort(key=lambda x: x.date, reverse=True)

        return upcoming, recent

    # ────────────────────────────────────────
    # MACRO CALENDAR
    # ────────────────────────────────────────

    def _get_macro_calendar(self) -> list[MacroEvent]:
        """
        Genera calendario eventi macro basato su date ricorrenti
        e valori recenti degli indicatori.
        """
        events = []
        now = datetime.now()
        year = now.year
        month = now.month

        # ── FEDERAL RESERVE MEETINGS (FOMC) 2026 ──
        fomc_dates_2026 = [
            ("28-29 Gennaio", "2026-01-29"),
            ("17-18 Marzo", "2026-03-18"),
            ("5-6 Maggio", "2026-05-06"),
            ("16-17 Giugno", "2026-06-17"),
            ("28-29 Luglio", "2026-07-29"),
            ("15-16 Settembre", "2026-09-16"),
            ("27-28 Ottobre", "2026-10-28"),
            ("15-16 Dicembre", "2026-12-16"),
        ]
        for label, date_str in fomc_dates_2026:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt > now - timedelta(days=7):  # Mostra recenti e futuri
                status = "✅ Passato" if dt < now else "📅 Prossimo"
                events.append(MacroEvent(
                    name=f"FOMC Meeting ({label})",
                    date=date_str,
                    description=f"Decisione tassi di interesse Fed. {status}",
                    importance="ALTA",
                    category="FED",
                ))

        # ── CPI (Consumer Price Index) - pubblicato ~12-13 di ogni mese ──
        for m in range(month, min(month + 4, 13)):
            cpi_date = f"{year}-{m:02d}-12"
            events.append(MacroEvent(
                name=f"CPI (Inflazione) - {datetime(year, m, 1).strftime('%B')}",
                date=cpi_date,
                description="Consumer Price Index USA. Impatto diretto su decisioni Fed e mercati.",
                importance="ALTA",
                category="CPI",
            ))

        # ── NFP (Non-Farm Payrolls) - primo venerdì di ogni mese ──
        for m in range(month, min(month + 4, 13)):
            # Calcola primo venerdì
            first_day = datetime(year, m, 1)
            days_until_fri = (4 - first_day.weekday()) % 7
            nfp_date = first_day + timedelta(days=days_until_fri)
            events.append(MacroEvent(
                name=f"NFP (Occupazione) - {datetime(year, m, 1).strftime('%B')}",
                date=nfp_date.strftime("%Y-%m-%d"),
                description="Non-Farm Payrolls USA. Indicatore chiave del mercato del lavoro.",
                importance="ALTA",
                category="NFP",
            ))

        # ── GDP (PIL USA) - fine mese (stima anticipata) ──
        gdp_months = [1, 4, 7, 10]  # Trimestrale
        for m in gdp_months:
            if m >= month or (m == 1 and month == 12):
                events.append(MacroEvent(
                    name=f"GDP USA (Stima) - Q{(m-1)//3 + 1}",
                    date=f"{year}-{m:02d}-28",
                    description="Prodotto Interno Lordo USA (stima anticipata trimestrale).",
                    importance="ALTA",
                    category="GDP",
                ))

        # ── PMI (Purchasing Managers Index) - primo giorno lavorativo del mese ──
        for m in range(month, min(month + 3, 13)):
            events.append(MacroEvent(
                name=f"PMI Manifatturiero - {datetime(year, m, 1).strftime('%B')}",
                date=f"{year}-{m:02d}-01",
                description="ISM Manufacturing PMI. Sopra 50 = espansione, sotto 50 = contrazione.",
                importance="MEDIA",
                category="PMI",
            ))

        # ── BCE (European Central Bank) ──
        bce_dates_2026 = [
            ("22 Gennaio", "2026-01-22"),
            ("5 Marzo", "2026-03-05"),
            ("16 Aprile", "2026-04-16"),
            ("5 Giugno", "2026-06-05"),
            ("17 Luglio", "2026-07-17"),
            ("11 Settembre", "2026-09-11"),
            ("23 Ottobre", "2026-10-23"),
            ("4 Dicembre", "2026-12-04"),
        ]
        for label, date_str in bce_dates_2026:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt > now - timedelta(days=7):
                status = "✅ Passato" if dt < now else "📅 Prossimo"
                events.append(MacroEvent(
                    name=f"BCE Tassi ({label})",
                    date=date_str,
                    description=f"Decisione tassi BCE. Impatto su EUR e mercati europei. {status}",
                    importance="ALTA",
                    category="BCE",
                ))

        # ── Valori recenti indicatori macro ──
        try:
            # VIX
            vix = yf.Ticker("^VIX").history(period="5d")
            if not vix.empty:
                events.append(MacroEvent(
                    name="VIX (Fear Index)",
                    date=now.strftime("%Y-%m-%d"),
                    description="Indice di volatilità. <20 = calmo, 20-30 = nervoso, >30 = panico.",
                    importance="MEDIA",
                    category="VOLATILITA",
                    latest_value=f"{vix['Close'].iloc[-1]:.1f}",
                ))

            # Treasury 10Y
            tnx = yf.Ticker("^TNX").history(period="5d")
            if not tnx.empty:
                events.append(MacroEvent(
                    name="Treasury 10Y (Rendimento)",
                    date=now.strftime("%Y-%m-%d"),
                    description="Rendimento del titolo di stato USA a 10 anni.",
                    importance="MEDIA",
                    category="BOND",
                    latest_value=f"{tnx['Close'].iloc[-1]:.2f}%",
                ))

            # DXY (Dollar Index)
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
            if not dxy.empty:
                events.append(MacroEvent(
                    name="US Dollar Index (DXY)",
                    date=now.strftime("%Y-%m-%d"),
                    description="Forza del dollaro. Dollaro forte = pressione su mercati emergenti e commodities.",
                    importance="MEDIA",
                    category="FOREX",
                    latest_value=f"{dxy['Close'].iloc[-1]:.1f}",
                ))

        except Exception as e:
            logger.warning(f"Errore recupero indicatori macro: {e}")

        # Ordina per data
        events.sort(key=lambda x: x.date)

        return events

    # ────────────────────────────────────────
    # CALENDARIO ECONOMICO (investing.com)
    # ────────────────────────────────────────

    def _fetch_economic_calendar(self) -> list[EconomicCalendarEvent]:
        """
        Recupera il calendario economico da investing.com via investpy.
        Include tutti i dati macro del giorno e della settimana corrente.
        Ha un timeout per evitare blocchi se investing.com non risponde.
        """
        events = []
        try:
            import investpy
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            now = datetime.now()
            from_date = now.strftime("%d/%m/%Y")
            to_date = (now + timedelta(days=7)).strftime("%d/%m/%Y")

            def _fetch():
                return investpy.economic_calendar(
                    time_zone="GMT +1:00",
                    countries=["united states", "euro zone", "germany", "italy",
                               "united kingdom", "japan", "china"],
                    from_date=from_date,
                    to_date=to_date,
                )

            # Timeout di 15 secondi per evitare blocchi
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch)
                try:
                    cal = future.result(timeout=15)
                except FuturesTimeout:
                    logger.warning("Timeout recupero calendario economico da investing.com (15s)")
                    return events

            if cal is not None and not cal.empty:
                for _, row in cal.iterrows():
                    imp = str(row.get("importance", "")).strip()
                    if imp == "high":
                        importance = "🔴 ALTA"
                    elif imp == "medium":
                        importance = "🟡 MEDIA"
                    else:
                        importance = "🟢 BASSA"

                    actual = str(row.get("actual", "")) if row.get("actual") else ""
                    forecast = str(row.get("forecast", "")) if row.get("forecast") else ""
                    previous = str(row.get("previous", "")) if row.get("previous") else ""

                    events.append(EconomicCalendarEvent(
                        date=str(row.get("date", "")),
                        time=str(row.get("time", "")),
                        country=str(row.get("zone", "")),
                        currency=str(row.get("currency", "")),
                        importance=importance,
                        event=str(row.get("event", "")),
                        actual=actual if actual != "None" else "",
                        forecast=forecast if forecast != "None" else "",
                        previous=previous if previous != "None" else "",
                    ))

                logger.info(f"Recuperati {len(events)} eventi economici da investing.com")

        except ImportError:
            logger.warning("investpy non installato. Installa con: pip install investpy")
        except Exception as e:
            logger.warning(f"Errore recupero calendario economico: {e}")

        return events

    # ────────────────────────────────────────
    # NEWS
    # ────────────────────────────────────────

    def _fetch_news(self) -> list[NewsItem]:
        """Recupera le news più recenti da Yahoo Finance per NASDAQ 100 e S&P 500."""
        all_news = []
        seen_titles = set()

        for ticker in ["^NDX", "^GSPC", "QQQ", "SWDA.MI"]:
            try:
                t = yf.Ticker(ticker)
                news_data = t.news
                if not news_data:
                    continue

                for item in news_data[:5]:
                    content = item.get("content", {})
                    title = content.get("title", "")

                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    # Estrai data
                    pub_date = content.get("pubDate", "")
                    if pub_date:
                        try:
                            dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                            date_str = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            date_str = pub_date[:16]
                    else:
                        date_str = ""

                    # Estrai URL
                    click_url = content.get("clickThroughUrl", {})
                    url = click_url.get("url", "") if isinstance(click_url, dict) else ""

                    # Estrai source
                    provider = content.get("provider", {})
                    source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"

                    all_news.append(NewsItem(
                        title=title,
                        summary=content.get("summary", "")[:300],
                        source=source,
                        date=date_str,
                        url=url,
                    ))

            except Exception as e:
                logger.debug(f"Errore news {ticker}: {e}")

        # Aggiungi news da CNN Business
        cnn_news = self._fetch_cnn_news()
        for n in cnn_news:
            if n.title not in seen_titles:
                seen_titles.add(n.title)
                all_news.append(n)

        # Aggiungi news da Reuters (breaking news, più rapide)
        reuters_news = self._fetch_reuters_news()
        for n in reuters_news:
            if n.title not in seen_titles:
                seen_titles.add(n.title)
                all_news.append(n)

        # Ordina per data (più recenti prima)
        all_news.sort(key=lambda x: x.date, reverse=True)

        return all_news  # Tutte le news, filtrate nella dashboard

    def _fetch_cnn_news(self) -> list[NewsItem]:
        """
        Recupera le news market-moving da CNN Business/Economy via scraping diretto.
        Scrape le pagine: edition.cnn.com/business e edition.cnn.com/economy.
        """
        import requests
        from bs4 import BeautifulSoup

        cnn_news = []
        seen_urls = set()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

        # Scraping diretto delle pagine CNN Business ed Economy
        pages = [
            "https://edition.cnn.com/business",
            "https://edition.cnn.com/economy",
        ]

        # Keyword per filtrare solo news market-moving
        market_keywords = [
            "market", "stock", "wall street", "fed", "rate",
            "inflation", "economy", "recession", "gdp", "jobs",
            "nasdaq", "dow", "s&p", "earnings", "trade", "tariff",
            "bitcoin", "crypto", "bank", "treasury", "oil", "gold",
            "dollar", "investor", "rally", "sell", "crash", "surge",
            "trump", "congress", "deficit", "debt", "tax",
            "ai", "tech", "nvidia", "apple", "microsoft", "amazon",
            "tesla", "meta", "google", "regulation", "dimon",
            "supreme court", "china", "sanction", "war",
        ]

        for page_url in pages:
            try:
                response = requests.get(page_url, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                # Trova tutti i link con href che puntano ad articoli
                links = soup.find_all("a", href=True)

                for link_tag in links:
                    href = link_tag.get("href", "")
                    text = link_tag.get_text(strip=True)

                    # Filtra: solo articoli con titolo significativo
                    if not text or len(text) < 25:
                        continue

                    # Deve essere un articolo business/economy con data nell'URL (es. /2026/02/24/)
                    if not any(seg in href for seg in ["/business/", "/economy/", "/markets/"]):
                        continue

                    # Costruisci URL completo
                    full_url = href if href.startswith("http") else f"https://edition.cnn.com{href}"

                    # Evita duplicati
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)

                    # Filtra per rilevanza di mercato
                    text_lower = text.lower()
                    is_relevant = any(kw in text_lower for kw in market_keywords)

                    # Estrai data dall'URL se possibile (es. /2026/02/24/)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    import re
                    date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", full_url)
                    if date_match:
                        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

                    # Scarta titoli che sono solo didascalie di immagini o spazzatura
                    if text.endswith("Images") or text.endswith("REUTERS") or "/" in text[:20]:
                        continue
                    if text.startswith("-") or text.startswith("—"):
                        continue

                    cnn_news.append(NewsItem(
                        title=text[:200],
                        summary="",
                        source="CNN Business" if is_relevant else "CNN",
                        date=date_str,
                        url=full_url,
                    ))

                    if len(cnn_news) >= 15:
                        break

            except Exception as e:
                logger.debug(f"Errore scraping CNN {page_url}: {e}")

        logger.info(f"Recuperate {len(cnn_news)} news da CNN Business")
        return cnn_news

    def _fetch_reuters_news(self) -> list[NewsItem]:
        """
        Recupera breaking news da fonti finanziarie via Google News RSS.
        Fonti: Reuters, Bloomberg, Financial Times, Investing.com,
        Walter Bloomberg (@DeItaone), First Squawk, Kobeissi Letter.
        """
        import requests
        from bs4 import BeautifulSoup

        reuters_news = []
        seen_titles = set()

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        }

        # Query Google News RSS per news market-moving da tutte le fonti
        _kw_markets = "markets+OR+oil+OR+fed+OR+economy+OR+stocks+OR+blackrock+OR+bitcoin+OR+treasury+OR+tariff+OR+opec"
        _kw_finance = "bank+OR+fund+OR+withdrawal+OR+crisis+OR+crash+OR+inflation+OR+recession+OR+gold+OR+dollar"
        _kw_geo = "hormuz+OR+war+OR+missile+OR+geopolit+OR+nuclear+OR+sanctions+OR+iran+OR+china+OR+russia"

        rss_feeds = [
            # Reuters
            (f"https://news.google.com/rss/search?q=site:reuters.com+{_kw_markets}&hl=en-US&gl=US&ceid=US:en", "Reuters"),
            (f"https://news.google.com/rss/search?q=site:reuters.com+{_kw_finance}&hl=en-US&gl=US&ceid=US:en", "Reuters"),
            (f"https://news.google.com/rss/search?q=site:reuters.com+{_kw_geo}&hl=en-US&gl=US&ceid=US:en", "Reuters Breaking"),
            # Bloomberg
            (f"https://news.google.com/rss/search?q=site:bloomberg.com+{_kw_markets}&hl=en-US&gl=US&ceid=US:en", "Bloomberg"),
            (f"https://news.google.com/rss/search?q=site:bloomberg.com+{_kw_finance}+OR+{_kw_geo}&hl=en-US&gl=US&ceid=US:en", "Bloomberg"),
            # Financial Times
            (f"https://news.google.com/rss/search?q=site:ft.com+{_kw_markets}+OR+{_kw_geo}&hl=en-US&gl=US&ceid=US:en", "Financial Times"),
            # Investing.com
            (f"https://news.google.com/rss/search?q=site:investing.com+{_kw_markets}+OR+breaking&hl=en-US&gl=US&ceid=US:en", "Investing.com"),
            # Walter Bloomberg (@DeItaone) + First Squawk + Kobeissi via aggregatori
            ("https://news.google.com/rss/search?q=from:DeItaone+markets+OR+breaking+OR+stocks&hl=en-US&gl=US&ceid=US:en", "Walter Bloomberg"),
            ("https://news.google.com/rss/search?q=from:FirstSquawk+breaking+OR+markets+OR+fed&hl=en-US&gl=US&ceid=US:en", "First Squawk"),
            ("https://news.google.com/rss/search?q=KobeissiLetter+markets+OR+stocks+OR+breaking&hl=en-US&gl=US&ceid=US:en", "Kobeissi Letter"),
            # Trump Truth Social (post market-moving riportati dalla stampa)
            ("https://news.google.com/rss/search?q=Trump+Truth+Social+post+markets+OR+tariff+OR+iran+OR+stocks+OR+economy+OR+oil&hl=en-US&gl=US&ceid=US:en", "Trump Truth"),
            ("https://news.google.com/rss/search?q=Trump+says+OR+Trump+announces+tariff+OR+iran+OR+economy+OR+oil+OR+markets&hl=en-US&gl=US&ceid=US:en", "Trump"),
            # Trump direct feed via Nitter (real-time, <1 min delay)
            ("https://nitter.net/realDonaldTrump/rss", "Trump (X/Truth)"),
        ]

        for rss_url, source_label in rss_feeds:
            try:
                response = requests.get(rss_url, headers=headers, timeout=15)
                if response.status_code != 200:
                    logger.debug(f"Google News RSS: HTTP {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "xml")
                items = soup.find_all("item")

                for item in items[:15]:
                    title_tag = item.find("title")
                    link_tag = item.find("link")
                    pub_tag = item.find("pubDate")
                    source_tag = item.find("source")

                    if not title_tag:
                        continue

                    raw_title = title_tag.get_text(strip=True)

                    # Pulisci titolo: rimuovi suffissi fonte comune
                    title = raw_title
                    for _sfx in [" - Reuters", " - Bloomberg.com", " - Financial Times",
                                 " - Investing.com", " - Forex Factory", " - Yahoo Finance",
                                 " - CNBC", " - MarketWatch"]:
                        if title.endswith(_sfx):
                            title = title[:-len(_sfx)].strip()
                            break

                    if title in seen_titles or len(title) < 15:
                        continue
                    seen_titles.add(title)

                    # Determina fonte reale dal tag <source> o dal suffisso
                    actual_source = source_label
                    if source_tag:
                        src_name = source_tag.get_text(strip=True)
                        if src_name in ("Reuters", "Bloomberg.com", "Financial Times",
                                        "Investing.com", "Forex Factory", "CNBC"):
                            actual_source = src_name.replace(".com", "")

                    # URL (Google News redirect, ma contiene l'articolo)
                    url = link_tag.get_text(strip=True) if link_tag else ""

                    # Data di pubblicazione
                    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    if pub_tag:
                        try:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(pub_tag.get_text(strip=True))
                            date_str = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            pass

                    # Sommario dalla description
                    desc_tag = item.find("description")
                    summary = ""
                    if desc_tag:
                        desc_soup = BeautifulSoup(desc_tag.get_text(), "html.parser")
                        summary = desc_soup.get_text(strip=True)[:300]

                    reuters_news.append(NewsItem(
                        title=title,
                        summary=summary,
                        source=actual_source,
                        date=date_str,
                        url=url,
                    ))

            except Exception as e:
                logger.debug(f"Errore Reuters RSS feed: {e}")

        logger.info(f"Recuperate {len(reuters_news)} news da Reuters")
        return reuters_news
