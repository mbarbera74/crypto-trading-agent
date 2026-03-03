"""
AI Report Definitions - Prompt e data gathering per ogni tipo di report.

Ogni report ha:
- id: identificativo univoco
- name: nome visualizzato
- icon: emoji
- description: descrizione breve
- get_context(): funzione per raccogliere dati di mercato specifici
- get_prompt(context, today): funzione che genera il prompt per l'AI
"""

from datetime import datetime
from typing import Optional
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("analysis.ai_report_definitions")


# ══════════════════════════════════════════════════════════════
# HELPER: raccolta dati comuni
# ══════════════════════════════════════════════════════════════

def _fetch_ticker_data(ticker: str, name: str, period: str = "5d") -> str:
    """Recupera dati base per un ticker."""
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty or len(data) < 2:
            return f"  {name}: dati non disponibili"
        close = data["Close"].iloc[-1]
        prev = data["Close"].iloc[-2]
        change = (close / prev - 1) * 100
        return f"  {name} ({ticker}): {close:,.2f} ({change:+.2f}%)"
    except Exception:
        return f"  {name}: errore"


def _fetch_ticker_info(ticker: str) -> dict:
    """Recupera info fondamentali (P/E, market cap, etc.)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name": info.get("shortName", ticker),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "peg": info.get("pegRatio"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "pb": info.get("priceToBook"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# REPORT 1: Daily Market Recap
# ══════════════════════════════════════════════════════════════

REPORT_DAILY_MARKET = {
    "id": "daily_market",
    "name": "Riassunto Giornaliero Mercati",
    "icon": "📊",
    "description": "Analisi quotidiana: macro, politica, mercati USA, valute, BTC",
}


def get_daily_market_context() -> str:
    tickers = {
        "^GSPC": "S&P 500", "^NDX": "NASDAQ 100", "^DJI": "Dow Jones",
        "^VIX": "VIX", "^TNX": "Treasury 10Y", "DX-Y.NYB": "DXY",
        "GC=F": "Gold", "CL=F": "WTI Oil", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    }
    sectors = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
               "XLV": "Healthcare", "XLC": "Communication", "XLY": "Consumer Disc."}
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d')}"]
    for t, n in tickers.items():
        parts.append(_fetch_ticker_data(t, n))
    parts.append("Settori:")
    for t, n in sectors.items():
        parts.append(_fetch_ticker_data(t, n))

    # News
    news_parts = []
    for ticker in ["^GSPC", "^NDX", "BTC-USD"]:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news:
                for item in news[:3]:
                    title = item.get("content", {}).get("title", "")
                    if title:
                        news_parts.append(f"  - {title}")
        except Exception:
            pass
    if news_parts:
        parts.append("News recenti:")
        parts.extend(news_parts)
    return "\n".join(parts)


def get_daily_market_prompt(context: str, today: str) -> str:
    return f"""Sei un analista finanziario esperto specializzato nei mercati americani.
Oggi è {today}. I mercati americani hanno appena chiuso.

Dati di mercato:
{context}

Scrivi un report giornaliero approfondito IN ITALIANO con questa struttura:

## 📊 Riassunto Giornaliero dei Mercati — {today}

### 🏛️ Eventi Macroeconomici
Principali eventi macro del giorno (dati economici, Fed, BCE). Se non ce ne sono, commenta l'attesa.

### 🏛️ Eventi Politici
Eventi politici che influenzano i mercati (dazi, geopolitica, legislazione).

### 📈 Mercati Azionari USA
- Performance indici (S&P 500, NASDAQ 100, Dow Jones)
- Settori forti/deboli, titoli protagonisti
- VIX, sentiment

### 💱 Valute & Commodities
DXY, Treasury yields, Oro, Petrolio.

### ₿ Bitcoin & Crypto
Performance BTC/ETH, catalizzatori, correlazioni, livelli tecnici chiave.

### 🔮 Outlook per Domani
Livelli chiave, eventi in calendario, sentiment atteso.

Sii specifico, usa i dati numerici. Scrivi in modo professionale ma comprensibile."""


# ══════════════════════════════════════════════════════════════
# REPORT 2: XEON Risk Analysis
# ══════════════════════════════════════════════════════════════

REPORT_XEON_RISK = {
    "id": "xeon_risk",
    "name": "XEON Risk Analysis",
    "icon": "🛡️",
    "description": "Analisi rischi XEON ETF (swap, controparti, CDS Deutsche Bank, ECB policy)",
}


def get_xeon_context() -> str:
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d')}"]

    # XEON ETF price
    parts.append(_fetch_ticker_data("XEON.DE", "XEON ETF (Xetra)"))
    parts.append(_fetch_ticker_data("XEON.MI", "XEON ETF (Milano)"))

    # Deutsche Bank (controparte swap principale)
    parts.append(_fetch_ticker_data("DBK.DE", "Deutsche Bank (Xetra)"))
    parts.append(_fetch_ticker_data("DB", "Deutsche Bank (NYSE)"))

    # Altre controparti potenziali
    parts.append(_fetch_ticker_data("BNP.PA", "BNP Paribas"))
    parts.append(_fetch_ticker_data("SAN.PA", "Société Générale"))
    parts.append(_fetch_ticker_data("BARC.L", "Barclays"))
    parts.append(_fetch_ticker_data("GS", "Goldman Sachs"))
    parts.append(_fetch_ticker_data("JPM", "JPMorgan"))

    # Tasso BCE (proxy: ESTR via short-term bond yields)
    parts.append(_fetch_ticker_data("^TNX", "US Treasury 10Y"))
    parts.append(_fetch_ticker_data("IBGS.L", "iShares EUR Govt Bond 1-3Y"))

    # Indicatori stress finanziario
    parts.append(_fetch_ticker_data("^VIX", "VIX"))
    parts.append(_fetch_ticker_data("HYG", "High Yield Bond ETF"))
    parts.append(_fetch_ticker_data("LQD", "Inv. Grade Bond ETF"))

    # Credit default swap info (proxy via bank stock CDS-like indicators)
    # CDS non disponibili su yfinance, usiamo i subordinated bond ETF come proxy
    parts.append(_fetch_ticker_data("AT1.PA", "iShares EUR AT1 Capital Bond"))

    # Info fondamentali Deutsche Bank
    db_info = _fetch_ticker_info("DB")
    if db_info:
        parts.append(f"\nDeutsche Bank Fondamentali:")
        for k, v in db_info.items():
            if v is not None:
                parts.append(f"  {k}: {v}")

    return "\n".join(parts)


def get_xeon_prompt(context: str, today: str) -> str:
    return f"""Sei un analista di rischio finanziario specializzato in ETF sintetici e strumenti derivati, con competenza specifica sui CDS (Credit Default Swap) delle banche europee.

Oggi è {today}. Devi produrre il report giornaliero di rischio su XEON (Xtrackers II EUR Overnight Rate Swap UCITS ETF 1C).

Dati di mercato attuali:
{context}

CONOSCENZE SUL FONDO XEON:
- Emittente: DWS (Deutsche Bank subsidiary)
- Tipo: ETF sintetico con replica swap-based
- Benchmark: Solactive €STR +8.5 bp Daily Index (replica il tasso overnight BCE €STR)
- Controparte swap principale: Deutsche Bank AG
- Collaterale: titoli di stato e obbligazioni investment grade in un conto segregato
- AUM: ~13-15 miliardi EUR (uno dei più grandi ETF monetari EUR)
- Rischio principale: controparte swap (Deutsche Bank) e qualità del collaterale

Scrivi il report IN ITALIANO con questa struttura:

## 🛡️ Report Rischio XEON — {today}

### 📋 Stato del Fondo
NAV corrente, tracking rispetto a €STR, eventuali anomalie.

### 🏦 Analisi Controparte: Deutsche Bank
- Prezzo azione e trend recente
- CDS Deutsche Bank: valuta il livello attuale rispetto alla media storica (CDS 5Y senior era ~55-80 bps nel 2024, valuta se è salito/sceso)
- Rating creditizio Deutsche Bank (i principali: Moody's, S&P, Fitch)
- Indicatori di stress: subordinated debt spread, AT1 bond performance, stock volatility
- Confronto con le altre grandi banche europee (BNP, SocGen, Barclays)

### 🏦 Altre Controparti e Banche Coinvolte
Valuta brevemente lo stato delle altre potenziali controparti del fondo (Goldman Sachs, JPMorgan, BNP Paribas).

### 💶 Politica Monetaria BCE e Impatto
- Tasso €STR corrente e attese
- Impatto di eventuali tagli/rialzi tassi ECB sul rendimento del fondo
- Curva forward dei tassi BCE

### ⚠️ Fattori di Rischio
1. Rischio controparte swap
2. Rischio collaterale (qualità, haircut)
3. Rischio liquidità (AUM, bid-ask spread)
4. Rischio regolamentare (UCITS limits, ESMA)
5. Rischio sistemico (contagio bancario)

### 📊 Valutazione CDS Controparti
Per ciascuna controparte rilevante (Deutsche Bank, BNP, SocGen, Goldman, JPM):
- Stima livello CDS 5Y attuale (usa la tua conoscenza + proxy dai dati)
- Trend (in aumento, stabile, in calo)
- Confronto con livelli di allarme storici

### 🚦 VERDETTO: RISCHIO ATTUALE

Alla fine del report, dai un giudizio CHIARO e SINTETICO con uno di questi livelli:

🟢 **RISCHIO BASSO** — Mantieni posizione. Nessuna criticità rilevata.
🟡 **RISCHIO MODERATO** — Monitora attentamente. [specifica cosa]
🟠 **RISCHIO ELEVATO** — Considera riduzione parziale. [specifica perché]
🔴 **RISCHIO CRITICO** — DISINVESTI IMMEDIATAMENTE. [specifica perché]

Il verdetto deve essere basato su: CDS Deutsche Bank, stress finanziario settore bancario, policy BCE, e qualsiasi fattore che potrebbe impattare il NAV.

Sii specifico e basati sui dati. Non minimizzare i rischi."""


# ══════════════════════════════════════════════════════════════
# REPORT 3: Entry Strategy Comparison
# ══════════════════════════════════════════════════════════════

REPORT_ENTRY_STRATEGY = {
    "id": "entry_strategy",
    "name": "Strategia di Ingresso (CSNDX/SWDA/VWCE)",
    "icon": "🎯",
    "description": "Confronto 3 strategie di ingresso + simulazione storica + SWDA vs VWCE",
}


def get_entry_strategy_context() -> str:
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d')}"]

    # Prezzi attuali
    for t, n in [("CNDX.MI", "CSNDX"), ("SWDA.MI", "SWDA"), ("VWCE.MI", "VWCE"),
                 ("^NDX", "NASDAQ 100"), ("SWDA.L", "SWDA London")]:
        parts.append(_fetch_ticker_data(t, n))

    # Dati storici per simulazione (max drawdown, recovery)
    for ticker, name in [("SWDA.MI", "SWDA"), ("CNDX.MI", "CSNDX"), ("VWCE.MI", "VWCE")]:
        try:
            data = yf.Ticker(ticker).history(period="5y")
            if not data.empty and len(data) > 100:
                high = data["Close"].max()
                current = data["Close"].iloc[-1]
                dd = (current / high - 1) * 100
                perf_1y = (current / data["Close"].iloc[-252] - 1) * 100 if len(data) >= 252 else 0
                perf_3y = (current / data["Close"].iloc[-756] - 1) * 100 if len(data) >= 756 else 0
                parts.append(f"  {name} storico: max={high:.2f}, attuale={current:.2f}, "
                             f"DD da max={dd:.1f}%, perf 1Y={perf_1y:.1f}%, perf 3Y={perf_3y:.1f}%")

                # Conta quante volte è sceso del 5%, 10%, 15%, 20%
                rolling_max = data["Close"].cummax()
                drawdowns = (data["Close"] / rolling_max - 1) * 100
                for lvl in [5, 10, 15, 20]:
                    count = (drawdowns <= -lvl).sum()
                    parts.append(f"    Giorni con DD >= {lvl}%: {count} su {len(data)}")
        except Exception:
            pass

    # Indicatori macro
    parts.append(_fetch_ticker_data("^VIX", "VIX"))
    parts.append(_fetch_ticker_data("^TNX", "US 10Y Yield"))

    # CAPE
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get("https://www.multpl.com/shiller-pe", headers={
            "User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            cape_el = soup.find("div", {"id": "current"})
            if cape_el:
                import re
                match = re.search(r"([\d.]+)", cape_el.get_text())
                if match:
                    parts.append(f"  CAPE S&P 500: {match.group(1)}")
    except Exception:
        pass

    return "\n".join(parts)


def get_entry_strategy_prompt(context: str, today: str) -> str:
    return f"""Sei un consulente finanziario quantitativo specializzato in strategie di ingresso a tranche su ETF europei.

Oggi è {today}. Un investitore con €130.000 di capitale vuole entrare sui mercati azionari.

Dati attuali:
{context}

Analizza IN ITALIANO queste 3 strategie, considerando i prezzi ATTUALI e il contesto macro:

## 🎯 Confronto Strategie di Ingresso — {today}

### Strategia A: "CSNDX Anticipato + SWDA Progressivo"
- 30% (€39.000) su CSNDX.MI al primo ritracciamento del -5% dai massimi
- 35% (€45.500) su SWDA.MI se il mercato scende del -10%
- 35% (€45.500) su SWDA.MI se il mercato scende del -15%

### Strategia B: "All-in SWDA su Crollo"
- Attendi un calo del -15/-20% di SWDA
- Entra col 100% (€130.000) su SWDA.MI in un'unica soluzione

### Strategia C: "DCA Progressivo SWDA"
- 33% (€42.900) su SWDA.MI a -5% dai massimi
- 33% (€42.900) su SWDA.MI a -10%
- 34% (€44.200) su SWDA.MI a -15/-20%

Per ciascuna strategia:
1. **Probabilità di esecuzione**: quanto è probabile che il mercato raggiunga quei livelli (usa dati storici)
2. **Simulazione storica**: simula su dati reali degli ultimi 5-10 anni (2016-2026). Quante volte si sarebbero attivati i trigger? Con quale rendimento?
3. **Rischio di restare cash**: quanto costa aspettare troppo (mercato che sale senza ritracciare)
4. **Rendimento atteso a 3-5 anni** per ciascun scenario

### Tabella Comparativa
Crea una tabella con: Strategia | Prob. esecuzione completa | Rendimento storico medio | Rischio | Score

### Verdetto: Strategia Migliore
Quale strategia è OTTIMALE considerando il contesto attuale?

---

## 📊 SWDA vs VWCE: Quale Scegliere?

Rifai TUTTA l'analisi precedente sostituendo SWDA.MI con VWCE.MI (Vanguard FTSE All-World) e dimmi:
- Differenze di composizione (SWDA = MSCI World developed, VWCE = FTSE All-World inclusi emergenti)
- TER e tracking difference
- Performance storica comparata
- Diversificazione geografica
- **Verdetto finale: meglio SWDA o VWCE per questa strategia?**

Sii quantitativo, usa numeri reali dai dati forniti."""


# ══════════════════════════════════════════════════════════════
# REPORT 4: Valuation / Overvalued Analysis
# ══════════════════════════════════════════════════════════════

REPORT_VALUATION = {
    "id": "valuation",
    "name": "Valutazione NDX/SWDA & Fair Value Big Tech",
    "icon": "📉",
    "description": "P/E, fair value big tech, CAPE, liquidità vs AI, sopravvalutazione",
}


def get_valuation_context() -> str:
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d')}"]

    # Indici
    for t, n in [("^NDX", "NASDAQ 100"), ("^GSPC", "S&P 500"), ("SWDA.MI", "SWDA"), ("^VIX", "VIX")]:
        parts.append(_fetch_ticker_data(t, n))

    # Big Tech fondamentali
    big_tech = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "AMD"]
    parts.append("\nFondamentali Big Tech:")
    for ticker in big_tech:
        info = _fetch_ticker_info(ticker)
        if info:
            mc = info.get("market_cap")
            mc_str = f"${mc/1e12:.2f}T" if mc and mc > 1e12 else (f"${mc/1e9:.0f}B" if mc else "N/A")
            parts.append(
                f"  {info.get('name', ticker)}: P/E trailing={info.get('pe_trailing', 'N/A')}, "
                f"P/E forward={info.get('pe_forward', 'N/A')}, PEG={info.get('peg', 'N/A')}, "
                f"P/S={info.get('ps', 'N/A')}, MCap={mc_str}, "
                f"RevGrowth={info.get('revenue_growth', 'N/A')}, EarningsGrowth={info.get('earnings_growth', 'N/A')}"
            )

    # QQQ P/E (proxy NASDAQ 100)
    qqq_info = _fetch_ticker_info("QQQ")
    if qqq_info:
        parts.append(f"\nQQQ (NASDAQ 100 ETF): P/E={qqq_info.get('pe_trailing', 'N/A')}")

    # SPY P/E
    spy_info = _fetch_ticker_info("SPY")
    if spy_info:
        parts.append(f"SPY (S&P 500 ETF): P/E={spy_info.get('pe_trailing', 'N/A')}")

    # Liquidità macro
    parts.append(_fetch_ticker_data("HYG", "High Yield Bond"))
    parts.append(_fetch_ticker_data("LQD", "Inv. Grade Bond"))
    parts.append(_fetch_ticker_data("TLT", "20Y+ Treasury Bond"))
    parts.append(_fetch_ticker_data("DX-Y.NYB", "Dollar Index"))

    return "\n".join(parts)


def get_valuation_prompt(context: str, today: str) -> str:
    return f"""Sei un analista fondamentale senior con 25 anni di esperienza in valutazione aziendale e analisi dei multipli di mercato, specializzato nel settore tecnologico e AI.

Oggi è {today}.

Dati di mercato e fondamentali:
{context}

Scrivi un report approfondito IN ITALIANO:

## 📉 Analisi Sopravvalutazione NDX / SWDA — {today}

### 📊 Multipli Attuali degli Indici
- NASDAQ 100: P/E, P/S, CAPE stimato, confronto con media storica
- S&P 500: P/E, CAPE, confronto storico
- MSCI World (SWDA): stima P/E, confronto
- Sono sopravvalutati? Di quanto rispetto alla media storica?

### 🏢 Fair Value delle Big Tech (una per una)
Per ciascuna delle principali (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA):
- P/E trailing e forward attuale
- PEG Ratio (P/E vs crescita utili)
- P/S (Price/Sales)
- Revenue growth e earnings growth dall'ultima trimestrale
- **Fair Value stimato** (basato su DCF semplificato o P/E storico medio)
- **% sopravvalutazione/sottovalutazione** rispetto al prezzo corrente
- Risultati ultima trimestrale: hanno battuto o mancato le stime?

### 🤖 Bolla AI? Analisi Multipli vs Risultati
- I risultati delle aziende AI (NVDA, MSFT Azure AI, GOOGL Cloud, META AI, AMZN Bedrock) giustificano i multipli attuali?
- Capex AI delle big tech vs revenue AI effettivo: il ROI c'è?
- Confronto con la bolla dot-com: somiglianze e differenze nei multipli
- La liquidità di mercato sta alimentando i multipli indipendentemente dai fondamentali?

### 💧 Liquidità e Multipli
- La liquidità (M2, Fed balance sheet, credit conditions) sta sostenendo i multipli?
- Se la liquidità contrae, quale impatto sui multipli del NASDAQ?
- Relazione storica tra VIX, liquidità e P/E del NASDAQ

### ⚖️ Fair Value Indici
- **NASDAQ 100 fair value stimato**: basato su fair value aggregato delle top 10 + media storica P/E
- **MSCI World (SWDA) fair value stimato**
- **% scostamento dal prezzo attuale**

### 🚦 VERDETTO FINALE
Il NASDAQ 100 e SWDA sono:
🟢 **SOTTOVALUTATI** — buon punto di ingresso
🟡 **FAIR VALUE** — prezzo ragionevole
🟠 **MODERATAMENTE SOPRAVVALUTATI** — cautela
🔴 **SIGNIFICATIVAMENTE SOPRAVVALUTATI** — alto rischio di correzione

Sii rigoroso con i numeri. Usa i dati delle trimestrali reali."""


# ══════════════════════════════════════════════════════════════
# REPORT 5: COT Report Analysis
# ══════════════════════════════════════════════════════════════

REPORT_COT = {
    "id": "cot_analysis",
    "name": "COT Report (S&P, NASDAQ, BTC)",
    "icon": "📋",
    "description": "Commitments of Traders: posizionamento speculatori, commercial, sentiment",
}


def get_cot_context() -> str:
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d')}"]

    # Prezzi correnti dei futures sottostanti
    for t, n in [("^GSPC", "S&P 500"), ("^NDX", "NASDAQ 100"), ("ES=F", "E-mini S&P Futures"),
                 ("NQ=F", "E-mini NASDAQ Futures"), ("BTC-USD", "Bitcoin"),
                 ("BTC=F", "BTC CME Futures")]:
        parts.append(_fetch_ticker_data(t, n))

    # VIX e indicatori sentiment
    parts.append(_fetch_ticker_data("^VIX", "VIX"))

    # Performance recente per contestualizzare
    for ticker, name in [("^GSPC", "S&P 500"), ("^NDX", "NASDAQ 100"), ("BTC-USD", "Bitcoin")]:
        try:
            data = yf.Ticker(ticker).history(period="3mo")
            if not data.empty and len(data) >= 20:
                current = data["Close"].iloc[-1]
                one_week = data["Close"].iloc[-5] if len(data) >= 5 else current
                one_month = data["Close"].iloc[-22] if len(data) >= 22 else current
                parts.append(f"  {name} performance: 1W={((current/one_week)-1)*100:+.2f}%, "
                             f"1M={((current/one_month)-1)*100:+.2f}%")
        except Exception:
            pass

    parts.append("\nNOTA: I dati COT effettivi (CFTC) non sono disponibili via yfinance.")
    parts.append("Usa la tua conoscenza dei COT report più recenti disponibili.")
    parts.append("Indica SEMPRE la data dei dati COT che stai usando.")

    return "\n".join(parts)


def get_cot_prompt(context: str, today: str) -> str:
    return f"""Sei un analista macro/derivati esperto di COT report (Commitments of Traders della CFTC), con 15 anni di esperienza nell'interpretazione del posizionamento dei futures.

Oggi è {today}.

Dati di mercato correnti:
{context}

ISTRUZIONI CRITICHE:
- Usa i dati COT più recenti che conosci (i report CFTC escono il venerdì con dati del martedì precedente)
- INDICA SEMPRE LA DATA DEI DATI COT CHE USI
- Se non hai dati recenti, indicalo chiaramente e basa l'analisi sugli ultimi disponibili

Scrivi il report IN ITALIANO:

## 📋 Analisi COT Report — {today}

Per ciascuno dei tre mercati (S&P 500, NASDAQ-100, Bitcoin futures CME):

### 📊 S&P 500 Futures (E-mini, contratto consolidato)
**Data COT:** [indica la data]

1. **Posizioni attuali:**
   - Non-Commercial: Long, Short, Spreads
   - Commercial: Long, Short
   - Net Non-Commercial = Long − Short
   - Net Commercial = Long − Short

2. **Variazione settimanale:** delta net rispetto al COT precedente

3. **Contesto storico (6-12 mesi):**
   - Net attuali vs range 6-12 mesi (estremo rialzista / neutrale / estremo ribassista)
   - Eventuali inversioni recenti (da net short a net long o viceversa)

4. **Sentiment:** rialzista / ribassista / neutrale con spiegazione

---

### 📊 NASDAQ-100 Futures (E-mini, CME consolidato)
[Stessa struttura del S&P 500]

---

### ₿ Bitcoin Futures CME
[Stessa struttura, specifica se usi contratto standard o micro]

---

### 🔍 Divergenze COT vs Prezzo
- Segnala eventuali divergenze: prezzo che sale ma speculatori che riducono long (o viceversa)
- Queste divergenze sono storicamente predittive?

### 🚦 CONCLUSIONE OPERATIVA

Per ciascun mercato, esprimi un bias (NON un segnale di trading):

| Mercato | Bias COT | Conferma/Contraddice Trend Prezzo | Note |
|---------|----------|-----------------------------------|------|
| S&P 500 | 🟢/🟡/🔴 Rialzista/Neutrale/Ribassista | ... | ... |
| NASDAQ-100 | 🟢/🟡/🔴 | ... | ... |
| Bitcoin | 🟢/🟡/🔴 | ... | ... |

⚠️ Evidenzia situazioni di ESTREMO nel posizionamento che storicamente anticipano inversioni.

Sii sintetico, usa sezioni distinte, dati numerici quando disponibili."""


# ══════════════════════════════════════════════════════════════
# REPORT 6: Pre-Market Analysis
# ══════════════════════════════════════════════════════════════

REPORT_PREMARKET = {
    "id": "premarket",
    "name": "Analisi Pre-Apertura USA + BTC",
    "icon": "🔔",
    "description": "Analisi pre-market dettagliata con BUY/SELL/HOLD su titoli specifici + BTC",
}


def get_premarket_context() -> str:
    parts = [f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

    # Futures pre-market
    for t, n in [("NQ=F", "NASDAQ 100 Futures"), ("ES=F", "S&P 500 Futures"),
                 ("YM=F", "Dow Jones Futures"), ("RTY=F", "Russell 2000 Futures")]:
        parts.append(_fetch_ticker_data(t, n))

    # VIX
    parts.append(_fetch_ticker_data("^VIX", "VIX"))
    parts.append(_fetch_ticker_data("^VIX9D", "VIX 9D (short-term)"))

    # Chiusure precedenti indici
    parts.append("\nChiusure precedenti:")
    for t, n in [("^GSPC", "S&P 500"), ("^NDX", "NASDAQ 100"), ("^DJI", "Dow Jones")]:
        parts.append(_fetch_ticker_data(t, n))

    # Mercati europei e asiatici (già aperti)
    parts.append("\nMercati Europei/Asiatici:")
    for t, n in [("^STOXX50E", "Euro Stoxx 50"), ("^FTSE", "FTSE 100"),
                 ("^GDAXI", "DAX"), ("^N225", "Nikkei 225"),
                 ("^HSI", "Hang Seng"), ("000001.SS", "Shanghai Composite")]:
        parts.append(_fetch_ticker_data(t, n))

    # Valute, commodities, bond
    parts.append("\nValute & Commodities:")
    for t, n in [("DX-Y.NYB", "Dollar Index"), ("EURUSD=X", "EUR/USD"),
                 ("^TNX", "US 10Y Yield"), ("^TYX", "US 30Y Yield"),
                 ("GC=F", "Gold"), ("CL=F", "WTI Oil"), ("NG=F", "Natural Gas")]:
        parts.append(_fetch_ticker_data(t, n))

    # BTC e crypto
    parts.append("\nCrypto:")
    for t, n in [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"),
                 ("SOL-USD", "Solana"), ("BTC=F", "BTC CME Futures")]:
        parts.append(_fetch_ticker_data(t, n))

    # Big tech pre-market (fondamentali + prezzo)
    parts.append("\nBig Tech (fondamentali):")
    big_tech = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
                "CRM", "UBER", "COIN", "PLTR", "SNOW", "ARM"]
    for ticker in big_tech:
        info = _fetch_ticker_info(ticker)
        if info and info.get("price"):
            pe = info.get("pe_forward", "N/A")
            rev_g = info.get("revenue_growth")
            rev_str = f"{rev_g:.1%}" if rev_g else "N/A"
            parts.append(f"  {info.get('name', ticker)}: ${info['price']:,.2f}, "
                         f"P/E fwd={pe}, RevGrowth={rev_str}")

    # Titoli con maggior pre-market movement (proxy: earnings recenti, news)
    parts.append("\nNews recenti (per catalizzatori):")
    for ticker in ["^GSPC", "^NDX", "AAPL", "NVDA", "TSLA"]:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news:
                for item in news[:2]:
                    title = item.get("content", {}).get("title", "")
                    if title:
                        parts.append(f"  - {title}")
        except Exception:
            pass

    return "\n".join(parts)


def get_premarket_prompt(context: str, today: str) -> str:
    return f"""Sei un trader professionista e analista di mercato con 20 anni di esperienza, specializzato in analisi pre-market. Gestisci un portafoglio personale e devi dare raccomandazioni CONCRETE e SPECIFICHE.

Oggi è {today}. I mercati americani stanno per aprire.

Dati pre-market e overnight:
{context}

Scrivi il report IN ITALIANO con questa struttura:

## 🔔 Analisi Pre-Apertura Mercati USA — {today}

### 📡 Stato Futures & Overnight
- Futures S&P 500, NASDAQ 100, Dow Jones: direzione, gap atteso
- Cosa è successo overnight (Asia, Europa)
- VIX e sentiment pre-market

### 🌍 Contesto Globale di Stamattina
- Performance mercati asiatici ed europei
- Movimento valute (DXY, EUR/USD)
- Bond: Treasury 10Y e 30Y yield
- Commodities: oro, petrolio, gas

### 📰 Catalizzatori del Giorno
- Dati economici in uscita oggi
- Earnings in programma (pre-market e after-hours)
- Discorsi Fed/BCE
- News market-moving dalla notte

### 📈 Previsione Apertura
Previsione dettagliata su:
- Direzione attesa (gap up/down/flat)
- Settori forti e deboli attesi
- Livelli chiave intraday per S&P 500 e NASDAQ 100 (supporti/resistenze)

### 🎯 RACCOMANDAZIONI OPERATIVE

⚠️ QUESTA È LA SEZIONE PIÙ IMPORTANTE. Devi dare indicazioni CONCRETE con TICKER SPECIFICI.

#### Azione Generale: BUY / SELL / HOLD
Indica chiaramente se oggi è un giorno per:
- 🟢 **BUY** — comprare aggressivamente
- 🟡 **HOLD** — mantenere posizioni, non aggiungere
- 🔴 **SELL** — ridurre esposizione / prendere profitto

#### 📗 TITOLI DA COMPRARE (se BUY)
Per ogni titolo indica:
- **Ticker e nome** (es. NVDA - NVIDIA)
- **Prezzo corrente** e **target intraday/settimanale**
- **Perché comprare oggi** (catalizzatore specifico)
- **Stop loss suggerito**

Suggerisci 3-5 titoli specifici basandoti su:
- Momentum tecnico favorevole
- Catalizzatori imminenti (earnings, upgrade, news)
- Valutazione attraente nel breve
- Setup tecnico (breakout, rimbalzo da supporto)

#### 📕 TITOLI DA VENDERE / EVITARE (se SELL)
Per ogni titolo indica:
- **Ticker e nome**
- **Perché vendere/evitare** (sopravvalutato, catalizzatore negativo, rottura supporto)
- Eventuali **short opportunity**

Suggerisci 3-5 titoli specifici

#### ⚡ TRADE IDEAS DEL GIORNO
2-3 idee di trade specifiche per la sessione di oggi, con:
- Entry, Target, Stop Loss
- Ratio rischio/rendimento
- Timeframe (intraday, swing)

### ₿ ANALISI PRE-APERTURA BITCOIN

Sezione dedicata a BTC:
- **Prezzo corrente e variazione 24h**
- **Livelli tecnici chiave** (supporti/resistenze intraday)
- **Correlazione con futures USA**: BTC si muove nella stessa direzione?
- **Catalizzatori crypto del giorno** (ETF flows, news, on-chain)
- **Raccomandazione BTC**: BUY / SELL / HOLD con motivazione
- **Target intraday BTC** (livello rialzista e ribassista)

---
DISCLAIMER: Questo è un report di analisi, non consulenza finanziaria.

Sii SPECIFICO con i ticker, i prezzi e i livelli. Non essere generico. I trade ideas devono essere actionable."""


# ══════════════════════════════════════════════════════════════
# REGISTRO REPORT
# ══════════════════════════════════════════════════════════════

AI_REPORTS = {
    "premarket": {
        **REPORT_PREMARKET,
        "get_context": get_premarket_context,
        "get_prompt": get_premarket_prompt,
    },
    "daily_market": {
        **REPORT_DAILY_MARKET,
        "get_context": get_daily_market_context,
        "get_prompt": get_daily_market_prompt,
    },
    "xeon_risk": {
        **REPORT_XEON_RISK,
        "get_context": get_xeon_context,
        "get_prompt": get_xeon_prompt,
    },
    "entry_strategy": {
        **REPORT_ENTRY_STRATEGY,
        "get_context": get_entry_strategy_context,
        "get_prompt": get_entry_strategy_prompt,
    },
    "valuation": {
        **REPORT_VALUATION,
        "get_context": get_valuation_context,
        "get_prompt": get_valuation_prompt,
    },
    "cot_analysis": {
        **REPORT_COT,
        "get_context": get_cot_context,
        "get_prompt": get_cot_prompt,
    },
}
