"""
Crypto Trading Agent - Punto di ingresso principale.

Utilizzo:
    python main.py backtest                  # Esegui backtest
    python main.py backtest --ml             # Backtest con ML
    python main.py live                      # Trading live (sandbox)
    python main.py live --no-ml              # Trading live senza ML
    python main.py dashboard                 # Avvia dashboard Streamlit
    python main.py analyze                   # Analisi segnale corrente
"""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def cmd_backtest(args):
    """Esegue il backtest della strategia su crypto o asset tradizionali."""
    from strategy.momentum import MomentumStrategy
    from engine.backtester import Backtester
    from config.settings import config

    asset = getattr(args, "asset", "btc").lower()

    # Mappa asset → configurazione
    ASSET_CONFIG = {
        "btc": {"name": "BTC/USDT", "currency": "$", "long_only": False, "commission": 0.001},
        "ndx": {"name": "NASDAQ 100 (^NDX)", "currency": "$", "long_only": False, "commission": 0.0005,
                "ticker": "^NDX", "fallbacks": ["QQQ"]},
        "swda": {"name": "SWDA.MI (iShares MSCI World)", "currency": "€", "long_only": False, "commission": 0.001,
                 "ticker": "SWDA.MI", "fallbacks": ["SWDA.L", "IWDA.AS"]},
        "csndx": {"name": "CSNDX (iShares NASDAQ 100)", "currency": "€", "long_only": False, "commission": 0.001,
                  "ticker": "CNDX.MI", "fallbacks": ["CSNDX.MI", "SXRV.DE"]},
    }

    if asset not in ASSET_CONFIG:
        console.print(f"[red]Asset '{asset}' non supportato. Usa: btc, ndx, swda, csndx[/red]")
        return

    cfg = ASSET_CONFIG[asset]
    console.print(Panel(
        f"🧪 [bold cyan]BACKTEST - {cfg['name']}[/bold cyan]\n"
        f"   Modalità: [bold]{'Long Only' if cfg['long_only'] else 'Long + Short'}[/bold]  |  "
        f"Commissione: {cfg['commission']*100:.2f}%  |  Valuta: {cfg['currency']}",
        style="bold",
    ))

    # Fetch dati
    if asset == "btc":
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        console.print(f"Scaricamento dati {config.trading.symbol} [{config.trading.timeframe}] "
                      f"per {args.days} giorni...")
        df = fetcher.fetch_historical(days=args.days)
    else:
        from data.stock_fetcher import StockFetcher
        fetcher = StockFetcher()
        ticker = cfg["ticker"]
        console.print(f"Scaricamento dati {cfg['name']} ({ticker}) per {args.days} giorni...")
        df = fetcher.fetch_historical(ticker, days=args.days)

        # Prova fallback tickers se necessario
        if df.empty and "fallbacks" in cfg:
            for fb in cfg["fallbacks"]:
                console.print(f"  [yellow]Tentativo con ticker alternativo: {fb}...[/yellow]")
                df = fetcher.fetch_historical(fb, days=args.days)
                if not df.empty:
                    break

    if df.empty:
        console.print("[red]Errore: nessun dato scaricato[/red]")
        return

    console.print(f"[green]Scaricate {len(df)} candele[/green]")

    # Per gli indici/ETF senza volume, genera un volume sintetico
    if asset != "btc" and ("volume" not in df.columns or df["volume"].sum() == 0):
        console.print("[yellow]Volume non disponibile: generazione volume sintetico basato su range[/yellow]")
        df["volume"] = ((df["high"] - df["low"]) / df["close"] * 1_000_000).clip(lower=100)

    strategy = MomentumStrategy(
        stop_loss_pct=config.trading.stop_loss_pct if asset == "btc" else 0.04,
        take_profit_pct=config.trading.take_profit_pct if asset == "btc" else 4.0,  # TP 400%
        min_confidence=0.5,  # Confidenza 50% per tutti gli asset
    )

    backtester = Backtester(
        strategy=strategy,
        initial_capital=args.capital,
        commission_pct=cfg["commission"],
        use_ml=args.ml,
        use_hmm=args.hmm,
        long_only=cfg["long_only"],
        asset_name=cfg["name"],
        currency=cfg["currency"],
    )

    result = backtester.run(df)
    backtester.print_report(result)


def cmd_live(args):
    """Avvia il trading live."""
    from strategy.momentum import MomentumStrategy
    from engine.live_trader import LiveTrader
    from notifications.notifier import TelegramNotifier
    from config.settings import config

    console.print(Panel("🚀 [bold green]TRADING LIVE[/bold green]", style="bold"))

    if not config.exchange.sandbox and not args.force:
        console.print("[bold red]⚠️  ATTENZIONE: Stai per avviare il trading in modalità LIVE![/bold red]")
        console.print("[yellow]Questo utilizzerà soldi REALI. Usa --force per confermare.[/yellow]")
        return

    strategy = MomentumStrategy(
        stop_loss_pct=config.trading.stop_loss_pct,
        take_profit_pct=config.trading.take_profit_pct,
    )

    notifier = TelegramNotifier()

    trader = LiveTrader(
        strategy=strategy,
        notifier=notifier,
        use_ml=not args.no_ml,
    )

    console.print(f"Simbolo: {config.trading.symbol}")
    console.print(f"Timeframe: {config.trading.timeframe}")
    console.print(f"Sandbox: {config.exchange.sandbox}")
    console.print(f"ML: {'ON' if not args.no_ml else 'OFF'}")
    console.print(f"Telegram: {'ON' if notifier.enabled else 'OFF'}")
    console.print()

    trader.start()


def cmd_dashboard(args):
    """Avvia la dashboard Streamlit."""
    console.print(Panel("📊 [bold magenta]DASHBOARD[/bold magenta]", style="bold"))

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    console.print(f"Avvio dashboard su http://localhost:{args.port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
    ])


def cmd_analyze(args):
    """Analizza il segnale corrente."""
    from data.fetcher import DataFetcher
    from indicators.technical import TechnicalIndicators
    from strategy.momentum import MomentumStrategy
    from ml.model import MLPredictor
    from config.settings import config
    from rich.table import Table

    console.print(Panel("🔍 [bold yellow]ANALISI SEGNALE[/bold yellow]", style="bold"))

    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv(limit=300)

    if df.empty:
        console.print("[red]Errore: nessun dato disponibile[/red]")
        return

    df_indicators = TechnicalIndicators.add_all(df)
    signals = TechnicalIndicators.get_current_signals(df_indicators)

    # Tabella indicatori
    table = Table(title=f"Indicatori Tecnici - {config.trading.symbol}", show_header=True)
    table.add_column("Indicatore", style="cyan")
    table.add_column("Valore", style="white", justify="right")
    table.add_column("Segnale", style="bold")

    for key, value in signals.items():
        signal_color = "green" if "bullish" in str(value).lower() else \
                       "red" if "bearish" in str(value).lower() else "yellow"
        table.add_row(key, str(value), f"[{signal_color}]{value}[/{signal_color}]")

    console.print(table)

    # Segnale della strategia
    strategy = MomentumStrategy(
        stop_loss_pct=config.trading.stop_loss_pct,
        take_profit_pct=config.trading.take_profit_pct,
    )

    ml_pred = None
    if not args.no_ml and config.ml.enabled:
        try:
            ml = MLPredictor()
            ml_pred = ml.predict(df_indicators)
            if ml_pred is not None:
                console.print(f"\n🧠 ML Prediction: {ml_pred:.4f}")
        except Exception:
            pass

    signal = strategy.analyze(df_indicators, ml_prediction=ml_pred)

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    console.print(f"\n{signal_emoji.get(signal.signal.value, '⚪')} "
                  f"[bold]Segnale: {signal.signal.value}[/bold] | "
                  f"Confidenza: {signal.confidence:.0%}")
    console.print(f"Prezzo: ${signal.price:,.2f}")

    if signal.stop_loss:
        console.print(f"Stop Loss: ${signal.stop_loss:,.2f}")
    if signal.take_profit:
        console.print(f"Take Profit: ${signal.take_profit:,.2f}")

    console.print(f"\nMotivi:")
    for reason in signal.reasons:
        console.print(f"  • {reason}")


def cmd_markets(args):
    """Analisi multi-asset: NDX, SWDA.MI, CSNDX, CAPE per-asset, Liquidità, Livelli di Ingresso."""
    from analysis.market_analyzer import MarketAnalyzer
    from rich.table import Table
    from rich.columns import Columns

    console.print(Panel(
        "🌍 [bold cyan]ANALISI MERCATI - NDX / SWDA.MI / CSNDX / CAPE / LIQUIDITÀ[/bold cyan]",
        style="bold",
    ))

    analyzer = MarketAnalyzer()

    with console.status("[bold green]Analisi in corso... Recupero dati da Yahoo Finance..."):
        report = analyzer.full_analysis(period=args.period)

    # ================================
    # SEZIONE 1: CAPE PER-ASSET
    # ================================
    console.print()

    # CAPE S&P 500 (base)
    if report.cape_sp500:
        c = report.cape_sp500
        console.print(Panel(
            f"[bold]CAPE Ratio S&P 500 (Base)[/bold]\n\n"
            f"  Valore attuale:     [bold yellow]{c.cape_value:.1f}[/bold yellow]\n"
            f"  Media storica:      {c.historical_mean:.1f}\n"
            f"  Deviazione:         {c.deviation_from_mean:+.0f}%\n"
            f"  Percentile:         {c.percentile:.0f}° percentile\n"
            f"  Rendimento 10Y:     ~{c.expected_10y_return:.1f}% annuo\n\n"
            f"  Valutazione:        [{c.entry_signal_color}][bold]{c.valuation_level}[/bold][/{c.entry_signal_color}]\n"
            f"  Segnale:            [{c.entry_signal_color}]{c.entry_signal}[/{c.entry_signal_color}]",
            title="📊 CAPE S&P 500 (Riferimento)",
            border_style="yellow",
        ))

    # CAPE per ciascun asset
    cape_table = Table(title="📊 CAPE per Asset (stime basate su S&P 500)", show_header=True, expand=True)
    cape_table.add_column("Metrica", style="cyan", width=22)
    cape_table.add_column("NDX (NASDAQ 100)", justify="center", width=22)
    cape_table.add_column("SWDA (MSCI World)", justify="center", width=22)
    cape_table.add_column("CSNDX (iShares NDX)", justify="center", width=22)

    assets_cape = [
        report.nasdaq100.cape_analysis if report.nasdaq100 else None,
        report.swda.cape_analysis if report.swda else None,
        report.csndx.cape_analysis if report.csndx else None,
    ]

    def fmt_cape(analysis, attr, fmt_str="{:.1f}"):
        if analysis is None:
            return "N/A"
        val = getattr(analysis, attr, None)
        if val is None:
            return "N/A"
        return fmt_str.format(val)

    def fmt_cape_colored(analysis):
        if analysis is None:
            return "N/A"
        c = analysis
        return f"[{c.entry_signal_color}]{c.valuation_level}[/{c.entry_signal_color}]"

    cape_table.add_row(
        "CAPE Stimato",
        fmt_cape(assets_cape[0], "cape_value"),
        fmt_cape(assets_cape[1], "cape_value"),
        fmt_cape(assets_cape[2], "cape_value"),
    )
    cape_table.add_row(
        "Media Storica",
        fmt_cape(assets_cape[0], "historical_mean"),
        fmt_cape(assets_cape[1], "historical_mean"),
        fmt_cape(assets_cape[2], "historical_mean"),
    )
    cape_table.add_row(
        "Deviazione",
        fmt_cape(assets_cape[0], "deviation_from_mean", "{:+.0f}%"),
        fmt_cape(assets_cape[1], "deviation_from_mean", "{:+.0f}%"),
        fmt_cape(assets_cape[2], "deviation_from_mean", "{:+.0f}%"),
    )
    cape_table.add_row(
        "Percentile",
        fmt_cape(assets_cape[0], "percentile", "{:.0f}°"),
        fmt_cape(assets_cape[1], "percentile", "{:.0f}°"),
        fmt_cape(assets_cape[2], "percentile", "{:.0f}°"),
    )
    cape_table.add_row(
        "Rend. 10Y Atteso",
        fmt_cape(assets_cape[0], "expected_10y_return", "~{:.1f}%"),
        fmt_cape(assets_cape[1], "expected_10y_return", "~{:.1f}%"),
        fmt_cape(assets_cape[2], "expected_10y_return", "~{:.1f}%"),
    )
    cape_table.add_row(
        "Valutazione",
        fmt_cape_colored(assets_cape[0]),
        fmt_cape_colored(assets_cape[1]),
        fmt_cape_colored(assets_cape[2]),
    )
    cape_table.add_row(
        "Segnale",
        assets_cape[0].entry_signal if assets_cape[0] else "N/A",
        assets_cape[1].entry_signal if assets_cape[1] else "N/A",
        assets_cape[2].entry_signal if assets_cape[2] else "N/A",
    )

    console.print(cape_table)

    # ================================
    # SEZIONE 2: LIQUIDITÀ
    # ================================
    if report.liquidity_analysis:
        l = report.liquidity_analysis
        console.print()

        indicators_text = ""
        for ind in l.indicators:
            color = "green" if ind.signal == "POSITIVO" else "red" if ind.signal == "NEGATIVO" else "yellow"
            indicators_text += (
                f"  [{color}]{'●' if ind.signal == 'POSITIVO' else '○' if ind.signal == 'NEUTRO' else '✗'}[/{color}] "
                f"{ind.name}: [{color}]{ind.signal}[/{color}] (score: {ind.score:+.2f})\n"
                f"     {ind.description}\n"
            )

        console.print(Panel(
            f"[bold]Analisi della Liquidità di Mercato[/bold]\n\n"
            f"{indicators_text}\n"
            f"  Score complessivo:  [bold]{l.overall_score:+.2f}[/bold]\n"
            f"  Livello liquidità:  [{l.entry_color}][bold]{l.liquidity_level}[/bold][/{l.entry_color}]\n"
            f"  Raccomandazione:    [{l.entry_color}]{l.entry_recommendation}[/{l.entry_color}]\n\n"
            f"  {l.description}",
            title="💧 Liquidità di Mercato",
            border_style="cyan",
        ))

        if l.positive_factors:
            console.print("[green]  Fattori positivi:[/green]")
            for f in l.positive_factors:
                console.print(f"    [green]✓[/green] {f}")

        if l.risk_factors:
            console.print("[red]  Fattori di rischio:[/red]")
            for f in l.risk_factors:
                console.print(f"    [red]✗[/red] {f}")

    # ================================
    # SEZIONE 3: ANALISI TECNICA ASSET
    # ================================
    console.print()
    table = Table(title="📈 Analisi Tecnica Asset", show_header=True, expand=True)
    table.add_column("Metrica", style="cyan", width=22)
    table.add_column("NDX (NASDAQ 100)", justify="center", width=22)
    table.add_column("SWDA.MI", justify="center", width=22)
    table.add_column("CSNDX", justify="center", width=22)

    assets = [report.nasdaq100, report.swda, report.csndx]
    asset_names = ["NDX", "SWDA.MI", "CSNDX"]

    def fmt_change(v):
        color = "green" if v > 0 else "red" if v < 0 else "white"
        return f"[{color}]{v:+.2f}%[/{color}]"

    def fmt_trend(t):
        color = "green" if "RIALZ" in t else "red" if "RIBASS" in t else "yellow"
        return f"[{color}]{t}[/{color}]"

    def fmt_val(a, attr, fmt_str="${:,.2f}", zero_check=True):
        if a is None:
            return "N/A"
        val = getattr(a, attr, 0)
        if zero_check and val == 0:
            return "N/A"
        return fmt_str.format(val)

    # Simbolo valuta: NDX è in USD, SWDA.MI e CSNDX in EUR
    currencies = ["$", "€", "€"]

    for row_name, attr, fmt in [
        ("Prezzo", "current_price", None),
        ("Variazione 1D", "price_change_1d", "change"),
        ("Variazione 1M", "price_change_1m", "change"),
        ("Variazione 3M", "price_change_3m", "change"),
        ("Variazione 1Y", "price_change_1y", "change"),
        ("SMA 50", "sma_50", None),
        ("SMA 200", "sma_200", None),
        ("RSI (14)", "rsi", "rsi"),
        ("Trend", "trend", "trend"),
        ("Golden Cross", "golden_cross", "bool"),
        ("Sopra SMA200", "above_sma200", "bool"),
        ("Score Tecnico", "technical_score", "score"),
    ]:
        vals = []
        for i, a in enumerate(assets):
            if a is None or a.current_price == 0:
                vals.append("N/A")
                continue
            if fmt == "change":
                vals.append(fmt_change(getattr(a, attr, 0)))
            elif fmt == "trend":
                vals.append(fmt_trend(getattr(a, attr, "N/A")))
            elif fmt == "bool":
                vals.append("✅" if getattr(a, attr, False) else "❌")
            elif fmt == "score":
                vals.append(f"{getattr(a, attr, 0):+.2f}")
            elif fmt == "rsi":
                vals.append(f"{getattr(a, attr, 50):.1f}")
            else:
                val = getattr(a, attr, 0)
                if val == 0:
                    vals.append("N/A")
                else:
                    vals.append(f"{currencies[i]}{val:,.2f}")
        table.add_row(row_name, *vals)

    console.print(table)

    # ================================
    # SEZIONE 4: LIVELLI DI INGRESSO
    # ================================
    for i, (asset, name) in enumerate(zip(assets, asset_names)):
        if asset is None or not asset.entry_levels or asset.current_price == 0:
            continue
        console.print()
        cur = currencies[i]
        entry_table = Table(
            title=f"🎯 Livelli di Ingresso - {name} (prezzo attuale: {cur}{asset.current_price:,.2f})",
            show_header=True,
            expand=True,
        )
        entry_table.add_column("Livello", style="cyan", width=18)
        entry_table.add_column("Prezzo", justify="right", width=14)
        entry_table.add_column("Distanza", justify="right", width=10)
        entry_table.add_column("Prob. 30gg", justify="center", width=12)
        entry_table.add_column("Prob. 90gg", justify="center", width=12)
        entry_table.add_column("Tipo", justify="center", width=16)

        for lvl in asset.entry_levels:
            dist_color = "green" if lvl.distance_pct < -3 else "yellow" if lvl.distance_pct < 0 else "white"

            # Colore probabilità
            def prob_color(p):
                if p > 0.7:
                    return "green"
                elif p > 0.4:
                    return "yellow"
                else:
                    return "red"

            p30_col = prob_color(lvl.prob_30d)
            p90_col = prob_color(lvl.prob_90d)

            entry_table.add_row(
                lvl.level,
                f"{cur}{lvl.price:,.2f}",
                f"[{dist_color}]{lvl.distance_pct:+.1f}%[/{dist_color}]",
                f"[{p30_col}]{lvl.prob_30d:.0%}[/{p30_col}]",
                f"[{p90_col}]{lvl.prob_90d:.0%}[/{p90_col}]",
                lvl.level_type.replace("_", " ").title(),
            )

        console.print(entry_table)

    # ================================
    # SEZIONE 5: RACCOMANDAZIONI
    # ================================
    console.print()
    rec_text = ""

    for name, score, rec, entry_type in [
        ("NASDAQ 100 (^NDX)", report.nasdaq100_score, report.nasdaq100_recommendation, report.nasdaq100_entry_type),
        ("SWDA.MI (iShares MSCI World)", report.swda_score, report.swda_recommendation, report.swda_entry_type),
        ("CSNDX (iShares NASDAQ 100)", report.csndx_score, report.csndx_recommendation, report.csndx_entry_type),
    ]:
        rec_text += (
            f"[bold]{name}[/bold]\n"
            f"  Score composito:  [bold]{score:+.2f}[/bold]\n"
            f"  Raccomandazione:  {rec}\n"
            f"  Tipo ingresso:    {entry_type}\n\n"
        )

    console.print(Panel(
        rec_text.strip(),
        title="🎯 Raccomandazioni di Ingresso",
        border_style="green",
    ))

    # Asset Allocation
    if report.suggested_allocation:
        console.print()
        alloc_table = Table(title="💼 Asset Allocation Consigliata", show_header=True)
        alloc_table.add_column("Asset", style="cyan")
        alloc_table.add_column("Allocazione %", justify="right", style="bold")

        for asset, pct in report.suggested_allocation.items():
            bar = "█" * int(pct / 2)
            color = "green" if "NASDAQ" in asset or "NDX" in asset else "cyan" if "SWDA" in asset or "CSNDX" in asset else "blue" if "Bond" in asset else "yellow"
            alloc_table.add_row(asset, f"[{color}]{bar} {pct}%[/{color}]")

        console.print(alloc_table)

    # Segnali tecnici dettagliati
    for asset, name in zip(assets, asset_names):
        if asset and asset.technical_signals:
            console.print(f"\n[bold cyan]Segnali tecnici {name}:[/bold cyan]")
            for sig in asset.technical_signals:
                console.print(f"  • {sig}")

    # ================================
    # SEZIONE 6: VERDETTO FINALE
    # ================================
    console.print()
    verdict_text = _build_verdict(report)
    console.print(Panel(
        verdict_text,
        title="✅ VERDETTO FINALE - Esito dell'Analisi",
        border_style="bold green",
    ))

    # Summary tecnico
    if report.summary:
        console.print()
        console.print(Panel(report.summary, title="📋 Sommario", border_style="white"))


def _build_verdict(report) -> str:
    """Costruisce il verdetto finale combinando CAPE, liquidità, tecnica e livelli."""
    lines = []

    # --- Regime di mercato ---
    cape_sp = report.cape_sp500
    liq = report.liquidity_analysis
    regime_color = "yellow"
    if cape_sp and cape_sp.cape_value > 35:
        regime = "MERCATO CARO"
        regime_color = "red"
    elif cape_sp and cape_sp.cape_value > 25:
        regime = "MERCATO MODERATAMENTE CARO"
        regime_color = "yellow"
    elif cape_sp and cape_sp.cape_value > 18:
        regime = "MERCATO NELLA NORMA"
        regime_color = "green"
    else:
        regime = "MERCATO SOTTOVALUTATO"
        regime_color = "green"

    lines.append(f"[{regime_color}][bold]🌐 REGIME: {regime}[/bold][/{regime_color}]")
    if cape_sp:
        lines.append(f"   CAPE S&P 500 a {cape_sp.cape_value:.1f} ({cape_sp.valuation_level})")
    if liq:
        liq_cls = "green" if liq.overall_score > 0.1 else "red" if liq.overall_score < -0.2 else "yellow"
        lines.append(f"   Liquidità: [{liq_cls}]{liq.liquidity_level}[/{liq_cls}] (score {liq.overall_score:+.2f})")
    lines.append("")

    # --- Per-asset verdict ---
    asset_data = [
        ("NASDAQ 100 (^NDX)", report.nasdaq100, report.nasdaq100_score,
         report.nasdaq100_recommendation, report.nasdaq100_entry_type, "$"),
        ("SWDA.MI (MSCI World)", report.swda, report.swda_score,
         report.swda_recommendation, report.swda_entry_type, "€"),
        ("CSNDX (iShares NDX)", report.csndx, report.csndx_score,
         report.csndx_recommendation, report.csndx_entry_type, "€"),
    ]

    for name, asset, score, rec, entry_type, cur in asset_data:
        if asset is None or asset.current_price == 0:
            continue

        # Emoji per score
        if score > 0.3:
            emoji = "🟢"
        elif score > 0:
            emoji = "🟡"
        elif score > -0.3:
            emoji = "🟠"
        else:
            emoji = "🔴"

        lines.append(f"{emoji} [bold]{name}[/bold]  |  Prezzo: {cur}{asset.current_price:,.2f}  |  Score: [bold]{score:+.2f}[/bold]")
        lines.append(f"   Trend: {asset.trend} | RSI: {asset.rsi:.0f} | "
                     f"1M: {asset.price_change_1m:+.1f}%")

        # CAPE per asset
        if asset.cape_analysis:
            ca = asset.cape_analysis
            lines.append(f"   CAPE: {ca.cape_value:.1f} ({ca.valuation_level}) → Rend. atteso 10Y: ~{ca.expected_10y_return:.1f}%")

        # Miglior livello di ingresso raggiungibile (prob > 60% a 90gg)
        best_levels = [l for l in asset.entry_levels if l.prob_90d > 0.60 and l.distance_pct < 0]
        if best_levels:
            best = min(best_levels, key=lambda l: l.distance_pct)  # livello più basso raggiungibile
            lines.append(f"   🎯 Livello consigliato: {cur}{best.price:,.2f} ({best.distance_pct:+.1f}%) "
                         f"- Prob. 90gg: {best.prob_90d:.0%}")
        elif asset.entry_levels:
            closest = min(asset.entry_levels, key=lambda l: abs(l.distance_pct))
            lines.append(f"   🎯 Livello più vicino: {cur}{closest.price:,.2f} ({closest.distance_pct:+.1f}%) "
                         f"- Prob. 90gg: {closest.prob_90d:.0%}")

        # Raccomandazione sintetica
        rec_color = "green" if "ACQUISTO" in rec or "FORTE" in rec else "red" if "EVITARE" in rec or "CAUTELA" in rec else "yellow"
        lines.append(f"   → [{rec_color}]{rec}[/{rec_color}]")
        lines.append(f"   → Strategia: {entry_type}")
        lines.append("")

    # --- Cosa fare ---
    scores = [s for s in [report.nasdaq100_score, report.swda_score, report.csndx_score] if s != 0]
    avg = sum(scores) / len(scores) if scores else 0

    lines.append("[bold]📌 AZIONE CONSIGLIATA:[/bold]")
    if avg > 0.3:
        lines.append("   [green]✔ Condizioni favorevoli. Procedere con ingressi secondo l'allocazione suggerita.[/green]")
    elif avg > 0:
        lines.append("   [green]✔ Ingresso graduale (DCA mensile) su tutti e 3 gli asset.[/green]")
        lines.append("   [yellow]⚠ Non investire tutto subito: distribuire su 3-6 mesi.[/yellow]")
    elif avg > -0.3:
        lines.append("   [yellow]⚠ Mercato caro. Privilegiare DCA diluito (trimestrale).[/yellow]")
        lines.append("   [yellow]⚠ Mantenere 25-30% in liquidità/bond in attesa di correzioni.[/yellow]")
        if report.swda and report.swda_score > report.nasdaq100_score:
            lines.append("   💡 SWDA.MI (MSCI World) è relativamente più attrattivo del NASDAQ 100.")
    else:
        lines.append("   [red]✘ Condizioni sfavorevoli. Accumulare liquidità, ridurre esposizione azionaria.[/red]")
        lines.append("   [red]✘ Considerare obbligazioni o asset difensivi.[/red]")

    return "\n".join(lines)


def cmd_monitor(args):
    """Monitor accumulo CSNDX e SWDA con notifiche WhatsApp."""
    from engine.accumulation_monitor import AccumulationMonitor

    console.print(Panel(
        "📱 [bold cyan]MONITOR ACCUMULO - CSNDX / SWDA.MI[/bold cyan]\n"
        "   Notifiche WhatsApp per segnali di acquisto",
        style="bold",
    ))

    monitor = AccumulationMonitor()
    notify = not getattr(args, "no_notify", False)

    if getattr(args, "schedule", False):
        interval = getattr(args, "interval", 4)
        console.print(f"[green]Modalità schedulata: analisi ogni {interval}h (orari mercato europeo)[/green]")
        console.print("[yellow]Ctrl+C per terminare[/yellow]")
        monitor.run_scheduled(interval_hours=interval)
    else:
        # Singola esecuzione
        signals = monitor.check_signals(notify=notify)

        if not signals:
            console.print("[yellow]Nessun segnale generato[/yellow]")
            return

        # Mostra risultati in console
        from rich.table import Table

        table = Table(title="📱 Segnali di Accumulo", show_header=True)
        table.add_column("Asset", style="cyan", width=25)
        table.add_column("Prezzo", style="green", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Azione", justify="center")
        table.add_column("Livello Target", justify="right")
        table.add_column("Prob. 90gg", justify="right")
        table.add_column("Trend", justify="center")

        for sig in signals:
            color = {"COMPRA": "green", "ATTENDI": "yellow", "EVITA": "red"}.get(sig.action, "white")
            emoji = {"COMPRA": "🟢", "ATTENDI": "🟡", "EVITA": "🔴"}.get(sig.action, "⚪")

            target = f"{sig.currency}{sig.best_entry_price:,.2f}" if sig.best_entry_price else "—"
            prob = f"{sig.best_entry_prob:.0%}" if sig.best_entry_prob else "—"

            table.add_row(
                sig.asset_name,
                f"{sig.currency}{sig.current_price:,.2f}",
                f"[{'green' if sig.score > 0 else 'red'}]{sig.score:+.2f}[/]",
                f"[{color}]{emoji} {sig.action}[/]",
                target,
                prob,
                sig.trend,
            )

        console.print(table)

        # Dettagli per asset
        for sig in signals:
            console.print(f"\n[bold]{sig.asset_name}[/bold]")
            console.print(f"  RSI: {sig.rsi:.0f} | 1M: {sig.price_change_1m:+.1f}%")
            if sig.cape_info:
                console.print(f"  CAPE: {sig.cape_info}")
            console.print(f"  → {sig.recommendation}")
            console.print(f"  → Strategia: {sig.entry_type}")

            if sig.crossed_levels:
                console.print(f"  [bold green]🎯 LIVELLI RAGGIUNTI:[/bold green]")
                for lvl in sig.crossed_levels:
                    console.print(f"     • {lvl['level']}: {sig.currency}{lvl['price']:,.2f}")

        # Stato notifiche
        if notify and monitor.notifier.enabled:
            console.print(f"\n[green]✅ Notifiche WhatsApp inviate a {monitor.notifier.to_number}[/green]")
        elif not monitor.notifier.enabled:
            console.print(f"\n[yellow]⚠️ WhatsApp non configurato. Configura TWILIO_ACCOUNT_SID e TWILIO_AUTH_TOKEN nel .env[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 Crypto Trading Agent - Trading automatico con ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Esegui backtest (BTC, NDX, SWDA, CSNDX)")
    bt_parser.add_argument("--days", type=int, default=90, help="Giorni di storico (default: 90)")
    bt_parser.add_argument("--capital", type=float, default=10000, help="Capitale iniziale (default: 10000)")
    bt_parser.add_argument("--ml", action="store_true", help="Abilita ML nel backtest")
    bt_parser.add_argument("--hmm", action="store_true", help="Abilita HMM Regime Detection nel backtest")
    bt_parser.add_argument("--asset", type=str, default="btc",
                           choices=["btc", "ndx", "swda", "csndx"],
                           help="Asset da testare: btc, ndx, swda, csndx (default: btc)")

    # Live
    live_parser = subparsers.add_parser("live", help="Avvia trading live")
    live_parser.add_argument("--no-ml", action="store_true", help="Disabilita ML")
    live_parser.add_argument("--force", action="store_true", help="Forza avvio in modalità live")

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Avvia dashboard Streamlit")
    dash_parser.add_argument("--port", type=int, default=8501, help="Porta (default: 8501)")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analizza segnale corrente")
    analyze_parser.add_argument("--no-ml", action="store_true", help="Disabilita ML")

    # Markets - Analisi multi-asset
    markets_parser = subparsers.add_parser("markets", help="Analisi NDX, SWDA.MI, CSNDX, CAPE per-asset, Liquidità, Livelli ingresso")
    markets_parser.add_argument("--period", type=str, default="1y",
                                choices=["3mo", "6mo", "1y", "2y", "5y"],
                                help="Periodo di storico (default: 1y)")

    # Monitor - Segnali di accumulo con notifiche WhatsApp
    monitor_parser = subparsers.add_parser("monitor", help="Monitor accumulo CSNDX/SWDA con notifiche WhatsApp")
    monitor_parser.add_argument("--schedule", action="store_true",
                                help="Esegui in loop continuo (ogni 4h durante orari di mercato)")
    monitor_parser.add_argument("--interval", type=int, default=4,
                                help="Intervallo in ore tra le analisi (default: 4)")
    monitor_parser.add_argument("--no-notify", action="store_true",
                                help="Disabilita notifiche WhatsApp (solo analisi)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "backtest": cmd_backtest,
        "live": cmd_live,
        "dashboard": cmd_dashboard,
        "analyze": cmd_analyze,
        "markets": cmd_markets,
        "monitor": cmd_monitor,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
