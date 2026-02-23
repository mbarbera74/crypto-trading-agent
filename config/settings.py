"""
Configurazione dell'agente di trading crypto.
Carica le variabili d'ambiente e definisce i parametri globali.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Carica il file .env dalla root del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ExchangeConfig:
    """Configurazione dell'exchange."""
    name: str = os.getenv("EXCHANGE_NAME", "binance")
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")
    sandbox: bool = os.getenv("USE_SANDBOX", "true").lower() == "true"


@dataclass
class TradingConfig:
    """Parametri di trading."""
    symbol: str = os.getenv("TRADING_SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TRADING_TIMEFRAME", "1h")
    amount: float = float(os.getenv("TRADING_AMOUNT", "0.001"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "10"))


@dataclass
class MLConfig:
    """Configurazione Machine Learning."""
    enabled: bool = os.getenv("ML_ENABLED", "true").lower() == "true"
    retrain_hours: int = int(os.getenv("ML_RETRAIN_HOURS", "24"))
    model_path: str = str(PROJECT_ROOT / "ml" / "saved_models")
    lookback_periods: int = 100
    train_test_split: float = 0.8
    n_estimators: int = 200
    max_depth: int = 6


@dataclass
class TelegramConfig:
    """Configurazione Telegram per le notifiche."""
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


@dataclass
class WhatsAppConfig:
    """Configurazione WhatsApp via Twilio."""
    account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number: str = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Sandbox Twilio
    to_number: str = os.getenv("TWILIO_WHATSAPP_TO", "whatsapp:+393316037980")
    enabled: bool = bool(os.getenv("TWILIO_ACCOUNT_SID", ""))


@dataclass
class AppConfig:
    """Configurazione principale dell'applicazione."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    whatsapp: WhatsAppConfig = field(default_factory=WhatsAppConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_dir: str = str(PROJECT_ROOT / "data" / "historical")


# Istanza globale della configurazione
config = AppConfig()
