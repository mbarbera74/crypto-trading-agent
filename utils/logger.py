"""
Logger utility basato su loguru per un logging strutturato e colorato.
"""

import sys
from pathlib import Path
from loguru import logger

from config.settings import config

# Rimuovi il logger di default
logger.remove()

# Console handler con colori
logger.add(
    sys.stdout,
    level=config.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    colorize=True,
)

# File handler per log persistenti
log_dir = Path(config.data_dir).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logger.add(
    str(log_dir / "trading_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="1 day",
    retention="30 days",
    compression="zip",
)

# File handler specifico per i trade
logger.add(
    str(log_dir / "trades_{time:YYYY-MM-DD}.log"),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    filter=lambda record: "TRADE" in record["message"],
    rotation="1 day",
    retention="90 days",
)


def get_logger(name: str = "trading_agent"):
    """Ritorna un logger con il nome specificato."""
    return logger.bind(name=name)
