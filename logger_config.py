"""
Centralised logging configuration for NeuroLingo.

Usage in any module:
    from logger_config import get_logger
    log = get_logger(__name__)
"""
import logging
import logging.handlers
import sys
from pathlib import Path

# Logs are always written next to this file (project root/logs/)
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "neurolingo.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "DEBUG") -> logging.Logger:
    """
    Configure the root logger with a rotating file handler and a console handler.

    Safe to call multiple times — existing handlers are cleared on each call
    so re-imports or test setups never produce duplicate log lines.

    Args:
        level: Root log level string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        The top-level "neurolingo" logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()  # Idempotent: prevents duplicate handlers

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

    # Rolling file: up to 5 MB × 3 backups ≈ 20 MB max on disk
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console: INFO+ only — DEBUG chatter stays in the log file
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger = logging.getLogger("neurolingo")
    logger.info("Logging initialised | file=%s | level=%s", LOG_FILE, level.upper())
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger namespaced under 'neurolingo'.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger with name "neurolingo.<name>".
    """
    return logging.getLogger(f"neurolingo.{name}")
