"""Colour‑rich, timestamped logger wrapper (stdout + file)."""
import datetime as dt
import logging
import sys
from pathlib import Path

RESET = "[0m"; CYAN = "[96m"; YELLOW = "[93m"; RED = "[91m"

class ColourFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: CYAN + "[%(levelname)s] %(message)s" + RESET,
        logging.INFO:  "[%(levelname)s] %(message)s",
        logging.WARNING: YELLOW + "[%(levelname)s] %(message)s" + RESET,
        logging.ERROR: RED + "[%(levelname)s] %(message)s" + RESET,
        logging.CRITICAL: RED + "[%(levelname)s] %(message)s" + RESET,
    }
    def format(self, record):
        fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt)
        return formatter.format(record)

def get_logger(name: str | None = None, log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(name or "phishsleuth")
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColourFormatter())
    logger.addHandler(ch)

    # File
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(Path(log_dir) / f"run_{timestamp}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger