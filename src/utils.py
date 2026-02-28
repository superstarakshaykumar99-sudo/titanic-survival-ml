"""
utils.py – shared path constants and logger factory.
"""

import logging
import os
from pathlib import Path

# ── Root of the project (one level above src/) ──────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Directory constants ──────────────────────────────────────────────────────
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
REPORTS_DIR   = ROOT_DIR / "reports"

# Ensure directories exist
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "titanic") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
