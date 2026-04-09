
"""
Centralized logging setup.
Import `logger` from this module everywhere.
"""
from __future__ import annotations

import logging
import sys

from config import Config


def _build_logger() -> logging.Logger:
    log = logging.getLogger("stock_eval")
    log.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
        ch.setFormatter(fmt)
        log.addHandler(ch)

        try:
            fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
            fh.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError:
            pass

    return log


logger = _build_logger()