"""Logging helpers for the momentum scanner."""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

_LOGGER_CONFIGURED = False


def _configure_root_logger(log_level: int = logging.INFO, log_dir: Optional[Path] = None) -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    log_dir = log_dir or Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "momentum_scanner.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _LOGGER_CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """Return a module-level logger configured for the project."""

    _configure_root_logger(log_level=level)
    return logging.getLogger(name)
