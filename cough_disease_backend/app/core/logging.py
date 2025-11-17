"""Logging helpers."""

import sys
from typing import Optional

from loguru import logger


def setup_logging(level: Optional[str] = "INFO") -> None:
  """Configure global Loguru logging."""
  logger.remove()
  log_level = (level or "INFO").upper()
  logger.add(
    sys.stderr,
    level=log_level,
    backtrace=False,
    diagnose=False,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
  )

