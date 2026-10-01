"""Structured logging helpers for DeepGPR.

The package never configures the root logger. Library code obtains child
loggers through :func:`get_logger`; applications opt in to console output
with :func:`configure_logging`.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from ..config.defaults import LOGGER_NAME

#: Default record format used by :func:`configure_logging`.
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger inside the ``DeepGPR`` hierarchy.

    Args:
        name: Dotted module name. Names already starting with ``DeepGPR`` are
            used unchanged; other names are nested below the package logger.

    Returns:
        The requested :class:`logging.Logger`.
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: Union[int, str] = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    *,
    stream=None,
) -> logging.Logger:
    """Attach a console handler to the ``DeepGPR`` logger.

    Calling the function repeatedly replaces the handler it installed before,
    so it is safe to use in notebooks.

    Args:
        level: Logging level name or number, e.g. ``"DEBUG"``.
        fmt: :mod:`logging` format string.
        stream: Output stream; defaults to ``sys.stderr``.

    Returns:
        The configured package logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_deepgpr_console_handler", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))
    handler._deepgpr_console_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["DEFAULT_LOG_FORMAT", "configure_logging", "get_logger"]
