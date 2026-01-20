"""
Centralized logging configuration for LLM4CP project.

This module provides a unified logging setup that:
- Configures logging format and levels
- Suppresses unwanted warnings from third-party libraries
- Provides convenient logger getter functions
"""

import logging
import sys
import warnings


# Suppress warnings BEFORE any imports that might trigger them
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None
) -> None:
    """
    Setup logging configuration for the entire project.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()