"""Logging configuration for market analyst."""

import logging


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the market analyst logger.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured logger instance for market_analyst.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("market_analyst")
