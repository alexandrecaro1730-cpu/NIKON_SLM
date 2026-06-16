"""
Logging utilities for production-style pipeline execution.

Why this matters:
- In production, every data run should leave an audit trail.
- Logs help explain whether a run passed, warned, or failed.
- This is especially useful for manufacturing environments where data quality
  problems may come from sensors, exports, or machine configuration changes.
"""

import logging
from pathlib import Path


def configure_logger(log_path: str | Path) -> logging.Logger:
    """
    Configure a file + console logger.

    Parameters
    ----------
    log_path:
        Where the log file should be saved.

    Returns
    -------
    logging.Logger
        Configured project logger.
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lpbf_quality")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger