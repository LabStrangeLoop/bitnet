"""Logging configuration for training experiments."""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup(output_dir: Path, resume: bool = False) -> None:
    """Configure logging to both console and file."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    log_file = output_dir / "train.log"
    file_mode = "a" if resume else "w"
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(file_handler)
