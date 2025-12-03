from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .. import __name__ as package_name

if TYPE_CHECKING:
    from rich import Console


def get_log_level() -> int:
    logger = logging.getLogger(package_name)
    return logger.level


def setup_logging(verbose: bool):
    logging.Formatter.converter = time.localtime
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%X')

    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger(package_name).setLevel(level)


def setup_file_logging(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    handler.setLevel(get_log_level())

    logging.getLogger().addHandler(handler)


def setup_rich_logging(console: Console):
    from rich.logging import RichHandler

    handler = RichHandler(console=console, level=get_log_level(), show_path=False, rich_tracebacks=True)
    formatter = logging.Formatter('%(message)s', datefmt='%X')
    handler.setFormatter(formatter)

    logging.getLogger().handlers = []
    logging.getLogger(package_name).addHandler(handler)
