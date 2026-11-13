"""Configuration loading. Everything configurable lives in ``config.yaml``."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT: Path = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (or the file named by $SIGNBRIDGE_CONFIG)."""
    cfg_path = Path(path or os.environ.get("SIGNBRIDGE_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def path_of(key: str) -> Path:
    """Return the absolute path for a key of the ``paths`` section."""
    return resolve(load_config()["paths"][key])


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    """Configure root logging to console and ``logs/signbridge.log``."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if to_file:
        try:
            path_of("logs").mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path_of("logs") / "signbridge.log", encoding="utf-8"))
        except OSError:
            pass
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
