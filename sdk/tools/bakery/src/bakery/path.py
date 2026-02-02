"""Path utilities for Bakery."""

import os
from pathlib import Path


def get_bakery_home() -> Path:
    """Get the Bakery home directory.

    Returns BAKERY_HOME environment variable if set, otherwise ~/.pie/bakery.
    """
    if bakery_home := os.environ.get("BAKERY_HOME"):
        return Path(bakery_home)
    return Path.home() / ".pie" / "bakery"


def get_config_path() -> Path:
    """Get the path to the bakery configuration file."""
    return get_bakery_home() / "config.toml"
