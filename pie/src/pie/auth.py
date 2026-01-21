"""Authorization utilities for Pie."""

from pathlib import Path

import toml


def load_authorized_users(auth_path: Path) -> dict:
    """Load authorized users from file, or return empty dict if not exists."""
    if not auth_path.exists():
        return {}
    return toml.loads(auth_path.read_text())


def save_authorized_users(auth_path: Path, users: dict) -> None:
    """Save authorized users to file."""
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(toml.dumps(users))
