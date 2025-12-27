"""Authorization management commands for Pie CLI.

Implements: pie-server auth add|remove|list
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import toml
import typer

from . import path as pie_path

app = typer.Typer(help="Manage authorized clients")


def load_authorized_users(auth_path: Path) -> dict:
    """Load authorized users from file, or return empty dict if not exists."""
    if not auth_path.exists():
        return {}
    return toml.loads(auth_path.read_text())


def save_authorized_users(auth_path: Path, users: dict) -> None:
    """Save authorized users to file."""
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(toml.dumps(users))


@app.command("add")
def auth_add(
    username: str = typer.Argument(..., help="Username of the user"),
    key_name: Optional[str] = typer.Argument(
        None, help="Optional name for this key (defaults to timestamp)"
    ),
) -> None:
    """Add an authorized user and its public key.

    The public key is read from stdin and can be in OpenSSH, PKCS#8 PEM, or PKCS#1 PEM format.
    """
    # Generate key name if not provided
    if key_name is None:
        key_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Show prompts only in interactive mode
    if sys.stdin.isatty():
        typer.echo("🔐 Adding authorized user...")
        typer.echo(f"   Username: {username}")
        typer.echo(f"   Key name: {key_name}")
        typer.echo()
        typer.echo("Enter public key (paste, then press Ctrl-D on a new line):")
        typer.echo("  Supported algorithms:")
        typer.echo("  - RSA (2048-8192 bits)")
        typer.echo("  - ED25519 (256 bits)")
        typer.echo("  - ECDSA (256, 384 bits)")
        typer.echo("  Supported formats:")
        typer.echo("  - OpenSSH (single line)")
        typer.echo("  - PKCS#8 PEM (multi-line)")
        typer.echo("  - PKCS#1 PEM (multi-line)")
        typer.echo("> ", nl=False)

    # Read public key from stdin
    public_key = sys.stdin.read().strip()

    auth_path = pie_path.get_authorized_users_path()
    users = load_authorized_users(auth_path)

    # Create user entry if it doesn't exist
    user_created = False
    if username not in users:
        users[username] = {}
        user_created = True

    if not public_key:
        typer.echo()
        typer.echo("⚠️ Warning: No public key provided")
        save_authorized_users(auth_path, users)
        if user_created:
            typer.echo(f"✅ Created user '{username}' without any keys")
        else:
            typer.echo(f"📋 User '{username}' already exists")
        return

    # Check if key name already exists
    if key_name in users[username]:
        typer.echo()
        typer.echo(
            f"❌ Key with name '{key_name}' already exists for user '{username}'",
            err=True,
        )
        raise typer.Exit(1)

    # Add the key (store the raw public key string)
    users[username][key_name] = public_key
    save_authorized_users(auth_path, users)

    typer.echo()
    if user_created:
        typer.echo(f"✅ Created user '{username}' and added key '{key_name}'")
    else:
        typer.echo(f"✅ Added key '{key_name}' to user '{username}'")


@app.command("remove")
def auth_remove(
    username: str = typer.Argument(..., help="Username of the user"),
    key_name: Optional[str] = typer.Argument(
        None, help="Optional name of the specific key to remove"
    ),
) -> None:
    """Remove an authorized user or a specific key.

    If key_name is provided, only that key is removed.
    If key_name is not provided, the entire user entry is removed.
    """
    auth_path = pie_path.get_authorized_users_path()

    if not auth_path.exists():
        typer.echo(
            f"❌ Authorized users file not found at {auth_path}. No users to remove.",
            err=True,
        )
        raise typer.Exit(1)

    users = load_authorized_users(auth_path)

    if username not in users:
        typer.echo(f"❌ User '{username}' not found", err=True)
        raise typer.Exit(1)

    if key_name:
        # Remove specific key
        if key_name not in users[username]:
            typer.echo(f"❌ Key '{key_name}' not found for user '{username}'", err=True)
            raise typer.Exit(1)

        del users[username][key_name]
        save_authorized_users(auth_path, users)
        typer.echo(f"✅ Removed key '{key_name}' from user '{username}'")
    else:
        # Remove entire user - prompt for confirmation in interactive mode
        key_count = len(users[username])
        if sys.stdin.isatty():
            confirm = typer.confirm(
                f"⚠️ This will remove user '{username}' and all {key_count} key(s). Continue?"
            )
            if not confirm:
                typer.echo("Operation cancelled.")
                raise typer.Exit(1)

        del users[username]
        save_authorized_users(auth_path, users)
        typer.echo(f"✅ Removed user '{username}' and all associated keys")


@app.command("list")
def auth_list() -> None:
    """List all authorized users and their keys."""
    auth_path = pie_path.get_authorized_users_path()

    if not auth_path.exists():
        typer.echo("📋 No authorized users found.")
        typer.echo(f"    (File not found at {auth_path})")
        return

    users = load_authorized_users(auth_path)

    if not users:
        typer.echo("📋 No authorized users found.")
        return

    typer.echo("📋 Authorized users:")
    typer.echo(f"    File: {auth_path}")
    typer.echo()

    for username in sorted(users.keys()):
        user_keys = users[username]
        key_count = len(user_keys)
        key_word = "key" if key_count == 1 else "keys"
        typer.echo(f"  {username} ({key_count} {key_word}):")

        for key_name in sorted(user_keys.keys()):
            typer.echo(f"    - {key_name}")

    typer.echo()
    typer.echo(f"Total: {len(users)} authorized user(s)")
