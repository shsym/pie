"""Serve command implementation for Pie CLI.

Implements: pie-server serve
Starts the Pie engine and optionally provides an interactive shell session.
"""

from pathlib import Path
from typing import Optional

import toml
import typer

from . import path as pie_path


def load_config(
    config_path: Optional[Path],
    no_auth: bool,
    host: Optional[str],
    port: Optional[int],
    verbose: bool,
    log: Optional[Path],
) -> tuple[dict, list[dict]]:
    """Load and merge configuration from file and CLI arguments.

    Returns:
        Tuple of (engine_config, backend_configs)
    """
    # Load config file
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        raise typer.Exit(
            typer.echo(
                f"❌ Configuration file not found at {file_path}. "
                "Run `pie-server config init` first.",
                err=True,
            )
            or 1
        )

    config = toml.loads(file_path.read_text())

    # Build engine config with CLI overrides
    engine_config = {
        "host": host or config.get("host", "127.0.0.1"),
        "port": port or config.get("port", 8080),
        "enable_auth": False if no_auth else config.get("enable_auth", True),
        "cache_dir": config.get("cache_dir", str(pie_path.get_pie_home() / "programs")),
        "verbose": verbose or config.get("verbose", False),
        "log": str(log) if log else config.get("log"),
    }

    backend_configs = config.get("backend", [])
    if not backend_configs:
        raise typer.Exit(
            typer.echo(
                "❌ No backend configurations found in the configuration file.",
                err=True,
            )
            or 1
        )

    return engine_config, backend_configs


def serve(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Network host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", help="Network port to use"),
    no_auth: bool = typer.Option(False, "--no-auth", help="Disable authentication"),
    log: Optional[Path] = typer.Option(None, "--log", help="Log file to write to"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose console logging"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable interactive shell mode"
    ),
) -> None:
    """Start the Pie engine and enter an interactive session.

    This command starts the Pie engine server along with configured backend
    services. In interactive mode, it provides a shell for running inferlets.
    """
    try:
        engine_config, backend_configs = load_config(
            config, no_auth, host, port, verbose, log
        )
    except typer.Exit:
        raise

    # Import here to avoid circular imports and allow module to load without Rust
    from . import manager

    if interactive:
        typer.echo("🚀 Starting Pie engine (interactive mode)...")
    else:
        typer.echo("🚀 Starting Pie engine...")

    server_handle = None
    backend_processes = []
    
    try:
        # Start engine and backends
        server_handle, backend_processes = manager.start_engine_and_backend(
            engine_config, backend_configs
        )

        if interactive:
            typer.echo(
                "Entering interactive session. Type 'help' for commands or use ↑/↓ for history."
            )
            manager.run_interactive_shell(engine_config, server_handle.internal_token)
        else:
            typer.echo("Press Ctrl+C to stop.")
            import signal

            try:
                signal.pause()
            except KeyboardInterrupt:
                pass

        # Cleanup
        typer.echo()
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        typer.echo("✅ Shutdown complete.")

    except KeyboardInterrupt:
        typer.echo()
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        typer.echo("✅ Shutdown complete.")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)

