"""Run command implementation for Pie CLI.

Implements: pie-server run <inferlet> [args]
Runs an inferlet with a one-shot Pie engine instance.
"""

from pathlib import Path
from typing import Optional

import typer

from . import path as pie_path
from . import serve as serve_module


def run(
    inferlet: Path = typer.Argument(..., help="Path to the .wasm inferlet file"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    log: Optional[Path] = typer.Option(None, "--log", help="Log file to write to"),
    arguments: Optional[list[str]] = typer.Argument(
        None, help="Arguments to pass to the inferlet"
    ),
) -> None:
    """Run an inferlet with a one-shot Pie engine.

    This command starts the Pie engine, runs the specified inferlet,
    waits for it to complete, and then shuts down.
    """
    # Verify inferlet exists
    if not inferlet.exists():
        typer.echo(f"❌ Inferlet file not found: {inferlet}", err=True)
        raise typer.Exit(1)

    try:
        engine_config, backend_configs = serve_module.load_config(
            config,
            no_auth=False,
            host=None,
            port=None,
            verbose=False,
            log=log,
        )
    except typer.Exit:
        raise

    from . import manager

    typer.echo("🚀 Starting Pie engine...")

    try:
        # Start engine and backends
        internal_token, backend_processes = manager.start_engine_and_backend(
            engine_config, backend_configs
        )
        typer.echo("✅ Engine started.")

        # Run the inferlet
        client_config = {
            "host": engine_config["host"],
            "port": engine_config["port"],
            "internal_auth_token": internal_token,
        }
        manager.submit_inferlet_and_wait(
            client_config,
            inferlet,
            arguments or [],
        )

        # Cleanup
        manager.terminate_engine_and_backend(backend_processes)
        typer.echo("✅ Shutdown complete.")

    except KeyboardInterrupt:
        typer.echo("\n⚠️ Interrupted.")
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
