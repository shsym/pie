"""Run command implementation for Pie CLI.

Implements: pie-server run <inferlet> [args]
Runs an inferlet with a one-shot Pie engine instance.
"""

from pathlib import Path
from typing import Optional

import typer

from . import serve as serve_module


def run(
    inferlet: Optional[str] = typer.Argument(
        None, help="Inferlet name from registry (e.g., 'std/text-completion@0.1.0')"
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p",
        help="Path to a local .wasm inferlet file"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    arguments: Optional[list[str]] = typer.Argument(
        None, help="Arguments to pass to the inferlet"
    ),
) -> None:
    """Run an inferlet with a one-shot Pie engine.

    This command starts the Pie engine, runs the specified inferlet,
    waits for it to complete, and then shuts down.
    
    You can specify an inferlet either by registry name or by path (mutually exclusive):
    
    - By registry: pie run std/text-completion@0.1.0
    - By path: pie run --path ./my_inferlet.wasm
    """
    # Validate mutually exclusive options
    if inferlet is None and path is None:
        typer.echo("❌ Error: Must specify either an inferlet name or --path", err=True)
        raise typer.Exit(1)
    
    if inferlet is not None and path is not None:
        typer.echo("❌ Error: Cannot specify both inferlet name and --path", err=True)
        raise typer.Exit(1)
    
    # Verify inferlet exists if using path
    if path is not None and not path.exists():
        typer.echo(f"❌ Inferlet file not found: {path}", err=True)
        raise typer.Exit(1)

    try:
        engine_config, backend_configs = serve_module.load_config(config)
    except typer.Exit:
        raise

    from . import manager

    typer.echo("🚀 Starting Pie engine...")

    server_handle = None
    backend_processes = []

    try:
        # Start engine and backends
        server_handle, backend_processes = manager.start_engine_and_backend(
            engine_config, backend_configs
        )

        # Run the inferlet
        client_config = {
            "host": engine_config["host"],
            "port": engine_config["port"],
            "internal_auth_token": server_handle.internal_token,
        }
        
        if path is not None:
            manager.submit_inferlet_and_wait(
                client_config,
                path,
                arguments or [],
            )
        else:
            # Launch from registry
            manager.submit_inferlet_from_registry_and_wait(
                client_config,
                inferlet,
                arguments or [],
            )

        # Cleanup
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        typer.echo("✅ Shutdown complete.")

    except KeyboardInterrupt:
        typer.echo("\n⚠️ Interrupted.")
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)
