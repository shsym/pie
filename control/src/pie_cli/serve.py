"""Serve command implementation for Pie CLI.

Implements: pie-server serve
Starts the Pie engine and optionally provides an interactive shell session.
"""

from pathlib import Path

import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import path as pie_path

console = Console()


def load_config(
    config_path: Path | None,
) -> tuple[dict, list[dict]]:
    """Load and merge configuration from file and CLI arguments.

    Returns:
        Tuple of (engine_config, model_configs)
    """
    # Load config file
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        console.print(f"[red]✗[/red] Configuration not found at {file_path}")
        console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    config = toml.loads(file_path.read_text())

    # Build engine config with CLI overrides
    engine_config = {
        "host": config.get("host", "127.0.0.1"),
        "port": config.get("port", 8080),
        "enable_auth": config.get("enable_auth", True),
        "cache_dir": config.get("cache_dir", str(pie_path.get_pie_home() / "cache")),
        "verbose": config.get("verbose", False),
        "log_dir": config.get("log_dir", str(pie_path.get_pie_home() / "logs")),
        "registry": config.get("registry", "https://registry.pie-project.org/"),
    }

    model_configs = config.get("model", [])
    if not model_configs:
        console.print("[red]✗[/red] No model configuration found")
        raise typer.Exit(1)

    return engine_config, model_configs


def serve(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
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
        engine_config, model_configs = load_config(config)
    except typer.Exit:
        raise

    # Import here to avoid circular imports and allow module to load without Rust
    from . import manager

    console.print()

    # Show config summary
    mode = "interactive" if interactive else "server"
    lines = Text()
    lines.append(f"{'Host':<15}", style="white")
    lines.append(f"{engine_config['host']}:{engine_config['port']}\n", style="dim")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model_configs[0].get('hf_repo', 'unknown')}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    device = model_configs[0].get("device", ["unknown"])
    device_str = ", ".join(device) if isinstance(device, list) else device
    lines.append(device_str, style="dim")

    console.print(
        Panel(
            lines, title=f"Pie Engine ({mode})", title_align="left", border_style="dim"
        )
    )
    console.print()

    server_handle = None
    backend_processes = []

    try:
        # Start engine and backends
        # manager.start_engine_and_backend handles the spinner UI
        server_handle, backend_processes = manager.start_engine_and_backend(
            engine_config, model_configs, console=console
        )

        # console.print("[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]")

        if interactive:
            console.print("[dim]Type 'help' for commands, ↑/↓ for history[/dim]")
            console.print()
            manager.run_interactive_shell(engine_config, server_handle.internal_token)
        else:
            import signal

            import time

            try:
                # Keep running while processes are alive
                while True:
                    if not manager.check_backend_processes(backend_processes):
                        typer.echo("[red]A backend process died. Shutting down.[/red]")
                        break

                    if server_handle and hasattr(server_handle, "is_running"):
                        if not server_handle.is_running():
                            typer.echo("[red]Engine process died. Shutting down.[/red]")
                            break

                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        # Cleanup
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")

    except KeyboardInterrupt:
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)
