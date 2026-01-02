"""Configuration management commands for Pie CLI.

Implements: pie-server config init|update|show
"""

from pathlib import Path
from typing import Optional

import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from . import path as pie_path

console = Console()
app = typer.Typer(help="Manage configuration")


def create_default_config_content(backend_type: str = "python") -> str:
    """Create the default configuration file content."""
    cache_dir = str(pie_path.get_pie_home() / "cache")
    log_dir = str(pie_path.get_pie_home() / "logs")
    config = {
        "host": "127.0.0.1",
        "port": 8080,
        "enable_auth": False,
        "cache_dir": cache_dir,
        "verbose": False,
        "log_dir": log_dir,
        "registry": "https://registry.pie-project.org/",
    }

    if backend_type == "dummy":
        config["backend"] = [{"backend_type": "dummy"}]
    else:
        config["backend"] = [
            {
                "backend_type": "python",
                "exec_path": "pie-backend",
                "model": "qwen-3-0.6b",
                "device": ["cuda:0"],
                "activation_dtype": "bfloat16",
                "kv_page_size": 16,
                "max_batch_tokens": 10240,
                "max_dist_size": 32,
                "max_num_embeds": 128,
                "max_num_adapters": 32,
                "max_adapter_rank": 8,
                "gpu_mem_utilization": 0.9,
                "enable_profiling": False,
                "random_seed": 42,
            }
        ]

    return toml.dumps(config)


@app.command("init")
def config_init(
    dummy: bool = typer.Option(False, "--dummy", help="Initialize with dummy backend"),
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Create a default config file."""
    config_path = Path(path) if path else pie_path.get_default_config_path()

    # Check if config file already exists
    if config_path.exists():
        overwrite = typer.confirm(
            f"Configuration already exists at {config_path}. Overwrite?"
        )
        if not overwrite:
            console.print("[dim]Aborted.[/dim]")
            return

    # Create the directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the default config file
    backend_type = "dummy" if dummy else "python"
    config_content = create_default_config_content(backend_type)
    config_path.write_text(config_content)

    console.print(f"[green]✓[/green] Created {config_path}")
    console.print()
    
    syntax = Syntax(config_content, "toml", theme="ansi_dark", line_numbers=False)
    console.print(Panel(syntax, title="Config", title_align="left", border_style="dim"))


@app.command("show")
def config_show(
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Show the content of the config file."""
    config_path = Path(path) if path else pie_path.get_default_config_path()

    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration not found at {config_path}")
        console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    config_content = config_path.read_text()
    syntax = Syntax(config_content, "toml", theme="ansi_dark", line_numbers=False)
    console.print(Panel(syntax, title=str(config_path), title_align="left", border_style="dim"))


@app.command("update")
def config_update(
    # Engine configuration options
    host: Optional[str] = typer.Option(None, "--host", help="Network host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", help="Network port to use"),
    enable_auth: Optional[bool] = typer.Option(
        None, "--enable-auth", help="Enable/disable authentication"
    ),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", help="Cache directory path"),
    verbose: Optional[bool] = typer.Option(None, "--verbose", help="Enable verbose logging"),
    log_dir: Optional[str] = typer.Option(None, "--log-dir", help="Log directory path"),
    registry: Optional[str] = typer.Option(None, "--registry", help="Inferlet registry URL"),
    # Backend configuration options
    backend_type: Optional[str] = typer.Option(None, "--backend-type", help="Backend type"),
    backend_exec_path: Optional[str] = typer.Option(
        None, "--backend-exec-path", help="Backend executable path"
    ),
    backend_model: Optional[str] = typer.Option(None, "--model", help="Model name"),
    backend_device: Optional[list[str]] = typer.Option(
        None, "--device", help="Device(s) (e.g., cuda:0 cuda:1)"
    ),
    backend_activation_dtype: Optional[str] = typer.Option(
        None, "--activation-dtype", help="Activation dtype (e.g., bfloat16)"
    ),
    backend_weight_dtype: Optional[str] = typer.Option(
        None, "--weight-dtype", help="Weight dtype (e.g., int4, int8)"
    ),
    backend_kv_page_size: Optional[int] = typer.Option(
        None, "--kv-page-size", help="KV page size"
    ),
    backend_max_batch_tokens: Optional[int] = typer.Option(
        None, "--max-batch-tokens", help="Maximum batch tokens"
    ),
    backend_max_dist_size: Optional[int] = typer.Option(
        None, "--max-dist-size", help="Maximum distribution size"
    ),
    backend_max_num_embeds: Optional[int] = typer.Option(
        None, "--max-num-embeds", help="Maximum number of embeddings"
    ),
    backend_max_num_adapters: Optional[int] = typer.Option(
        None, "--max-num-adapters", help="Maximum number of adapters"
    ),
    backend_max_adapter_rank: Optional[int] = typer.Option(
        None, "--max-adapter-rank", help="Maximum adapter rank"
    ),
    backend_gpu_mem_utilization: Optional[float] = typer.Option(
        None, "--gpu-mem-utilization", help="GPU memory utilization (0.0 to 1.0)"
    ),
    backend_enable_profiling: Optional[bool] = typer.Option(
        None, "--enable-profiling", help="Enable profiling"
    ),
    backend_random_seed: Optional[int] = typer.Option(
        None, "--backend-random-seed", help="Random seed for backend"
    ),
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Update the entries of the config file."""
    # Collect engine updates
    engine_updates = {
        k: v
        for k, v in {
            "host": host,
            "port": port,
            "enable_auth": enable_auth,
            "cache_dir": cache_dir,
            "verbose": verbose,
            "log_dir": log_dir,
            "registry": registry,
        }.items()
        if v is not None
    }

    # Collect backend updates
    backend_updates = {
        k: v
        for k, v in {
            "backend_type": backend_type,
            "exec_path": backend_exec_path,
            "model": backend_model,
            "device": backend_device,
            "activation_dtype": backend_activation_dtype,
            "weight_dtype": backend_weight_dtype,
            "kv_page_size": backend_kv_page_size,
            "max_batch_tokens": backend_max_batch_tokens,
            "max_dist_size": backend_max_dist_size,
            "max_num_embeds": backend_max_num_embeds,
            "max_num_adapters": backend_max_num_adapters,
            "max_adapter_rank": backend_max_adapter_rank,
            "gpu_mem_utilization": backend_gpu_mem_utilization,
            "enable_profiling": backend_enable_profiling,
            "random_seed": backend_random_seed,
        }.items()
        if v is not None
    }

    if not engine_updates and not backend_updates:
        console.print("[yellow]![/yellow] No options provided")
        console.print("[dim]Run 'pie config update --help' to see available options.[/dim]")
        return

    config_path = Path(path) if path else pie_path.get_default_config_path()

    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration not found at {config_path}")
        console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    # Read and parse the existing config
    config = toml.loads(config_path.read_text())

    # Update engine configuration
    for key, value in engine_updates.items():
        config[key] = value
        console.print(f"[green]✓[/green] {key} = {value}")

    # Update backend configuration (first backend entry)
    if backend_updates:
        if not config.get("backend"):
            console.print("[red]✗[/red] No backend configuration found")
            raise typer.Exit(1)

        for key, value in backend_updates.items():
            config["backend"][0][key] = value
            console.print(f"[green]✓[/green] backend.{key} = {value}")

    # Write the updated config
    config_path.write_text(toml.dumps(config))
    console.print()
    console.print(f"[dim]Saved to {config_path}[/dim]")
