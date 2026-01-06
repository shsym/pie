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


def create_default_config_content() -> str:
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
        "model": [
            {
                "hf_repo": "Qwen/Qwen3-0.6B",
                "device": ["cuda:0"],
                "activation_dtype": "bfloat16",
                "weight_dtype": "auto",
                "kv_page_size": 16,
                "max_batch_tokens": 10240,
                "max_dist_size": 32,
                "max_num_embeds": 128,
                "max_num_adapters": 32,
                "max_adapter_rank": 8,
                "gpu_mem_utilization": 0.9,
                "max_batch_size": 128,
                "enable_profiling": False,
                "random_seed": 42,
                "use_cuda_graphs": True,
            }
        ],
    }

    return toml.dumps(config)


@app.command("init")
def config_init(
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
    config_content = create_default_config_content()
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
    console.print(
        Panel(syntax, title=str(config_path), title_align="left", border_style="dim")
    )


@app.command("update")
def config_update(
    # Engine configuration options
    host: Optional[str] = typer.Option(None, "--host", help="Network host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", help="Network port to use"),
    enable_auth: Optional[bool] = typer.Option(
        None, "--enable-auth", help="Enable/disable authentication"
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Cache directory path"
    ),
    verbose: Optional[bool] = typer.Option(
        None, "--verbose", help="Enable verbose logging"
    ),
    log_dir: Optional[str] = typer.Option(None, "--log-dir", help="Log directory path"),
    registry: Optional[str] = typer.Option(
        None, "--registry", help="Inferlet registry URL"
    ),
    # Model configuration options
    model_hf_repo: Optional[str] = typer.Option(
        None,
        "--hf-repo",
        help="HuggingFace repo (e.g., meta-llama/Llama-3.2-1B-Instruct)",
    ),
    model_device: Optional[list[str]] = typer.Option(
        None, "--device", help="Device(s) (e.g., cuda:0 cuda:1)"
    ),
    model_activation_dtype: Optional[str] = typer.Option(
        None, "--activation-dtype", help="Activation dtype (e.g., bfloat16)"
    ),
    model_weight_dtype: Optional[str] = typer.Option(
        None,
        "--weight-dtype",
        help="Weight dtype: auto, float32, float16, bfloat16, int4, int8, float8",
    ),
    model_kv_page_size: Optional[int] = typer.Option(
        None, "--kv-page-size", help="KV page size"
    ),
    model_max_batch_tokens: Optional[int] = typer.Option(
        None, "--max-batch-tokens", help="Maximum batch tokens"
    ),
    model_max_dist_size: Optional[int] = typer.Option(
        None, "--max-dist-size", help="Maximum distribution size"
    ),
    model_max_num_embeds: Optional[int] = typer.Option(
        None, "--max-num-embeds", help="Maximum number of embeddings"
    ),
    model_max_num_adapters: Optional[int] = typer.Option(
        None, "--max-num-adapters", help="Maximum number of adapters"
    ),
    model_max_adapter_rank: Optional[int] = typer.Option(
        None, "--max-adapter-rank", help="Maximum adapter rank"
    ),
    model_gpu_mem_utilization: Optional[float] = typer.Option(
        None, "--gpu-mem-utilization", help="GPU memory utilization (0.0 to 1.0)"
    ),
    model_max_batch_size: Optional[int] = typer.Option(
        None, "--max-batch-size", help="Maximum batch size (requests)"
    ),
    model_enable_profiling: Optional[bool] = typer.Option(
        None, "--enable-profiling", help="Enable profiling"
    ),
    model_random_seed: Optional[int] = typer.Option(
        None, "--random-seed", help="Random seed for model"
    ),
    model_use_cuda_graphs: Optional[bool] = typer.Option(
        None,
        "--use-cuda-graphs/--no-use-cuda-graphs",
        help="Enable/disable CUDA graphs",
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

    # Collect model updates
    model_updates = {
        k: v
        for k, v in {
            "hf_repo": model_hf_repo,
            "device": model_device,
            "activation_dtype": model_activation_dtype,
            "weight_dtype": model_weight_dtype,
            "kv_page_size": model_kv_page_size,
            "max_batch_tokens": model_max_batch_tokens,
            "max_dist_size": model_max_dist_size,
            "max_num_embeds": model_max_num_embeds,
            "max_num_adapters": model_max_num_adapters,
            "max_adapter_rank": model_max_adapter_rank,
            "gpu_mem_utilization": model_gpu_mem_utilization,
            "max_batch_size": model_max_batch_size,
            "enable_profiling": model_enable_profiling,
            "random_seed": model_random_seed,
            "use_cuda_graphs": model_use_cuda_graphs,
        }.items()
        if v is not None
    }

    if not engine_updates and not model_updates:
        console.print("[yellow]![/yellow] No options provided")
        console.print(
            "[dim]Run 'pie config update --help' to see available options.[/dim]"
        )
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

    # Update model configuration (first model entry)
    if model_updates:
        if not config.get("model"):
            console.print("[red]✗[/red] No model configuration found")
            raise typer.Exit(1)

        for key, value in model_updates.items():
            config["model"][0][key] = value
            console.print(f"[green]✓[/green] model.{key} = {value}")

    # Write the updated config
    config_path.write_text(toml.dumps(config))
    console.print()
    console.print(f"[dim]Saved to {config_path}[/dim]")
