"""Model management commands for Pie CLI.

Implements: pie-server model list|add|remove|search|info
"""

import re
from pathlib import Path
from typing import Optional

import httpx
import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from rich.text import Text

from . import path as pie_path

console = Console()
app = typer.Typer(help="Manage local models")

MODEL_INDEX_BASE = "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main"
GITHUB_API_URL = "https://api.github.com/repos/pie-project/model-index/contents"


@app.command("list")
def model_list() -> None:
    """List downloaded models."""
    models_dir = pie_path.get_pie_cache_home() / "models"
    
    if not models_dir.exists():
        console.print(Panel("[dim]No models found[/dim]", title="Local Models", title_align="left", border_style="dim"))
        return

    models = [entry.name for entry in models_dir.iterdir() if entry.is_dir()]
    
    if not models:
        console.print(Panel("[dim]No models found[/dim]", title="Local Models", title_align="left", border_style="dim"))
        return

    lines = "\n".join(sorted(models))
    console.print(Panel(lines, title="Local Models", title_align="left", border_style="dim"))


@app.command("add")
def model_add(model_name: str = typer.Argument(..., help="Name of the model to add")) -> None:
    """Download a model from the model registry."""
    console.print()
    console.print(f"[bold]Adding model:[/bold] {model_name}")

    models_root = pie_path.get_pie_cache_home() / "models"
    model_files_dir = models_root / model_name
    metadata_path = models_root / f"{model_name}.toml"

    # Check if model already exists
    if metadata_path.exists() or model_files_dir.exists():
        overwrite = typer.confirm(
            f"Model '{model_name}' already exists. Overwrite?"
        )
        if not overwrite:
            console.print("[dim]Aborted.[/dim]")
            return

    model_files_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Storing at {model_files_dir}[/dim]")

    # Download metadata
    metadata_url = f"{MODEL_INDEX_BASE}/{model_name}.toml"
    try:
        metadata_raw = download_file_with_progress(metadata_url, "metadata")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            console.print(f"[red]✗[/red] Model '{model_name}' not found in registry")
            raise typer.Exit(1)
        raise

    metadata_str = metadata_raw.decode("utf-8")
    metadata = toml.loads(metadata_str)
    metadata_path.write_text(metadata_str)

    # Download source files
    if "source" in metadata:
        for name, url in metadata["source"].items():
            file_data = download_file_with_progress(url, name)
            (model_files_dir / name).write_bytes(file_data)

    console.print()
    console.print(f"[green]✓[/green] Model '{model_name}' added")


@app.command("remove")
def model_remove(model_name: str = typer.Argument(..., help="Name of the model to remove")) -> None:
    """Delete a downloaded model."""
    models_root = pie_path.get_pie_cache_home() / "models"
    model_files_dir = models_root / model_name
    metadata_path = models_root / f"{model_name}.toml"

    was_removed = False

    if model_files_dir.exists():
        import shutil
        shutil.rmtree(model_files_dir)
        was_removed = True

    if metadata_path.exists():
        metadata_path.unlink()
        was_removed = True

    if was_removed:
        console.print(f"[green]✓[/green] Model '{model_name}' removed")
    else:
        console.print(f"[red]✗[/red] Model '{model_name}' not found locally")
        raise typer.Exit(1)


@app.command("search")
def model_search(
    pattern: Optional[str] = typer.Argument(None, help="Optional regex pattern to filter models"),
) -> None:
    """Search for models in the model registry."""
    with console.status("[dim]Searching registry...[/dim]"):
        try:
            with httpx.Client() as client:
                response = client.get(
                    GITHUB_API_URL,
                    params={"ref": "main"},
                    headers={"User-Agent": "pie-index-list/1.0"},
                )
                response.raise_for_status()
                items = response.json()
        except httpx.HTTPError as e:
            console.print(f"[red]✗[/red] Failed to fetch model index: {e}")
            raise typer.Exit(1)

    # Compile regex if provided
    regex = re.compile(pattern) if pattern else None

    # Collect matching models
    models = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if not name.endswith(".toml") or name == "traits.toml":
            continue

        model_name = name.removesuffix(".toml")
        if regex is None or regex.search(model_name):
            models.append(model_name)

    if not models:
        console.print(Panel("[dim]No models found[/dim]", title="Registry", title_align="left", border_style="dim"))
        return

    lines = Text()
    for i, model in enumerate(sorted(models)):
        if i > 0:
            lines.append("\n")
        lines.append(model, style="white")
    
    title = "Registry" if not pattern else f"Registry ({pattern})"
    console.print(Panel(lines, title=title, title_align="left", border_style="dim"))


@app.command("info")
def model_info(model_name: str = typer.Argument(..., help="Name of the model")) -> None:
    """Show information about a model from the model registry."""
    with console.status(f"[dim]Fetching info...[/dim]"):
        url = f"{MODEL_INDEX_BASE}/{model_name}.toml"
        try:
            with httpx.Client() as client:
                response = client.get(url, headers={"User-Agent": "pie-index-info/1.0"})
                response.raise_for_status()
                toml_content = response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print(f"[red]✗[/red] Model '{model_name}' not found in registry")
                raise typer.Exit(1)
            raise

    parsed = toml.loads(toml_content)

    # Architecture info
    lines = Text()
    if "architecture" in parsed:
        arch = parsed["architecture"]
        first = True
        for key, value in arch.items():
            if not first:
                lines.append("\n")
            first = False
            lines.append(f"{key:<20}", style="white")
            lines.append(str(value), style="dim")
    
    console.print(Panel(lines, title=model_name, title_align="left", border_style="dim"))

    # Download status
    models_root = pie_path.get_pie_cache_home() / "models"
    model_files_dir = models_root / model_name
    metadata_path = models_root / f"{model_name}.toml"

    console.print()
    if model_files_dir.exists() and metadata_path.exists():
        console.print(f"[green]✓[/green] Downloaded at [dim]{model_files_dir}[/dim]")
    else:
        console.print(f"[dim]○ Not downloaded. Run:[/dim] pie model add {model_name}")


def download_file_with_progress(url: str, name: str) -> bytes:
    """Download a file with a progress bar."""
    with httpx.Client(follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(name, total=total_size or None)
                content = bytearray()

                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    progress.update(task, advance=len(chunk))

            return bytes(content)
