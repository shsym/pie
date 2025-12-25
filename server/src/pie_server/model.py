"""Model management commands for Pie CLI.

Implements: pie-server model list|add|remove|search|info
"""

import re
from pathlib import Path
from typing import Optional

import httpx
import toml
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

from . import path as pie_path

app = typer.Typer(help="Manage local models")

MODEL_INDEX_BASE = "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main"
GITHUB_API_URL = "https://api.github.com/repos/pie-project/model-index/contents"


@app.command("list")
def model_list() -> None:
    """List downloaded models."""
    typer.echo("📚 Available local models:")

    models_dir = pie_path.get_pie_cache_home() / "models"
    if not models_dir.exists():
        typer.echo("  No models found.")
        return

    found = False
    for entry in models_dir.iterdir():
        if entry.is_dir():
            typer.echo(f"  - {entry.name}")
            found = True

    if not found:
        typer.echo("  No models found.")


@app.command("add")
def model_add(model_name: str = typer.Argument(..., help="Name of the model to add")) -> None:
    """Download a model from the model registry."""
    typer.echo(f"➕ Adding model: {model_name}")

    models_root = pie_path.get_pie_cache_home() / "models"
    model_files_dir = models_root / model_name
    metadata_path = models_root / f"{model_name}.toml"

    # Check if model already exists
    if metadata_path.exists() or model_files_dir.exists():
        overwrite = typer.confirm(
            f"⚠️ Model '{model_name}' already exists. Overwrite?"
        )
        if not overwrite:
            typer.echo("Aborted by user.")
            return

    model_files_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Parameters will be stored at {model_files_dir}")

    # Download metadata
    metadata_url = f"{MODEL_INDEX_BASE}/{model_name}.toml"
    try:
        metadata_raw = download_file_with_progress(metadata_url, "Downloading metadata...")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            typer.echo(f"❌ Model '{model_name}' not found in the official index.", err=True)
            raise typer.Exit(1)
        raise

    metadata_str = metadata_raw.decode("utf-8")
    metadata = toml.loads(metadata_str)
    metadata_path.write_text(metadata_str)

    # Download source files
    if "source" in metadata:
        for name, url in metadata["source"].items():
            file_data = download_file_with_progress(url, f"Downloading {name}...")
            (model_files_dir / name).write_bytes(file_data)

    typer.echo(f"✅ Model '{model_name}' added successfully!")


@app.command("remove")
def model_remove(model_name: str = typer.Argument(..., help="Name of the model to remove")) -> None:
    """Delete a downloaded model."""
    typer.echo(f"🗑️ Removing model: {model_name}")

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
        typer.echo(f"✅ Model '{model_name}' removed.")
    else:
        typer.echo(f"❌ Model '{model_name}' not found locally.", err=True)
        raise typer.Exit(1)


@app.command("search")
def model_search(
    pattern: Optional[str] = typer.Argument(None, help="Optional regex pattern to filter models"),
) -> None:
    """Search for models in the model registry."""
    typer.echo("🔍 Searching for models...")

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
        typer.echo(f"❌ Failed to fetch model index: {e}", err=True)
        raise typer.Exit(1)

    # Compile regex if provided
    regex = re.compile(pattern) if pattern else None

    # Print matching model names
    for item in items:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if not name.endswith(".toml") or name == "traits.toml":
            continue

        model_name = name.removesuffix(".toml")
        if regex is None or regex.search(model_name):
            typer.echo(model_name)


@app.command("info")
def model_info(model_name: str = typer.Argument(..., help="Name of the model")) -> None:
    """Show information about a model from the model registry."""
    typer.echo(f"📋 Getting model information for '{model_name}'...")

    # Fetch the TOML file
    url = f"{MODEL_INDEX_BASE}/{model_name}.toml"
    try:
        with httpx.Client() as client:
            response = client.get(url, headers={"User-Agent": "pie-index-info/1.0"})
            response.raise_for_status()
            toml_content = response.text
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            typer.echo(f"❌ Model '{model_name}' not found in the registry", err=True)
            raise typer.Exit(1)
        raise

    parsed = toml.loads(toml_content)

    # Display architecture section
    if "architecture" in parsed:
        typer.echo("\n🏗️  Architecture:")
        _print_toml_table(parsed["architecture"], indent=1)
    else:
        typer.echo("❌ No architecture section found in the model configuration")

    # Check if downloaded locally
    models_root = pie_path.get_pie_cache_home() / "models"
    model_files_dir = models_root / model_name
    metadata_path = models_root / f"{model_name}.toml"

    typer.echo("\n📦 Download Status:")
    if model_files_dir.exists() and metadata_path.exists():
        typer.echo("  ✅ Downloaded locally")
        typer.echo(f"  📁 Location: {model_files_dir}")
    else:
        typer.echo("  ❌ Not downloaded")
        typer.echo(f"  💡 Use `pie-server model add {model_name}` to download")


def download_file_with_progress(url: str, message: str) -> bytes:
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
            ) as progress:
                task = progress.add_task(message, total=total_size or None)
                content = bytearray()

                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    progress.update(task, advance=len(chunk))

            return bytes(content)


def _print_toml_table(table: dict, indent: int = 0) -> None:
    """Print a TOML table with proper indentation."""
    prefix = "  " * indent
    for key, value in table.items():
        if isinstance(value, dict):
            typer.echo(f"{prefix}{key}:")
            _print_toml_table(value, indent + 1)
        else:
            typer.echo(f"{prefix}{key}: {value}")
