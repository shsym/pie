"""Model management commands for Pie CLI.

Implements: pie-server model list|download|remove
Uses HuggingFace Hub as the source for models.
"""

import json
from pathlib import Path

import typer
from huggingface_hub import scan_cache_dir, snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
app = typer.Typer(help="Manage models from HuggingFace")

# Mapping from HuggingFace model_type to Pie architecture
HF_TO_PIE_ARCH = {
    "llama": "llama3",
    "qwen2": "qwen2", 
    "qwen3": "qwen3",
    "gptoss": "gpt_oss",
}


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    return Path.home() / ".cache" / "huggingface" / "hub"


def parse_repo_id_from_dirname(dirname: str) -> str | None:
    """Parse HuggingFace repo ID from cache directory name.
    
    HF cache uses format: models--{org}--{repo}
    Returns: org/repo or None if not a valid model directory
    """
    if not dirname.startswith("models--"):
        return None
    parts = dirname[8:].split("--")  # Remove "models--" prefix
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    elif len(parts) == 1:
        return parts[0]  # No org, just repo name
    return None


def get_model_config(cache_dir: Path, repo_id: str) -> dict | None:
    """Get config.json from cached model snapshot."""
    # Convert repo_id to cache dirname format
    dirname = "models--" + repo_id.replace("/", "--")
    model_cache = cache_dir / dirname
    
    if not model_cache.exists():
        return None
    
    # Find snapshot directory
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None
    
    # Use first available snapshot
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None
    
    config_path = snapshots[0] / "config.json"
    if not config_path.exists():
        return None
    
    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_pie_compatibility(config: dict | None) -> tuple[bool, str]:
    """Check if a model is compatible with Pie.
    
    Returns: (is_compatible, arch_name or reason)
    """
    if config is None:
        return False, "no config"
    
    model_type = config.get("model_type", "")
    if model_type in HF_TO_PIE_ARCH:
        return True, HF_TO_PIE_ARCH[model_type]
    
    return False, f"unsupported type: {model_type}"


@app.command("list")
def model_list() -> None:
    """List locally cached HuggingFace models."""
    cache_dir = get_hf_cache_dir()
    
    if not cache_dir.exists():
        console.print(Panel("[dim]No HuggingFace cache found[/dim]", 
                           title="Models", title_align="left", border_style="dim"))
        return
    
    # Collect models
    models: list[tuple[str, bool, str]] = []  # (repo_id, compatible, info)
    
    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        repo_id = parse_repo_id_from_dirname(entry.name)
        if repo_id is None:
            continue
        
        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)
        models.append((repo_id, compatible, info))
    
    if not models:
        console.print(Panel("[dim]No models found[/dim]", 
                           title="Models", title_align="left", border_style="dim"))
        return
    
    # Build display
    lines = Text()
    for i, (repo_id, compatible, info) in enumerate(sorted(models)):
        if i > 0:
            lines.append("\n")
        
        if compatible:
            lines.append("✓ ", style="green")
            lines.append(repo_id, style="white")
            lines.append(f" ({info})", style="dim")
        else:
            lines.append("○ ", style="dim")
            lines.append(repo_id, style="dim")
            lines.append(f" ({info})", style="dim")
    
    console.print(Panel(lines, title="Models", title_align="left", border_style="dim"))


@app.command("download")
def model_download(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g., meta-llama/Llama-3.2-1B-Instruct)")
) -> None:
    """Download a model from HuggingFace."""
    console.print()
    console.print(f"[bold]Downloading:[/bold] {repo_id}")
    
    try:
        with console.status("[dim]Downloading from HuggingFace...[/dim]"):
            local_path = snapshot_download(
                repo_id,
                local_files_only=False,
            )
        
        console.print(f"[green]✓[/green] Downloaded to {local_path}")
        
        # Check compatibility
        cache_dir = get_hf_cache_dir()
        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)
        
        console.print()
        if compatible:
            console.print(f"[green]✓[/green] Pie compatible (arch: {info})")
            console.print(f"[dim]Add to config.toml:[/dim]")
            console.print(f'  repo_id = "{repo_id}"')
            console.print(f'  arch = "{info}"')
        else:
            console.print(f"[yellow]![/yellow] Not Pie compatible ({info})")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Download failed: {e}")
        raise typer.Exit(1)


@app.command("remove")
def model_remove(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID to remove")
) -> None:
    """Remove a locally cached model."""
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to scan cache: {e}")
        raise typer.Exit(1)
    
    # Find the repo in cache
    target_repo = None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            target_repo = repo
            break
    
    if target_repo is None:
        console.print(f"[red]✗[/red] Model '{repo_id}' not found in cache")
        raise typer.Exit(1)
    
    # Confirm deletion
    size_mb = target_repo.size_on_disk / (1024 * 1024)
    if not typer.confirm(f"Remove {repo_id} ({size_mb:.1f} MB)?"):
        console.print("[dim]Aborted.[/dim]")
        return
    
    # Delete using huggingface_hub
    try:
        delete_strategy = cache_info.delete_revisions(*[rev.commit_hash for rev in target_repo.revisions])
        delete_strategy.execute()
        console.print(f"[green]✓[/green] Removed {repo_id}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to remove: {e}")
        raise typer.Exit(1)
