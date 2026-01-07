"""Model management commands for Pie CLI.

Implements: pie model list|download|remove
Uses HuggingFace Hub as the source for models.
"""

import typer
from huggingface_hub import scan_cache_dir, snapshot_download
from rich.console import Console
from rich.panel import Panel

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

from pie.model import (
    get_hf_cache_dir,
    parse_repo_id_from_dirname,
    get_model_config,
    check_pie_compatibility,

)
import typing


class TqdmProgress:
    """Adapter to map tqdm calls to a global rich Progress instance.
    
    Aggregates all byte downloads into a single master progress bar.
    """

    _progress: typing.Optional[Progress] = None
    _master_task_id: typing.Optional[int] = None
    _total_bytes: int = 0
    _completed_bytes: int = 0
    _lock: typing.Optional["threading.Lock"] = None

    def __init__(
        self,
        iterable=None,
        desc: str | None = None,
        total: float | None = None,
        unit: str = "it",
        unit_scale: bool = False,
        miniters: int | None = None,
        ncols: int | None = None,
        **kwargs,
    ):
        import threading
        self.iterable = iterable
        self.desc = desc
        self.total = total or 0
        self.unit = unit
        self.task_id = None  # Individual tasks are hidden
        self._is_byte_task = (unit == "B")

        # We only care about byte tasks (actual file downloads)
        # Skip "Fetching X files" bar and other non-byte bars
        if not self._is_byte_task:
            return

        # Update master task total
        if self._progress and TqdmProgress._master_task_id is not None:
            with self.get_lock():
                TqdmProgress._total_bytes += int(self.total)
                self._progress.update(
                    TqdmProgress._master_task_id,
                    total=TqdmProgress._total_bytes,
                )

    def update(self, n: float = 1) -> None:
        if not self._is_byte_task:
            return
        if self._progress and TqdmProgress._master_task_id is not None:
            with self.get_lock():
                TqdmProgress._completed_bytes += int(n)
                self._progress.update(
                    TqdmProgress._master_task_id,
                    completed=TqdmProgress._completed_bytes,
                )

    def close(self) -> None:
        pass  # Master task is closed by the context manager

    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
                self.update(1)

    @classmethod
    def get_lock(cls):
        import threading
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def refresh(self, nolock=False, lock_args=None):
        if self._progress:
            self._progress.refresh()

    def reset(self, total=None):
        pass  # Not applicable for aggregated progress

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        console.print(s, end=end)

    def set_description(self, desc=None, refresh=True):
        pass  # Description is fixed for master task

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        pass

    def disable(self):
        pass


class SmartDownloadColumn(ProgressColumn):
    """Renders file size if unit is 'B', otherwise 'completed/total unit'."""

    def render(self, task: "Task") -> Text:
        unit = task.fields.get("unit", "it")
        if unit == "B":
            return DownloadColumn().render(task)
        
        if task.total is None:
            return Text(f"{int(task.completed)} {unit}", style="progress.download")
        
        return Text(f"{int(task.completed)}/{int(task.total)} {unit}", style="progress.download")


class SmartTransferSpeedColumn(ProgressColumn):
    """Renders transfer speed if unit is 'B', otherwise 'speed unit/s'."""

    def render(self, task: "Task") -> Text:
        unit = task.fields.get("unit", "it")
        if unit == "B":
            return TransferSpeedColumn().render(task)
        
        if task.speed is None:
            return Text("?", style="progress.data.speed")
        
        return Text(f"{task.speed:.1f} {unit}/s", style="progress.data.speed")





console = Console()
app = typer.Typer(help="Manage models from HuggingFace")


@app.command("list")
def model_list() -> None:
    """List locally cached HuggingFace models."""
    cache_dir = get_hf_cache_dir()

    if not cache_dir.exists():
        console.print(
            Panel(
                "[dim]No HuggingFace cache found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
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
        console.print(
            Panel(
                "[dim]No models found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
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
    repo_id: str = typer.Argument(
        ..., help="HuggingFace repo ID (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
) -> None:
    """Download a model from HuggingFace."""
    console.print()
    console.print(f"[bold]Downloading:[/bold] {repo_id}")

    try:
        # Configuration for the progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        # Reset aggregation state
        TqdmProgress._progress = progress
        TqdmProgress._total_bytes = 0
        TqdmProgress._completed_bytes = 0

        with progress:
            # Create master task for aggregated byte progress
            TqdmProgress._master_task_id = progress.add_task(
                description="Downloading...",
                total=None,  # Unknown until files start downloading
            )

            local_path = snapshot_download(
                repo_id,
                local_files_only=False,
                tqdm_class=TqdmProgress,
            )

        # Cleanup global references
        TqdmProgress._progress = None
        TqdmProgress._master_task_id = None

        console.print(f"[green]✓[/green] Downloaded to {local_path}")

        # Check compatibility
        cache_dir = get_hf_cache_dir()
        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)

        console.print()
        if compatible:
            console.print(f"[green]✓[/green] Pie compatible (arch: {info})")
            console.print("Add to config.toml:")
            console.print(f'  hf_repo = "{repo_id}"')
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
        delete_strategy = cache_info.delete_revisions(
            *[rev.commit_hash for rev in target_repo.revisions]
        )
        delete_strategy.execute()
        console.print(f"[green]✓[/green] Removed {repo_id}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to remove: {e}")
        raise typer.Exit(1)
