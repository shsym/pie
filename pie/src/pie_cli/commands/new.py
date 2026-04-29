"""New-project command for Pie CLI.

Implements: pie new <name> [--ts]

Thin wrapper around bakery's `create` pipeline. Defaults to Rust to
match `bakery create`; pass --ts for TypeScript. (Python template TBD;
follows whenever bakery grows one.)
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from bakery import create as create_cmd

console = Console()


def new(
    name: str = typer.Argument(..., help="Name of the inferlet project."),
    ts: bool = typer.Option(
        False, "--ts", "-t", help="Create a TypeScript project instead of Rust."
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory."
    ),
) -> None:
    """Create a new inferlet project (Rust by default, or TypeScript with --ts)."""
    try:
        create_cmd.handle_create_command(name=name, rust=not ts, output=output)
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
