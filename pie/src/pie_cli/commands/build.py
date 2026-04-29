"""Build command implementation for Pie CLI.

Implements: pie build <path> -o <output>

Thin wrapper around bakery's build pipeline so users don't need a
separate `bakery` binary on PATH. Auto-detects Rust / Python / JS / TS
projects.
"""

from pathlib import Path

import typer
from rich.console import Console

from bakery import build as build_cmd

console = Console()


def build(
    input_path: Path = typer.Argument(
        ..., help="Project directory or source file to build."
    ),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Output .wasm file path."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug build (JS/Python: include source maps)."
    ),
) -> None:
    """Build an inferlet to a WebAssembly component.

    Auto-detects the platform: Rust (Cargo.toml), Python
    (pyproject.toml or main.py), or JavaScript/TypeScript (package.json
    or single .js/.ts file).
    """
    try:
        build_cmd.handle_build_command(
            input_path=input_path.expanduser(),
            output=output.expanduser(),
            debug=debug,
        )
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
