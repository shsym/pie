"""LLM Serving Monitor - A text-based monitoring tool for LLM serving systems."""

__version__ = "0.1.0"

import typer

app = typer.Typer(
    name="llm-monitor",
    help="Text-based monitoring tool for LLM serving systems",
    add_completion=False,
)


@app.command()
def run(
    gpus: int = typer.Option(8, "--gpus", "-g", help="Number of GPUs to simulate"),
    tp_groups: int = typer.Option(8, "--tp-groups", "-t", help="Number of TP groups"),
    refresh: float = typer.Option(
        0.5, "--refresh", "-r", help="Refresh rate in seconds"
    ),
):
    """Launch the LLM serving monitor TUI."""
    from .app import LLMMonitorApp

    app_instance = LLMMonitorApp(
        num_gpus=gpus,
        num_tp_groups=tp_groups,
        refresh_rate=refresh,
    )
    app_instance.run()


@app.command()
def version():
    """Show the version."""
    typer.echo(f"llm-monitor version {__version__}")


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
