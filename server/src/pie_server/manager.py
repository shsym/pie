"""Engine and backend management for Pie CLI.

This module handles the lifecycle of the Pie engine and backend services,
mirroring Rust cli/manager.rs functionality.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from . import path as pie_path


def start_engine_and_backend(
    engine_config: dict,
    backend_configs: list[dict],
) -> tuple[str, list[subprocess.Popen]]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        backend_configs: List of backend configurations

    Returns:
        Tuple of (internal_auth_token, list of backend processes)
    """
    from . import pie_server_rs

    # Load authorized users if auth is enabled
    authorized_users_path = None
    if engine_config.get("enable_auth", True):
        auth_path = pie_path.get_authorized_users_path()
        if auth_path.exists():
            authorized_users_path = str(auth_path)

    # Create server config
    server_config = pie_server_rs.ServerConfig(
        host=engine_config.get("host", "127.0.0.1"),
        port=engine_config.get("port", 8080),
        enable_auth=engine_config.get("enable_auth", True),
        cache_dir=engine_config.get("cache_dir"),
        verbose=engine_config.get("verbose", False),
        log_path=engine_config.get("log"),
    )

    # Start the engine
    internal_token = pie_server_rs.start_server(server_config, authorized_users_path)

    # Launch backend processes
    backend_processes: list[subprocess.Popen] = []
    num_backends = 0

    for backend_config in backend_configs:
        backend_type = backend_config.get("backend_type")

        if backend_type == "dummy":
            # Dummy backend is handled internally by the Rust code
            # We just count it for the wait
            num_backends += 1
            typer.echo("- Starting dummy backend")
            # Note: We'd need to call into Rust for the actual dummy backend
            # For now, skip it
            continue

        exec_path = backend_config.get("exec_path")
        if not exec_path:
            typer.echo(f"⚠️ Backend config missing exec_path: {backend_config}", err=True)
            continue

        # Build command with arguments
        cmd = [exec_path]
        cmd.extend(["--host", engine_config.get("host", "127.0.0.1")])
        cmd.extend(["--port", str(engine_config.get("port", 8080))])
        cmd.extend(["--internal_auth_token", internal_token])

        # Add additional backend config options
        for key, value in backend_config.items():
            if key in ("backend_type", "exec_path"):
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        typer.echo(f"- Spawning backend: {exec_path}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
            )
            backend_processes.append(process)
            num_backends += 1
        except Exception as e:
            typer.echo(f"❌ Failed to spawn backend: {e}", err=True)
            # Clean up already-started backends
            for p in backend_processes:
                p.terminate()
            raise typer.Exit(1)

    # TODO: Wait for backends to be ready
    # The Rust implementation uses signals and internal communication
    # For now, we'll just return immediately

    return internal_token, backend_processes


def terminate_engine_and_backend(backend_processes: list[subprocess.Popen]) -> None:
    """Terminate the engine and backend processes.

    Args:
        backend_processes: List of backend subprocess.Popen objects
    """
    import signal

    typer.echo("🔄 Terminating backend processes...")

    for process in backend_processes:
        if process.poll() is None:  # Still running
            pid = process.pid
            typer.echo(f"🔄 Terminating backend process with PID: {pid}")
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                typer.echo(f"  Force killing process {pid}")
                process.kill()
            except Exception as e:
                typer.echo(f"  Error terminating process {pid}: {e}")

    # Note: The Rust engine will shut down when it loses connection
    # In a full implementation, we'd send a shutdown signal to the engine


def run_interactive_shell(engine_config: dict, internal_token: str) -> None:
    """Run the interactive shell session.

    Args:
        engine_config: Engine configuration dict
        internal_token: Internal authentication token
    """
    # Simple readline-based shell
    try:
        import readline
    except ImportError:
        pass  # readline not available on all platforms

    # Load history
    history_path = pie_path.get_shell_history_path()
    try:
        if history_path.exists():
            readline.read_history_file(str(history_path))
    except (OSError, NameError):
        pass

    client_config = {
        "host": engine_config.get("host", "127.0.0.1"),
        "port": engine_config.get("port", 8080),
        "internal_auth_token": internal_token,
    }

    typer.echo("Available commands:")
    typer.echo("  run <path> [ARGS]... - Run a .wasm inferlet with optional arguments")
    typer.echo("  stat                 - Query the backend statistics")
    typer.echo("  exit                 - Exit the Pie session")
    typer.echo("  help                 - Show this help message")

    while True:
        try:
            line = input("pie> ")
        except EOFError:
            typer.echo("Exiting...")
            break
        except KeyboardInterrupt:
            typer.echo("\n(To exit, type 'exit' or press Ctrl-D)")
            continue

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0]
        args = parts[1:]

        if command == "exit":
            typer.echo("Exiting...")
            break
        elif command == "help":
            typer.echo("Available commands:")
            typer.echo("  run <path> [ARGS]... - Run a .wasm inferlet")
            typer.echo("  stat                 - Query backend statistics")
            typer.echo("  exit                 - Exit the session")
            typer.echo("  help                 - Show this help message")
        elif command == "run":
            if not args:
                typer.echo("Usage: run <inferlet_path> [ARGS]...")
                continue
            inferlet_path = Path(args[0]).expanduser()
            inferlet_args = args[1:]
            try:
                submit_inferlet_and_wait(client_config, inferlet_path, inferlet_args)
            except Exception as e:
                typer.echo(f"Error running inferlet: {e}")
        elif command == "stat":
            typer.echo("(stat command not yet implemented)")
        else:
            typer.echo(f"Unknown command: '{command}'. Type 'help' for a list of commands.")

    # Save history
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(history_path))
    except (OSError, NameError):
        pass


def submit_inferlet_and_wait(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
) -> None:
    """Submit an inferlet to the engine and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        arguments: Arguments to pass to the inferlet
    """
    # Check inferlet exists
    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")

    # Read and hash the inferlet
    import hashlib
    inferlet_blob = inferlet_path.read_bytes()
    hash_hex = hashlib.blake2b(inferlet_blob, digest_size=32).hexdigest()
    typer.echo(f"Inferlet hash: {hash_hex}")

    # TODO: Connect to engine and run the inferlet
    # This requires implementing the WebSocket client protocol
    # For now, we'll just print a placeholder message
    typer.echo(f"(Would run inferlet {inferlet_path.name} with args {arguments})")
    typer.echo("Note: Full inferlet execution requires pie-client Python bindings")
