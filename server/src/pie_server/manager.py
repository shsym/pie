"""Engine and backend management for Pie CLI.

This module handles the lifecycle of the Pie engine and backend services,
mirroring Rust cli/manager.rs functionality.
"""

import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import typer

from . import path as pie_path

if TYPE_CHECKING:
    from . import pie_server_rs


def start_engine_and_backend(
    engine_config: dict,
    backend_configs: list[dict],
    timeout: float = 60.0,
) -> tuple["pie_server_rs.ServerHandle", list[subprocess.Popen]]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        backend_configs: List of backend configurations
        timeout: Maximum time to wait for backends to connect (seconds)

    Returns:
        Tuple of (ServerHandle, list of backend processes)
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

    # Start the engine - returns a ServerHandle
    server_handle = pie_server_rs.start_server(server_config, authorized_users_path)
    typer.echo(f"✅ Engine started (token: {server_handle.internal_token[:8]}...)")

    # Count expected backends
    expected_backends = 0
    
    # Launch backend processes
    backend_processes: list["multiprocessing.Process | subprocess.Popen"] = []

    for backend_config in backend_configs:
        backend_type = backend_config.get("backend_type")

        if backend_type == "dummy":
            # Dummy backend is handled internally by the Rust code
            expected_backends += 1
            typer.echo("- Starting dummy backend")
            # TODO: Call into Rust to start dummy backend
            continue

        if backend_type == "python":
            # Spawn pie-backend directly using multiprocessing
            typer.echo("- Spawning Python backend (pie-backend)")
            try:
                process = spawn_python_backend(
                    engine_config, 
                    backend_config, 
                    server_handle.internal_token
                )
                backend_processes.append(process)
                expected_backends += 1
            except Exception as e:
                typer.echo(f"❌ Failed to spawn backend: {e}", err=True)
                for p in backend_processes:
                    p.terminate()
                server_handle.shutdown()
                raise typer.Exit(1)
            continue

        # Fallback: external backend via exec_path (for custom backends)
        exec_path = backend_config.get("exec_path")
        if not exec_path:
            typer.echo(f"⚠️ Backend config missing exec_path: {backend_config}", err=True)
            continue

        # Build command with arguments
        cmd = [exec_path]
        cmd.extend(["--host", engine_config.get("host", "127.0.0.1")])
        cmd.extend(["--port", str(engine_config.get("port", 8080))])
        cmd.extend(["--internal-auth-token", server_handle.internal_token])

        # Add additional backend config options
        for key, value in backend_config.items():
            if key in ("backend_type", "exec_path"):
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        typer.echo(f"- Spawning external backend: {exec_path}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            backend_processes.append(process)
            expected_backends += 1
        except Exception as e:
            typer.echo(f"❌ Failed to spawn backend: {e}", err=True)
            # Clean up already-started backends
            for p in backend_processes:
                p.terminate()
            server_handle.shutdown()
            raise typer.Exit(1)

    # Wait for backends to register with the engine
    if expected_backends > 0:
        typer.echo(f"⏳ Waiting for {expected_backends} backend(s) to connect...")
        if not wait_for_backends(server_handle, expected_backends, timeout, backend_processes):
            typer.echo("❌ Timeout waiting for backends to connect", err=True)
            for p in backend_processes:
                p.terminate()
            server_handle.shutdown()
            raise typer.Exit(1)
        typer.echo(f"✅ {expected_backends} backend(s) connected")

    return server_handle, backend_processes


def spawn_python_backend(
    engine_config: dict,
    backend_config: dict,
    internal_token: str,
) -> multiprocessing.Process:
    """Spawn a Python backend process directly using multiprocessing.

    This avoids the overhead of subprocess and directly imports pie_backend,
    which is faster and more reliable than shelling out.

    Args:
        engine_config: Engine configuration dict (host, port)
        backend_config: Backend configuration dict
        internal_token: Internal authentication token

    Returns:
        multiprocessing.Process object
    """
    # Build kwargs for pie_backend.main()
    backend_kwargs = {
        "host": engine_config.get("host", "127.0.0.1"),
        "port": engine_config.get("port", 8080),
        "internal_auth_token": internal_token,
        "model": backend_config.get("model"),
        "device": backend_config.get("device"),
        "cache_dir": backend_config.get("cache_dir"),
        "kv_page_size": backend_config.get("kv_page_size", 16),
        "max_dist_size": backend_config.get("max_dist_size", 64),
        "max_num_embeds": backend_config.get("max_num_embeds", 128),
        "max_batch_tokens": backend_config.get("max_batch_tokens", 10240),
        "max_num_adapters": backend_config.get("max_num_adapters", 48),
        "max_adapter_rank": backend_config.get("max_adapter_rank", 8),
        "gpu_mem_utilization": backend_config.get("gpu_mem_utilization", 0.9),
        "activation_dtype": backend_config.get("activation_dtype", "bfloat16"),
        "weight_dtype": backend_config.get("weight_dtype"),
        "enable_profiling": backend_config.get("enable_profiling", False),
    }
    
    # Remove None values
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}

    # Use spawn context for CUDA compatibility
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_run_backend_process,
        kwargs=backend_kwargs,
        daemon=False,  # Allow cleanup
    )
    process.start()
    return process


def _run_backend_process(**kwargs):
    """Target function for the backend process.
    
    This runs in a separate process and imports/calls pie_backend.main().
    """
    from pie_backend.__main__ import main
    main(**kwargs)


def wait_for_backends(
    server_handle: "pie_server_rs.ServerHandle",
    expected_count: int,
    timeout: float,
    backend_processes: list[subprocess.Popen],
) -> bool:
    """Wait for the expected number of backends to register with the engine.

    Args:
        server_handle: The server handle to query
        expected_count: Number of backends we expect to connect
        timeout: Maximum time to wait in seconds
        backend_processes: List of backend processes to check for early exit

    Returns:
        True if all backends connected, False if timeout
    """
    start_time = time.time()
    poll_interval = 0.5  # seconds
    
    while time.time() - start_time < timeout:
        # Check if any backend process has died
        for i, process in enumerate(backend_processes):
            if process.poll() is not None:
                # Process has exited
                stderr = process.stderr.read().decode() if process.stderr else ""
                typer.echo(f"❌ Backend process exited unexpectedly (exit code {process.returncode})", err=True)
                if stderr:
                    typer.echo(f"   stderr: {stderr[:500]}", err=True)
                return False
        
        # Check registered models
        models = server_handle.registered_models()
        if len(models) >= expected_count:
            return True
        
        time.sleep(poll_interval)
    
    return False


def terminate_engine_and_backend(
    server_handle: "pie_server_rs.ServerHandle | None",
    backend_processes: list[subprocess.Popen],
) -> None:
    """Terminate the engine and backend processes.

    Args:
        server_handle: The server handle (or None if already shut down)
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

    # Gracefully shut down the engine
    if server_handle is not None:
        try:
            if server_handle.is_running():
                typer.echo("🔄 Shutting down engine...")
                server_handle.shutdown()
        except Exception as e:
            typer.echo(f"  Error shutting down engine: {e}")


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
    import asyncio

    asyncio.run(_submit_inferlet_async(client_config, inferlet_path, arguments))


async def _submit_inferlet_async(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
) -> None:
    """Async implementation of submit_inferlet_and_wait."""
    import blake3
    from pie_client import PieClient, Event

    # Check inferlet exists
    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")

    # Read and hash the inferlet
    inferlet_blob = inferlet_path.read_bytes()
    program_hash = blake3.blake3(inferlet_blob).hexdigest()
    typer.echo(f"Inferlet hash: {program_hash}")

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    async with PieClient(server_uri) as client:
        # Authenticate with internal token
        await client.internal_authenticate(internal_token)

        # Check if program already exists, upload if not
        if not await client.program_exists(program_hash):
            typer.echo("Uploading inferlet...")
            await client.upload_program(inferlet_blob)
        else:
            typer.echo("Inferlet already cached on server.")

        # Launch the instance
        typer.echo(f"Launching {inferlet_path.name}...")
        cmd_name = inferlet_path.stem  # Use filename without extension
        instance = await client.launch_instance(
            program_hash=program_hash,
            arguments=arguments,
            cmd_name=cmd_name,
            detached=False,
        )
        typer.echo(f"Instance launched: {instance.instance_id}")

        # Stream events until completion
        while True:
            event, message = await instance.recv()

            if event == Event.Stdout:
                # Stream stdout without extra newline
                print(message, end="", flush=True)
            elif event == Event.Stderr:
                # Stream stderr to stderr
                import sys
                print(message, end="", file=sys.stderr, flush=True)
            elif event == Event.Message:
                typer.echo(f"[Message] {message}")
            elif event == Event.Completed:
                typer.echo(f"✅ Instance completed: {message}")
                break
            elif event == Event.Aborted:
                typer.echo(f"⚠️ Instance aborted: {message}")
                break
            elif event == Event.Exception:
                typer.echo(f"❌ Instance exception: {message}", err=True)
                break
            elif event == Event.ServerError:
                typer.echo(f"❌ Server error: {message}", err=True)
                break
            elif event == Event.OutOfResources:
                typer.echo(f"❌ Out of resources: {message}", err=True)
                break
            elif event == Event.Blob:
                typer.echo(f"[Received blob: {len(message)} bytes]")
            else:
                typer.echo(f"[Unknown event {event}]: {message}")

