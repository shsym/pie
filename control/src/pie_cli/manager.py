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
    from . import pie_rs


def start_engine_and_backend(
    engine_config: dict,
    model_configs: list[dict],
    timeout: float = 60.0,
    console: Optional["typer.rich_utils.Console"] = None,
) -> tuple["pie_rs.ServerHandle", list[subprocess.Popen]]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations (formerly backend_configs)
        timeout: Maximum time to wait for backends to connect (seconds)

    Returns:
        Tuple of (ServerHandle, list of backend processes)
    """
    from . import pie_rs
    from rich.console import Console
    from rich.control import Control, ControlType

    # Use passed console or create new one
    if console is None:
        console = Console()

    # Load authorized users if auth is enabled
    authorized_users_path = None
    if engine_config.get("enable_auth", True):
        auth_path = pie_path.get_authorized_users_path()
        if auth_path.exists():
            authorized_users_path = str(auth_path)

    # Create server config
    server_config = pie_rs.ServerConfig(
        host=engine_config.get("host", "127.0.0.1"),
        port=engine_config.get("port", 8080),
        enable_auth=engine_config.get("enable_auth", True),
        cache_dir=engine_config.get("cache_dir"),
        verbose=engine_config.get("verbose", False),
        log_dir=engine_config.get("log_dir"),
        registry=engine_config.get("registry", "https://registry.pie-project.org/"),
    )

    # Start the engine - returns a ServerHandle
    server_handle = pie_rs.start_server(server_config, authorized_users_path)
    # typer.echo(f"✅ Engine started (token: {server_handle.internal_token[:8]}...)") - Removed for minimalism

    # Count expected backends
    expected_backends = 0
    
    # Create log queue for backend communication
    # We use a multiprocessing Manager Queue to ensure it works across process boundaries reliably
    manager_obj = multiprocessing.Manager()  # Renamed to avoid shadowing 'manager' module
    log_queue = manager_obj.Queue()

    # Launch backend processes
    backend_processes: list["multiprocessing.Process | subprocess.Popen"] = []

    # Start log monitor thread
    import threading
    log_monitor_thread = threading.Thread(
        target=backend_log_monitor,
        args=(log_queue, console),
        daemon=True
    )
    log_monitor_thread.start()
    
    with console.status("Starting engine...", spinner="dots") as status:
        for model_config in model_configs:
            # Spawn pie-backend directly using multiprocessing
            status.update(f"Spawning backend (pie-backend)...")
            try:
                process = spawn_python_backend(
                    engine_config, 
                    model_config, 
                    server_handle.internal_token,
                    log_queue
                )
                backend_processes.append(process)
                expected_backends += 1
            except Exception as e:
                console.print(f"❌ Failed to spawn backend: {e}")
                for p in backend_processes:
                    p.terminate()
                server_handle.shutdown()
                raise typer.Exit(1)

        # Wait for backends to register with the engine
        if expected_backends > 0:
            status.update(f"Waiting for {expected_backends} backend(s) to connect...")
            if not wait_for_backends(server_handle, expected_backends, timeout, backend_processes):
                console.print("❌ Timeout waiting for backends to connect")
                for p in backend_processes:
                    p.terminate()
                server_handle.shutdown()
                raise typer.Exit(1)
            # Backend connected
            
    # Final success message
    console.print("[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]")

    return server_handle, backend_processes


def spawn_python_backend(
    engine_config: dict,
    model_config: dict,
    internal_token: str,
    log_queue: multiprocessing.Queue,
) -> multiprocessing.Process:
    """Spawn a Python backend process directly using multiprocessing.

    This avoids the overhead of subprocess and directly imports pie_backend,
    which is faster and more reliable than shelling out.

    Args:
        engine_config: Engine configuration dict (host, port)
        model_config: Model configuration dict (formerly backend_config)
        internal_token: Internal authentication token
        log_queue: Queue for sending log messages back to the controller

    Returns:
        multiprocessing.Process object
    """
    # Build kwargs for pie_backend.main()
    backend_kwargs = {
        "host": engine_config.get("host", "127.0.0.1"),
        "port": engine_config.get("port", 8080),
        "internal_auth_token": internal_token,
        "hf_repo": model_config.get("hf_repo"),
        "device": model_config.get("device"),
        "cache_dir": model_config.get("cache_dir"),
        "kv_page_size": model_config.get("kv_page_size", 16),
        "max_dist_size": model_config.get("max_dist_size", 64),
        "max_num_embeds": model_config.get("max_num_embeds", 128),
        "max_batch_tokens": model_config.get("max_batch_tokens", 10240),
        "max_num_adapters": model_config.get("max_num_adapters", 48),
        "max_adapter_rank": model_config.get("max_adapter_rank", 8),
        "gpu_mem_utilization": model_config.get("gpu_mem_utilization", 0.9),
        "activation_dtype": model_config.get("activation_dtype", "bfloat16"),
        "weight_dtype": model_config.get("weight_dtype"),
        "enable_profiling": model_config.get("enable_profiling", False),
        "random_seed": model_config.get("random_seed", 42),
        "log_queue": log_queue,
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


def backend_log_monitor(log_queue: multiprocessing.Queue, console: "Console"):
    """Monitor loop for backend logs."""
    import queue
    
    while True:
        try:
            # Block for a short time to allow check for exit
            record = log_queue.get(timeout=1.0)
            
            level = record.get("level", "INFO")
            msg = record.get("message", "")
            
            if level == "DEBUG":
                # Suppress DEBUG logs completely for cleaner output
                # If we really want them, we could add a verbose flag, but for now user wants silence
                continue
            elif level == "SUCCESS":
                 console.print(f"  ✅ {msg}")
            elif level == "WARNING":
                 console.print(f"  ⚠️ {msg}")
            elif level == "ERROR":
                 # Errors are important, make them visible
                 console.print(f"  ❌ [bold red][backend: error][/bold red] {msg}")
            else:
                 # Standard INFO
                 level_str = "info"
            
            # If msg is "Starting server..." we might want to skip it if it's redundant
            if "Starting server" in msg:
                continue

            # Default dimmed formatting
            console.print(f"[dim]  [backend: {level_str}] {msg}[/dim]")
                     
        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError):
            break
        except Exception:
            # Ignore monitoring errors to avoid crashing main thread
            pass


def wait_for_backends(
    server_handle: "pie_rs.ServerHandle",
    expected_count: int,
    timeout: float,
    backend_processes: list,
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
        if not check_backend_processes(backend_processes):
            return False
        
        # Check registered models
        models = server_handle.registered_models()
        if len(models) >= expected_count:
            return True
        
        time.sleep(poll_interval)
    
    return False



def check_backend_processes(backend_processes: list) -> bool:
    """Check if all backend processes are still alive.
    
    Args:
        backend_processes: List of backend processes to check
        
    Returns:
        True if all processes are alive, False if any have died
    """
    import subprocess
    
    all_alive = True
    for process in backend_processes:
        is_dead = False
        return_code = None
        stderr = ""

        if isinstance(process, subprocess.Popen):
            if process.poll() is not None:
                is_dead = True
                return_code = process.returncode
                stderr = process.stderr.read().decode() if process.stderr else ""
        else:
            # Assume multiprocessing.Process
            if not process.is_alive():
                is_dead = True
                return_code = process.exitcode
        
        if is_dead:
            all_alive = False
            typer.echo(f"❌ Backend process exited unexpectedly (exit code {return_code})", err=True)
            if stderr:
                typer.echo(f"   stderr: {stderr[:500]}", err=True)
            
    return all_alive


def terminate_engine_and_backend(
    server_handle: "pie_rs.ServerHandle | None",
    backend_processes: list,
) -> None:
    """Terminate the engine and backend processes.

    Args:
        server_handle: The server handle (or None if already shut down)
        backend_processes: List of backend subprocess.Popen objects
    """
    import signal


    for process in backend_processes:
        is_running = False
        pid = process.pid
        
        if isinstance(process, subprocess.Popen):
            is_running = process.poll() is None
        else:
            is_running = process.is_alive()

        if is_running:  # Still running
            try:
                if isinstance(process, subprocess.Popen):
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                else:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        raise subprocess.TimeoutExpired(cmd=str(pid), timeout=5)
            except subprocess.TimeoutExpired:
                typer.echo(f"  Force killing process {pid}")
                process.kill()
            except Exception as e:
                typer.echo(f"  Error terminating process {pid}: {e}")

    # Gracefully shut down the engine
    if server_handle is not None:
        try:
            if server_handle.is_running():
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
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
) -> None:
    """Submit an inferlet to the engine and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        arguments: Arguments to pass to the inferlet
    """
    import asyncio

    asyncio.run(_submit_inferlet_async(
        client_config, 
        inferlet_path, 
        arguments,
        server_handle,
        backend_processes
    ))


async def _submit_inferlet_async(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
) -> None:
    """Async implementation of submit_inferlet_and_wait."""
    import blake3
    import asyncio
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

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(_monitor_processes_task(server_handle, backend_processes))

    try:
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
            instance = await client.launch_instance(
                program_hash=program_hash,
                arguments=arguments,
                detached=False,
            )
            typer.echo(f"Instance launched: {instance.instance_id}")

            # Stream events until completion
            while True:
                # Wait for either new message or monitor failure
                recv_task = asyncio.create_task(instance.recv())
                
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)
                
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    # Monitor task finished (meaning it raised exception)
                    monitor_task.result()  # Re-raise exception
                
                # If we get here, recv_task must be done
                event, message = recv_task.result()

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

    finally:
        # If we have a monitor task, cancel it
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


async def _monitor_processes_task(
    server_handle: "pie_rs.ServerHandle | None",
    backend_processes: list | None,
):
    """Async task to monitor backend processes."""
    import asyncio
    
    if not backend_processes:
        return

    while True:
        if not check_backend_processes(backend_processes):
            # If any backend dies, we raise an exception to cancel the run
            raise RuntimeError("Backend process died")
        
        # Also check engine if possible
        if server_handle and hasattr(server_handle, 'is_running'):
            if not server_handle.is_running():
                raise RuntimeError("Engine process died")

        await asyncio.sleep(1.0)



def submit_inferlet_from_registry_and_wait(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
) -> None:
    """Submit an inferlet from the registry and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_name: Inferlet name (e.g., "std/text-completion@0.1.0")
        arguments: Arguments to pass to the inferlet
    """
    import asyncio

    asyncio.run(_submit_inferlet_from_registry_async(
        client_config, 
        inferlet_name, 
        arguments, 
        server_handle, 
        backend_processes
    ))


async def _submit_inferlet_from_registry_async(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
) -> None:
    """Async implementation of submit_inferlet_from_registry_and_wait."""
    import asyncio
    from pie_client import PieClient, Event

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(_monitor_processes_task(server_handle, backend_processes))

    try:
        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Launch the instance from registry
            typer.echo(f"Launching {inferlet_name} from registry...")
            instance = await client.launch_instance_from_registry(
                inferlet=inferlet_name,
                arguments=arguments,
                detached=False,
            )
            typer.echo(f"Instance launched: {instance.instance_id}")

            # Stream events until completion
            while True:
                # Wait for either new message or monitor failure
                recv_task = asyncio.create_task(instance.recv())
                
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)
                
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    # Monitor task finished (meaning it raised exception)
                    monitor_task.result()  # Re-raise exception
                
                # If we get here, recv_task must be done
                event, message = recv_task.result()

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
    except Exception:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
        raise
