"""Engine and backend management for Pie.

This module handles the lifecycle of the Pie engine and backend services.
"""

import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from . import path as pie_path

if TYPE_CHECKING:
    from . import pie_rs


class EngineError(Exception):
    """Exception raised for engine/backend errors."""

    pass


def start_engine_and_backend(
    engine_config: dict,
    model_configs: list[dict],
    timeout: float = 300.0,
    console: Optional[Any] = None,
    on_status: Optional[callable] = None,
    on_message: Optional[callable] = None,
) -> tuple["pie_rs.ServerHandle", list]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations (formerly backend_configs)
        timeout: Maximum time to wait for backends to connect (seconds)
        console: Optional rich.console.Console for output
        on_status: Optional callback for status updates: (status_message: str) -> None
        on_message: Optional callback for log messages: (level: str, message: str) -> None

    Returns:
        Tuple of (ServerHandle, list of backend processes)

    Raises:
        EngineError: If engine or backend fails to start
    """
    from . import pie_rs

    # Setup console if available
    use_rich = console is not None
    if use_rich:
        from rich.console import Console

    def status_update(msg: str):
        if on_status:
            on_status(msg)

    def log_message(level: str, msg: str):
        if on_message:
            on_message(level, msg)

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

    # Count expected backends
    expected_backends = 0

    # Create log queue for backend communication
    manager_obj = multiprocessing.Manager()
    log_queue = manager_obj.Queue()

    # Launch backend processes
    backend_processes: list = []

    # Start log monitor thread if console available
    if console is not None:
        import threading

        log_monitor_thread = threading.Thread(
            target=backend_log_monitor, args=(log_queue, console), daemon=True
        )
        log_monitor_thread.start()

    # Print initial status (progress bar will provide loading feedback)
    if console is not None:
        console.print("[dim]Starting engine...[/dim]")

    try:
        for model_config in model_configs:
            status_update("Spawning backend (pie-backend)...")
            try:
                process = spawn_python_backend(
                    engine_config, model_config, server_handle.internal_token, log_queue
                )
                backend_processes.append(process)
                expected_backends += 1
            except Exception as e:
                for p in backend_processes:
                    p.terminate()
                server_handle.shutdown()
                raise EngineError(f"Failed to spawn backend: {e}") from e

        # Wait for backends to register with the engine
        if expected_backends > 0:
            status_update(f"Waiting for {expected_backends} backend(s) to connect...")
            if not wait_for_backends(
                server_handle, expected_backends, timeout, backend_processes
            ):
                for p in backend_processes:
                    p.terminate()
                server_handle.shutdown()
                raise EngineError("Timeout waiting for backends to connect")
    except Exception:
        raise

    # Final success message
    if console is not None:
        console.print(
            "[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]"
        )

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
        "max_batch_size": model_config.get("max_batch_size", 128),
        "activation_dtype": model_config.get("activation_dtype", "bfloat16"),
        "weight_dtype": model_config.get("weight_dtype"),
        "enable_profiling": model_config.get("enable_profiling", False),
        "random_seed": model_config.get("random_seed", 42),
        "use_cuda_graphs": model_config.get("use_cuda_graphs", True),
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


def backend_log_monitor(log_queue: multiprocessing.Queue, console: Any):
    """Monitor loop for backend logs with progress bar support."""
    import queue
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )

    progress_ctx = None
    progress_task = None

    while True:
        try:
            # Block for a short time to allow check for exit
            record = log_queue.get(timeout=1.0)

            level = record.get("level", "INFO")
            msg = record.get("message", "")

            if level == "PROGRESS":
                # Handle progress updates from backend weight loading
                current = record.get("current", 0)
                total = record.get("total", 100)
                desc = record.get("description", "")

                if progress_ctx is None:
                    # Create progress bar on first PROGRESS message
                    progress_ctx = Progress(
                        SpinnerColumn(style="cyan"),
                        TextColumn("[bold cyan]Loading weights[/bold cyan]"),
                        BarColumn(
                            bar_width=40, style="cyan", complete_style="bright_cyan"
                        ),
                        TaskProgressColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True,
                    )
                    progress_ctx.start()
                    progress_task = progress_ctx.add_task("Loading", total=total)

                # Update progress
                progress_ctx.update(progress_task, completed=current)

            elif level == "PROGRESS_DONE":
                # Complete and close progress bar
                if progress_ctx is not None:
                    progress_ctx.stop()
                    progress_ctx = None
                    progress_task = None
                console.print("[green]✓[/green] Model weights loaded")

            elif level == "DEBUG":
                # Suppress DEBUG logs completely for cleaner output
                continue
            elif level == "SUCCESS":
                console.print(f"  ✅ {msg}")
            elif level == "WARNING":
                console.print(f"  ⚠️ {msg}")
            elif level == "ERROR":
                console.print(f"  ❌ [bold red][backend: error][/bold red] {msg}")
            else:
                level_str = "info"
                if "Starting server" in msg:
                    continue
                console.print(f"[dim]  [backend: {level_str}] {msg}[/dim]")

        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError):
            if progress_ctx is not None:
                progress_ctx.stop()
            break
        except Exception:
            if progress_ctx is not None:
                progress_ctx.stop()
                progress_ctx = None
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


def check_backend_processes(
    backend_processes: list, on_error: Optional[callable] = None
) -> bool:
    """Check if all backend processes are still alive.

    Args:
        backend_processes: List of backend processes to check
        on_error: Optional callback for error messages: (message: str) -> None

    Returns:
        True if all processes are alive, False if any have died
    """
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
            error_msg = f"Backend process exited unexpectedly (exit code {return_code})"
            if stderr:
                error_msg += f" stderr: {stderr[:500]}"
            if on_error:
                on_error(error_msg)
            else:
                print(f"❌ {error_msg}", file=sys.stderr)

    return all_alive


def terminate_engine_and_backend(
    server_handle: "pie_rs.ServerHandle | None",
    backend_processes: list,
    on_message: Optional[callable] = None,
) -> None:
    """Terminate the engine and backend processes.

    Args:
        server_handle: The server handle (or None if already shut down)
        backend_processes: List of backend subprocess.Popen objects
        on_message: Optional callback for status messages: (message: str) -> None
    """
    import signal

    def log(msg: str):
        if on_message:
            on_message(msg)

    for process in backend_processes:
        is_running = False
        pid = process.pid

        if isinstance(process, subprocess.Popen):
            is_running = process.poll() is None
        else:
            is_running = process.is_alive()

        if is_running:
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
                log(f"Force killing process {pid}")
                process.kill()
            except Exception as e:
                log(f"Error terminating process {pid}: {e}")

    # Gracefully shut down the engine
    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception as e:
            log(f"Error shutting down engine: {e}")


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

    print("Available commands:")
    print("  run <path> [ARGS]... - Run a .wasm inferlet with optional arguments")
    print("  stat                 - Query the backend statistics")
    print("  exit                 - Exit the Pie session")
    print("  help                 - Show this help message")

    while True:
        try:
            line = input("pie> ")
        except EOFError:
            print("Exiting...")
            break
        except KeyboardInterrupt:
            print("\n(To exit, type 'exit' or press Ctrl-D)")
            continue

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0]
        args = parts[1:]

        if command == "exit":
            print("Exiting...")
            break
        elif command == "help":
            print("Available commands:")
            print("  run <path> [ARGS]... - Run a .wasm inferlet")
            print("  stat                 - Query backend statistics")
            print("  exit                 - Exit the session")
            print("  help                 - Show this help message")
        elif command == "run":
            if not args:
                print("Usage: run <inferlet_path> [ARGS]...")
                continue
            inferlet_path = Path(args[0]).expanduser()
            inferlet_args = args[1:]
            try:
                submit_inferlet_and_wait(client_config, inferlet_path, inferlet_args)
            except Exception as e:
                print(f"Error running inferlet: {e}")
        elif command == "stat":
            print("(stat command not yet implemented)")
        else:
            print(f"Unknown command: '{command}'. Type 'help' for a list of commands.")

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
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet to the engine and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
        on_event: Optional callback for events: (event_type: str, message: str) -> None
    """
    import asyncio

    asyncio.run(
        _submit_inferlet_async(
            client_config,
            inferlet_path,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


async def _submit_inferlet_async(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Async implementation of submit_inferlet_and_wait."""
    import blake3
    import asyncio
    from pie_client import PieClient, Event

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    # Check inferlet exists
    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")

    # Read and hash the inferlet
    inferlet_blob = inferlet_path.read_bytes()
    program_hash = blake3.blake3(inferlet_blob).hexdigest()
    emit("info", f"Inferlet hash: {program_hash}")

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(
            _monitor_processes_task(server_handle, backend_processes)
        )

    try:
        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Check if program already exists, upload if not
            if not await client.program_exists(program_hash):
                emit("info", "Uploading inferlet...")
                await client.upload_program(inferlet_blob)
            else:
                emit("info", "Inferlet already cached on server.")

            # Launch the instance
            emit("info", f"Launching {inferlet_path.name}...")
            instance = await client.launch_instance(
                program_hash=program_hash,
                arguments=arguments,
                detached=False,
            )
            emit("info", f"Instance launched: {instance.instance_id}")

            # Stream events until completion
            while True:
                recv_task = asyncio.create_task(instance.recv())
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    monitor_task.result()

                event, message = recv_task.result()

                if event == Event.Stdout:
                    print(message, end="", flush=True)
                elif event == Event.Stderr:
                    print(message, end="", file=sys.stderr, flush=True)
                elif event == Event.Message:
                    emit("message", f"[Message] {message}")
                elif event == Event.Completed:
                    emit("completed", f"{message}")
                    break
                elif event == Event.Aborted:
                    emit("aborted", f"⚠️ Instance aborted: {message}")
                    break
                elif event == Event.Exception:
                    emit("exception", f"❌ Instance exception: {message}")
                    break
                elif event == Event.ServerError:
                    emit("error", f"❌ Server error: {message}")
                    break
                elif event == Event.OutOfResources:
                    emit("error", f"❌ Out of resources: {message}")
                    break
                elif event == Event.Blob:
                    emit("blob", f"[Received blob: {len(message)} bytes]")
                else:
                    emit("unknown", f"[Unknown event {event}]: {message}")

    finally:
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
            raise RuntimeError("Backend process died")

        if server_handle and hasattr(server_handle, "is_running"):
            if not server_handle.is_running():
                raise RuntimeError("Engine process died")

        await asyncio.sleep(1.0)


def submit_inferlet_from_registry_and_wait(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet from the registry and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_name: Inferlet name (e.g., "std/text-completion@0.1.0")
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
        on_event: Optional callback for events: (event_type: str, message: str) -> None
    """
    import asyncio

    asyncio.run(
        _submit_inferlet_from_registry_async(
            client_config,
            inferlet_name,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


async def _submit_inferlet_from_registry_async(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "pie_rs.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Async implementation of submit_inferlet_from_registry_and_wait."""
    import asyncio
    from pie_client import PieClient, Event

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(
            _monitor_processes_task(server_handle, backend_processes)
        )

    try:
        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Launch the instance from registry
            instance = await client.launch_instance_from_registry(
                inferlet=inferlet_name,
                arguments=arguments,
                detached=False,
            )

            # Stream events until completion
            while True:
                recv_task = asyncio.create_task(instance.recv())
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    monitor_task.result()

                event, message = recv_task.result()

                if event == Event.Stdout:
                    print(message, end="", flush=True)
                elif event == Event.Stderr:
                    print(message, end="", file=sys.stderr, flush=True)
                elif event == Event.Message:
                    emit("message", f"[Message] {message}")
                elif event == Event.Completed:
                    emit("completed", f"{message}")
                    break
                elif event == Event.Aborted:
                    emit("aborted", f"⚠️ Instance aborted: {message}")
                    break
                elif event == Event.Exception:
                    emit("exception", f"❌ Instance exception: {message}")
                    break
                elif event == Event.ServerError:
                    emit("error", f"❌ Server error: {message}")
                    break
                elif event == Event.OutOfResources:
                    emit("error", f"❌ Out of resources: {message}")
                    break
                elif event == Event.Blob:
                    emit("blob", f"[Received blob: {len(message)} bytes]")
                else:
                    emit("unknown", f"[Unknown event {event}]: {message}")
    except Exception:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
        raise
