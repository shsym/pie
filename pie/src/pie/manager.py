"""Engine and backend management for Pie.

This module handles the lifecycle of the Pie engine and backend services.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from . import path as pie_path

if TYPE_CHECKING:
    from . import _pie


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
) -> tuple["_pie.ServerHandle", list]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations
        timeout: Maximum time to wait for backends to connect (seconds)
        console: Optional rich.console.Console for output
        on_status: Optional callback for status updates: (status_message: str) -> None
        on_message: Optional callback for log messages: (level: str, message: str) -> None

    Returns:
        Tuple of (ServerHandle, list of backend processes - empty for FFI mode)

    Raises:
        EngineError: If engine or backend fails to start
    """
    from . import _pie

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
    # Get telemetry config from engine_config (loaded from [telemetry] section)
    telemetry_config = engine_config.get("telemetry", {})
    server_config = _pie.ServerConfig(
        host=engine_config.get("host", "127.0.0.1"),
        port=engine_config.get("port", 8080),
        enable_auth=engine_config.get("enable_auth", True),
        cache_dir=engine_config.get("cache_dir"),
        verbose=engine_config.get("verbose", False),
        log_dir=engine_config.get("log_dir"),
        registry=engine_config.get("registry", "https://registry.pie-project.org/"),
        telemetry_enabled=telemetry_config.get("enabled", False),
        telemetry_endpoint=telemetry_config.get("endpoint", "http://localhost:4317"),
        telemetry_service_name=telemetry_config.get("service_name", "pie-runtime"),
    )

    # FFI MODE: Queue-based communication for high throughput
    if console is not None:
        console.print("[dim]Starting engine...[/dim]")

    try:
        # Currently supports single-model configurations
        if len(model_configs) > 1:
            raise EngineError(
                "Currently only single-model configurations are supported"
            )

        model_config = model_configs[0]
        status_update("Initializing backend in-process...")

        # Build the full config for pie_backend
        full_config = _build_backend_config(
            engine_config, model_config, authorized_users_path
        )

        # Initialize Python backend - returns Runtime
        from pie_worker.server import start_ffi_worker

        runtime = _pie.initialize_backend(full_config)

        # Create the FfiQueue FIRST via Python
        ffi_queue = _pie.FfiQueue()

        # Start the Python worker thread BEFORE starting server
        # This allows the worker to respond to handshake during Model::new
        _ffi_worker = start_ffi_worker(ffi_queue, runtime)

        # Now start server with FFI mode - the worker is ready to handle requests
        server_handle = _pie.start_server_with_ffi(
            server_config,
            authorized_users_path,
            ffi_queue,
        )

    except Exception as e:
        raise EngineError(f"Failed to initialize backend: {e}") from e

    # Final success message
    if console is not None:
        console.print(
            "[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]"
        )

    # Return empty list for backend_processes (no separate processes in FFI mode)
    return server_handle, []


def _build_backend_config(
    engine_config: dict, model_config: dict, internal_token: str
) -> dict:
    """Build the configuration dict for RuntimeConfig.from_args().

    Args:
        engine_config: Engine configuration dict (host, port) - not used for RuntimeConfig
        model_config: Model configuration dict
        internal_token: Internal authentication token - not used for RuntimeConfig

    Returns:
        Configuration dict suitable for RuntimeConfig.from_args(**kwargs)
    """
    # Note: host, port, internal_auth_token are for pycrust server registration,
    # not RuntimeConfig. They're excluded here because in FFI mode we don't
    # need to register with the engine - we call Python directly.

    # Handle device field - can be string or list
    # RuntimeConfig.from_args expects: device (str) OR devices (list[str])
    device_value = model_config.get("device")
    device_key = None
    if device_value is not None:
        if isinstance(device_value, list):
            device_key = "devices"
        else:
            device_key = "device"

    config = {
        "hf_repo": model_config.get("hf_repo"),
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
        "telemetry_enabled": engine_config.get("telemetry", {}).get("enabled", False),
        "telemetry_endpoint": engine_config.get("telemetry", {}).get(
            "endpoint", "http://localhost:4317"
        ),
        "telemetry_service_name": engine_config.get("telemetry", {}).get(
            "service_name", "pie"
        ),
        "random_seed": model_config.get("random_seed", 42),
        "use_cuda_graphs": model_config.get("use_cuda_graphs", True),
    }

    # Add device with correct key
    if device_key is not None:
        config[device_key] = device_value

    # Remove None values to use from_args() defaults
    return {k: v for k, v in config.items() if v is not None}


def wait_for_backends(
    server_handle: "_pie.ServerHandle",
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
        # Check if the engine is still running
        if not server_handle.is_running():
            print("❌ Engine stopped unexpectedly", file=sys.stderr)
            return False

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
        # In FFI mode, backend_processes may contain dispatcher functions (not processes)
        # Skip anything that's not a process
        if not hasattr(process, "pid") or not hasattr(process, "is_alive"):
            continue

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
    server_handle: "_pie.ServerHandle | None",
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
        # In FFI mode, backend_processes may contain dispatcher functions (not processes)
        # Skip anything that's not a process
        if not hasattr(process, "pid"):
            continue

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
    server_handle: "_pie.ServerHandle | None" = None,
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
    server_handle: "_pie.ServerHandle | None" = None,
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
    server_handle: "_pie.ServerHandle | None",
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
    server_handle: "_pie.ServerHandle | None" = None,
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
    server_handle: "_pie.ServerHandle | None" = None,
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
