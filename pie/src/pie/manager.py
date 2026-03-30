"""Engine and backend management for Pie.

This module handles the lifecycle of the Pie engine and backend services.
"""

import sys
import time
import os
import signal
import subprocess
import random
import asyncio
import warnings
from pathlib import Path


from typing import Optional, Any
import queue  # For Queue type hint logic if needed, but Queue is from MP


class EngineError(Exception):
    """Exception raised for engine/backend errors."""

    pass


class FfiWorkerHandle:
    """Wrapper for FFI worker thread to look like a process for cleanup."""

    def __init__(self, thread, stop_event):
        self.thread = thread
        self.stop_event = stop_event
        self.pid = -1  # Pseudo-PID

    def terminate(self):
        self.stop_event.set()

    def kill(self):
        """Simulate kill by ensuring stop event is set (threads cannot be SIGKILLed)."""
        self.stop_event.set()

    def join(self, timeout=None):
        self.thread.join(timeout=timeout)

    def is_alive(self):
        return self.thread.is_alive()


def start_engine_and_backend(
    engine_config: dict,
    model_configs: list[dict],
    timeout: float = 1200.0,
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

    def status_update(msg: str):
        if on_status:
            on_status(msg)

    def log_message(level: str, msg: str):
        if on_message:
            on_message(level, msg)

    from . import path as pie_path
    from . import _pie
    import torch

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
        python_snapshot=engine_config.get("python_snapshot", True),
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

        # Detect backend
        backend = model_config.get("backend", "native")

        # Detect multi-GPU configuration
        device_value = model_config.get("device")
        world_size = len(device_value) if isinstance(device_value, list) else 1

        # Validate that all configured devices are accessible
        devices_to_validate = (
            device_value if isinstance(device_value, list) else [device_value]
        )
        available_gpus = torch.cuda.device_count()

        for device in devices_to_validate:
            if device and device.startswith("cuda:"):
                device_idx = int(device.split(":")[1])
                if device_idx >= available_gpus:
                    raise EngineError(
                        f"Device '{device}' is not accessible. "
                        f"Only {available_gpus} GPU(s) are visible (cuda:0 to cuda:{available_gpus - 1}). "
                        f"Check CUDA_VISIBLE_DEVICES environment variable."
                    )

        # Select the spawn function based on backend
        _start_backend = (
            _start_vllm_ffi_backend if backend == "vllm"
            else _start_multi_gpu_ffi_backend
        )

        if world_size > 1:
            # Multi-GPU FFI mode: spawn worker processes
            backend_processes = _start_backend(
                engine_config,
                model_config,
                server_config,
                authorized_users_path,
                device_value,
                world_size,
                console,
                status_update,
                timeout,
            )
            server_handle = backend_processes.pop(0)  # First element is server_handle
        else:
            # Single-GPU mode: use same IPC architecture as multi-GPU with world_size=1
            status_update("Initializing single-GPU backend...")

            # Treat as multi-GPU with 1 device for unified code path
            backend_processes = _start_backend(
                engine_config,
                model_config,
                server_config,
                authorized_users_path,
                [device_value] if isinstance(device_value, str) else device_value,
                1,  # world_size = 1
                console,
                status_update,
                timeout,
            )
            server_handle = backend_processes.pop(0)  # First element is server_handle

    except Exception as e:
        raise EngineError(f"Failed to initialize backend: {e}") from e

    # Final success message
    if console is not None:
        console.print(
            "[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]"
        )

    return server_handle, backend_processes


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
        "adapter_path": model_config.get("adapter_path"),
        "gpu_mem_utilization": model_config.get("gpu_mem_utilization", 0.9),
        "max_model_len": model_config.get("max_model_len"),
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
        "use_cuda_graphs": model_config.get("use_cuda_graphs", False),
        "dummy_mode": model_config.get("dummy_mode", False),
        "backend": model_config.get("backend", "native"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }

    # Add device with correct key
    if device_key is not None:
        config[device_key] = device_value

    # Remove None values to use from_args() defaults
    return {k: v for k, v in config.items() if v is not None}


# =============================================================================
# Multi-GPU FFI Mode Helpers
# =============================================================================


def _init_distributed(rank: int, world_size: int, master_port: int, device: str):
    """Initialize torch.distributed for a given rank.

    Sets up CUDA device and process group using FileStore for rendezvous.
    """
    import datetime
    import torch
    import torch.distributed as dist

    # Set CUDA device
    torch.cuda.set_device(device)

    # Suppress harmless warnings
    warnings.filterwarnings(
        "ignore", message=".*barrier.*device under current context.*"
    )

    # Use FileStore for more robust rendezvous (avoids port conflicts)
    store_path = f"/tmp/pie_dist_store_{master_port}"
    store = dist.FileStore(store_path, world_size)
    timeout = datetime.timedelta(seconds=300)

    # Initialize process group with NCCL
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Extract device index for device_id parameter
    device_id = None
    if device.startswith("cuda:"):
        device_id = torch.device(device)

    dist.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        device_id=device_id,
    )


def _setup_process_groups(rank: int, group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for each execution group (Rank 0 + Group Workers)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        # Comm group includes Rank 0 (Controller) + Group Workers
        comm_ranks = sorted(list(set([0] + group_ranks)))

        # Create group (Collective: all ranks/participants must call this)
        # Note: Depending on backend, might need global participation.
        # For safety in NCCL, everyone calls new_group for all groups.
        pg = dist.new_group(comm_ranks)

        pg_map[i] = pg

    return pg_map


def _setup_compute_process_groups(rank: int, group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for Tensor Parallel computation (TP ranks only)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        # Compute group includes ONLY the TP workers for this group
        # Sort to ensure consistent ordering
        comm_ranks = sorted(list(set(group_ranks)))

        # Create group
        pg = dist.new_group(comm_ranks)

        pg_map[i] = pg

    return pg_map


def _create_runtime(
    config_dict: dict,
    devices: list[str],
    rank: int,
    world_size: int,
    group_topology: list[list[int]],
    result_queue: Any | None = None,
    result_queues: list | None = None,  # All queues (for Rank 0)
    pg_map: dict | None = None,
    compute_pg_map: dict | None = None,
):
    """Create a Runtime instance for the given rank."""

    from pie_worker.config import RuntimeConfig
    from pie_worker.runtime import Runtime

    # Remove device/devices from config to avoid duplicate argument
    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=devices,
        rank=rank,
        world_size=world_size,
    )

    # Determine my group ID
    my_group_id = 0
    for i, group in enumerate(group_topology):
        if rank in group:
            my_group_id = i
            break

    rt = Runtime(
        config,
        log_queue=None,
        group_id=my_group_id,
        result_queue=result_queue,
        result_queues=result_queues,
        process_groups=pg_map,
        compute_process_groups=compute_pg_map,
        group_topology=group_topology,
    )
    return rt


# =============================================================================
# Multi-GPU FFI Mode Entry Points
# =============================================================================


def _start_multi_gpu_ffi_backend(
    engine_config: dict,
    model_config: dict,
    server_config,
    authorized_users_path: str | None,
    devices: list[str],
    world_size: int,
    console,
    status_update: callable,
    timeout: float,
) -> list:
    """Start multi-GPU backend with Coordinator + All Workers architecture.

    Main Process (Coordinator):
        - Only runs the Rust server
        - Does NOT participate in torch.distributed
        - Does NOT load any model

    Worker Processes (0..world_size-1):
        - All run identical code
        - Each participates in torch.distributed
        - Each loads model on its assigned GPU
        - Each connects to IPC server

    Returns:
        List where first element is ServerHandle, rest are worker processes
    """

    status_update(f"Initializing multi-GPU backend ({world_size} devices)...")

    import torch.multiprocessing as mp
    from . import _pie
    from . import path as pie_path

    # Use 'spawn' context for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Generate master port
    master_port = 29500 + random.randint(0, 1000)

    # Build config dict for all ranks
    full_config = _build_backend_config(
        engine_config, model_config, authorized_users_path
    )

    # Determine Tensor Parallel degree and topology
    tp_degree = model_config.get(
        "tensor_parallel_size", engine_config.get("tensor_parallel_size")
    )
    if tp_degree is None:
        # Default to world_size (TP across all devices) to avoid OOM by default
        # If users want DP, they should explicitly set tensor_parallel_size=1
        tp_degree = world_size
        if console:
            console.print(
                f"[yellow]![/yellow] tensor_parallel_size not set, defaulting to {tp_degree} (use all GPUs)"
            )

    group_topology = _calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    status_update(f"  Topology: {num_groups} groups (TP={tp_degree})")

    # Phase 1: Start server and get IPC server names for ALL groups
    # Server starts immediately, workers will connect later
    partial_handle, ipc_server_names = _pie.start_server_phase1(
        server_config, authorized_users_path, num_groups
    )

    # Create ready queue for workers to signal when they've connected to IPC
    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    # Spawn ALL worker processes (ranks 0..world_size-1)
    # All workers run identical code - just with different rank/device
    ctx = mp.spawn(
        _ipc_worker_process,
        args=(
            world_size,
            devices,
            master_port,
            full_config,
            group_topology,
            ipc_server_names,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,  # Workers die automatically when main process exits
    )

    # Wait for ALL workers to signal they've connected to IPC
    # Each worker sends its rank when ready
    # Wait for ALL workers to signal they've connected to IPC
    # Monitor processes while waiting to catch early exits (e.g. OOM)
    connected_ranks = set()
    start_wait = time.time()

    # We need to wait for world_size ranks
    while len(connected_ranks) < world_size:
        # 1. Check if processes are alive to catch early exits (e.g. OOM)
        for p in ctx.processes:
            if not p.is_alive():
                exitcode = p.exitcode
                if exitcode != 0:
                    raise RuntimeError(
                        f"Worker process {p.pid} died unexpectedly with exit code {exitcode}"
                    )

        # 2. Check for timeout
        if time.time() - start_wait > timeout:
            # Clean up
            ready_queue.close()
            ready_queue.join_thread()
            raise TimeoutError(f"Timed out waiting for {world_size} workers to connect")

        # 3. Try access queue
        try:
            # Use non-blocking get
            rank = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if console:
                status_update(
                    f"  Worker {rank} ready ({len(connected_ranks)}/{world_size})"
                )
        except queue.Empty:
            continue

    # Clean up the ready_queue to prevent semaphore leak
    ready_queue.close()
    ready_queue.join_thread()

    # Phase 2: Complete initialization (blocks until handshake succeeds)
    # All workers are now connected via IPC
    server_handle = partial_handle.complete()

    # Return server handle and worker context
    return [server_handle, ctx]


def _calculate_topology(world_size: int, tp_degree: int) -> list[list[int]]:
    """Calculate process groups based on world size and TP degree.

    Returns:
        List of groups, where each group is a list of ranks.
        Example: world_size=4, tp=2 -> [[0, 1], [2, 3]]
    """
    if world_size % tp_degree != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by TP degree ({tp_degree})"
        )

    num_groups = world_size // tp_degree
    topology = []
    for g in range(num_groups):
        group_ranks = list(range(g * tp_degree, (g + 1) * tp_degree))
        topology.append(group_ranks)

    return topology


def _ipc_worker_process(
    local_rank: int,  # mp.spawn passes 0-indexed local rank (= actual rank for us)
    world_size: int,
    devices: list[str],
    master_port: int,
    config_dict: dict,
    group_topology: list[list[int]],
    ipc_server_names: list[str],
    ready_queue,  # Queue to signal when connected
):
    """Worker process for Coordinator + All Workers architecture.

    All workers run this identical code path. Each worker:
    1. Initializes torch.distributed
    2. Loads model on its assigned GPU
    3. Connects to the IPC server for its group
    4. Signals ready via ready_queue
    5. Runs the IPC worker loop

    Args:
        local_rank: Rank of this worker (0 to world_size-1)
        world_size: Total number of workers
        devices: List of device strings
        master_port: Port for torch.distributed rendezvous
        config_dict: Runtime configuration
        group_topology: List of groups, each containing ranks
        ipc_server_names: IPC server names for each group
        ready_queue: Queue to signal when ready
    """
    from pie import _pie
    from pie_worker.runtime import Runtime
    from pie_worker.config import RuntimeConfig
    import torch.distributed as dist
    import torch

    rank = local_rank  # With nprocs=world_size, local_rank IS the actual rank

    # Wrap entire worker body in try/finally for CUDA cleanup.
    # On Jetson unified memory, CMA allocations leak if not explicitly freed
    # before process exit — the driver does NOT reclaim on process death.
    try:
        _ipc_worker_body(
            rank, world_size, devices, master_port, config_dict,
            group_topology, ipc_server_names, ready_queue,
            _pie, Runtime, RuntimeConfig, dist, torch,
        )
    except Exception:
        import traceback, sys
        traceback.print_exc()
        raise
    finally:
        # Explicit CUDA cleanup — critical for Jetson unified memory (CMA).
        # Without this, leaked CMA persists until host reboot.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reset all CUDA state for this device
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        import gc
        gc.collect()


def _ipc_worker_body(
    rank, world_size, devices, master_port, config_dict,
    group_topology, ipc_server_names, ready_queue,
    _pie, Runtime, RuntimeConfig, dist, torch,
):
    """Inner body of _ipc_worker_process, separated for cleanup guarantee."""

    try:
        # Determine my group and TP rank within it
        my_group_id = 0
        tp_rank = 0
        for i, group in enumerate(group_topology):
            if rank in group:
                my_group_id = i
                tp_rank = group.index(rank)  # My position within the TP group
                break

        tp_degree = len(group_topology[my_group_id])
    except Exception as e:
        with open("/tmp/worker_startup_error.log", "w") as f:
            import traceback
            f.write(f"Worker startup failed before dist init: {e}\n{traceback.format_exc()}")
        raise

    try:
        # Initialize distributed
        if world_size > 1:
             _init_distributed(rank, world_size, master_port, devices[rank])
        else:
             # Skip distributed for single device to avoid NCCL/socket issues
            device_str = devices[rank]
            if device_str.startswith("cuda"):
                torch.cuda.set_device(device_str)

        # Setup process groups (collective ops - all ranks must participate)
        # Capture the mappings to pass to Runtime
        if world_size > 1:
            pg_map = _setup_process_groups(rank, group_topology)
            compute_pg_map = _setup_compute_process_groups(rank, group_topology)
        else:
            pg_map = {}
            compute_pg_map = {}

    except Exception as e:
        with open("/tmp/worker_startup_error.log", "w") as f:
            import traceback
            f.write(f"Worker startup failed during dist init: {e}\n{traceback.format_exc()}")
        raise

    # Create runtime config
    # For TP>1: each worker needs ALL devices in its TP group so devices[tp_rank] works
    # Get the device list for this TP group (e.g., ["cuda:0", "cuda:1"] for group 0)
    my_group_ranks = group_topology[my_group_id]
    group_devices = [devices[r] for r in my_group_ranks]

    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }
    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=group_devices,  # All devices in this TP group
        rank=tp_rank,  # Position within TP group
        world_size=tp_degree,  # Size of TP group
        tensor_parallel_size=tp_degree,  # Ensure sharding is enabled!
    )

    # Create runtime (loads model on this GPU)
    # Pass process groups for TP communication
    runtime = Runtime(
        config,
        group_id=my_group_id,
        process_groups=pg_map,
        compute_process_groups=compute_pg_map,
        group_topology=group_topology,
    )

    # Sync all workers before connecting to server
    if dist.is_initialized():
        dist.barrier()

    # Check if I'm a group leader (first rank in my TP group)
    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            # Group leader: connect to IPC and handle requests
            server_name = ipc_server_names[my_group_id]
            ipc_queue = _pie.FfiIpcQueue.connect(server_name, my_group_id)

            # Signal that we're connected and ready
            ready_queue.put(rank)

            # Run IPC worker loop (handles requests from Rust server)
            _run_ipc_worker_loop(ipc_queue, runtime)
        else:
            # Non-leader: signal ready, then run worker loop waiting for commands from leader
            ready_queue.put(rank)
            runtime.worker_loop()
    finally:
        # Cleanup - ensure process group is destroyed even on termination
        if dist.is_initialized():
            dist.destroy_process_group()


def _start_vllm_ffi_backend(
    engine_config: dict,
    model_config: dict,
    server_config,
    authorized_users_path: str | None,
    devices: list[str],
    world_size: int,
    console,
    status_update: callable,
    timeout: float,
) -> list:
    """Start vLLM-backed backend using Pie's IPC architecture.

    Same process topology as _start_multi_gpu_ffi_backend but spawns
    PieVllmRuntime instead of native Runtime.

    Args:
        engine_config: Engine configuration dict
        model_config: Model configuration dict
        server_config: Rust server config
        authorized_users_path: Path to authorized users file
        devices: List of device strings
        world_size: Total number of workers
        console: Optional rich console
        status_update: Status callback
        timeout: Maximum time to wait for workers

    Returns:
        List where first element is ServerHandle, rest are worker processes
    """
    status_update(f"Initializing vLLM backend ({world_size} device(s))...")

    import torch.multiprocessing as mp
    from . import _pie

    mp.set_start_method("spawn", force=True)

    master_port = 29500 + random.randint(0, 1000)

    full_config = _build_backend_config(
        engine_config, model_config, authorized_users_path
    )

    # Determine Tensor Parallel degree and topology
    tp_degree = model_config.get(
        "tensor_parallel_size", engine_config.get("tensor_parallel_size")
    )
    if tp_degree is None:
        tp_degree = world_size
        if console:
            console.print(
                f"[yellow]![/yellow] tensor_parallel_size not set, defaulting to {tp_degree} (use all GPUs)"
            )

    group_topology = _calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    status_update(f"  Topology: {num_groups} group(s) (TP={tp_degree})")

    # Phase 1: Start server and get IPC server names for ALL groups
    partial_handle, ipc_server_names = _pie.start_server_phase1(
        server_config, authorized_users_path, num_groups
    )

    # Create ready queue for workers to signal when they've connected to IPC
    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    # Create per-group TP queues for leader→follower signaling.
    # Uses mp.Queue instead of NCCL broadcast_object_list to avoid the
    # 600s NCCL idle timeout that kills both workers when the engine is idle.
    tp_queues = None
    if tp_degree > 1:
        tp_queues = {
            group_id: spawn_ctx.Queue()
            for group_id in range(num_groups)
        }

    # Spawn ALL worker processes (ranks 0..world_size-1)
    ctx = mp.spawn(
        _vllm_worker_process,
        args=(
            world_size,
            devices,
            master_port,
            full_config,
            group_topology,
            ipc_server_names,
            ready_queue,
            tp_queues,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,
    )

    # Wait for ALL workers to signal they've connected to IPC
    connected_ranks = set()
    start_wait = time.time()

    while len(connected_ranks) < world_size:
        # Check if processes are alive to catch early exits (e.g. OOM)
        for p in ctx.processes:
            if not p.is_alive():
                exitcode = p.exitcode
                if exitcode != 0:
                    raise RuntimeError(
                        f"vLLM worker process {p.pid} died unexpectedly with exit code {exitcode}"
                    )

        if time.time() - start_wait > timeout:
            ready_queue.close()
            ready_queue.join_thread()
            raise TimeoutError(f"Timed out waiting for {world_size} vLLM workers to connect")

        try:
            rank = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if console:
                status_update(
                    f"  vLLM worker {rank} ready ({len(connected_ranks)}/{world_size})"
                )
        except queue.Empty:
            continue

    # Clean up the ready_queue to prevent semaphore leak
    ready_queue.close()
    ready_queue.join_thread()

    # Phase 2: Complete initialization (blocks until handshake succeeds)
    server_handle = partial_handle.complete()

    return [server_handle, ctx]


def _vllm_worker_process(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    config_dict: dict,
    group_topology: list[list[int]],
    ipc_server_names: list[str],
    ready_queue,
    tp_queues=None,
):
    """Worker process that creates PieVllmRuntime instead of native Runtime.

    Same structure as _ipc_worker_process, but imports and instantiates
    PieVllmRuntime for vLLM-backed inference.

    Args:
        local_rank: Rank of this worker (0 to world_size-1)
        world_size: Total number of workers
        devices: List of device strings
        master_port: Port for torch.distributed rendezvous
        config_dict: Runtime configuration
        group_topology: List of groups, each containing ranks
        ipc_server_names: IPC server names for each group
        ready_queue: Queue to signal when ready
        tp_queues: Per-group queues for TP leader→follower signaling
    """
    from pie import _pie
    from pie_worker.vllm_runtime import PieVllmRuntime
    from pie_worker.config import RuntimeConfig
    import torch.distributed as dist
    import torch

    rank = local_rank

    try:
        # Determine my group and TP rank within it
        my_group_id = 0
        tp_rank = 0
        for i, group in enumerate(group_topology):
            if rank in group:
                my_group_id = i
                tp_rank = group.index(rank)
                break

        tp_degree = len(group_topology[my_group_id])
    except Exception as e:
        with open("/tmp/vllm_worker_startup_error.log", "w") as f:
            import traceback
            f.write(f"vLLM worker startup failed before dist init: {e}\n{traceback.format_exc()}")
        raise

    try:
        # Skip PIE's distributed init for vLLM backend — vLLM handles its own
        # NCCL init internally via Worker.init_device()
        # Just set the CUDA device for this rank.
        torch.cuda.set_device(devices[rank])

        # vLLM manages its own process groups — skip PIE's setup
        pg_map = {}
        compute_pg_map = {}

    except Exception as e:
        with open("/tmp/vllm_worker_startup_error.log", "w") as f:
            import traceback
            f.write(f"vLLM worker startup failed during dist init: {e}\n{traceback.format_exc()}")
        raise

    try:
        # Create runtime config
        my_group_ranks = group_topology[my_group_id]
        group_devices = [devices[r] for r in my_group_ranks]

        filtered_config = {
            k: v for k, v in config_dict.items() if k not in ("device", "devices")
        }
        config = RuntimeConfig.from_args(
            **filtered_config,
            devices=group_devices,
            rank=tp_rank,
            world_size=tp_degree,
            tensor_parallel_size=tp_degree,
            master_port=master_port,
        )

        # Resolve the TP queue for this group (if available).
        tp_queue = tp_queues.get(my_group_id) if tp_queues else None

        # Create vLLM runtime (loads model via vLLM's GPUWorker)
        runtime = PieVllmRuntime(
            config,
            group_id=my_group_id,
            process_groups=pg_map,
            compute_process_groups=compute_pg_map,
            group_topology=group_topology,
            tp_queue=tp_queue,
        )

        # Sync all workers before connecting to server
        if dist.is_initialized():
            dist.barrier()
    except Exception as e:
        import traceback
        msg = f"vLLM worker rank {rank} failed during runtime init: {e}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr, flush=True)
        with open(f"/tmp/vllm_worker_rank{rank}_error.log", "w") as f:
            f.write(msg)
        raise

    # Check if I'm a group leader (first rank in my TP group)
    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            # Group leader: connect to IPC and handle requests
            server_name = ipc_server_names[my_group_id]
            ipc_queue = _pie.FfiIpcQueue.connect(server_name, my_group_id)

            # Signal that we're connected and ready
            ready_queue.put(rank)

            # Run IPC worker loop (handles requests from Rust server)
            _run_ipc_worker_loop(ipc_queue, runtime)
        else:
            # Non-leader: signal ready, then run worker loop waiting for commands from leader
            ready_queue.put(rank)
            runtime.worker_loop()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_ipc_worker_loop(ipc_queue, runtime):
    """Step-level GPU scheduler with Python-side continuations.

    Replaces the synchronous poll→fire_batch→respond loop with a step-level
    scheduler that:
      1. Drains the IPC queue between every GPU step
      2. Merges new fire_batch arrivals with ongoing continuations
      3. Executes one GPU step via ThreadPoolExecutor
      4. Responds only when all N steps complete for a request group

    For multi-step requests (max_decode_steps > 1), Python loops internally
    instead of round-tripping through Rust for each step.  This eliminates
    the batch fragmentation at c=16 where 16 requests split into 2 groups
    of 8 that alternate on the GPU.

    Args:
        ipc_queue: FfiIpcQueue instance connected to Rust
        runtime: Runtime instance (PieVllmRuntime) to dispatch calls to
    """
    import msgpack
    import time as _time
    from dataclasses import dataclass, field
    import numpy as np

    from pie_worker.vllm_runtime import PieVllmRuntime, DecodedBatchArrays
    from pie_worker.vllm_batch_translator import PieVllmBatchTranslator
    from pie_worker.batch_merger import compute_batch_generate_counts

    # Non-fire_batch RPC dispatch
    methods = {
        "handshake": runtime.handshake_rpc,
        "query": runtime.query_rpc,
        "format_chat": runtime.format_chat_rpc,
        "embed_image": runtime.embed_image_rpc,
        "initialize_adapter": runtime.initialize_adapter_rpc,
        "update_adapter": runtime.update_adapter_rpc,
        "upload_adapter": runtime.upload_adapter_rpc,
        "download_adapter": runtime.download_adapter_rpc,
    }

    _ipc_timing = os.environ.get("PIE_IPC_TIMING", "")
    _batching_debug = os.environ.get("PIE_BATCH_DEBUG", "")

    # --- Data structures for continuation tracking ---

    @dataclass
    class PendingRequest:
        """A fire_batch RPC waiting to be processed."""
        pycrust_request_id: int
        kwargs: dict
        num_requests: int
        max_decode_steps: int
        generate_count: int  # non-flush requests

    @dataclass
    class ActiveGroup:
        """A group of requests being stepped through the continuation loop."""
        # Mutable per-step arrays
        token_ids: np.ndarray          # current step's input tokens
        kv_last_page_lens: np.ndarray  # KV page fill levels
        seq_lens: np.ndarray           # total sequence lengths
        # Immutable across steps (shared references)
        kv_page_indices: np.ndarray
        kv_page_indptr: np.ndarray
        sampling_params_list: list
        adapter_indices: list
        kwargs: dict                   # original kwargs (for sampler_types etc.)
        # Accumulation state
        accumulated_tokens: list       # [list[int]] per request
        accumulated_dists: list        # [list] per request
        remaining_steps: int
        # Back-reference for response routing
        pending: PendingRequest

    # --- State ---
    shutdown_requested = False
    poll_timeout_ms = 100
    parent_pid = os.getppid()
    check_parent_every = 100
    loop_count = 0

    pending_new: list[PendingRequest] = []
    active_groups: list[ActiveGroup] = []
    # GPU step result: None means idle, _SENTINEL means error was handled,
    # otherwise holds the result from the last fire_batch/execute_step.
    # Runs synchronously on the main thread to avoid CUDA thread-safety issues.
    gpu_result = None
    gpu_error = None  # Exception from last step, if any
    current_slices = None  # [(group_ref, start_idx, count)] from last merge

    def _drain_queue():
        """Non-blocking drain of all pending IPC requests."""
        nonlocal shutdown_requested
        while True:
            try:
                request = ipc_queue.poll_blocking(0)
            except Exception:
                shutdown_requested = True
                return
            if request is None:
                return
            request_id, method, payload = request
            try:
                args = msgpack.unpackb(payload)
                if method == "shutdown":
                    shutdown_requested = True
                    ipc_queue.respond(request_id, msgpack.packb(None))
                    return
                if method == "fire_batch":
                    num_reqs = len(PieVllmBatchTranslator.decode_binary_array(
                        args["qo_indptr"], np.uint32)) - 1
                    mds = args.get("max_decode_steps", 1)
                    gc = compute_batch_generate_counts([args])[0]
                    pending_new.append(PendingRequest(
                        pycrust_request_id=request_id,
                        kwargs=args,
                        num_requests=num_reqs,
                        max_decode_steps=mds,
                        generate_count=gc,
                    ))
                    if _batching_debug:
                        print(f"[BATCH-ACC] req_id={request_id} reqs={num_reqs} "
                              f"mds={mds} pending_new={len(pending_new)} "
                              f"active={len(active_groups)}",
                              file=sys.stderr, flush=True)
                    continue
                # Non-fire_batch: handle immediately
                fn = methods.get(method)
                if fn is None:
                    ipc_queue.respond(request_id, msgpack.packb(f"Method not found: {method}"))
                    continue
                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)
                ipc_queue.respond(request_id, msgpack.packb(result))
            except Exception as e:
                import traceback
                print(f"[IPC Worker Error] {method}: {e}\n{traceback.format_exc()}")
                ipc_queue.respond(request_id, msgpack.packb(str(e)))

    def _build_merged_arrays():
        """Merge active continuations + new arrivals into one batch.

        Returns (merged_arrays: DecodedBatchArrays, merged_kwargs: dict,
                 slices: list[(ref, start, count)]).
        """
        all_token_ids = []
        all_kv_page_indices = []
        all_kv_page_indptr_diffs = []
        all_kv_last_page_lens = []
        all_sampling_params = []
        all_adapter_indices = []
        slices = []
        idx = 0

        # Active continuations (already decoded numpy arrays)
        for grp in active_groups:
            n = grp.pending.num_requests
            all_token_ids.append(grp.token_ids)
            all_kv_page_indices.append(grp.kv_page_indices)
            all_kv_page_indptr_diffs.append(np.diff(grp.kv_page_indptr))
            all_kv_last_page_lens.append(grp.kv_last_page_lens)
            all_sampling_params.extend(grp.sampling_params_list)
            all_adapter_indices.extend(grp.adapter_indices)
            slices.append((grp, idx, n))
            idx += n

        # New arrivals (need decoding)
        for pend in pending_new:
            arrays = PieVllmRuntime.decode_batch_arrays(pend.kwargs, runtime.kv_page_size)
            n = arrays.num_requests
            all_token_ids.append(arrays.token_ids)
            all_kv_page_indices.append(arrays.kv_page_indices)
            all_kv_page_indptr_diffs.append(np.diff(arrays.kv_page_indptr))
            all_kv_last_page_lens.append(arrays.kv_last_page_lens)
            all_sampling_params.extend(arrays.sampling_params_list)
            all_adapter_indices.extend(arrays.adapter_indices)
            slices.append((pend, idx, n))
            idx += n

        # Build merged arrays
        merged_token_ids = np.concatenate(all_token_ids)
        merged_kv_page_indices = np.concatenate(all_kv_page_indices)
        merged_kv_last_page_lens = np.concatenate(all_kv_last_page_lens)

        kv_diffs = np.concatenate(all_kv_page_indptr_diffs)
        merged_kv_page_indptr = np.empty(len(kv_diffs) + 1, dtype=np.int32)
        merged_kv_page_indptr[0] = 0
        np.cumsum(kv_diffs, out=merged_kv_page_indptr[1:])

        # For decode steps: 1 token per request
        merged_qo_indptr = np.arange(idx + 1, dtype=np.int32)

        tokens_per_req = [1] * idx
        blocks_per_req = PieVllmBatchTranslator.pages_to_blocks(
            merged_kv_page_indices, merged_kv_page_indptr
        )
        seq_lens = PieVllmBatchTranslator.compute_seq_lens(
            merged_kv_page_indptr, merged_kv_last_page_lens, runtime.kv_page_size
        )

        merged_arrays = DecodedBatchArrays(
            token_ids=merged_token_ids,
            qo_indptr=merged_qo_indptr,
            kv_page_indices=merged_kv_page_indices,
            kv_page_indptr=merged_kv_page_indptr,
            kv_last_page_lens=merged_kv_last_page_lens,
            num_requests=idx,
            tokens_per_req=tokens_per_req,
            blocks_per_req=blocks_per_req,
            seq_lens=seq_lens,
            sampling_params_list=all_sampling_params,
            adapter_indices=all_adapter_indices,
        )

        # Build merged kwargs for prepare_step (needs sampler_types, masks, etc.)
        # Use first available kwargs as base, override arrays
        base_kwargs = (active_groups[0].kwargs if active_groups
                       else pending_new[0].kwargs)
        merged_kwargs = dict(base_kwargs)
        merged_kwargs["single_token_mode"] = True
        merged_kwargs["flattened_masks"] = b""
        merged_kwargs["mask_indptr"] = b""
        # Rebuild sampler SoA arrays from merged sampling_params
        # (prepare_step reads these from kwargs for multi-position logits)
        rns_parts = []
        st_parts = []
        for grp in active_groups:
            rns_parts.append(PieVllmBatchTranslator.decode_binary_array(
                grp.kwargs.get("request_num_samplers", b""), np.uint32))
            st_parts.append(PieVllmBatchTranslator.decode_binary_array(
                grp.kwargs.get("sampler_types", b""), np.uint32))
        for pend in pending_new:
            rns_parts.append(PieVllmBatchTranslator.decode_binary_array(
                pend.kwargs.get("request_num_samplers", b""), np.uint32))
            st_parts.append(PieVllmBatchTranslator.decode_binary_array(
                pend.kwargs.get("sampler_types", b""), np.uint32))
        if rns_parts:
            merged_kwargs["request_num_samplers"] = np.concatenate(rns_parts).tobytes()
        if st_parts:
            merged_kwargs["sampler_types"] = np.concatenate(st_parts).tobytes()
        # Concatenate sampler parameter arrays
        for key in ("sampler_temperatures", "sampler_top_k", "sampler_top_p", "sampler_min_p"):
            parts = []
            for grp in active_groups:
                parts.append(PieVllmBatchTranslator.decode_binary_array(
                    grp.kwargs.get(key, b""), np.float32 if "temperature" in key or "top_p" in key or "min_p" in key else np.uint32))
            for pend in pending_new:
                parts.append(PieVllmBatchTranslator.decode_binary_array(
                    pend.kwargs.get(key, b""), np.float32 if "temperature" in key or "top_p" in key or "min_p" in key else np.uint32))
            if parts:
                merged_kwargs[key] = np.concatenate(parts).tobytes()

        return merged_arrays, merged_kwargs, slices

    def _collect_and_advance(model_output):
        """After GPU step: extract tokens, advance continuations, retire finished."""
        nonlocal active_groups, current_slices

        sampled_tokens = runtime.extract_sampled_tokens(model_output)

        # Check if any group needs distribution capture
        has_dists = False
        for ref, start, count in current_slices:
            kw = ref.kwargs if isinstance(ref, ActiveGroup) else ref.kwargs
            st = PieVllmBatchTranslator.decode_binary_array(
                kw.get("sampler_types", b""), np.uint32)
            if len(st) > 0 and np.any(st == 0):
                has_dists = True
                break

        # If distributions needed, run full packaging for this step
        step_results = None
        if has_dists:
            # Build kwargs for _package_response from the merged batch
            total_reqs = sum(count for _, _, count in current_slices)
            merged_kwargs = active_groups[0].kwargs if active_groups else pending_new[0].kwargs
            step_results = runtime._package_response(
                model_output, merged_kwargs, total_reqs)

        continuing = []
        retired = []

        for ref, start, count in current_slices:
            group_tokens = sampled_tokens[start:start + count]

            if isinstance(ref, ActiveGroup):
                # Existing continuation
                grp = ref
                for i, token in enumerate(group_tokens):
                    grp.accumulated_tokens[i].append(token)
                if has_dists and step_results:
                    for i in range(count):
                        if start + i < len(step_results):
                            grp.accumulated_dists[i].extend(step_results[start + i].get("dists", []))
                grp.remaining_steps -= 1
            else:
                # New arrival's first step — create ActiveGroup
                pend = ref
                arrays = PieVllmRuntime.decode_batch_arrays(pend.kwargs, runtime.kv_page_size)
                grp = ActiveGroup(
                    token_ids=arrays.token_ids,  # will be updated below
                    kv_last_page_lens=arrays.kv_last_page_lens.copy(),
                    seq_lens=arrays.seq_lens.copy(),
                    kv_page_indices=arrays.kv_page_indices,
                    kv_page_indptr=arrays.kv_page_indptr,
                    sampling_params_list=arrays.sampling_params_list,
                    adapter_indices=arrays.adapter_indices,
                    kwargs=pend.kwargs,
                    accumulated_tokens=[[t] for t in group_tokens],
                    accumulated_dists=[[] for _ in range(count)],
                    remaining_steps=pend.max_decode_steps - 1,
                    pending=pend,
                )
                if has_dists and step_results:
                    for i in range(count):
                        if start + i < len(step_results):
                            grp.accumulated_dists[i].extend(step_results[start + i].get("dists", []))

            if grp.remaining_steps <= 0:
                retired.append(grp)
            else:
                # Update arrays for next step (replicates model.rs:855-868)
                grp.token_ids = np.array(group_tokens, dtype=np.int64)
                grp.kv_last_page_lens = grp.kv_last_page_lens + 1
                overflow = grp.kv_last_page_lens > runtime.kv_page_size
                grp.kv_last_page_lens[overflow] = 1
                grp.seq_lens = grp.seq_lens + 1
                continuing.append(grp)

        active_groups = continuing
        return retired

    def _respond_retired(retired_groups):
        """Send accumulated multi-token responses for finished groups."""
        for grp in retired_groups:
            results = []
            for i in range(grp.pending.num_requests):
                # Only include results for generate (non-flush) requests
                rns = PieVllmBatchTranslator.decode_binary_array(
                    grp.kwargs.get("request_num_samplers", b""), np.uint32)
                if i < len(rns) and rns[i] == 0:
                    continue  # flush request — skip
                tokens = grp.accumulated_tokens[i] if i < len(grp.accumulated_tokens) else []
                dists = grp.accumulated_dists[i] if i < len(grp.accumulated_dists) else []
                results.append({"tokens": tokens, "dists": dists})

            response = {"results": results, "metrics": {
                "batch_size": grp.pending.num_requests,
                "continuation_steps": len(grp.accumulated_tokens[0]) if grp.accumulated_tokens else 0,
            }}

            if _batching_debug:
                print(f"[BATCH-RETIRE] req_id={grp.pending.pycrust_request_id} "
                      f"reqs={grp.pending.num_requests} "
                      f"tokens_per_req={len(grp.accumulated_tokens[0]) if grp.accumulated_tokens else 0}",
                      file=sys.stderr, flush=True)

            ipc_queue.respond(grp.pending.pycrust_request_id,
                              msgpack.packb(response))

    def _fire_step():
        """Merge active + pending into one batch and run on GPU (main thread).

        Runs synchronously on the main thread to ensure CUDA context safety.
        vLLM's model runner initializes CUDA on the main thread, and GPU ops
        from other threads cause segfaults.
        """
        nonlocal active_groups, pending_new, gpu_result, gpu_error, current_slices

        gpu_error = None

        # Check if all requests are single-step (fast path)
        all_single = (not active_groups and
                      all(p.max_decode_steps <= 1 for p in pending_new))

        if all_single and len(pending_new) == 1:
            # Single fire_batch, single step — use original fire_batch directly
            pend = pending_new.pop()
            current_slices = [(pend, 0, pend.num_requests)]
            try:
                gpu_result = runtime.fire_batch(**pend.kwargs)
            except Exception as e:
                gpu_error = e
            return

        if all_single and len(pending_new) > 1:
            # Multiple single-step fire_batches — merge kwargs and fire_batch
            from pie_worker.batch_merger import merge_fire_batch_kwargs
            kwargs_list = [p.kwargs for p in pending_new]
            merged = merge_fire_batch_kwargs(kwargs_list)
            current_slices = []
            idx = 0
            for p in pending_new:
                current_slices.append((p, idx, p.num_requests))
                idx += p.num_requests
            pending_new = []
            try:
                gpu_result = runtime.fire_batch(**merged)
            except Exception as e:
                gpu_error = e
            return

        # Multi-step path: build merged arrays and run one GPU step
        merged_arrays, merged_kwargs, current_slices = _build_merged_arrays()
        active_groups = []
        pending_new = []

        total_reqs = merged_arrays.num_requests
        if _batching_debug:
            n_active = sum(1 for ref, _, _ in current_slices if isinstance(ref, ActiveGroup))
            n_new = sum(1 for ref, _, _ in current_slices if isinstance(ref, PendingRequest))
            print(f"[BATCH-STEP] reqs={total_reqs} active_groups={n_active} "
                  f"new_groups={n_new}",
                  file=sys.stderr, flush=True)

        import torch
        try:
            with torch.inference_mode():
                scheduler_output = runtime.prepare_step(merged_arrays, merged_kwargs)
                gpu_result = runtime.execute_step(scheduler_output)
        except Exception as e:
            gpu_error = e

    def _handle_single_step_completion():
        """Handle completion of single-step fire_batch (fast path)."""
        nonlocal gpu_result, gpu_error, current_slices

        if gpu_error is not None:
            import traceback
            print(f"[BATCH-ERROR] fire_batch failed: {gpu_error}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            for ref, _, _ in current_slices:
                if isinstance(ref, PendingRequest):
                    ipc_queue.respond(ref.pycrust_request_id,
                                      msgpack.packb(str(gpu_error)))
            gpu_result = None
            gpu_error = None
            current_slices = None
            return

        result = gpu_result

        # Single-step results — respond immediately
        if len(current_slices) == 1:
            ref, _, _ = current_slices[0]
            ipc_queue.respond(ref.pycrust_request_id, msgpack.packb(result))
        else:
            # Multiple merged single-step batches — split results
            from pie_worker.batch_merger import split_fire_batch_results
            gen_counts = [ref.generate_count for ref, _, _ in current_slices]
            responses = split_fire_batch_results(result, gen_counts)
            for (ref, _, _), resp in zip(current_slices, responses):
                ipc_queue.respond(ref.pycrust_request_id, msgpack.packb(resp))

        gpu_result = None
        current_slices = None

    def _handle_multistep_completion():
        """Handle completion of a multi-step GPU step."""
        nonlocal gpu_result, gpu_error, current_slices

        if gpu_error is not None:
            import traceback
            print(f"[BATCH-ERROR] execute_step failed: {gpu_error}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            # Respond with error to all groups
            for ref, _, _ in current_slices:
                pend = ref.pending if isinstance(ref, ActiveGroup) else ref
                ipc_queue.respond(pend.pycrust_request_id,
                                  msgpack.packb(str(gpu_error)))
            gpu_result = None
            gpu_error = None
            current_slices = None
            active_groups.clear()
            return

        retired = _collect_and_advance(gpu_result)
        _respond_retired(retired)

        gpu_result = None
        # current_slices consumed by _collect_and_advance

    def _is_multistep_batch():
        """Check if current batch is a multi-step batch (vs single-step fast path)."""
        if current_slices is None:
            return False
        return any(isinstance(ref, ActiveGroup) for ref, _, _ in current_slices) or \
               any(isinstance(ref, PendingRequest) and ref.max_decode_steps > 1
                   for ref, _, _ in current_slices)

    # --- Main loop ---
    # With synchronous GPU execution on the main thread, the loop is:
    #   1. DRAIN IPC queue (non-blocking)
    #   2. FIRE step if work pending (runs GPU synchronously)
    #   3. COLLECT results and respond
    #   4. WAIT for next IPC request if idle
    try:
        while not shutdown_requested:
            loop_count += 1
            if loop_count % check_parent_every == 0:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break

            # 1. DRAIN
            _drain_queue()
            if shutdown_requested:
                break

            # 2. FIRE + COLLECT — GPU runs synchronously, result is immediate
            if (active_groups or pending_new):
                _fire_step()
                # Result is available immediately after _fire_step returns
                if _is_multistep_batch():
                    _handle_multistep_completion()
                else:
                    _handle_single_step_completion()
                continue  # drain again immediately (new IPC may have arrived during GPU)

            # 3. WAIT — no work pending, block for next IPC request
            if not pending_new and not active_groups:
                try:
                    request = ipc_queue.poll_blocking(poll_timeout_ms)
                except Exception:
                    break
                if request is not None:
                    request_id, method, payload = request
                    try:
                        args = msgpack.unpackb(payload)
                        if method == "fire_batch":
                            num_reqs = len(PieVllmBatchTranslator.decode_binary_array(
                                args["qo_indptr"], np.uint32)) - 1
                            mds = args.get("max_decode_steps", 1)
                            gc = compute_batch_generate_counts([args])[0]
                            pending_new.append(PendingRequest(
                                pycrust_request_id=request_id,
                                kwargs=args,
                                num_requests=num_reqs,
                                max_decode_steps=mds,
                                generate_count=gc,
                            ))
                        elif method == "shutdown":
                            shutdown_requested = True
                            ipc_queue.respond(request_id, msgpack.packb(None))
                        else:
                            fn = methods.get(method)
                            if fn:
                                if isinstance(args, dict):
                                    result = fn(**args)
                                elif isinstance(args, (list, tuple)):
                                    result = fn(*args)
                                else:
                                    result = fn(args)
                                ipc_queue.respond(request_id, msgpack.packb(result))
                    except Exception as e:
                        import traceback
                        print(f"[IPC Worker Error] {method}: {e}\n{traceback.format_exc()}")
                        ipc_queue.respond(request_id, msgpack.packb(str(e)))
    finally:
        runtime.shutdown()


def _ipc_group_worker(
    server_name: str,
    group_id: int,
    config_dict: dict,
    device: str,
):
    """IPC worker process for symmetric all-IPC architecture.

    This runs in a separate subprocess with its own GIL.
    It loads the model on the specified device and connects to the IPC server.

    Args:
        server_name: IPC server name to connect to
        group_id: Group ID for this worker
        config_dict: Runtime configuration
        device: Device string (e.g., "cuda:0")
    """
    from pie import _pie
    from pie_worker.runtime import Runtime
    from pie_worker.config import RuntimeConfig

    # Create runtime config for this group
    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=[device],
        rank=0,  # Local rank in this process
        world_size=1,  # Single GPU in this process
    )

    # Create runtime (loads model on this GPU)
    runtime = Runtime(config, group_id=group_id)

    # Connect to IPC and run worker loop
    ipc_queue = _pie.FfiIpcQueue.connect(server_name, group_id)
    _run_ipc_worker_loop(ipc_queue, runtime)


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
                # Check for SpawnContext
                if hasattr(process, "processes"):
                    # For SpawnContext, is_alive checks if any process is alive
                    # Context is "dead" if all processes are dead
                    pass

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
    # Suppress semaphore leak warning during shutdown (cosmetic, happens when workers are killed)
    warnings.filterwarnings(
        "ignore", message=".*leaked semaphore.*", category=UserWarning
    )

    from pie_worker import utils as pie_utils

    def log(msg: str):
        if on_message:
            on_message(msg)
        else:
            # sys imported globally
            pass
            # print(f"[Manager] {msg}", file=sys.stderr)

    # 1. Shut down the server FIRST - this sends shutdown signal to workers via IPC
    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception as e:
            log(f"Error shutting down engine: {e}")

    # 2. Give workers time to shut down gracefully after receiving IPC shutdown
    time.sleep(1.0)

    # 3. Broadcast STOP signal via control channel (legacy, may not be in use)
    try:
        if pie_utils._control_channel is not None:
            pie_utils._control_channel.send("STOP")
            time.sleep(0.5)
    except ImportError:
        pass
    except Exception as e:
        log(f"Error sending STOP signal: {e}")

    for process in backend_processes:
        # Check for SpawnContext (multiprocessing spawn context)
        # It doesn't have a pid attribute directly, but has .processes and .join
        if hasattr(process, "join") and hasattr(process, "processes"):
            try:
                # Terminate all worker processes in the spawn context
                for p in process.processes:
                    if p.is_alive():
                        p.terminate()
                # Wait briefly for termination
                for p in process.processes:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()  # Force kill if still alive
                # Finally join the context itself
                process.join(timeout=1)
            except Exception as e:
                log(f"Error terminating SpawnContext: {e}")
            continue

        # In FFI mode, backend_processes may contain dispatcher functions (not processes)
        # Skip anything that's not a process or context
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

    # Finalize control channel queues if they exist
    # This prevents "leaked semaphore" warnings from multiprocessing.resource_tracker
    try:
        if pie_utils._control_channel is not None:
            pie_utils._control_channel.cleanup()
            pie_utils._control_channel = None

    except ImportError:
        pass  # pie_worker might not be installed or importable
    except Exception as e:
        log(f"Error cleaning up control channel: {e}")


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

    from . import path as pie_path

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
    manifest_path: Path,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet to the engine and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        manifest_path: Path to the manifest TOML file
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
    """
    asyncio.run(
        _submit_inferlet_async(
            client_config,
            inferlet_path,
            manifest_path,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


async def _submit_inferlet_async(
    client_config: dict,
    inferlet_path: Path,
    manifest_path: Path,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Async implementation of submit_inferlet_and_wait."""

    import tomllib
    from pie_client import PieClient, Event

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    # Check inferlet exists
    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")

    # Check manifest exists
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

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
        # Parse manifest to get inferlet name
        manifest_content = manifest_path.read_text()
        manifest = tomllib.loads(manifest_content)
        package_name = manifest["package"]["name"]
        version = manifest["package"]["version"]
        inferlet_name = f"{package_name}@{version}"
        emit("info", f"Inferlet: {inferlet_name}")

        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Check if program already exists, install if not
            if not await client.program_exists(inferlet_name, inferlet_path, manifest_path):
                emit("info", "Installing inferlet...")
                await client.install_program(inferlet_path, manifest_path)
            else:
                emit("info", "Inferlet already cached on server.")

            # Launch the instance
            emit("info", f"Launching {inferlet_path.name}...")
            instance = await client.launch_instance(
                inferlet_name,
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
        inferlet_name: Inferlet name (e.g., "text-completion@0.1.0")
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
