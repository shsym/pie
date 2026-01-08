import sys
import os


import fire
from .server import start_server
from .runtime import RuntimeConfig, Runtime
import sys
import platform
import importlib.metadata
import torch
import time
import signal
import warnings


# Main entry point for the server
def main(
    hf_repo: str | None = None,
    host: str = "localhost",
    port: int = 9123,
    internal_auth_token: str | None = None,
    cache_dir: str | None = None,
    kv_page_size: int = 16,
    max_dist_size: int = 64,
    max_num_embeds: int = 128,
    max_batch_tokens: int = 10240,
    max_batch_size: int = 128,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    gpu_mem_utilization: float = 0.9,
    device: list[str] | str | None = None,
    activation_dtype: str = "bfloat16",
    weight_dtype: str | None = None,
    enable_profiling: bool = False,
    random_seed: int = 42,
    use_cuda_graphs: bool = True,
    test: bool = False,
    log_queue: (
        object | None
    ) = None,  # multiprocessing.Queue, using object to avoid type issues with fire
):
    """
    Runs the application with configuration provided as command-line arguments.

    Args:
        hf_repo: HuggingFace repo (e.g., "meta-llama/Llama-3.2-1B-Instruct") (required).
        host: Hostname for the ZMQ service to bind to.
        port: Port for the ZMQ service to bind to.
        internal_auth_token: Internal authentication token for connecting to the controller.
        cache_dir: Directory for model cache. Defaults to PIE_HOME env var,
                   then the platform-specific user cache dir.
        kv_page_size: The size of each page in the key-value cache.
        max_dist_size: Maximum distance for embeddings.
        gpu_mem_utilization: Proportion of GPU memory to use for KV cache (0.0 - 1.0).
        max_num_embeds: Maximum number of embeddings to store.
        max_batch_tokens: Maximum number of tokens in a batch.
        max_num_adapters: Maximum number of adapters that can be loaded.
        max_adapter_rank: Maximum rank for any loaded adapter.
        device: The device(s) to run the model on (e.g., 'mps', 'cuda:0', 'cpu',
                or ['cuda:0', 'cuda:1'] for future multi-GPU support).
                If a list is provided, only the first device is used for now.
        activation_dtype: The data type for activations (e.g., 'bfloat16', 'float16').
        weight_dtype: The data type for weights. If different from activation_dtype,
                      weights will be quantized. Options: 'int4', 'int8', 'float8'.
                      If None, uses activation_dtype (no quantization).
        enable_profiling: Enable unified profiler (timing + tensor tracking) (default: False).
        test: Run embedded test client after server starts (default: False).
        log_queue: Optional multiprocessing.Queue for sending logs back to controller.
    """

    if hf_repo is None:
        raise ValueError("The 'hf_repo' argument is required.")
    # Parse device argument
    if device is None:
        device_list = None
    elif isinstance(device, list):
        device_list = device
    else:  # str
        if "," in device:
            device_list = [d.strip() for d in device.split(",")]
        else:
            device_list = [device]

    # Check if we need to spawn multiple processes
    if device_list and len(device_list) > 1:
        world_size = len(device_list)
        import torch.multiprocessing as mp
        import random

        # Use 'spawn' context for CUDA compatibility
        mp.set_start_method("spawn", force=True)

        # Helper: signal
        import signal

        # Remove the previous simple ignore if present, or just implement new logic
        # We need the parent to handle SIGTERM to kill children, but children need to ignore it initially
        # until Rank 0 tells them to stop (or parent kills them).

        # ACTUALLY, simpler approach based on plan:
        # Parent: Catch SIGTERM -> ctx.terminate() -> join()
        # Children: Ignore SIGTERM. Rank 0 re-enables it.

        # So in main (parent):
        # Generate master port BEFORE spawn so all processes use same port
        master_port = 29500 + random.randint(0, 1000)

        # Create IPC control channels BEFORE spawn so all processes share the queues
        from .control_channel import create_control_channels

        control_queues = create_control_channels(world_size)

        ctx = mp.spawn(
            init_process,
            args=(
                world_size,
                device_list,
                master_port,  # Pass port to ensure all processes use same port
                control_queues,  # IPC control channels
                hf_repo,
                host,
                port,
                internal_auth_token,
                cache_dir,
                kv_page_size,
                max_dist_size,
                max_num_embeds,
                max_batch_tokens,
                max_batch_size,
                max_num_adapters,
                max_adapter_rank,
                gpu_mem_utilization,
                activation_dtype,
                weight_dtype,
                enable_profiling,
                random_seed,
                use_cuda_graphs,
                test,
                log_queue,
            ),
            nprocs=world_size,
            join=False,  # We manage join manually
        )

        # Cleanup function to kill all children immediately
        def cleanup_children():
            for p in ctx.processes:
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.join(timeout=1.0)

        # Register atexit handler
        import atexit

        atexit.register(cleanup_children)

        def sigterm_handler(signum, frame):
            # Forward signal to children and exit
            cleanup_children()
            sys.exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigterm_handler)

        # Wait for children
        try:
            while not ctx.join():
                pass
        finally:
            cleanup_children()
    else:
        # Single process mode (backward compatibility)
        single_device = device_list[0] if device_list else None
        init_process(
            0,  # rank
            1,  # world_size
            (
                [single_device] if single_device else []
            ),  # devices (will be resolved in config)
            0,  # master_port (unused in single process mode)
            None,  # control_queues (not needed for single process)
            hf_repo,
            host,
            port,
            internal_auth_token,
            cache_dir,
            kv_page_size,
            max_dist_size,
            max_num_embeds,
            max_batch_tokens,
            max_batch_size,
            max_num_adapters,
            max_adapter_rank,
            gpu_mem_utilization,
            activation_dtype,
            weight_dtype,
            enable_profiling,
            random_seed,
            use_cuda_graphs,
            test,
            log_queue,
        )


def init_process(
    rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,  # Port for distributed coordination (shared by all processes)
    control_queues: list | None,  # IPC control channel queues (created by parent)
    hf_repo: str,
    host: str,
    port: int,
    internal_auth_token: str | None,
    cache_dir: str | None,
    kv_page_size: int,
    max_dist_size: int,
    max_num_embeds: int,
    max_batch_tokens: int,
    max_batch_size: int,
    max_num_adapters: int,
    max_adapter_rank: int,
    gpu_mem_utilization: float,
    activation_dtype: str,
    weight_dtype: str | None,
    enable_profiling: bool,
    random_seed: int,
    use_cuda_graphs: bool,
    test: bool,
    log_queue: object | None,
):
    """
    Initialize the distributed process and start the runtime.
    """
    import os
    import torch.distributed as dist
    import signal

    # Ignore SIGTERM initially so we don't die when parent forwards it.
    # Rank 0 will re-enable a handler in start_server.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Setup distributed environment if world_size > 1
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        # Set NCCL timeout so operations fail instead of hanging forever
        # This allows recovery when one rank dies mid-operation
        os.environ["NCCL_TIMEOUT"] = "300"  # 5 minutes
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        # CRITICAL: Set CUDA device BEFORE init_process_group so NCCL detects correct device
        local_device = (
            devices[rank] if devices and rank < len(devices) else f"cuda:{rank}"
        )
        torch.cuda.set_device(local_device)

        # Suppress harmless barrier() device warning
        import warnings

        warnings.filterwarnings(
            "ignore", message=".*barrier.*device under current context.*"
        )

        # Use NCCL for CUDA, GLOO for CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        pg_options = None
        if backend == "nccl":
            try:
                from torch.distributed import ProcessGroupNCCL

                pg_options = ProcessGroupNCCL.Options()
                pg_options.config.capture_safe = True
            except (ImportError, AttributeError):
                pass

        if pg_options:
            dist.init_process_group(
                backend, rank=rank, world_size=world_size, pg_options=pg_options
            )
        else:
            dist.init_process_group(backend, rank=rank, world_size=world_size)

        # Initialize IPC control channel (replaces GLOO for metadata broadcasts)
        import pie_backend.utils as pie_utils
        from .control_channel import ControlChannel

        pie_utils._control_channel = ControlChannel(rank, world_size, control_queues)

    # Determine local device for this rank
    # If devices list is empty (auto-detect), RuntimeConfig will handle it for rank 0
    # For multi-gpu, we expect explicit devices list usually, but let's handle it safely
    device_str = devices[rank] if devices and rank < len(devices) else None

    # Create configuration
    config = RuntimeConfig.from_args(
        hf_repo=hf_repo,
        cache_dir=cache_dir,
        kv_page_size=kv_page_size,
        max_dist_size=max_dist_size,
        max_num_embeds=max_num_embeds,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
        max_num_adapters=max_num_adapters,
        max_adapter_rank=max_adapter_rank,
        gpu_mem_utilization=gpu_mem_utilization,
        devices=devices,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        enable_profiling=enable_profiling,
        random_seed=random_seed,
        use_cuda_graphs=use_cuda_graphs,
        rank=rank,
        world_size=world_size,
    )

    # Log trace to queue if available
    trace_msg = f"[TRACE rank={rank}] Config devices={config.devices}, my device={config.device}"
    log_queue.put({"level": "DEBUG", "message": trace_msg})

    # Initialize Runtime
    service = Runtime(config, log_queue=log_queue)

    # Synchronize all ranks before starting server/worker loop
    # This prevents workers from spinning on NCCL broadcast before rank 0 is ready
    if world_size > 1:
        dist.barrier()

    if rank == 0:

        start_server(
            host=host,
            port=port,
            auth_token=internal_auth_token,
            service=service,
            run_tests=test,
            log_queue=log_queue,
        )

        # Shutdown workers
        # Handled inside start_server -> service.shutdown()

    else:
        # Workers run the worker loop
        service.worker_loop()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def entrypoint():
    fire.Fire(main)


if __name__ == "__main__":
    entrypoint()
