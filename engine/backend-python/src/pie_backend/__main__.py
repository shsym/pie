import fire
from .server import start_server
from .runtime import RuntimeConfig, Runtime


# Main entry point for the server
def main(
    model: str,
    host: str = "localhost",
    port: int = 9123,
    internal_auth_token: str | None = None,
    cache_dir: str | None = None,
    kv_page_size: int = 16,
    max_dist_size: int = 64,
    max_num_embeds: int = 128,
    max_batch_tokens: int = 10240,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    gpu_mem_utilization: float = 0.9,
    device: list[str] | str | None = None,
    activation_dtype: str = "bfloat16",
    weight_dtype: str | None = None,
    enable_profiling: bool = False,
    test: bool = False,
):
    """
    Runs the application with configuration provided as command-line arguments.

    Args:
        model: Name of the model to load (required).
        host: Hostname for the ZMQ service to bind to.
        port: Port for the ZMQ service to bind to.
        controller_host: Hostname of the controller to register with.
        controller_port: Port of the controller to register with.
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
    """
    # Parse device argument
    if device is None:
        device_list = None
    elif isinstance(device, list):
        device_list = device
    else: # str
        if "," in device:
            device_list = [d.strip() for d in device.split(",")]
        else:
            device_list = [device]

    # Check if we need to spawn multiple processes
    if device_list and len(device_list) > 1:
        world_size = len(device_list)
        import torch.multiprocessing as mp
        
        # Use 'spawn' context for CUDA compatibility
        mp.set_start_method("spawn", force=True)
        
        print(f"Spawning {world_size} processes for devices: {device_list}")
        mp.spawn(
            init_process,
            args=(
                world_size,
                device_list,
                model,
                host,
                port,
                internal_auth_token,
                cache_dir,
                kv_page_size,
                max_dist_size,
                max_num_embeds,
                max_batch_tokens,
                max_num_adapters,
                max_adapter_rank,
                gpu_mem_utilization,
                activation_dtype,
                weight_dtype,
                enable_profiling,
                test,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single process mode (backward compatibility)
        single_device = device_list[0] if device_list else None
        init_process(
            0, # rank
            1, # world_size
            [single_device] if single_device else [], # devices (will be resolved in config)
            model,
            host,
            port,
            internal_auth_token,
            cache_dir,
            kv_page_size,
            max_dist_size,
            max_num_embeds,
            max_batch_tokens,
            max_num_adapters,
            max_adapter_rank,
            gpu_mem_utilization,
            activation_dtype,
            weight_dtype,
            enable_profiling,
            test,
        )

def init_process(
    rank: int,
    world_size: int,
    devices: list[str],
    model: str,
    host: str,
    port: int,
    internal_auth_token: str | None,
    cache_dir: str | None,
    kv_page_size: int,
    max_dist_size: int,
    max_num_embeds: int,
    max_batch_tokens: int,
    max_num_adapters: int,
    max_adapter_rank: int,
    gpu_mem_utilization: float,
    activation_dtype: str,
    weight_dtype: str | None,
    enable_profiling: bool,
    test: bool,
):
    """
    Initialize the distributed process and start the runtime.
    """
    import os
    import torch.distributed as dist

    # Setup distributed environment if world_size > 1
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port + 100) # Use a different port for coordination
        
        # Initialize process group
        # Use NCCL for CUDA, GLOO for CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Determine local device for this rank
    # If devices list is empty (auto-detect), RuntimeConfig will handle it for rank 0
    # For multi-gpu, we expect explicit devices list usually, but let's handle it safely
    device_str = devices[rank] if devices and rank < len(devices) else None

    # Create configuration
    config = RuntimeConfig.from_args(
        model=model,
        cache_dir=cache_dir,
        kv_page_size=kv_page_size,
        max_dist_size=max_dist_size,
        max_num_embeds=max_num_embeds,
        max_batch_tokens=max_batch_tokens,
        max_num_adapters=max_num_adapters,
        max_adapter_rank=max_adapter_rank,
        gpu_mem_utilization=gpu_mem_utilization,
        devices=devices,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        enable_profiling=enable_profiling,
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        config.print()

    # Initialize Runtime
    service = Runtime(config)

    if rank == 0:
        # Rank 0 runs the server
        start_server(host=host, port=port, auth_token=internal_auth_token, service=service, run_tests=test)
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
