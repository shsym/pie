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
    max_num_kv_pages: int | None = None,
    gpu_mem_headroom: float | None = None,
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
        max_num_kv_pages: Maximum number of pages in the key-value cache.
        gpu_mem_headroom: TODO
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
    # For now, take the first device if a list is provided (future multi-GPU support)
    if isinstance(device, list):
        device = device[0] if device else None

    config = RuntimeConfig.from_args(
        model=model,
        cache_dir=cache_dir,
        kv_page_size=kv_page_size,
        max_dist_size=max_dist_size,
        max_num_embeds=max_num_embeds,
        max_batch_tokens=max_batch_tokens,
        max_num_adapters=max_num_adapters,
        max_adapter_rank=max_adapter_rank,
        max_num_kv_pages=max_num_kv_pages,
        gpu_mem_headroom=gpu_mem_headroom,
        device=device,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        enable_profiling=enable_profiling,
    )
    config.print()

    service = Runtime(config)
    start_server(host=host, port=port, auth_token=internal_auth_token, service=service, run_tests=test)



def entrypoint():
    fire.Fire(main)


if __name__ == "__main__":
    entrypoint()
