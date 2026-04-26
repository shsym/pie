"""Worker entry point for the sglang backend.

Mirrors `pie_backend_vllm.worker` exactly — the leader/follower RPC loops,
distributed init, and process-group setup are reused directly from
`pie_backend.worker`. Only the engine class changes.
"""

from __future__ import annotations


# Re-export helpers used by pie/server.py
from pie_backend.worker import calculate_topology  # noqa: F401


def worker_main(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    group_topology: list[list[int]],
    ready_queue,
):
    """Worker process entry point for `mp.spawn` — sglang variant."""
    import torch

    try:
        _worker_body(
            local_rank, world_size, devices, master_port,
            model_config, group_topology, ready_queue,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        import gc
        gc.collect()


def _worker_body(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    group_topology: list[list[int]],
    ready_queue,
):
    import inspect
    import threading

    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    from pie import _runtime as pie_runtime
    from pie_backend.config import RuntimeConfig
    from pie_backend.worker import (
        _init_distributed,
        _setup_process_groups,
        _setup_compute_process_groups,
        _leader_loop,
        _follower_loop,
    )

    from .engine import SGLangEngine

    tqdm.set_lock(threading.RLock())

    rank = local_rank

    my_group_id = 0
    tp_rank = 0
    for i, group in enumerate(group_topology):
        if rank in group:
            my_group_id = i
            tp_rank = group.index(rank)
            break
    tp_degree = len(group_topology[my_group_id])

    if world_size > 1:
        _init_distributed(rank, world_size, master_port, devices[rank])
    else:
        device_str = devices[rank]
        if device_str.startswith("cuda"):
            torch.cuda.set_device(device_str)

    if world_size > 1:
        pg_map = _setup_process_groups(group_topology)
        compute_pg_map = _setup_compute_process_groups(group_topology)
    else:
        pg_map = {}
        compute_pg_map = {}

    group_devices = [devices[r] for r in group_topology[my_group_id]]

    valid_keys = set(inspect.signature(RuntimeConfig.from_args).parameters.keys())
    filtered_config = {
        k: v for k, v in model_config.items()
        if k in valid_keys
        and k not in ("device", "devices", "tensor_parallel_size")
        and v is not None
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=group_devices,
        rank=tp_rank,
        world_size=tp_degree,
        tensor_parallel_size=tp_degree,
    )

    compute_pg = compute_pg_map.get(my_group_id)
    engine = SGLangEngine.load(config, compute_process_group=compute_pg)

    if dist.is_initialized():
        dist.barrier()

    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            server = pie_runtime.RpcServer.create()
            server_name = server.server_name()

            metadata = {
                "total_pages": getattr(config, "max_num_kv_pages", 0),
                "max_batch_tokens": getattr(config, "max_batch_tokens", 10240),
                "max_batch_size": getattr(config, "max_batch_size", 128),
                "arch_name": engine.arch_type,
                "snapshot_dir": str(engine.snapshot_dir) if engine.snapshot_dir else "",
                "swap_pool_size": engine.swap_pool_size,
                # Tell pie's bootstrap which page_size sglang chose, so the
                # Rust runtime is aligned with the backend's KV geometry.
                "kv_page_size": int(config.kv_page_size),
            }

            ready_queue.put((rank, server_name, metadata))

            stop_event = threading.Event()
            _leader_loop(
                engine=engine,
                server=server,
                stop_event=stop_event,
                compute_pg=compute_pg,
                group_topology=group_topology,
                group_id=my_group_id,
            )
        else:
            ready_queue.put((rank, None, None))
            _follower_loop(
                engine=engine,
                compute_pg=compute_pg,
                leader_rank=group_topology[my_group_id][0],
                group_id=my_group_id,
                config=config,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
