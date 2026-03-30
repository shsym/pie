"""
FFI server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using direct FFI calls via PyO3.

GPU-aware batching (Python-side batching):
  When multiple fire_batch RPCs arrive while the GPU is busy, this server
  accumulates them and merges into a single execute_model call.  This
  eliminates the serial fire_batch bottleneck at high concurrency and
  achieves near-optimal GPU utilization.

  Flow:
    1. Non-blocking drain: collect all available fire_batch RPCs
    2. If GPU idle + requests pending: merge kwargs and fire
    3. GPU runs in ThreadPoolExecutor (GIL released during CUDA kernels)
    4. Main thread continues accumulating while GPU is busy
    5. When GPU finishes: split results and respond to each pycrust request_id
"""

from __future__ import annotations

import os
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import msgpack

from .batch_merger import (
    compute_batch_generate_counts,
    merge_fire_batch_kwargs,
    split_fire_batch_results,
)
from .runtime import Runtime

# Status codes for FFI dispatch (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


def poll_ffi_queue(
    ffi_queue, service: "Runtime | Any", stop_event: threading.Event, poll_timeout_ms: int = 100
) -> None:
    """Poll the Rust FfiQueue with GPU-aware fire_batch accumulation.

    Replaces the synchronous poll→handle→respond loop with an accumulation
    loop that batches multiple fire_batch RPCs into one execute_model call.

    Non-fire_batch RPCs (handshake, query, format_chat, etc.) are handled
    immediately and synchronously — they don't touch the GPU.

    Args:
        ffi_queue: _pie.FfiQueue instance from start_server_with_ffi
        service: Runtime instance to dispatch calls to
        stop_event: Event to signal shutdown
        poll_timeout_ms: How long to block waiting for requests when idle (ms)
    """
    # Method dispatch table for non-fire_batch RPCs
    methods = {
        "handshake": service.handshake_rpc,
        "query": service.query_rpc,
        "format_chat": service.format_chat_rpc,
        "embed_image": service.embed_image_rpc,
        "initialize_adapter": service.initialize_adapter_rpc,
        "update_adapter": service.update_adapter_rpc,
        "upload_adapter": service.upload_adapter_rpc,
        "download_adapter": service.download_adapter_rpc,
    }

    _ipc_timing = os.environ.get("PIE_IPC_TIMING", "")
    _batching_debug = os.environ.get("PIE_BATCH_DEBUG", "")

    # GPU-aware batching state
    pending_fire_batches: list[tuple[int, dict]] = []  # [(request_id, kwargs)]
    gpu_future = None           # Future from ThreadPoolExecutor
    prev_batch_info = None      # (generate_counts, request_ids) for response splitting
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pie-gpu")

    def _route_request(request) -> None:
        """Route a single request: accumulate fire_batch, handle others immediately."""
        nonlocal pending_fire_batches

        request_id, method, payload = request

        if _ipc_timing:
            t_received_ns = _time.clock_gettime_ns(_time.CLOCK_MONOTONIC)

        try:
            args = msgpack.unpackb(payload)

            if method == "fire_batch":
                # Accumulate fire_batch for GPU-aware batching
                pending_fire_batches.append((request_id, args))
                if _batching_debug:
                    print(f"[BATCH-ACC] Accumulated fire_batch req_id={request_id}, "
                          f"pending={len(pending_fire_batches)}, "
                          f"gpu_busy={gpu_future is not None and not gpu_future.done()}",
                          file=sys.stderr, flush=True)
                return

            # Non-fire_batch RPCs: handle immediately
            fn = methods.get(method)
            if fn is None:
                response = msgpack.packb(f"Method not found: {method}")
                ffi_queue.respond(request_id, response)
                return

            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, (list, tuple)):
                result = fn(*args)
            else:
                result = fn(args)

            response = msgpack.packb(result)
            ffi_queue.respond(request_id, response)

            if _ipc_timing and method == "format_chat":
                t_done_ns = _time.clock_gettime_ns(_time.CLOCK_MONOTONIC)
                total_ms = (t_done_ns - t_received_ns) / 1e6
                print(f"[IPC-TIMING] method={method} total={total_ms:.2f}ms",
                      file=sys.stderr, flush=True)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[FFI Queue Error] {method}: {e}\n{tb}")
            response = msgpack.packb(str(e))
            ffi_queue.respond(request_id, response)

    def _fire_pending() -> None:
        """Merge pending fire_batch requests and submit to GPU thread."""
        nonlocal pending_fire_batches, gpu_future, prev_batch_info

        batch = pending_fire_batches
        pending_fire_batches = []

        request_ids = [req_id for req_id, _ in batch]
        kwargs_list = [kw for _, kw in batch]

        if _batching_debug or _ipc_timing:
            print(f"[BATCH-FIRE] Merging {len(batch)} fire_batch RPCs, "
                  f"req_ids={request_ids}",
                  file=sys.stderr, flush=True)

        if len(kwargs_list) == 1:
            # Single batch: no merge needed, call fire_batch directly
            generate_counts = compute_batch_generate_counts(kwargs_list)
            prev_batch_info = (generate_counts, request_ids, _time.perf_counter())

            def _run_single():
                return service.fire_batch(**kwargs_list[0])

            gpu_future = executor.submit(_run_single)
        else:
            # Multiple batches: merge and fire
            generate_counts = compute_batch_generate_counts(kwargs_list)
            merged_kwargs = merge_fire_batch_kwargs(kwargs_list)
            prev_batch_info = (generate_counts, request_ids, _time.perf_counter())

            def _run_merged():
                return service.fire_batch(**merged_kwargs)

            gpu_future = executor.submit(_run_merged)

    def _handle_gpu_completion() -> None:
        """Collect GPU results, split, and respond to each pycrust request."""
        nonlocal gpu_future, prev_batch_info

        generate_counts, request_ids, t_submit = prev_batch_info

        try:
            merged_result = gpu_future.result()
        except Exception as e:
            # GPU execution failed — send error to all pending requests
            import traceback
            tb = traceback.format_exc()
            print(f"[BATCH-ERROR] fire_batch failed: {e}\n{tb}",
                  file=sys.stderr, flush=True)
            error_response = msgpack.packb(str(e))
            for req_id in request_ids:
                ffi_queue.respond(req_id, error_response)
            gpu_future = None
            prev_batch_info = None
            return

        if _ipc_timing:
            t_done = _time.perf_counter()
            total_ms = (t_done - t_submit) * 1000
            print(f"[BATCH-TIMING] gpu_total={total_ms:.1f}ms "
                  f"num_merged={len(request_ids)} "
                  f"generate_counts={generate_counts}",
                  file=sys.stderr, flush=True)

        # Split and respond
        if len(request_ids) == 1:
            # Single batch: respond directly
            response = msgpack.packb(merged_result)
            ffi_queue.respond(request_ids[0], response)
        else:
            # Multiple batches: split results
            responses = split_fire_batch_results(merged_result, generate_counts)
            for req_id, resp in zip(request_ids, responses):
                ffi_queue.respond(req_id, msgpack.packb(resp))

        gpu_future = None
        prev_batch_info = None

    # ---- Main accumulation loop ----
    try:
        while not stop_event.is_set():
            # 1. Non-blocking drain: collect all available requests
            drained = False
            while True:
                request = ffi_queue.poll_blocking(0)
                if request is None:
                    break
                _route_request(request)
                drained = True

            # 2. Check GPU completion
            if gpu_future is not None and gpu_future.done():
                _handle_gpu_completion()

            # 3. Fire pending batch if GPU is idle
            if gpu_future is None and pending_fire_batches:
                _fire_pending()
                continue  # Immediately drain again (new requests may have arrived)

            # 4. Wait for next event
            if gpu_future is not None:
                # GPU busy: short poll to check for new requests and detect
                # GPU completion quickly.  1ms balances responsiveness vs spin.
                request = ffi_queue.poll_blocking(1)
                if request is not None:
                    _route_request(request)
            elif not pending_fire_batches:
                # Fully idle: block for next request
                request = ffi_queue.poll_blocking(poll_timeout_ms)
                if request is not None:
                    _route_request(request)

    finally:
        # Ensure cleanup when thread stops
        executor.shutdown(wait=False)
        print("[FFI Worker] Shutting down Runtime...")
        service.shutdown()


def start_ffi_worker(
    ffi_queue, service: "Runtime | Any", thread_name: str = "pie-ffi-worker"
) -> tuple[threading.Thread, threading.Event]:
    """Start the FFI worker thread that polls the Rust queue.

    Args:
        ffi_queue: _pie.FfiQueue instance
        service: Runtime instance to dispatch calls to
        thread_name: Name for the worker thread (for debugging)

    Returns:
        tuple (thread, stop_event) where thread is already started.
    """
    stop_event = threading.Event()

    def worker():
        poll_ffi_queue(ffi_queue, service, stop_event)

    thread = threading.Thread(target=worker, name=thread_name, daemon=True)
    thread.start()
    return thread, stop_event
