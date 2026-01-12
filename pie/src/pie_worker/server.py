"""
FFI server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using direct FFI calls via PyO3.
"""

from __future__ import annotations

import queue
import threading

import msgpack

from .runtime import Runtime

# Status codes for FFI dispatch (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


class ThreadedDispatcher:
    """FFI dispatcher that processes all requests on a dedicated Python thread.

    CRITICAL: CUDA requires all operations happen on the SAME thread.
    A lock alone is not sufficient - we need a dedicated worker thread.
    """

    def __init__(self, service: Runtime):
        self.service = service
        self.request_queue: queue.Queue = queue.Queue()
        self.running = True

        # Method dispatch table
        self.methods = {
            "handshake": service.handshake_rpc,
            "query": service.query_rpc,
            "fire_batch": service.fire_batch,
            "embed_image": service.embed_image_rpc,
            "initialize_adapter": service.initialize_adapter_rpc,
            "update_adapter": service.update_adapter_rpc,
            "upload_adapter": service.upload_adapter_rpc,
            "download_adapter": service.download_adapter_rpc,
        }

        # Start worker thread - this thread owns all CUDA state
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="pie-ffi-worker", daemon=True
        )
        self.worker_thread.start()

    def _worker_loop(self):
        """Process requests on this dedicated thread (owns CUDA state)."""
        import time

        # Timing accumulators (in seconds)
        call_count = 0
        total_queue_wait = 0.0
        total_unpack = 0.0
        total_method_call = 0.0
        total_pack = 0.0

        while self.running:
            try:
                request = self.request_queue.get(timeout=1.0)
                if request is None:
                    break

                method, payload, result_holder = request

                # Measure queue wait time
                dequeue_time = time.perf_counter()
                queue_wait = dequeue_time - result_holder.get(
                    "enqueue_time", dequeue_time
                )

                try:
                    # Measure unpack time
                    unpack_start = time.perf_counter()
                    args = msgpack.unpackb(payload)
                    unpack_time = time.perf_counter() - unpack_start

                    # Measure method call time (CUDA work)
                    call_start = time.perf_counter()
                    fn = self.methods.get(method)
                    if fn is None:
                        result_holder["status"] = STATUS_METHOD_NOT_FOUND
                        result_holder["response"] = msgpack.packb(
                            f"Method not found: {method}"
                        )
                        result_holder["event"].set()
                        continue

                    if isinstance(args, dict):
                        result = fn(**args)
                    elif isinstance(args, (list, tuple)):
                        result = fn(*args)
                    else:
                        result = fn(args)
                    call_time = time.perf_counter() - call_start

                    # Measure pack time
                    pack_start = time.perf_counter()
                    response = msgpack.packb(result)
                    pack_time = time.perf_counter() - pack_start

                    result_holder["status"] = STATUS_OK
                    result_holder["response"] = response
                    result_holder["event"].set()

                    # Accumulate timing (only for fire_batch which is the hot path)
                    if method == "fire_batch":
                        call_count += 1
                        total_queue_wait += queue_wait
                        total_unpack += unpack_time
                        total_method_call += call_time
                        total_pack += pack_time

                        # Print every 100 calls
                        if call_count % 100 == 0:
                            print(
                                f"[FFI Timing] {call_count} calls avg: "
                                f"queue={total_queue_wait/call_count*1000:.2f}ms, "
                                f"unpack={total_unpack/call_count*1000:.2f}ms, "
                                f"method={total_method_call/call_count*1000:.2f}ms, "
                                f"pack={total_pack/call_count*1000:.2f}ms"
                            )

                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    print(f"[FFI Worker Error] {method}: {e}\n{tb}")
                    result_holder["status"] = STATUS_INTERNAL_ERROR
                    result_holder["response"] = msgpack.packb(str(e))
                    result_holder["event"].set()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FFI Worker] Error: {e}")

    def __call__(self, method: str, payload: bytes) -> tuple[int, bytes]:
        """Entry point called from Rust FFI."""
        import time

        # Time from Rust call to queue put
        enqueue_start = time.perf_counter()
        event = threading.Event()
        result_holder = {
            "status": 0,
            "response": b"",
            "event": event,
            "enqueue_time": enqueue_start,  # Track when request was queued
            "timing": {},  # Will be filled by worker
        }
        self.request_queue.put((method, payload, result_holder))

        # Wait for completion
        event.wait()

        return (result_holder["status"], result_holder["response"])

    def _dispatch(self, method: str, payload: bytes) -> tuple[int, bytes]:
        """Execute the actual method call."""
        fn = self.methods.get(method)
        if fn is None:
            return (
                STATUS_METHOD_NOT_FOUND,
                msgpack.packb(f"Method not found: {method}"),
            )

        try:
            args = msgpack.unpackb(payload)

            # Handle different argument formats
            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, (list, tuple)):
                result = fn(*args)
            else:
                result = fn(args)

            return (STATUS_OK, msgpack.packb(result))

        except TypeError as e:
            return (STATUS_INVALID_PARAMS, msgpack.packb(str(e)))
        except Exception as e:
            return (STATUS_INTERNAL_ERROR, msgpack.packb(str(e)))

    def shutdown(self):
        """Signal the worker thread to stop."""
        self.running = False
        self.request_queue.put(None)
        self.worker_thread.join(timeout=5.0)


def create_dispatcher(service: Runtime) -> ThreadedDispatcher:
    """Create a threaded dispatcher for FFI calls from Rust.

    This creates a ThreadedDispatcher that handles all method calls on a
    dedicated Python thread, ensuring CUDA/PyTorch thread safety.

    Args:
        service: Runtime instance to dispatch calls to

    Returns:
        ThreadedDispatcher instance (callable)
    """
    return ThreadedDispatcher(service)


def poll_ffi_queue(
    ffi_queue, service: Runtime, stop_event: threading.Event, poll_timeout_ms: int = 100
) -> None:
    """Poll the Rust FfiQueue and process requests.

    This is the new high-performance worker loop that polls a Rust queue
    directly without Python queue overhead. Should be called from a dedicated
    Python thread that owns all CUDA state.

    Args:
        ffi_queue: _pie.FfiQueue instance from start_server_with_ffi
        service: Runtime instance to dispatch calls to
        stop_event: Event to signal shutdown
        poll_timeout_ms: How long to block waiting for requests (ms)
    """
    # Method dispatch table
    methods = {
        "handshake": service.handshake_rpc,
        "query": service.query_rpc,
        "fire_batch": service.fire_batch,
        "embed_image": service.embed_image_rpc,
        "initialize_adapter": service.initialize_adapter_rpc,
        "update_adapter": service.update_adapter_rpc,
        "upload_adapter": service.upload_adapter_rpc,
        "download_adapter": service.download_adapter_rpc,
    }

    while not stop_event.is_set():
        # Poll the Rust queue (releases GIL while waiting)
        request = ffi_queue.poll_blocking(poll_timeout_ms)
        if request is None:
            continue  # Timeout, try again

        request_id, method, payload = request

        try:
            # Unpack args
            args = msgpack.unpackb(payload)

            # Get handler
            fn = methods.get(method)
            if fn is None:
                response = msgpack.packb(f"Method not found: {method}")
                ffi_queue.respond(request_id, response)
                continue

            # Call handler
            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, (list, tuple)):
                result = fn(*args)
            else:
                result = fn(args)

            # Pack and respond
            response = msgpack.packb(result)
            ffi_queue.respond(request_id, response)

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"[FFI Queue Error] {method}: {e}\n{tb}")
            response = msgpack.packb(str(e))
            ffi_queue.respond(request_id, response)


def start_ffi_worker(
    ffi_queue, service: Runtime
) -> tuple[threading.Thread, threading.Event]:
    """Start the FFI worker thread that polls the Rust queue.

    Returns:
        tuple (thread, stop_event) where thread is already started.
    """
    stop_event = threading.Event()

    def worker():
        poll_ffi_queue(ffi_queue, service, stop_event)

    thread = threading.Thread(target=worker, name="pie-ffi-worker", daemon=True)
    thread.start()
    return thread, stop_event
