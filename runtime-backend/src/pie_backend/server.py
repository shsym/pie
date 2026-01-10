"""
Pycrust-based server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using pycrust (iceoryx2 shared memory IPC) or direct FFI calls.
"""

from __future__ import annotations

import queue
import random
import signal
import threading
from typing import Callable

import msgpack
import msgspec
from websockets.sync.client import connect

from pycrust import RpcEndpoint

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


def poll_ffi_queue(ffi_queue, service: Runtime, poll_timeout_ms: int = 100) -> None:
    """Poll the Rust FfiQueue and process requests.

    This is the new high-performance worker loop that polls a Rust queue
    directly without Python queue overhead. Should be called from a dedicated
    Python thread that owns all CUDA state.

    Args:
        ffi_queue: _pie.FfiQueue instance from start_server_with_ffi
        service: Runtime instance to dispatch calls to
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

    while True:
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


def start_ffi_worker(ffi_queue, service: Runtime) -> threading.Thread:
    """Start the FFI worker thread that polls the Rust queue.

    Returns the worker thread (already started).
    """

    def worker():
        poll_ffi_queue(ffi_queue, service)

    thread = threading.Thread(target=worker, name="pie-ffi-worker", daemon=True)
    thread.start()
    return thread


def start_server(
    host: str,
    port: int,
    auth_token: str,
    service: Runtime,
    *,
    run_tests: bool = False,
    log_queue: object | None = None,
):
    """Spin up the backend service using pycrust RPC.

    Args:
        host: Controller host to register with
        port: Controller port to register with
        auth_token: Authentication token for controller
        service: Runtime implementation to handle requests
        run_tests: If True, spawn embedded test client after server starts
        log_queue: Optional queue for logging to controller
    """

    import os

    # Use PID + time-based unique ID to avoid collisions across forked processes
    unique_id = f"{os.getpid()}-{int(random.random() * 1000000)}"
    service_name = f"pie-backend-{unique_id}"

    # Event to signal shutdown
    shutdown_event = threading.Event()

    # Create pycrust RPC endpoint
    endpoint = RpcEndpoint(service_name)

    # Register RPC methods
    endpoint.register("handshake", service.handshake_rpc)
    endpoint.register("query", service.query_rpc)
    endpoint.register("fire_batch", service.fire_batch)
    endpoint.register("embed_image", service.embed_image_rpc)
    endpoint.register("initialize_adapter", service.initialize_adapter_rpc)
    endpoint.register("update_adapter", service.update_adapter_rpc)
    endpoint.register("upload_adapter", service.upload_adapter_rpc)
    endpoint.register("download_adapter", service.download_adapter_rpc)

    # Start registration thread (connects to controller via WebSocket)
    reg_t = threading.Thread(
        target=register_thread,
        args=(host, port, auth_token, service_name, shutdown_event, log_queue),
        daemon=True,
    )
    reg_t.start()

    # Spawn embedded test client if requested
    test_t = None
    if run_tests:
        from .test_client import run_embedded_tests

        test_t = threading.Thread(
            target=run_embedded_tests,
            args=(service_name,),
            daemon=True,
        )
        test_t.start()

    # Setup shutdown signal handlers
    def shutdown_handler(signum, _frame):
        if not shutdown_event.is_set():
            msg = f"\nReceived signal {signum}, shutting down server..."
            if log_queue:
                log_queue.put({"level": "DEBUG", "message": msg})
            shutdown_event.set()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        if log_queue:
            log_queue.put(
                {
                    "level": "DEBUG",
                    "message": f"Starting pycrust endpoint: {service_name}",
                }
            )

        # This blocks until shutdown (Ctrl+C or SIGTERM)
        endpoint.listen()

    except KeyboardInterrupt:
        pass
    finally:
        if log_queue:
            log_queue.put(
                {
                    "level": "DEBUG",
                    "message": "Waiting for background threads to finish...",
                }
            )

        # Shutdown the runtime (stops worker ranks via NCCL)
        try:
            service.shutdown()
        except Exception as e:
            if log_queue:
                log_queue.put(
                    {"level": "ERROR", "message": f"Error during service shutdown: {e}"}
                )

        # Stop remaining threads
        if reg_t.is_alive():
            reg_t.join(timeout=2.0)
        if test_t and test_t.is_alive():
            test_t.join(timeout=2.0)

        if log_queue:
            log_queue.put({"level": "DEBUG", "message": "Server shutdown complete."})


def register_thread(
    host: str,
    port: int,
    auth_token: str,
    service_name: str,
    shutdown_event: threading.Event,
    log_queue: object | None = None,
) -> None:
    """Register this service with the controller.

    Args:
        host: Controller host
        port: Controller port
        auth_token: Authentication token
        service_name: Pycrust service name (used by Rust runtime to connect)
        shutdown_event: Event to signal shutdown
        log_queue: Optional queue for logging
    """
    import time

    # Wait for pycrust endpoint to start listening before registering.
    # The endpoint.listen() call happens in the main thread after this
    # thread starts, so we need a small delay to ensure it's ready.
    time.sleep(0.5)

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    controller_addr = f"ws://{host}:{port}"
    try:
        with connect(controller_addr) as websocket:
            # Authenticate
            auth_msg = encoder.encode(
                {
                    "type": "internal_authenticate",
                    "corr_id": 0,
                    "token": auth_token,
                }
            )

            if auth_msg is not None:
                websocket.send(auth_msg)

            auth_response = decoder.decode(websocket.recv())
            if not auth_response.get("successful"):
                err_msg = f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                if log_queue:
                    log_queue.put({"level": "ERROR", "message": err_msg})
                shutdown_event.set()
                return

            # Register service with pycrust service name as endpoint
            reg_msg = encoder.encode(
                {
                    "type": "attach_remote_service",
                    "corr_id": 0,
                    "endpoint": service_name,  # Pycrust service name
                    "service_name": f"service-{random.randint(1000000, 9999999)}",
                    "service_type": "model",
                }
            )

            if reg_msg is not None:
                websocket.send(reg_msg)

            reg_response = decoder.decode(websocket.recv())
            if not reg_response.get("successful"):
                err_msg = f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                if log_queue:
                    log_queue.put({"message": err_msg, "level": "ERROR"})
                shutdown_event.set()
                return

            success_msg = f"Registered with controller at {controller_addr} (service: {service_name})"
            if log_queue:
                log_queue.put({"message": success_msg, "level": "DEBUG"})

            # Keep connection alive until shutdown
            while not shutdown_event.is_set():
                shutdown_event.wait(timeout=1.0)

    except (ConnectionRefusedError, TimeoutError) as exc:
        err_msg = (
            f"Failed to connect to the controller at {controller_addr}. Error: {exc}"
        )
        if log_queue:
            log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()

    except (OSError, ValueError, RuntimeError) as exc:
        err_msg = (
            f"An unexpected error occurred during registration: {exc}. Terminating."
        )
        if log_queue:
            log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()
