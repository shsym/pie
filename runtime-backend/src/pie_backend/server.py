from __future__ import annotations
import enum
import os
import random
import signal
import struct
import sys
import queue
import threading
import time
import msgspec
import zmq
import torch
import traceback
from websockets.sync.client import connect

from .message import (
    DownloadAdapterRequest,
    EmbedImageRequest,
    ForwardPassRequest,
    HandshakeRequest,
    InitializeAdapterRequest,
    QueryRequest,
    UpdateAdapterRequest,
    UploadAdapterRequest,
)

from .runtime import Runtime
from .utils import terminate




def start_server(
    host: str,
    port: int,
    auth_token: str,
    service: Runtime,
    *,
    run_tests: bool = False,
    log_queue: object | None = None,
):
    """Spin up the backend service using the provided handler implementation.

    Args:
        host: Controller host to register with
        port: Controller port to register with
        auth_token: Authentication token for controller
        service: Runtime implementation to handle requests
        run_tests: If True, spawn embedded test client after server starts
        log_queue: Optional queue for logging to controller
    """

    unique_id = random.randint(1000000, 9999999)
    endpoint = f"ipc:///tmp/pie-model-service-{unique_id}"

    # Queues for internal communication
    work_request_queue = queue.Queue()
    response_queue = queue.Queue()
    
    # Event to signal all threads to stop
    shutdown_event = threading.Event()

    # Thread Structure (Refactored for ZMQ Safety):
    #
    # +-----------------------------+
    # |         io_thread           |  <-- Owns ZMQ socket (Reads & Writes)
    # +-----------------------------+
    #   ^  |                      ^
    #   |  | (work req queue)     | (response queue)
    #   |  v                      |
    #   v +------------------+    |
    # +------------------+   |    |
    # |  worker_thread   |---+----+
    # +------------------+

    worker_t = threading.Thread(
        target=worker_thread,
        args=(work_request_queue, response_queue, service, shutdown_event),
        daemon=True,
    )
    worker_t.start()

    # Replaces zmq_response_thread and zmq_listen_thread
    io_t = threading.Thread(
        target=io_thread,
        args=(endpoint, work_request_queue, response_queue, shutdown_event),
        daemon=True,
    )
    io_t.start()

    reg_t = threading.Thread(
        target=register_thread,
        args=(host, port, auth_token, endpoint, shutdown_event, log_queue),
        daemon=True,
    )
    reg_t.start()

    # Spawn embedded test client if requested
    test_t = None
    if run_tests:
        from .test_client import run_embedded_tests

        test_t = threading.Thread(
            target=run_embedded_tests,
            args=(endpoint,),
            daemon=True,
        )
        test_t.start()

    # Setup shutdown flag and signal handlers
    def shutdown_handler(signum, _frame):
        if not shutdown_event.is_set():
            msg = f"\nReceived signal {signum}, shutting down server..."
            log_queue.put({"level": "DEBUG", "message": msg})
            shutdown_event.set()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        # Block until shutdown signal
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        msg = "Waiting for background threads to finish..."
        log_queue.put({"level": "DEBUG", "message": msg})
            
        # 1. Stop the worker thread first so it stops processing requests/using NCCL
        if worker_t.is_alive():
            worker_t.join(timeout=5.0)
            
        # 2. Now that worker is stopped, it's safe for Rank 0 to use NCCL to stop other ranks
        try:
            service.shutdown()
        except Exception as e:
            err_msg = f"Error during service shutdown: {e}"
            log_queue.put({"level": "ERROR", "message": err_msg})

        # 3. Stop remaining threads
        if io_t.is_alive():
            io_t.join(timeout=2.0)
        if reg_t.is_alive():
            reg_t.join(timeout=2.0)
        if test_t and test_t.is_alive():
            test_t.join(timeout=2.0)
            
        final_msg = "Server shutdown complete."
        log_queue.put({"level": "DEBUG", "message": final_msg})


def register_thread(
    host: str, 
    port: int, 
    auth_token: str, 
    endpoint: str, 
    shutdown_event: threading.Event,
    log_queue: object | None = None,
) -> None:
    """Register this service with the controller."""

    # Using msgspec for consistency
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    controller_addr = f"ws://{host}:{port}"
    try:
        with connect(controller_addr) as websocket:
            auth_msg = encoder.encode({
                    "type": "internal_authenticate",
                    "corr_id": 0,
                    "token": auth_token,
                })

            if auth_msg is not None:
                websocket.send(auth_msg)

            auth_response = decoder.decode(websocket.recv())
            if not auth_response.get("successful"):
                err_msg = f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                log_queue.put({"level": "ERROR", "message": err_msg})
                shutdown_event.set()
                return

            reg_msg = encoder.encode({
                    "type": "attach_remote_service",
                    "corr_id": 0,
                    "endpoint": endpoint,
                    "service_name": f"service-{random.randint(1000000, 9999999)}",
                    "service_type": "model",
                })
            
            if reg_msg is not None:
                websocket.send(reg_msg)

            reg_response = decoder.decode(websocket.recv())
            if not reg_response.get("successful"):
                err_msg = f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                log_queue.put({"message": err_msg, "level": "ERROR"})
                shutdown_event.set()
                return

            success_msg = f"Registered with controller at {controller_addr}"
            log_queue.put({"message": success_msg, "level": "DEBUG"})

    except (ConnectionRefusedError, TimeoutError) as exc:
        err_msg = f"Failed to connect to the controller at {controller_addr}. Error: {exc}"
        log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()

    except (OSError, ValueError, RuntimeError) as exc:
        err_msg = f"An unexpected error occurred during registration: {exc}. Terminating."
        log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()





class HandlerId(enum.Enum):
    """Enumeration of handler message types."""

    HANDSHAKE = 0

    QUERY = 2
    FORWARD_PASS = 3
    EMBED_IMAGE = 4
    INITIALIZE_ADAPTER = 5
    UPDATE_ADAPTER = 6
    UPLOAD_HANDLER = 7
    DOWNLOAD_HANDLER = 8

@torch.inference_mode()
def worker_thread(
    work_request_queue: queue.Queue, 
    response_queue: queue.Queue, 
    service: Runtime,
    shutdown_event: threading.Event
) -> None:
    """Worker thread that processes incoming requests from the controller."""

    try:
        while not shutdown_event.is_set():
            try:
                item = work_request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            client_identity, corr_id_bytes, handler_id_bytes, handler_id, reqs = item

            resps = []
            match handler_id:
                case HandlerId.HANDSHAKE.value:
                    resps = service.handshake(reqs)
                case HandlerId.QUERY.value:
                    resps = service.query(reqs)
                case HandlerId.FORWARD_PASS.value:
                    resps = service.forward_pass_handler(reqs)
                case HandlerId.EMBED_IMAGE.value:
                    service.embed_image(reqs)
                    # Original behavior: No response sent for this handler
                case HandlerId.INITIALIZE_ADAPTER.value:
                    service.initialize_adapter(reqs)
                case HandlerId.UPDATE_ADAPTER.value:
                    service.update_adapter(reqs)
                case HandlerId.UPLOAD_HANDLER.value:
                    service.upload_adapter(reqs)
                case HandlerId.DOWNLOAD_HANDLER.value:
                    resps = service.download_adapter(reqs)

                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, resps)
                )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Unhandled error occurred in the worker thread: {exc}")
        traceback.print_exc()
        shutdown_event.set()


def io_thread(
    endpoint: str,
    work_queue: queue.Queue,
    response_queue: queue.Queue,
    shutdown_event: threading.Event
) -> None:
    """Thread that handles ALL ZMQ I/O (listen and response) to ensure thread safety."""

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    
    try:
        socket.bind(endpoint)
    except zmq.ZMQError as e:
        print(f"Failed to bind to {endpoint}: {e}")
        shutdown_event.set()
        return

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    msgpack_encoder = msgspec.msgpack.Encoder()

    decoders = {
        HandlerId.HANDSHAKE.value: msgspec.msgpack.Decoder(HandshakeRequest),

        HandlerId.QUERY.value: msgspec.msgpack.Decoder(QueryRequest),
        HandlerId.FORWARD_PASS.value: msgspec.msgpack.Decoder(ForwardPassRequest),
        HandlerId.EMBED_IMAGE.value: msgspec.msgpack.Decoder(EmbedImageRequest),
        HandlerId.INITIALIZE_ADAPTER.value: msgspec.msgpack.Decoder(
            InitializeAdapterRequest
        ),
        HandlerId.UPDATE_ADAPTER.value: msgspec.msgpack.Decoder(UpdateAdapterRequest),
        HandlerId.UPLOAD_HANDLER.value: msgspec.msgpack.Decoder(UploadAdapterRequest),
        HandlerId.DOWNLOAD_HANDLER.value: msgspec.msgpack.Decoder(
            DownloadAdapterRequest
        ),
    }

    try:
        while not shutdown_event.is_set():
            # 1. Process Outgoing Responses
            try:
                # Drain up to 50 responses at a time to prevent starvation
                for _ in range(50):
                    resp_item = response_queue.get_nowait()
                    client_identity, corr_id_bytes, handler_id_bytes, resps = resp_item
                    
                    response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                        msgpack_encoder.encode(r) for r in resps
                    ]
                    socket.send_multipart(response_msg)
            except queue.Empty:
                pass

            # 2. Process Incoming Requests
            # Poll with timeout to allow checking response_queue and shutdown_event
            socks = dict(poller.poll(timeout=10)) 

            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv_multipart()

                if len(message) < 3:
                    print(f"[!] Received invalid message: {message}", file=sys.stderr)
                    continue

                client_identity, corr_id_bytes, handler_id_bytes = message[:3]
                try:
                    # corr_id extracted but not used
                    _ = struct.unpack(">I", corr_id_bytes)[0]
                    handler_id = struct.unpack(">I", handler_id_bytes)[0]
                    
                    if handler_id in decoders:
                        reqs = [decoders[handler_id].decode(m) for m in message[3:]]
                    else:
                        print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)
                        continue

                except (struct.error, KeyError, msgspec.DecodeError) as exc:
                    print(
                        f"[!] Error decoding request header or payload: {exc}",
                        file=sys.stderr,
                    )
                    continue

                if not reqs:
                    print("[!] Received empty request body", file=sys.stderr)
                    continue

                # Dispatch
                work_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, handler_id, reqs)
                )

    except zmq.error.ZMQError as exc:
        if exc.errno in {zmq.ETERM, zmq.ENOTSOCK}:
            return
        print(f"Unhandled error occurred in the ZMQ I/O thread: {exc}")
        shutdown_event.set()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Unhandled error occurred in the I/O thread: {exc}")
        shutdown_event.set()
    finally:
        socket.close()
        context.term()