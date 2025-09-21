"""Shared server utilities for PIE backends.

This module hosts the transport loop, registration logic, and configuration
helpers that are agnostic to the underlying compute backend. Individual
backends provide their own handler classes and model loading routines while
reusing this shared infrastructure.
"""

from __future__ import annotations

import os
import sys
import random
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type

import msgpack
import msgspec
import torch
import zmq
from platformdirs import user_cache_dir
from websockets.sync.client import connect

from config.common import ModelInfo
from message import (
    EmbedImageRequest,
    ForwardPassRequest,
    HandshakeRequest,
    InitializeAdapterRequest,
    QueryRequest,
    UpdateAdapterRequest,
)


@dataclass
class ServerConfig:
    """Lightweight view of server configuration options."""

    model: str
    host: str
    port: int
    controller_host: str
    controller_port: int
    auth_token: str | None
    cache_dir: str | None
    kv_page_size: int
    max_dist_size: int
    max_num_kv_pages: int
    max_num_embeds: int
    max_num_adapters: int
    max_adapter_rank: int
    device: str
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve the cache directory using CLI arg > env var > default."""

    return cache_dir or os.environ.get("PIE_HOME") or str(Path(user_cache_dir("pie")))


def build_config(**kwargs: Any) -> Dict[str, Any]:
    """Normalize server configuration dictionary and resolve cache directory."""

    config = dict(kwargs)
    config["cache_dir"] = resolve_cache_dir(config.get("cache_dir"))
    return config


def print_config(config: Dict[str, Any]) -> None:
    """Utility to print configuration in a consistent format."""

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("----------------------")


def start_service(
    *,
    config: Dict[str, Any],
    handler_cls: Type,
    model: Any,
    model_info: ModelInfo,
    register_with_controller: bool = True,
) -> None:
    """Spin up the backend service using the provided handler implementation."""

    if config["controller_host"] in ["127.0.0.1", "localhost"]:
        unique_id = random.randint(1000, 9999)
        endpoint = f"ipc:///tmp/pie-service-{unique_id}"
        real_endpoint = endpoint
    else:
        endpoint = f"tcp://{config['host']}:{config['port']}"
        real_endpoint = f"tcp://*:{config['port']}"

    handler = handler_cls(
        model=model,
        model_info=model_info,
        kv_page_size=config["kv_page_size"],
        max_dist_size=config["max_dist_size"],
        max_num_kv_pages=config["max_num_kv_pages"],
        max_num_embeds=config["max_num_embeds"],
        max_num_adapters=config["max_num_adapters"],
        max_adapter_rank=config["max_adapter_rank"],
        dtype=getattr(torch, config["dtype"]),
        device=config["device"],
    )

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(real_endpoint)

    threading.Thread(target=run_zmq_server, args=(socket, handler), daemon=True).start()

    if register_with_controller:
        threading.Thread(
            target=register,
            args=(config, endpoint),
            daemon=True,
        ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()
        print("Server shutdown complete.")


def register(config: Dict[str, Any], endpoint: str) -> None:
    """Register this service with the controller."""

    controller_addr = f"ws://{config['controller_host']}:{config['controller_port']}"
    try:
        with connect(controller_addr) as websocket:
            websocket.send(
                msgpack.packb(
                    {
                        "type": "authenticate",
                        "corr_id": 0,
                        "token": config["auth_token"],
                    },
                    use_bin_type=True,
                )
            )
            auth_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not auth_response.get("successful"):
                print(
                    f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            websocket.send(
                msgpack.packb(
                    {
                        "type": "attach_remote_service",
                        "corr_id": 0,
                        "endpoint": endpoint,
                        "service_name": config["model"],
                        "service_type": "model",
                    },
                    use_bin_type=True,
                )
            )
            reg_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not reg_response.get("successful"):
                print(
                    f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            print(f"Registered with controller at {controller_addr}")

    except ConnectionRefusedError:
        print(f"Failed to connect to the controller at {controller_addr}.")
        print("Please ensure the controller is running and accessible.")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"An error occurred during registration: {exc}")


def run_zmq_server(socket: zmq.Socket, handler: Any) -> None:
    """Core ZMQ service loop dispatching requests to the handler."""

    msgpack_encoder = msgspec.msgpack.Encoder()
    decoders = {
        0: msgspec.msgpack.Decoder(HandshakeRequest),
        1: msgspec.msgpack.Decoder(QueryRequest),
        2: msgspec.msgpack.Decoder(ForwardPassRequest),
        3: msgspec.msgpack.Decoder(EmbedImageRequest),
        4: msgspec.msgpack.Decoder(InitializeAdapterRequest),
        5: msgspec.msgpack.Decoder(UpdateAdapterRequest),
    }

    while True:
        try:
            message = socket.recv_multipart()

            if len(message) < 3:
                print(f"[!] Received invalid message: {message}", file=sys.stderr)
                continue

            client_identity, corr_id_bytes, handler_id_bytes = message[:3]
            try:
                corr_id = struct.unpack(">I", corr_id_bytes)[0]
                handler_id = struct.unpack(">I", handler_id_bytes)[0]
                reqs = [decoders[handler_id].decode(m) for m in message[3:]]
            except (struct.error, KeyError, msgspec.DecodeError) as exc:
                print(
                    f"[!] Error decoding request header or payload: {exc}",
                    file=sys.stderr,
                )
                continue

            if not reqs:
                print(f"[!] Received empty request body", file=sys.stderr)
                continue

            resps = []
            match handler_id:
                case 0:
                    resps = handler.handshake(reqs)
                case 1:
                    resps = handler.query(reqs)
                case 2:
                    resps = handler.forward_pass(reqs)
                case 3:
                    handler.embed_image(reqs)
                case 4:
                    handler.initialize_adapter(reqs)
                case 5:
                    handler.update_adapter(reqs)
                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                    msgpack_encoder.encode(r) for r in resps
                ]
                socket.send_multipart(response_msg)

        except zmq.ZMQError as exc:
            print(f"ZMQ Error in server loop: {exc}", file=sys.stderr)
            break


__all__ = [
    "ServerConfig",
    "build_config",
    "print_config",
    "resolve_cache_dir",
    "run_zmq_server",
    "start_service",
    "register",
]
