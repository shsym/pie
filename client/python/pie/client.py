import asyncio
import msgpack
import websockets
import blake3
import subprocess
import tempfile
from pathlib import Path
import uuid

class Instance:
    """Represents a running instance of a program on the server."""

    def __init__(self, client, instance_id: str):
        self.client = client
        self.instance_id = instance_id
        # Use .get() to be safe, though an exception is better if it must exist
        self.event_queue = self.client.inst_event_queues.get(instance_id)
        if self.event_queue is None:
            raise Exception(f"Internal error: No event queue for instance {instance_id}")

    async def send(self, message: str):
        """Send a message to the instance."""
        await self.client.signal_instance(self.instance_id, message)

    async def recv(self) -> tuple[str, str]:
        """Receive an event from the instance. Blocks until an event is available."""
        if self.event_queue is None:
            raise Exception("Event queue is not available for this instance.")
        event, msg = await self.event_queue.get()
        return event, msg

    async def terminate(self):
        """Request termination of the instance."""
        await self.client.terminate_instance(self.instance_id)


class PieClient:
    """
    An asynchronous client for interacting with the Pie WebSocket server.
    This client is designed to be used as an async context manager.
    """

    def __init__(self, server_uri: str):
        """
        Initialize the client.
        :param server_uri: The WebSocket server URI (e.g., "ws://127.0.0.1:8080").
        """
        self.server_uri = server_uri
        self.ws = None
        self.listener_task = None
        self.corr_id_counter = 0
        self.pending_requests = {}
        self.inst_event_queues = {}

    async def __aenter__(self):
        """Enter the async context, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, closing the connection cleanly."""
        await self.close()

    async def connect(self):
        """Establish a WebSocket connection and start the background listener."""
        self.ws = await websockets.connect(self.server_uri)
        print(f"[PieClient] Connected to {self.server_uri}")
        self.listener_task = asyncio.create_task(self._listen_to_server())

    async def _listen_to_server(self):
        """Background task to receive and process all incoming server messages."""
        try:
            async for raw_msg in self.ws:
                if isinstance(raw_msg, bytes):
                    try:
                        message = msgpack.unpackb(raw_msg, raw=False)
                        await self._process_server_message(message)
                    except msgpack.UnpackException as e:
                        print(f"[PieClient] Failed to decode messagepack: {e}")
                else:
                    print(f"[PieClient] Received unexpected non-binary message: {raw_msg}")
        except websockets.ConnectionClosedOK:
            print("[PieClient] Connection closed normally.")
        except websockets.ConnectionClosedError as e:
            print(f"[PieClient] Connection closed with error: {e}")
        except Exception as e:
            print(f"[PieClient] Listener task encountered an unexpected error: {e}")

    async def _process_server_message(self, message: dict):
        """Route incoming server messages based on their type."""
        msg_type = message.get("type")
        if msg_type == "response":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_requests:
                future = self.pending_requests.pop(corr_id)
                future.set_result((message.get("successful"), message.get("result")))
        elif msg_type == "instance_event":
            instance_id = message.get("instance_id")
            if instance_id in self.inst_event_queues:
                event = message.get("event")
                msg = message.get("message")
                await self.inst_event_queues[instance_id].put((event, msg))
        elif msg_type == "server_event":
            print(f"[PieClient] Received server event: {message.get('message')}")
        else:
            print(f"[PieClient] Received unknown message type: {msg_type}")

    async def close(self):
        """Gracefully close the WebSocket connection and shut down background tasks."""
        if self.ws and not self.ws.closed:
            await self.ws.close()

        if self.listener_task:
            # Wait for the listener task to finish after the connection is closed.
            try:
                await asyncio.wait_for(self.listener_task, timeout=2.0)
            except asyncio.TimeoutError:
                print("[PieClient] Timeout waiting for listener task to close, cancelling.")
                self.listener_task.cancel()
            except asyncio.CancelledError:
                pass  # Task was already cancelled.

        print("[PieClient] Client has been shut down.")

    def _get_next_corr_id(self):
        """Generate a unique correlation ID for a request."""
        self.corr_id_counter += 1
        return self.corr_id_counter

    async def _send_msg_and_wait(self, msg: dict) -> tuple[bool, str]:
        """Send a message that expects a response and wait for it."""
        corr_id = self._get_next_corr_id()
        msg["corr_id"] = corr_id

        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future

        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

        return await future

    async def authenticate(self, token: str) -> tuple[bool, str]:
        """Authenticate the client with the server using a token."""
        msg = {"type": "authenticate", "token": token}
        successful, result = await self._send_msg_and_wait(msg)

        if successful:
            print("[PieClient] Authenticated successfully.")
        else:
            print(f"[PieClient] Authentication failed: {result}")
        return successful, result

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a generic query to the server."""
        msg = {"type": "query", "subject": subject, "record": record}
        return await self._send_msg_and_wait(msg)


    async def program_exists(self, program_hash: str) -> bool:
        """Check if a program with the given hash exists on the server."""
        successful, result = await self.query("program_exists", program_hash)
        if successful:
            return result == "true"
        else:
            raise Exception(f"Query for program_exists failed: {result}")

    async def upload_program(self, program_bytes: bytes):
        """Upload a program to the server in chunks."""
        program_hash = blake3.blake3(program_bytes).hexdigest()
        chunk_size = 256 * 1024  # 256 KiB, must match server
        total_size = len(program_bytes)
        total_chunks = (total_size + chunk_size - 1) // chunk_size

        # A single correlation ID is used for the entire upload sequence.
        corr_id = self._get_next_corr_id()

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, total_size)
            msg = {
                "type": "upload_program",
                "corr_id": corr_id,
                "program_hash": program_hash,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_data": program_bytes[start:end],
            }
            encoded = msgpack.packb(msg, use_bin_type=True)
            await self.ws.send(encoded)

        # After sending the last chunk, wait for the final response.
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future
        if successful:
            print(f"[PieClient] Program uploaded successfully: {result}")
        else:
            raise Exception(f"Program upload failed: {result}")

    async def launch_instance(self, program_hash: str, arguments: list[str] = None) -> Instance:
        """Launch an instance of a program."""
        msg = {
            "type": "launch_instance",
            "program_hash": program_hash,
            "arguments": arguments if arguments is not None else [],
        }
        successful, result = await self._send_msg_and_wait(msg)
        if successful:
            instance_id = result
            self.inst_event_queues[instance_id] = asyncio.Queue()
            return Instance(self, instance_id)
        else:
            raise Exception(f"Failed to launch instance: {result}")

    async def launch_server_instance(self, program_hash: str, port: int, arguments: list[str] = None):
        """Launch a server instance of a program on a specific port."""
        msg = {
            "type": "launch_server_instance",
            "port": port,
            "program_hash": program_hash,
            "arguments": arguments if arguments is not None else [],
        }
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Failed to launch server instance: {result}")

    async def signal_instance(self, instance_id: str, message: str):
        """Send a signal/message to a running instance (fire-and-forget)."""
        msg = {"type": "signal_instance", "instance_id": instance_id, "message": message}
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

    async def terminate_instance(self, instance_id: str):
        """Request the server to terminate a running instance (fire-and-forget)."""
        msg = {"type": "terminate_instance", "instance_id": instance_id}
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)




def _compile_rust_sync(rust_code: str, cargo_toml_content: str, package_name: str) -> bytes:
    """
    [Internal Synchronous Helper] Compiles rust code in a temporary directory.
    This function is blocking and should be run in a separate thread.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        src_path = project_path / "src"
        src_path.mkdir()

        # Write the configuration and source files
        (project_path / "Cargo.toml").write_text(cargo_toml_content)
        (src_path / "lib.rs").write_text(rust_code)

        # Define and run the compilation command
        command = ["cargo", "build", "--target", "wasm32-wasip2", "--release"]

        try:
            print(f"🚀 Compiling crate '{package_name}'...")
            # This is a blocking call
            subprocess.run(
                command,
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Error: `cargo` command not found. Ensure Rust is installed and in your PATH. "
                "You also need the wasm32-wasip2 target. Install it with: `rustup target add wasm32-wasip2`"
            )
        except subprocess.CalledProcessError as e:
            error_message = f"❌ Rust compilation failed.\n\n--- COMPILER OUTPUT ---\n{e.stderr}"
            raise RuntimeError(error_message)

        # Locate and read the compiled .wasm file
        wasm_file_name = f"{package_name.replace('-', '_')}.wasm"
        wasm_path = project_path / "target" / "wasm32-wasip2" / "release" / wasm_file_name

        if not wasm_path.exists():
            raise RuntimeError(f"Build succeeded but could not find WASM file at {wasm_path}")

        print("✅ Compilation successful! Reading WASM binary.")
        return wasm_path.read_bytes()

async def compile_program(source: str|Path, dependencies: list[str]) -> bytes:
    """
    Compiles Rust source into a WASM binary and returns the bytes.

    This function dynamically creates a temporary Cargo project, compiles the code
    with the specified dependencies to the `wasm32-wasip2` target, reads the
    resulting .wasm file into memory, and cleans up all intermediate files.

    Args:
        source: A string of Rust code or a Path object pointing to a .rs file.
        dependencies: A list of dependency strings, e.g., ["tokio = \"1\"", "serde = \"1.0\""].

    Returns:
        The compiled WASM binary as a bytes object.

    Raises:
        FileNotFoundError: If the source path does not exist.
        RuntimeError: If compilation fails or required tools are missing.
    """
    # 1. Resolve source code content from string or file path
    rust_code: str
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith('.rs')):
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found at: {source_path}")
        rust_code = source_path.read_text()
    else:
        rust_code = source

    # 2. Prepare the Cargo.toml configuration
    package_name = f"pie-temp-crate-{uuid.uuid4().hex[:8]}"
    deps_str = "\n".join(dependencies)
    cargo_toml_content = f"""
[package]
name = "{package_name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
{deps_str}
"""

    # 3. Run the blocking compilation in a thread pool to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    wasm_bytes = await loop.run_in_executor(
        None,  # Use the default thread pool executor
        _compile_rust_sync,
        rust_code,
        cargo_toml_content,
        package_name
    )
    return wasm_bytes