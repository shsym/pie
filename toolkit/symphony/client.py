import asyncio
import msgpack
import websockets
import blake3

class Instance:
    """Represents a running instance of a program on the server."""
    def __init__(self, client, instance_id: str):
        self.client = client
        self.instance_id = instance_id
        self.event_queue = client.inst_event_queues.get(instance_id)
        if self.event_queue is None:
            raise Exception(f"No event queue for instance {instance_id}")

    async def send(self, message: str):
        """Send a message to the instance."""
        await self.client.signal_instance(self.instance_id, message)

    async def recv(self) -> tuple[str, str]:
        """Receive an event from the instance."""
        if self.event_queue is None:
            raise Exception("Event queue closed")
        event, msg = await self.event_queue.get()
        return event, msg

    async def terminate(self):
        """Terminate the instance."""
        await self.client.terminate_instance(self.instance_id)

class SymphonyClient:
    """A client for interacting with the Symphony WebSocket server."""
    def __init__(self, server_uri: str):
        """
        Initialize the client.

        :param server_uri: The WebSocket server URI, e.g., "ws://127.0.0.1:9000"
        """
        self.server_uri = server_uri
        self.ws = None
        self.listener_task = None
        self.corr_id_counter = 0
        self.pending_requests = {}
        self.inst_event_queues = {}

    async def connect(self):
        """Establish a WebSocket connection and start listening for messages."""
        self.ws = await websockets.connect(self.server_uri)
        print(f"[SymphonyClient] Connected to {self.server_uri}")
        self.listener_task = asyncio.create_task(self._listen_to_server())

    async def _listen_to_server(self):
        """Background task to receive and process server messages."""
        try:
            async for raw_msg in self.ws:
                if isinstance(raw_msg, bytes):
                    try:
                        message = msgpack.unpackb(raw_msg, raw=False)
                        await self._process_server_message(message)
                    except Exception as e:
                        print(f"[SymphonyClient] Failed to decode messagepack: {e}")
                else:
                    print(f"[SymphonyClient] Received non-binary message: {raw_msg}")
        except websockets.ConnectionClosed:
            print("[SymphonyClient] Connection closed")

    async def _process_server_message(self, message):
        """Process incoming server messages based on their type."""
        msg_type = message.get("type")
        if msg_type == "response":
            corr_id = message["corr_id"]
            if corr_id in self.pending_requests:
                future = self.pending_requests.pop(corr_id)
                future.set_result((message["successful"], message["result"]))
        elif msg_type == "instance_event":
            instance_id = message["instance_id"]
            event = message["event"]
            msg = message["message"]
            if instance_id in self.inst_event_queues:
                await self.inst_event_queues[instance_id].put((event, msg))
        elif msg_type == "server_event":
            # Handle server events if needed
            pass
        else:
            print(f"[SymphonyClient] Unknown message type: {msg_type}")

    async def close(self):
        """Close the WebSocket connection and clean up."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        print("[SymphonyClient] Connection closed/cancelled.")

    def _get_next_corr_id(self):
        """Generate a unique correlation ID."""
        self.corr_id_counter += 1
        return self.corr_id_counter

    async def _send_msg(self, msg: dict):
        """Serialize and send a message to the server."""
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a query to the server and await the response."""
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "query",
            "corr_id": corr_id,
            "subject": subject,
            "record": record,
        }
        await self._send_msg(msg)
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future
        return successful, result

    async def program_exists(self, program_hash: str) -> bool:
        """Check if a program exists on the server."""
        successful, result = await self.query("program_exists", program_hash)
        if successful:
            return result == "true"
        else:
            raise Exception(f"Query failed: {result}")

    async def upload_program(self, program_bytes: bytes):
        """Upload a program to the server in chunks."""
        program_hash = blake3.blake3(program_bytes).hexdigest()
        chunk_size = 256 * 1024  # 256 KiB, matching server CHUNK_SIZE_BYTES
        total_size = len(program_bytes)
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        corr_id = self._get_next_corr_id()

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, total_size)
            chunk_data = program_bytes[start:end]
            msg = {
                "type": "upload_program",
                "corr_id": corr_id,
                "program_hash": program_hash,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_data": chunk_data,
            }
            await self._send_msg(msg)

        # Wait for the response after the last chunk
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future
        if successful:
            print(f"[SymphonyClient] Program uploaded successfully: {result}")
        else:
            raise Exception(f"Program upload failed: {result}")

    async def launch_instance(self, program_hash: str) -> Instance:
        """Launch an instance of a program and return an Instance object."""
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "launch_instance",
            "corr_id": corr_id,
            "program_hash": program_hash,
        }
        await self._send_msg(msg)
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future
        if successful:
            instance_id = result
            self.inst_event_queues[instance_id] = asyncio.Queue()
            return Instance(self, instance_id)
        else:
            raise Exception(f"Failed to launch instance: {result}")

    async def launch_server_instance(self, program_hash: str, port: int):
        """Launch a server instance of a program on a specific port."""
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "launch_server_instance",
            "corr_id": corr_id,
            "port": port,
            "program_hash": program_hash,
        }
        await self._send_msg(msg)
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        successful, result = await future
        if not successful:
            raise Exception(f"Failed to launch server instance: {result}")

    async def signal_instance(self, instance_id: str, message: str):
        """Send a signal/message to a running instance."""
        msg = {
            "type": "signal_instance",
            "instance_id": instance_id,
            "message": message,
        }
        await self._send_msg(msg)

    async def terminate_instance(self, instance_id: str):
        """Terminate a running instance."""
        msg = {
            "type": "terminate_instance",
            "instance_id": instance_id,
        }
        await self._send_msg(msg)

# Example usage
async def main():
    client = SymphonyClient("ws://127.0.0.1:9000")
    await client.connect()

    # Upload a program
    with open("example.wasm", "rb") as f:
        program_bytes = f.read()
    await client.upload_program(program_bytes)

    # Check if program exists
    program_hash = blake3.blake3(program_bytes).hexdigest()
    exists = await client.program_exists(program_hash)
    print(f"Program exists: {exists}")

    # Launch an instance
    instance = await client.launch_instance(program_hash)

    # Interact with the instance
    await instance.send("Hello from client")
    event, msg = await instance.recv()
    print(f"Received event: {event}, message: {msg}")

    # Terminate the instance
    await instance.terminate()

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())