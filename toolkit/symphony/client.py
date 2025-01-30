import asyncio
import msgpack
import websockets
import blake3

CHUNK_SIZE = 64 * 1024


class SymphonyClient:
    """
    A client library for interacting with the Symphony WebSocket server.
    """

    def __init__(self, server_uri: str):
        """
        :param server_uri: e.g. "ws://127.0.0.1:9000"
        """
        self.server_uri = server_uri
        self.ws = None
        self.listener_task = None

        # We store *all* incoming messages from server in a queue.
        # If you want a "wait for a single response" pattern, you can pop from this queue.
        self._incoming_messages = asyncio.Queue()

    async def connect(self):
        """Establish the WebSocket connection and start listening."""
        self.ws = await websockets.connect(self.server_uri)
        print("[SymphonyClient] Connected to", self.server_uri)

        # Start a background listener to receive server messages asynchronously.
        self.listener_task = asyncio.create_task(self._listen_to_server())

    async def _listen_to_server(self):
        """Background task that receives messages from the server and pushes them to a queue."""
        try:
            async for raw_msg in self.ws:
                # raw_msg is binary if the server uses msgpack. If text, the server might
                # have returned an error or different frame. We'll handle both cases.
                if isinstance(raw_msg, bytes):
                    try:
                        message = msgpack.unpackb(raw_msg, raw=False)
                    except Exception as e:
                        print("[SymphonyClient] Failed to decode messagepack:", e)
                        continue
                else:
                    print("[SymphonyClient] Received non-binary message:", raw_msg)
                    continue

                # Put it on the queue so user code can process it
                await self._incoming_messages.put(message)

        except websockets.ConnectionClosed:
            print("[SymphonyClient] Connection closed")

    async def close(self):
        """Close the WebSocket and cancel the background listener."""
        if self.ws and not self.ws.closed:
            await self.ws.close()

        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass

        print("[SymphonyClient] Connection closed/cancelled.")

    ### Utility methods for sending/receiving ###

    async def _send_msg(self, msg: dict):
        """Serialize dict to MessagePack and send via WebSocket."""
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

    async def wait_for_next_message(self) -> dict:
        """
        Wait for the next incoming server message.
        Returns a dictionary decoded from MessagePack.
        """
        return await self._incoming_messages.get()

    ### High-level actions ###

    async def query_existence(self, program_hash: str) -> dict:
        """
        Sends `query_existence` to the server and returns the first server reply.
        If the server might send multiple messages, you can do a more advanced approach
        (like reading from the queue until you find the matching 'type').
        """
        request_msg = {
            "type": "query_existence",
            "hash": program_hash,
        }
        await self._send_msg(request_msg)
        # Return the next server message
        response = await self.wait_for_next_message()
        return response

    async def upload_program(self, wasm_bytes: bytes, program_hash: str):
        """
        Upload the given wasm_bytes to the server in chunked form.
        Will print any server responses in real-time.

        :param wasm_bytes: The raw content of the .wasm file
        :param program_hash: The computed BLAKE3 hash for the file
        """
        total_size = len(wasm_bytes)
        total_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_index in range(total_chunks):
            start = chunk_index * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total_size)
            chunk_data = wasm_bytes[start:end]

            upload_msg = {
                "type": "upload_program",
                "hash": program_hash,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_data": chunk_data,
            }
            await self._send_msg(upload_msg)

            # Wait for server's ack each time
            resp = await self.wait_for_next_message()
            print(f"[SymphonyClient] Upload chunk {chunk_index}/{total_chunks} response:", resp)

    async def start_program(self, program_hash: str) -> dict:
        """
        Sends `start_program` to the server for the given hash.
        Returns the immediate server response, e.g. "program_launched".
        """
        request_msg = {
            "type": "start_program",
            "hash": program_hash
        }
        await self._send_msg(request_msg)
        resp = await self.wait_for_next_message()
        return resp

    async def send_event(self, instance_id: str, event_data: str):
        """
        Send an event to a running instance.
        """
        request_msg = {
            "type": "send_event",
            "instance_id": instance_id,
            "event_data": event_data
        }
        await self._send_msg(request_msg)
        # Depending on your flow, the server might or might not respond here.
        # If you want to see if the server responds, do:
        # resp = await self.wait_for_next_message()
        # return resp

    async def terminate_program(self, instance_id: str) -> dict:
        """
        Request program termination.
        Returns the server response.
        """
        request_msg = {
            "type": "terminate_program",
            "instance_id": instance_id
        }
        await self._send_msg(request_msg)
        resp = await self.wait_for_next_message()
        return resp

    @property
    def incoming_messages(self):
        return self._incoming_messages