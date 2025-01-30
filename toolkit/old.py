import asyncio
import websockets
import msgpack
import blake3

PROGRAM_CACHE_DIR = "../example-apps/target/wasm32-wasip2/release/"

async def main():
    uri = "ws://127.0.0.1:9000"
    async with websockets.connect(uri) as ws:
        print("Connected to Symphony server.")

        # 1) Query existence of a sample hash
        query_msg = {
            "type": "query_existence",
            "hash": "fd4cf7193c818fc5fc464d441406ca29182c9e86966ed5c54a25bce720d14a44"
        }
        await ws.send(msgpack.packb(query_msg, use_bin_type=True))
        response_data = await ws.recv()  # This is binary
        response = msgpack.unpackb(response_data, raw=False)
        print("query_existence response:", response)

        # 2) Upload a local .wasm (renamed as “program”) file in chunks
        program_path = f"{PROGRAM_CACHE_DIR}helloworld.wasm"
        with open(program_path, "rb") as f:
            program_bytes = f.read()

        # Compute BLAKE3
        file_hash = blake3.blake3(program_bytes).hexdigest()
        print("Program BLAKE3:", file_hash)

        chunk_size = 64 * 1024
        total_chunks = (len(program_bytes) + chunk_size - 1) // chunk_size

        for i in range(total_chunks):
            chunk = program_bytes[i*chunk_size : (i+1)*chunk_size]
            msg = {
                "type": "upload_program",
                "hash": file_hash,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_data": chunk,  # raw bytes
            }
            await ws.send(msgpack.packb(msg, use_bin_type=True))

            # Read server ack
            resp_data = await ws.recv()
            resp = msgpack.unpackb(resp_data, raw=False)
            print("Upload chunk response:", resp)

        # Read server ack
        resp_data = await ws.recv()
        resp = msgpack.unpackb(resp_data, raw=False)
        print("Final upload response:", resp)
        # 3) Start the program

        start_msg = {
            "type": "start_program",
            "hash": file_hash,
            "configuration": {},  # could contain CPU/memory limits
        }
        await ws.send(msgpack.packb(start_msg, use_bin_type=True))
        start_resp_data = await ws.recv()
        start_resp = msgpack.unpackb(start_resp_data, raw=False)
        print("Start response:", start_resp)
        return;

        print("Start response:", start_resp)

        if start_resp.get("type") == "program_launched":
            instance_id = start_resp.get("instance_id")
            # 4) Send an event
            event_msg = {
                "type": "send_event",
                "hash": file_hash,
                "instance_id": instance_id,
                "event_data": {"my_event": "hello program!"}
            }
            await ws.send(msgpack.packb(event_msg, use_bin_type=True))
            event_resp_data = await ws.recv()
            event_resp = msgpack.unpackb(event_resp_data, raw=False)
            print("Event response:", event_resp)

            # 5) Terminate the program
            term_msg = {
                "type": "terminate_program",
                "hash": file_hash,
                "instance_id": instance_id
            }
            await ws.send(msgpack.packb(term_msg, use_bin_type=True))
            term_resp_data = await ws.recv()
            term_resp = msgpack.unpackb(term_resp_data, raw=False)
            print("Termination response:", term_resp)

asyncio.run(main())