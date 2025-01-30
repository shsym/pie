import asyncio
import os
import blake3

from symphony import SymphonyClient

PROGRAM_PATH = "../example-apps/target/wasm32-wasip2/release/helloworld.wasm"


async def demo_sequence():
    # 1) Create and connect the client
    client = SymphonyClient("ws://127.0.0.1:9000")
    await client.connect()

    # 2) Compute BLAKE3 for the local file
    with open(PROGRAM_PATH, "rb") as f:
        wasm_bytes = f.read()
    file_hash = blake3.blake3(wasm_bytes).hexdigest()
    print("[Demo] Program file hash:", file_hash)

    # 3) Query existence
    query_resp = await client.query_existence(file_hash)
    print("[Demo] query_existence response:", query_resp)

    # 4) If not present, upload
    if not query_resp.get("exists", False):
        print("[Demo] Program not found on server, uploading now...")
        await client.upload_program(wasm_bytes, file_hash)

        # The last chunk upload typically yields a "upload_complete"
        # But let's read any subsequent messages too if needed.
        # For example, if the server sends "upload_complete" after the last chunk ack.
        # The above library code receives *one* message after each chunk,
        # which in many server flows includes the final "upload_complete" message.

    else:
        print("[Demo] Program already exists on server, skipping upload.")

    # 5) Start the program
    start_resp = await client.start_program(file_hash)
    print("[Demo] start_program response:", start_resp)

    if start_resp.get("type") == "program_launched":
        instance_id = start_resp["instance_id"]
        print(f"[Demo] Program launched with instance_id = {instance_id}")

        # 6) Send a couple of events
        await client.send_event(instance_id, "Hello from Python client - event #1")
        await client.send_event(instance_id, "Another event #2")

        # We might want to see if the server sends back any "program_event" messages.
        # Since the server can send them asynchronously, let's wait briefly:
        await asyncio.sleep(2.0)
        while not client.incoming_messages.empty():
            msg = await client.wait_for_next_message()
            print("[Demo] Received async event:", msg)

        # 7) Terminate the program
        term_resp = await client.terminate_program(instance_id)
        print("[Demo] terminate_program response:", term_resp)

    else:
        print("[Demo] Program launch failed or was not recognized.")

    # 8) Close the connection
    await client.close()


def main():
    asyncio.run(demo_sequence())


if __name__ == "__main__":
    main()
