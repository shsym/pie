import asyncio
import time
from pathlib import Path
from blake3 import blake3
from symphony import SymphonyClient, Instance  # Assuming these are defined elsewhere
import random
async def main():
    # Define the program name and construct the file path
    program_name = "agent_react"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = SymphonyClient(server_uri)
    await client.connect()

    # Read the program file and compute its hash
    with open(program_path, "rb") as f:
        program_bytes = f.read()
    program_hash = blake3(program_bytes).hexdigest()
    print(f"Program file hash: {program_hash}")

    # Check if the program exists on the server; upload if not
    if not await client.program_exists(program_hash):
        print("Program not found on server, uploading now...")
        await client.upload_program(program_bytes)
        print("Program uploaded successfully!")

    # Launch 200 instances
    NUM_INSTANCES = 200
    NUM_PROMPTS = 1#200
    instances = []
    for _ in range(NUM_INSTANCES):
        instance = await client.launch_instance(program_hash)
        #print(f"Instance {instance.instance_id} launched.")
        instances.append(instance)

    # Define a function to handle each instance's send/receive operations and measure latency
    async def handle_instance(instance: Instance):
        instance_start = time.monotonic()
        try:
            # Send two messages to the instance
            await instance.send("please tell me about this natural number:" + str(random.randint(1, 1000000)))
            await instance.send("32") # max_num_outputs
            await instance.send(str(NUM_PROMPTS)) # num_prompts

            # Listen for events until termination
            while True:
                event, message = await instance.recv()
                # if event == "terminated":
                    
                #     print(f"Instance {instance.instance_id} terminated. Reason: {message}. Latency: {latency:.4f} seconds")
                # else:
                instance_end = time.monotonic()
                latency = instance_end - instance_start
                #print(f"Instance {instance.instance_id} received message: {message}")
                return latency

        except Exception as e:
            print(f"Error handling instance {instance.instance_id}: {e}")
            return None

    # Record overall start time before launching tasks
    overall_start = time.monotonic()

    # Create concurrent tasks for each instance and collect latencies
    tasks = [asyncio.create_task(handle_instance(instance)) for instance in instances]
    latencies = await asyncio.gather(*tasks)

    # Record overall end time after tasks complete
    overall_end = time.monotonic()

    num_prompts = NUM_PROMPTS * NUM_INSTANCES

    # Filter out any None values from failed instances
    valid_latencies = [lat for lat in latencies if lat is not None]
    if valid_latencies:
        average_latency = sum(valid_latencies) / num_prompts
        print(f"Average latency per instance: {average_latency:.4f} seconds")
    else:
        print("No valid latency measurements collected.")

    print(f"Total time: {overall_end - overall_start:.4f} seconds")
    # Calculate throughput (instances completed per second)
    total_time = overall_end - overall_start
    throughput = num_prompts / total_time if total_time > 0 else 0
    print(f"Overall throughput: {throughput:.2f} instances per second")

    # Close the client connection
    #await client.close()
    print("Client connection closed.")
    
    # exit the program
    exit(0)

if __name__ == "__main__":
    asyncio.run(main())