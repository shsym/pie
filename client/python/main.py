import asyncio
import time
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance  # Assuming these are defined elsewhere
import random

async def main():
    # Define the program name and construct the file path
    program_name = "graph_of_thought"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = PieClient(server_uri)
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
    NUM_INSTANCES = 60
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
                if event == "terminated":
                    print(f"Instance {instance.instance_id} terminated. Reason: {message}")
                    ...
                else:
                    ...
                    
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
    

    
    
async def main_swarm():
    
    # Define the program name and construct the file path
    program_name = "agent_swarm"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = PieClient(server_uri)
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
    NUM_SWARMS = 60
    SWARM_SIZE = 4#200
    instances = []
    for _ in range(NUM_SWARMS * SWARM_SIZE):
        instance = await client.launch_instance(program_hash)
        print(f"Instance {instance.instance_id} launched.")
        instances.append(instance)

    # Define a function to handle each instance's send/receive operations and measure latency
    async def handle_instance(idx, instance: Instance):
        
        group = idx // SWARM_SIZE
        role_idx = idx % SWARM_SIZE
        
        if role_idx == 0:
            role = "idea_generator"
        elif role_idx == 1:
            role = "plot_developer"
        elif role_idx == 2:
            role = "character_creator"
        elif role_idx == 3:
            role = "dialogue_writer"
        else:
            raise ValueError(f"Invalid role index: {role_idx}")
        
        try:
            instance_start = time.monotonic()

            # Send two messages to the instance
            await instance.send(role)
            await instance.send(str(group)) # max_num_outputs

            if role == "idea_generator":
                await instance.send("please tell me about this natural number:" + str(random.randint(1, 1000000)))


            # Listen for events until termination
            while True:
                event, message = await instance.recv()
                if event == "terminated":
                    #print(f"Instance {instance.instance_id} terminated. Reason: {message}.")
                    ...
                else:
                    print(f"Instance {instance.instance_id} received message: {message}")
                    
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
    tasks = [asyncio.create_task(handle_instance(i, instance)) for i, instance in enumerate(instances)]
    latencies = await asyncio.gather(*tasks)

    # Record overall end time after tasks complete
    overall_end = time.monotonic()

    num_prompts = NUM_SWARMS * SWARM_SIZE

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
    



    
    
async def main_bench_agent():
    
     # Define the program name and construct the file path
    program_name = "agent_react_bench"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = PieClient(server_uri)
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
    NUM_INSTANCES = 1
    NUM_PROMPTS = 96#200
    instances = []
    for _ in range(NUM_INSTANCES):
        instance = await client.launch_instance(program_hash)
        #print(f"Instance {instance.instance_id} launched.")
        instances.append(instance)

    # Define a function to handle each instance's send/receive operations and measure latency
    async def handle_instance(instance: Instance):
        instance_start = time.monotonic()
        try:
            await instance.send("Explain the LLM decoding process ELI5." * 10) # input_prompt
            # Send two messages to the instance
            await instance.send("32") # max_num_outputs
            await instance.send("2") # num_fc
            await instance.send(str(NUM_PROMPTS)) # num_insts
            await instance.send("true") # use_cache
            await instance.send("true") # use_asyncfc
            await instance.send("true") # use_ctx_mask

            # Listen for events until termination
            while True:
                event, message = await instance.recv()
                if event == "terminated":
                    print(f"Instance {instance.instance_id} terminated. Reason: {message}")
                    ...
                else:
                    ...
                    
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
    
    
    

    
async def main_rt_bench():
    
     # Define the program name and construct the file path
    program_name = "rt_bench"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = PieClient(server_uri)
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
    for num_instances in [10]:
        print(f"Running experiment with {num_instances} instances")

        # Repeat the experiment 10 times and collect all latencies
        all_latencies = []
        for _ in range(5):
            instances = []
            for _ in range(num_instances):
                instance = await client.launch_instance(program_hash)
                instances.append(instance)

            # Define a function to handle each instance's send/receive operations and measure latency
            async def handle_instance(instance: Instance):
                try:
                    await instance.send("ping")
                    # Listen for events until termination
                    micros = 0
                    while True:
                        try:
                            event, message = await instance.recv()
                            if event == "terminated":
                                ...
                            else:
                                micros += int(message)
                                return micros
                        except asyncio.TimeoutError:
                            return 0
                except Exception as e:
                    return None

            # Record overall start time before launching tasks
            overall_start = time.monotonic()

            # Create concurrent tasks for each instance and collect latencies
            tasks = [asyncio.create_task(handle_instance(instance)) for instance in instances]
            latencies = await asyncio.gather(*tasks)

            # Record overall end time after tasks complete
            overall_end = time.monotonic()

            # Filter out any None values from failed instances
            valid_latencies = [lat for lat in latencies if lat > 0]
            all_latencies.extend(valid_latencies)

        # Calculate the average latency over all experiments
        if all_latencies:
            # 24 is pings per instance
            average_latency = (sum(all_latencies) / 24) / len(all_latencies)
            print(f"Average latency per instance over 10 runs: {average_latency} microseconds")
        else:
            print("No valid latency measurements collected.")





async def main_count_calls():
    program_name = "beam_search"
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")
    server_uri = "ws://127.0.0.1:9123"

    client = PieClient(server_uri)
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

        # Repeat the experiment 10 times and collect all latencies

    instance = await client.launch_instance(program_hash)
    await instance.send("please tell me about this natural number:" + str(random.randint(1, 1000000)))
    await instance.send("64") # max_num_outputs
    await instance.send("1") # num_prompts

    event, message = await instance.recv()
    print(message)
    
    



async def main_startup_time():
    
     # Define the program name and construct the file path
    program_name = "rt_bench"#"text_completion"# # # 
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    # Check if the program file exists
    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    # Server URI (matching the Rust code)
    server_uri = "ws://127.0.0.1:9123"
    print(f"Using program: {program_name}")

    # Initialize and connect the client
    client = PieClient(server_uri)
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
    
    for num_instances in [641]:
        print(f"Running experiment with {num_instances} instances")

        # Repeat the experiment 10 times and collect all latencies
        all_latencies = []
        for _ in range(5):
            instances = []
            for _ in range(num_instances):
                instance = await client.launch_instance(program_hash)
                
                instances.append(instance)

            # Define a function to handle each instance's send/receive operations and measure latency
            async def handle_instance(instance: Instance):
                instance_start = time.monotonic()
                await instance.send("ping")
                await instance.recv()
                return time.monotonic() - instance_start
                

            # Create concurrent tasks for each instance and collect latencies
            tasks = [asyncio.create_task(handle_instance(instance)) for instance in instances]
            latencies = await asyncio.gather(*tasks)

            # Filter out any None values from failed instances
            valid_latencies = [lat for lat in latencies if lat > 0]
            all_latencies.extend(valid_latencies)

        # Calculate the average latency over all experiments
        if all_latencies:
            # 24 is pings per instance
            average_latency = sum(all_latencies) / len(all_latencies)
            print(f"Average latency per instance over 10 runs: {average_latency * 1000} milliseconds")
        else:
            print("No valid latency measurements collected.")


if __name__ == "__main__":
    asyncio.run(main())