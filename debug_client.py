import asyncio
from pie_client import PieClient, Event

async def main():
    async with PieClient("ws://127.0.0.1:8082") as client:
        print("Connected.")
        try:
             await client.authenticate("main-user")
        except:
             pass
        
        # Launch text completion inferlet
        print("Launching text-completion...")
        prompt = "Hello, world!"
        args = ["--prompt", prompt, "--max-tokens", "10"]
        instance = await client.launch_instance_from_registry("text-completion", arguments=args)
        
        # print(f"Sending prompt: {prompt}")
        # await instance.send(prompt)
        
        while True:
            event, msg = await instance.recv()
            print(f"Event: {event}, Msg: {msg}")
            if event == Event.Completed:
                break

if __name__ == "__main__":
    asyncio.run(main())
