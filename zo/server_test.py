import asyncio
import time
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance  # Assuming these are defined elsewhere
import random

async def main():

    # Server URI (matching the Rust code)
    server_uri = "ws://gh071:8080"

    # Initialize and connect the client
    client = PieClient(server_uri)
    await client.connect()

    print("Client connected")


if __name__ == "__main__":
    asyncio.run(main())