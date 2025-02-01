import asyncio
import time
from typing import Any

from symphony import wrpc
from symphony.command import Transform, Yield, Terminate, Next, Read
from symphony.common import ThreadId
from symphony.engine import Engine
from symphony.model import Model, load_model

VERSION = (0, 1, 0)


class ServerConfig:
    model_name: str
    device: str

    # service engine settings
    num_blocks: int = 6000
    block_size: int = 32  # 32 tokens per block
    dist_num_vars: int = 32  # only keep top 32 token ids in the next token distribution

    max_thread_idle_time: int = 100
    max_batch_size: int = 128
    blocks_per_batch_item = 32

    verbose: bool = False

    def __init__(self,
                 model_name: str,
                 device: str,
                 num_blocks: int = 6000,
                 block_size: int = 32,
                 dist_num_vars: int = 32,
                 max_thread_idle_time: int = 100,
                 max_batch_size: int = 128,
                 blocks_per_batch_item: int = 32,
                 verbose: bool = False
                 ):
        self.model_name = model_name
        self.device = device

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dist_num_vars = dist_num_vars
        self.max_thread_idle_time = max_thread_idle_time
        self.max_batch_size = max_batch_size
        self.blocks_per_batch_item = blocks_per_batch_item
        self.verbose = verbose


class Server(wrpc.Server):
    config: ServerConfig

    model: Model
    engine: Engine

    threads: dict[wrpc.SessionId, list[ThreadId]]

    def __init__(self, config: ServerConfig):
        super().__init__()

        self.config = config
        self.threads = {}

        if config.verbose:
            print(f"Symphony ({VERSION[0]}.{VERSION[1]}.{VERSION[2]})")
            print(f"Loading model {config.model_name}...")

        self.model = load_model(config.model_name, config.device)

        if config.verbose:
            print(f"Instantiating engine...")

        self.engine = Engine(
            model=self.model,
            gpu_device=config.device,
            gpu_num_blocks=config.num_blocks,
            cpu_num_blocks=config.num_blocks,
            block_size=config.block_size,
            dist_num_vars=config.dist_num_vars,
            max_thread_idle_time=config.max_thread_idle_time,
            max_batch_size=config.max_batch_size,
            blocks_per_batch_item=config.blocks_per_batch_item,
            verbose=config.verbose
        )

    async def run(self, host: str, port: int):

        if self.config.verbose:
            print(f"Starting server on {host}:{port}...")

        await self.start(host, port)
        service_task = asyncio.create_task(self.service_loop())

        # wait for the worker to finish
        await service_task

        # await asyncio.Future()

    async def service_loop(self):

        last_time = time.time()
        idle_time = 0

        sleep_threshold = 10
        sleeping = False
        while True:

            idle = not self.engine.step()
            elapsed_time = time.time() - last_time
            last_time = time.time()

            if self.config.verbose and not idle:
                print(f"[server] tick ({elapsed_time:.2f}s)")

            if idle:
                idle_time += elapsed_time
                await asyncio.sleep(0.01)
            else:

                if sleeping:
                    print(f"[server] woke up. total idle time: {idle_time:.2f}s")

                sleeping = False
                idle_time = 0

            if idle_time > sleep_threshold:

                if self.config.verbose and not sleeping:
                    # save energy
                    print(f"[server] sleeping...")
                sleeping = True
                await asyncio.sleep(1)

            await asyncio.sleep(0)

    async def begin_session(self, s: wrpc.Session):
        self.threads[s.sid] = []
        print(f"[server] session started: {s.sid}")

    async def end_session(self, s: wrpc.Session):
        # destroy all threads owned by the session
        print(f"[server] session ended: {s.sid}")
        for tid in self.threads[s.sid]:
            await self.engine.command(tid, Terminate(), immediate=True)

    @wrpc.handle("info")
    async def metadata(self, _) -> dict:
        return {
            "model_name": self.config.model_name
        }

    @wrpc.handle("create_thread")
    async def create_thread(self, s: wrpc.Session, parent_tid: ThreadId | None = None) -> ThreadId:
        # ensure that the session owns the parent thread
        if parent_tid is not None:
            if parent_tid not in self.threads[s.sid]:
                return -1  # error code

        tid = self.engine.create_thread(parent_tid)
        self.threads[s.sid].append(tid)
        return tid

    @wrpc.handle("destroy_thread")
    async def destroy_thread(self, s: wrpc.Session, tid: ThreadId) -> bool:
        if tid not in self.threads[s.sid]:
            return False

        # terminate the thread gracefully
        await self.engine.command(tid, Terminate(), immediate=True)
        return True

    @wrpc.handle("suspend_thread")
    async def suspend_thread(self, s: wrpc.Session, tid: ThreadId) -> bool:
        if tid not in self.threads[s.sid]:
            return False

        await self.engine.command(tid, Yield(), immediate=True)
        return True

    @wrpc.handle("command")
    async def command(self, s: wrpc.Session, tid: ThreadId, command: str, **args) -> Any:
        if tid not in self.threads[s.sid]:
            return False

        match command:
            case "transform":
                if "token_ids" not in args:
                    return False
                cmd = Transform(args["token_ids"])
            case "yield":
                cmd = Yield()
            case "terminate":
                cmd = Terminate()
            case "next":
                if "n" not in args:
                    return False
                cmd = Next(args["n"])

            case "read":
                if "indices" not in args:
                    return False
                cmd = Read(args["indices"])

            case _:
                return False
        # print(f"[server] [thread {tid}] command queued: {command}", cmd)

        response = await self.engine.command(tid, cmd)
        # print(f"[server] [thread {tid}] command response: {response} for {command}")
        return response
