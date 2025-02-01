from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

import numpy as np

from symphony import wrpc
from symphony.common import ThreadId
from symphony.tokenizer import Tokenizer, load_tokenizer


class Session(wrpc.Client):
    host: str
    port: int

    tokenizer: Tokenizer

    def __init__(self, url: str, port: int):
        super().__init__()

        self.host = url
        self.port = port

    async def __aenter__(self) -> Session:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def open(self):
        await self.connect(self.host, self.port)

        info = await self.query("info", {})
        self.tokenizer = load_tokenizer(info["model_name"])

    async def close(self):
        await self.disconnect()

    async def new_thread(self) -> Thread:
        thread = Thread(self)
        await thread.create()
        return thread

    async def create_thread(self, parent_tid: ThreadId | None = None) -> ThreadId:
        tid = await self.query("create_thread", {
            "parent_tid": parent_tid
        })
        return tid

    def destroy_thread(self, tid: ThreadId) -> bool:
        return self.send("destroy_thread", {
            "tid": tid
        })

    def suspend_thread(self, tid: ThreadId) -> bool:
        return self.send("suspend_thread", {
            "tid": tid
        })

    def command(self, tid: ThreadId, command: str, **args):
        self.send("command", {
            "tid": tid,
            "command": command,
            **args
        })

    async def command_response(self, tid: ThreadId, command: str, **args) -> Any:
        return await self.query("command", {
            "tid": tid,
            "command": command,
            **args
        })


class ThreadState(Enum):
    RUNNING = 0
    INITIALIZED = 1
    TERMINATED = 2


class Thread:
    session: Session

    state: ThreadState
    tid: ThreadId

    def __init__(self, session: Session):
        self.session = session
        self.state = ThreadState.INITIALIZED

    async def create(self):

        if self.state != ThreadState.INITIALIZED:
            raise ValueError("Thread already created")

        self.tid = await self.session.create_thread()
        if self.tid == -1:
            raise ValueError("Thread creation failed")
        self.state = ThreadState.RUNNING

    def destroy(self):
        if self.state == ThreadState.RUNNING:
            self.session.destroy_thread(self.tid)
        self.state = ThreadState.TERMINATED

    def feed(self, token_ids: list[int] | int | str):
        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")

        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if isinstance(token_ids, str):
            token_ids = self.session.tokenizer.encode(token_ids, add_special_tokens=False)

        self.session.command(self.tid, "transform", token_ids=token_ids)

    async def next(self, n: int = 1) -> TokenDistribution | list[TokenDistribution]:

        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")

        response = await self.session.command_response(self.tid, "next", n=n)
        #
        # print(response["token_ids"])
        # print(response["probs"])
        # print(response["probs_rem"])

        token_ids = np.array(response["token_ids"])
        probs = np.array(response["probs"])
        probs_remaining = np.array(response["probs_rem"])

        dist = [TokenDistribution(token_ids[i], probs[i], probs_remaining[i]) for i in range(n)]

        if n == 1:
            return dist[0]
        return dist

    async def read(self, indices: list[int]) -> TokenDistribution | list[TokenDistribution]:
        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")

        response = await self.session.command_response(self.tid, "read", indices=indices)
        token_ids = np.array(response["token_ids"])
        probs = np.array(response["probs"])
        probs_remaining = np.array(response["probs_rem"])

        dist = [TokenDistribution(token_ids[i], probs[i], probs_remaining[i]) for i in range(n)]

        if len(dist) == 1:
            return dist[0]
        return dist

    def yield_(self):
        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")
        self.session.command(self.tid, "yield")

    def terminate(self):
        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")
        self.session.command(self.tid, "terminate")

    async def fork(self) -> Thread:
        if self.state != ThreadState.RUNNING:
            raise ValueError("Thread not running")
        thread = Thread(self.session)
        thread.tid = await self.session.create_thread(self.tid)
        thread.state = ThreadState.RUNNING
        return thread


class TokenDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    probs_remaining: float

    def __init__(self, token_ids: np.ndarray, probs: np.ndarray, probs_remaining: float):
        self.token_ids = token_ids
        self.probs = probs
        self.probs_remaining = probs_remaining

    def top(self) -> int:
        return int(self.token_ids[np.argmax(self.probs)])

    def top_k(self, k: int) -> list[int]:
        return self.token_ids[np.argsort(self.probs)[-k:]].tolist()
